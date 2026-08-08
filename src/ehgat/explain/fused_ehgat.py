"""Physics-fused E-HGATv2: a tropical Max-Plus DP makespan head + exact additive energy.

The standard ehgat.surrogate.ehgatv2.EHGATv2 pools node embeddings into a smooth
MLP that regresses (C_max, E). That MLP smears gradients across all features, so
input-attribution of C_max is not faithful to the schedule's true critical path.

FusedEHGATv2 keeps the frozen heterogeneous message-passing core but replaces
the makespan MLP with a physics-anchored, natively differentiable head:

1. Tropical projection heads map node embeddings to local physical attributes:
   - an edge/leg head FC(h_u || h_v) -> (empty_t, loaded_t, empty_e, loaded_e) per
     task (indexed by its single incoming AGV arc), and
   - a node delay head FC(h_v) -> d_v per task (the quay-crane handling tau).
2. These local attributes are composed into C_max by the exact max-plus DP
   (ehgat.explain.event_dag.assemble_event_dag + tropical longest path), so
   dC_max/d(local attribute) is the exact binary critical path -- faithful by
   construction, with no Jacobian smearing.
3. Energy is exact-additive: E = sum_j (empty_e_j + loaded_e_j) over the predicted
   leg energies -- a strictly linear head, so dE/d(leg energy) = 1.

Identifiability. Because the raw max-plus output only constrains the active critical
path, the local attributes are made identifiable by direct physics anchoring: training
(ehgat.explain.train_fused) supervises empty_t/loaded_t/empty_e/
loaded_e/d_v against their exact physical values, guaranteeing dense gradients on
every leg even when it is off the current critical path.

This is built non-destructively -- it wraps a trained core and the original scalar head
remains fully functional via core.forward / core.predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from ehgat.environment.physics import SPEED_TABLE, SpeedLevel
from ehgat.explain.event_dag import (
    EventDag,
    assemble_coupled_event_dag,
    assemble_event_dag,
    extract_precedence,
)
from ehgat.explain.tropical_dp import tropical_longest_path
from ehgat.surrogate.ehgatv2 import EDGE_DIM, EHGATv2
from ehgat.surrogate.graph import AGV_EDGE, NODE_TYPE, QC_EDGE

__all__ = ["FusedEHGATv2", "FusedPrediction"]

_N_LEG_TIMES = 2  # (empty_t, loaded_t) -- the GNN-predicted transport overheads
_T_TRAVEL_COL = 0  # EDGE_FEATURES index of Travel_Time (= empty_t + loaded_t)
_E_EMPTY_COL = 1  # EDGE_FEATURES index of Empty_Energy
_E_LOADED_COL = 2  # EDGE_FEATURES index of Loaded_Energy

# Physics-unrolled coupling: each AGV travel leg supplies the wait head with a contention
# vector read off the previous iterate's tentative timing -- own power, concurrent power
# demand over the leg's active interval, budget excess, and number of overlapping legs.
# Two legs per task (empty, loaded) give 2 * 4 features. The wait is a timing fixed-point
# and is therefore unavailable to a single static pass; see the unrolled forward.
_CONT_PER_LEG = 6
_CONT_DIM = 2 * _CONT_PER_LEG
_CONT_EPS = 1e-6
_CONT_CNT_SCALE = 4.0  # normaliser for the peak concurrent-leg count (bounded ~ #AGVs)
_CONT_CLAMP = 6.0      # cap on budget-normalised contention features (anti-blowup safeguard)


def _peak_contention(starts: Tensor, ends: Tensor, powers: Tensor, p_max: float) -> Tensor:
    """Bounded per-leg contention features from leg intervals + powers.

    Operates on the last axis as the leg axis M = 2n (empty legs then loaded legs),
    with an optional leading batch axis, covering both the per-graph ([2N]) and batched
    ([S, 2n]) paths. Six bounded features per leg describe the power pressure and the
    priority/queue structure by which the simulator resolves it:
      0 own power, 1 peak concurrent-others power, 2 budget excess, 3 peak concurrent count
      (bounded by fleet concurrency and by the budget, hence stable in N),
      4 start-time rank (fraction of legs starting earlier, the leg's queue position),
      5 time-to-next-budget-free (soonest completion of a concurrent leg after this leg's
        start, normalised by mean leg duration), which sets the wait length.

    Returns [..., n, 12] = per task [empty(6), loaded(6)].
    """
    finite = p_max != float("inf")
    scale = p_max if finite else 1.0
    dt = powers.dtype
    neg = torch.finfo(dt).min
    pos = torch.finfo(dt).max
    # Event times are leg starts. act[..., k, j] = leg j running at event time t_k = starts[k].
    t_k = starts.unsqueeze(-1)
    s_j = starts.unsqueeze(-2)
    e_j = ends.unsqueeze(-2)
    act = (s_j <= t_k + _CONT_EPS) & (t_k < e_j - _CONT_EPS)
    actf = act.to(dt)
    inst = (actf * powers.unsqueeze(-2)).sum(-1)   # [..., k] total power at each event
    cnt = actf.sum(-1)                             # [..., k] concurrent legs at each event
    # Event k inside leg i's interval [s_i, e_i): inint[..., i, k].
    ts = starts.unsqueeze(-2)
    s_i = starts.unsqueeze(-1)
    e_i = ends.unsqueeze(-1)
    inint = (s_i <= ts + _CONT_EPS) & (ts < e_i - _CONT_EPS)
    peak_power = torch.where(inint, inst.unsqueeze(-2), torch.full_like(inint, neg, dtype=dt)).amax(-1)
    peak_cnt = torch.where(inint, cnt.unsqueeze(-2), torch.zeros_like(inint, dtype=dt)).amax(-1)
    conc_others = (peak_power - powers).clamp_min(0.0)
    excess = (peak_power - p_max).clamp_min(0.0) if finite else torch.zeros_like(powers)
    # Priority/queue structure. rank[i] = fraction of legs starting strictly earlier than i.
    rank = (s_j < s_i - _CONT_EPS).to(dt).mean(-1)
    # time-to-next-free: among legs running at i's start (excluding i), soonest finish - start.
    runs_at_start = (s_j <= s_i + _CONT_EPS) & (s_i < e_j - _CONT_EPS)
    self_mask = torch.eye(starts.shape[-1], dtype=torch.bool, device=starts.device)
    runs_at_start = runs_at_start & ~self_mask
    remaining = (e_j - s_i).clamp_min(0.0)
    ttf = torch.where(runs_at_start, remaining, torch.full_like(remaining, pos)).amin(-1)
    ttf = torch.where(ttf >= pos * 0.5, torch.zeros_like(ttf), ttf)  # no concurrent leg -> 0
    mean_dur = (ends - starts).clamp_min(_CONT_EPS).mean(-1, keepdim=True)
    feats = torch.stack(
        [
            (powers / scale).clamp(0.0, _CONT_CLAMP),
            (conc_others / scale).clamp(0.0, _CONT_CLAMP),
            (excess / scale).clamp(0.0, _CONT_CLAMP),
            (peak_cnt / _CONT_CNT_SCALE).clamp(0.0, _CONT_CLAMP),
            rank,
            (ttf / mean_dur).clamp(0.0, _CONT_CLAMP),
        ],
        dim=-1,
    )  # [..., M, 6]
    half = feats.shape[-2] // 2
    return torch.cat([feats[..., :half, :], feats[..., half:, :]], dim=-1)  # [..., n, 12]

# Empty and loaded legs take independent speed levels. The time split is therefore not a
# smooth function of the arc features, but it is exactly recoverable. Each leg's energy pins its
# level (empty_t = empty_e / empty_power(level)); the admissible (empty_level, loaded_level)
# pair is the one whose leg times sum to the arc's Travel_Time. The inversion is performed
# over the 3x3 discrete level grid; the selected branch remains differentiable in the
# energies.
_EMPTY_POWERS = tuple(SPEED_TABLE[lvl].empty_power for lvl in SpeedLevel)   # (7.8, 10, 13.2)
_LOADED_POWERS = tuple(SPEED_TABLE[lvl].loaded_power for lvl in SpeedLevel)  # (11.7, 15, 19.8)


def _leg_time_prior(travel_time: Tensor, empty_e: Tensor, loaded_e: Tensor) -> tuple[Tensor, Tensor]:
    """Exact (empty_t, loaded_t) split via discrete inversion of the per-leg powers."""
    pe = travel_time.new_tensor(_EMPTY_POWERS)   # [3]
    pl = travel_time.new_tensor(_LOADED_POWERS)  # [3]
    empty_cand = empty_e[:, None] / pe[None, :]   # [N, 3] candidate empty times
    loaded_cand = loaded_e[:, None] / pl[None, :]  # [N, 3] candidate loaded times
    total = empty_cand[:, :, None] + loaded_cand[:, None, :]            # [N, 3, 3]
    n = travel_time.shape[0]
    flat = (total - travel_time[:, None, None]).abs().reshape(n, 9).argmin(dim=1)  # [N]
    ei, li = flat // 3, flat % 3
    empty_t = empty_cand.gather(1, ei[:, None]).squeeze(1)
    loaded_t = loaded_cand.gather(1, li[:, None]).squeeze(1)
    return empty_t, loaded_t


@dataclass(slots=True)
class FusedPrediction:
    """One graph's fused outputs and the differentiable local attributes behind them.

    The leg and delay tensors are retained as graph nodes, from which the explainer reads
    exact dC_max/d(leg) and dE/d(leg) after backward.
    """

    makespan: Tensor          # scalar
    energy: Tensor            # scalar
    node_delay: Tensor        # [N]  predicted tau
    empty_t: Tensor           # [N]
    loaded_t: Tensor          # [N]
    empty_e: Tensor           # [N]
    loaded_e: Tensor          # [N]
    dag: EventDag             # assembled event DAG (edge_weights carry leg-time grads)
    wait_empty: Tensor        # [N]  predicted empty-leg power wait (0 when uncoupled)
    wait_loaded: Tensor       # [N]  predicted loaded-leg power wait (0 when uncoupled)

    @property
    def objectives(self) -> Tensor:
        """[2] physical (C_max, E) vector for loss/metrics."""
        return torch.stack([self.makespan, self.energy])

    @property
    def leg_times(self) -> Tensor:
        """[N, 2] GNN-predicted (empty_t, loaded_t) for time anchoring."""
        return torch.stack([self.empty_t, self.loaded_t], dim=-1)

    @property
    def leg_waits(self) -> Tensor:
        """[N, 2] GNN-predicted (empty_wait, loaded_wait) for power-wait anchoring."""
        return torch.stack([self.wait_empty, self.wait_loaded], dim=-1)


class FusedEHGATv2(nn.Module):
    """Wrap a trained EHGATv2 core with tropical-DP makespan + additive energy.

    Physics-anchored residual heads. The projection heads receive the frozen structural
    embeddings together with the physical priors that determine the targets in closed
    form -- the AGV arc's (travel_time, empty_e, loaded_e) and the node's handling time --
    standardised with the core's own buffers. Prediction is in standardised leg space and
    is de-normalised by the registered leg_* and tau_* buffers, set from training statistics
    by ehgat.explain.train_fused.train_fused. The head therefore learns an O(1) residual
    rather than reaching physical scale from a cold softplus, which renders the otherwise
    non-injective max-plus map identifiable and its gradient dense.
    """

    leg_mean: Tensor
    leg_std: Tensor
    tau_mean: Tensor
    tau_std: Tensor
    wait_mean: Tensor
    wait_std: Tensor

    peak_power: Tensor

    def __init__(
        self,
        core: EHGATv2,
        *,
        use_physics_prior: bool = False,
        coupled: bool = False,
        unroll_steps: int = 0,
        peak_power: float | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        # Physics-unrolled refinement steps for the coupled wait head (0 = single static
        # pass). Each step recomputes the tentative max-plus timing and
        # feeds the resulting per-leg contention back into the wait head -- a learned,
        # differentiable analogue of the simulator's iterative power-contention resolution.
        self.unroll_steps = int(unroll_steps)
        # When True the leg head learns a residual around the exact closed-form leg split
        # (faithful-by-construction baseline, but the GNN barely contributes). When False
        # (default) the GNN must predict the leg times itself from its embeddings + the
        # standardised arc priors -- the learned heads carry the prediction; the max-plus layer still
        # guarantees faithful critical-path gradients regardless.
        self.use_physics_prior = use_physics_prior
        # Under peak-power coupling the GNN additionally predicts a per-leg power wait; the
        # leg's effective max-plus weight is leg_time + wait and the makespan is composed
        # over the precedence-only coupled activity DAG (no explicit resolution arcs needed --
        # the continuous waits reproduce the coupled critical path exactly when correct).
        self.coupled = coupled
        hidden = core.config.hidden
        # Heads see [h_src || h_dst || standardised arc priors] and [h || handling prior].
        # Only the leg times are learned; leg energies are read exactly from the inputs.
        self.leg_head = nn.Sequential(
            nn.Linear(2 * hidden + EDGE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, _N_LEG_TIMES)
        )
        self.delay_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        # Power-wait head: a per-leg (empty, loaded) resource-contention delay. It sees the
        # leg context PLUS the physics-unrolled contention vector (own/concurrent power, budget
        # excess, overlap count) read off the previous iterate's tentative timing -- the
        # timing-dependent signal that a single static pass cannot express.
        self.wait_head = nn.Sequential(
            nn.Linear(2 * hidden + EDGE_DIM + _CONT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, _N_LEG_TIMES),
        )
        # Zero the output layer so the initial (pre-training) wait prediction is the mean wait
        # (de-normalised from the zero output), not a large random value -- the physics-unrolled
        # feedback (waits -> timing -> contention -> waits) is otherwise prone to diverge.
        nn.init.zeros_(self.wait_head[-1].weight)
        nn.init.zeros_(self.wait_head[-1].bias)
        # De-normalisation of standardised head outputs into physical units.
        self.register_buffer("leg_mean", torch.zeros(_N_LEG_TIMES))
        self.register_buffer("leg_std", torch.ones(_N_LEG_TIMES))
        self.register_buffer("tau_mean", torch.zeros(1))
        self.register_buffer("tau_std", torch.ones(1))
        self.register_buffer("wait_mean", torch.zeros(_N_LEG_TIMES))
        self.register_buffer("wait_std", torch.ones(_N_LEG_TIMES))
        # Fleet power budget (kW); used to form the contention budget-excess feature. Inf when
        # uncoupled so the feature is well-defined even if the head is never exercised.
        self.register_buffer(
            "peak_power", torch.tensor(float(peak_power) if peak_power is not None else float("inf"))
        )

    def set_leg_normalization(
        self,
        *,
        leg_mean: Tensor,
        leg_std: Tensor,
        tau_mean: Tensor,
        tau_std: Tensor,
        wait_mean: Tensor | None = None,
        wait_std: Tensor | None = None,
    ) -> None:
        """Populate leg/delay (and optional wait) de-normalisation buffers from train stats."""
        eps = 1e-6
        self.leg_mean.copy_(leg_mean)
        self.leg_std.copy_(leg_std.clamp_min(eps))
        self.tau_mean.copy_(tau_mean.reshape(1))
        self.tau_std.copy_(tau_std.reshape(1).clamp_min(eps))
        if wait_mean is not None and wait_std is not None:
            self.wait_mean.copy_(wait_mean)
            self.wait_std.copy_(wait_std.clamp_min(eps))

    def freeze_core(self) -> None:
        """Lock the heterogeneous message-passing layers; train only the new heads."""
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.core.eval()

    def head_parameters(self) -> list[nn.Parameter]:
        """Trainable projection-head parameters (the only ones the fine-tuner optimises)."""
        params = list(self.leg_head.parameters()) + list(self.delay_head.parameters())
        if self.coupled:
            params += list(self.wait_head.parameters())
        return params

    def static_features(
        self, data: HeteroData, h: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Embedding+prior features that DON'T depend on the heads (so they can be cached).

        Returns (leg_in, delay_in, empty_e, loaded_e, arc) with rows in task order
        (AGV arc per task, sorted by dst). leg_in feeds the leg/wait heads, delay_in
        the tau head; empty_e/loaded_e are the exact arc leg energies.
        """
        x = data[NODE_TYPE].x
        agv_index = data[AGV_EDGE].edge_index
        agv_attr = data[AGV_EDGE].edge_attr
        agv_prior = (agv_attr - self.core.agv_mean) / self.core.agv_std
        hand_prior = (x[:, 0:1] - self.core.node_mean[0]) / self.core.node_std[0]

        order = torch.argsort(agv_index[1])  # row k <-> task k
        src = agv_index[0][order]
        dst = agv_index[1][order]
        arc = agv_attr[order]
        empty_e = arc[:, _E_EMPTY_COL]
        loaded_e = arc[:, _E_LOADED_COL]
        leg_in = torch.cat([h[src], h[dst], agv_prior[order]], dim=-1)  # [N, 2H + EDGE_DIM]
        delay_in = torch.cat([h, hand_prior], dim=-1)                   # [N, H + 1]
        return leg_in, delay_in, empty_e, loaded_e, arc

    @torch.no_grad()
    def encode_cached(self, data: HeteroData) -> Tensor:
        """Frozen-core node embeddings h [N, hidden], detached for reuse.

        The core is frozen, so h is identical across every fine-tuning epoch -- caching
        it once and passing it to forward(data, h=...) skips the GNN message passing (the
        single most expensive part of the per-graph step) on every subsequent epoch.
        """
        self.core.eval()
        return self.core.encode(data).detach()

    def _local_attributes(
        self, data: HeteroData, h: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project embeddings + physics priors to the wait-independent local attributes.

        Returns (empty_t, loaded_t, empty_e, loaded_e, d, leg_in). Leg times are
        GNN-predicted (anchored, de-normalised); leg energies are read exactly from the
        AGV arc features; leg_in is returned so the (unrolled) wait head can be driven
        with appended contention features. Power waits are computed separately because they
        depend on the tentative timing (see _predict_waits / forward).
        """
        if h is None:
            h = self.core.encode(data)  # [N, hidden]
        leg_in, delay_in, empty_e, loaded_e, arc = self.static_features(data, h)
        if self.use_physics_prior:
            # Baseline: residual around the exact closed-form split (GNN barely contributes).
            handling = data[NODE_TYPE].x[:, 0]
            empty_prior, loaded_prior = _leg_time_prior(arc[:, _T_TRAVEL_COL], empty_e, loaded_e)
            resid = self.leg_head(leg_in) * self.leg_std
            empty_t = (empty_prior + resid[:, 0]).clamp_min(0.0)
            loaded_t = (loaded_prior + resid[:, 1]).clamp_min(0.0)
            node_delay = (handling + self.delay_head(delay_in).squeeze(-1) * self.tau_std).clamp_min(0.0)
        else:
            # GNN-does-the-work: predict the leg times / delay from embeddings (+ priors as
            # input features), de-normalised by the leg/tau buffers. The split is a learnable
            # function of (Travel_Time, energies); the GNN must infer it.
            times = (self.leg_head(leg_in) * self.leg_std + self.leg_mean).clamp_min(0.0)
            empty_t, loaded_t = times[:, 0], times[:, 1]
            node_delay = (
                self.delay_head(delay_in).squeeze(-1) * self.tau_std + self.tau_mean
            ).clamp_min(0.0)
        return empty_t, loaded_t, empty_e, loaded_e, node_delay, leg_in

    def _predict_waits(self, leg_in: Tensor, contention: Tensor) -> Tensor:
        """Per-leg [N, 2] power waits from the leg context + a contention vector."""
        wait_in = torch.cat([leg_in, contention], dim=-1)
        return (self.wait_head(wait_in) * self.wait_std + self.wait_mean).clamp_min(0.0)

    def _contention_features(
        self,
        e_end: Tensor,
        l_end: Tensor,
        empty_eff: Tensor,
        loaded_eff: Tensor,
        p_empty: Tensor,
        p_loaded: Tensor,
    ) -> Tensor:
        """Per-task [N, 2*4] contention vector from one iterate's tentative leg timing.

        Each of the 2N AGV legs contributes its active interval [end - dur, end] and its
        power, from which the budget pressure is summarised as the simulator resolves it:
        the peak instantaneous concurrent power over the leg's interval (bounded by fleet
        concurrency, hence stable in N), the concurrent power of other legs at that peak,
        the budget excess, and the peak concurrent leg count. All inputs are detached
        (read off the previous iterate), so gradients flow only through the final composition.
        """
        return _peak_contention(
            torch.cat([e_end - empty_eff, l_end - loaded_eff]),
            torch.cat([e_end, l_end]),
            torch.cat([p_empty, p_loaded]),
            float(self.peak_power),
        )

    def forward(self, data: HeteroData, h: Tensor | None = None) -> FusedPrediction:
        """Predict one graph's (C_max, E) via tropical DP + additive energy.

        data must be a single HeteroData (the tropical DP is assembled per
        graph). Batching is handled by iterating graphs in the trainer/explainer. h is an
        optional cached frozen-core embedding (see encode_cached).

        Coupled mode runs unroll_steps physics-unrolled refinements: predict waits ->
        recompose the max-plus timing -> read the per-leg contention off that timing -> refine
        the waits, mirroring the simulator's iterative power-contention resolution.
        """
        n = int(data[NODE_TYPE].x.shape[0])
        is_load = data[NODE_TYPE].x[:, 1] > 0.5
        agv_prev, qc_prev = extract_precedence(
            data[AGV_EDGE].edge_index, data[QC_EDGE].edge_index, n
        )
        empty_t, loaded_t, empty_e, loaded_e, node_delay, leg_in = self._local_attributes(data, h=h)
        energy = (empty_e + loaded_e).sum()  # strictly additive -> exact, dE/d(leg_e) = 1

        if not self.coupled:
            dag = assemble_event_dag(is_load, agv_prev, qc_prev, empty_t, loaded_t, node_delay)
            completion = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
            makespan = completion[dag.completion_nodes].max()
            zeros = empty_t.new_zeros(n)
            return FusedPrediction(
                makespan=makespan, energy=energy, node_delay=node_delay, empty_t=empty_t,
                loaded_t=loaded_t, empty_e=empty_e, loaded_e=loaded_e, dag=dag,
                wait_empty=zeros, wait_loaded=zeros,
            )

        # Per-leg power draw (kW) = exact leg energy / predicted travel time; detached for the
        # contention features (the wait is idle time, not part of the power-drawing travel).
        p_empty = (empty_e / empty_t.clamp_min(_CONT_EPS)).detach()
        p_loaded = (loaded_e / loaded_t.clamp_min(_CONT_EPS)).detach()
        e_idx = 1 + 4 * torch.arange(n, device=empty_t.device)
        l_idx = e_idx + 1
        contention = empty_t.new_zeros(n, _CONT_DIM)
        waits = self._predict_waits(leg_in, contention)
        for step in range(self.unroll_steps + 1):
            if step > 0:
                waits = self._predict_waits(leg_in, contention)
            empty_eff = empty_t + waits[:, 0]
            loaded_eff = loaded_t + waits[:, 1]
            dag = assemble_coupled_event_dag(
                is_load, agv_prev, qc_prev, empty_eff, loaded_eff, node_delay, []
            )
            completion = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
            if step < self.unroll_steps:
                contention = self._contention_features(
                    completion[e_idx].detach(), completion[l_idx].detach(),
                    empty_eff.detach(), loaded_eff.detach(), p_empty, p_loaded,
                )
        makespan = completion[dag.completion_nodes].max()
        wait_empty, wait_loaded = waits[:, 0], waits[:, 1]
        return FusedPrediction(
            makespan=makespan,
            energy=energy,
            node_delay=node_delay,
            empty_t=empty_t,
            loaded_t=loaded_t,
            empty_e=empty_e,
            loaded_e=loaded_e,
            dag=dag,
            wait_empty=wait_empty,
            wait_loaded=wait_loaded,
        )
