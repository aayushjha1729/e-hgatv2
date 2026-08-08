"""Surrogate-assisted NSGA-II with optional attribution-guided mutation.

A (mu + lambda) NSGA-II (Deb et al. 2002) extended in two independent ways, each of
which can be enabled or disabled separately:

Offspring screening (screening_factor = k > 1). The search produces k*lambda
candidates per generation, ranks them by surrogate-predicted (C_max, E) using
non-dominated sorting and crowding distance, and passes only the best lambda to the
exact evaluator. The exact-evaluation count per generation is therefore lambda,
unchanged from the unassisted algorithm.

Guided mutation (guidance). Rather than sampling the mutated task and the operator
uniformly, both may be drawn from a criticality signal: the surrogate's edge attention
(attention), the tropical subgradient of the fused head (tape), an adaptive
operator-utility controller (reward), or the exact critical path computed without the
network (oracle, retained as a ceiling reference).

Representation and operators
The population holds decoded ehgat.environment.decoder.Schedule objects so the
surrogate can score them directly. Crossover is performed in random-key space: parents are
re-encoded with encode_canonical, recombined by biased uniform crossover, and decoded
again, which yields an acyclic schedule by the SPV construction. Mutation applies one of

1. speed -- shift the empty or loaded speed level, trading C_max against E
   without altering precedence;
2. reassign -- move the task to a different AGV and re-project the global order;
3. swap_agv / swap_qc -- exchange the task with its predecessor in one resource
   chain. Reordering one chain independently of the other can introduce an AGV/QC
   deadlock; acyclicity is re-checked with Kahn's algorithm
   (build_precedence -> ScheduleCycleError) and the mutation rejected on a
   cycle.

The swap operators are the reason acyclicity is re-validated at all. The invariant that
every schedule admitted to the population is acyclic is property-tested.

The search maintains a deduplicated external non-dominated archive and a per-generation
front_history for convergence analysis, and is deterministic given a single seeded
NumPy generator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from torch_geometric.data import Batch

from ehgat.benchmark.faithfulness import critical_path_binding
from ehgat.environment.decoder import NUM_BLOCKS, Schedule, decode, encode_canonical
from ehgat.environment.evaluator import ScheduleCycleError, build_precedence, evaluate
from ehgat.environment.instance import Instance
from ehgat.environment.physics import SpeedLevel
from ehgat.search.nsga2 import (
    crowding_distance,
    fast_non_dominated_sort,
    order_by_rank_crowding,
)
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.search.tape_guidance import tape_predict_objectives, tape_signals_batch
from ehgat.surrogate.ehgatv2 import EHGATv2
from ehgat.surrogate.graph import AGV_EDGE, NODE_TYPE, QC_EDGE, build_hetero_graph
from ehgat.utils.seeding import make_rng

__all__ = [
    "AdaptivePursuit",
    "AttentionNSGA2Config",
    "AttentionNSGA2Result",
    "attention_bottleneck_task",
    "attention_bottleneck_type",
    "attention_task_probabilities",
    "default_config",
    "mutate_reassign_agv",
    "mutate_speed",
    "mutate_swap_on_agv",
    "mutate_swap_on_qc",
    "operator_probabilities",
    "operator_reward",
    "run_attention_nsga2",
]

Objectives = tuple[float, float]
_SPEED_LEVELS: tuple[SpeedLevel, ...] = tuple(SpeedLevel)
_ARCHIVE_ROUND = 6  # dedup objective-key precision (mirrors the BRKGA archive)
_MUTATION_OPS = ("speed", "reassign", "swap_agv", "swap_qc")
# speed is the universal makespan-energy lever and also tunes AGV travel time, the
# makespan driver in the AGV-bound case. Its score is held at or above the structural
# operators' to prevent crowd-out: at high agv_bias a 0.5 baseline pushes speed below
# uniform and starves the operator that generates the hypervolume spread. Default 1.0.
_SPEED_WEIGHT = 1.0
# Channel-B operator-selection sources. random is the uniform null; attention /
# oracle are structural priors that map a bottleneck-type signal to operator scores;
# reward is the field-standard online AOS (Thierens 2005 Adaptive Pursuit driven by a
# measured fitness-improvement credit) -- the genuine operator-utility baseline/ceiling,
# distinct from the bottleneck-identity oracle (which assumes the bottleneck->operator map).
_OPERATOR_SOURCES = ("random", "attention", "oracle", "reward")
# Channel-A guidance signal: attention is the bare HAN readout (learned, empirically
# near-random faithfulness); tape is the fused model's native critical-path gradient
# (dC_max/d(leg) -- faithful by construction). With tape the same signal that
# attributes a schedule also steers the mutation and screening.
_GUIDANCE_SOURCES = ("attention", "tape")
_AGGREGATION_WINDOWS = ("full", "front", "best")
# Channel-B granularity: a single population-averaged bias per generation ("population")
# vs the per-task bottleneck of the individual being mutated ("per_task"). The latter
# avoids the averaging that mis-routes offspring whose own bottleneck differs from the mean.
_OPERATOR_GRANULARITIES = ("population", "per_task")
# Adaptive Pursuit (Thierens, "An adaptive pursuit strategy for allocating operator
# probabilities", GECCO 2005) hyper-parameters for the reward AOS arm. alpha is the
# recency weight of the operator-quality estimate, beta the pursuit rate of the
# selection probabilities toward the current best (p_max) / the floor (p_min).
_AP_ALPHA = 0.3
_AP_BETA = 0.3
_AP_PMIN = 0.05


@dataclass(frozen=True, slots=True)
class AttentionNSGA2Config:
    """Hyper-parameters for the attention-guided NSGA-II."""

    pop_size: int
    generations: int = 100
    crossover_prob: float = 0.9  # else a parent is cloned before mutation
    mutation_prob: float = 0.9  # prob. a child undergoes attention-guided mutation
    inherit_prob: float = 0.7  # biased uniform crossover bias toward parent A
    tournament_size: int = 2
    random_mutation: bool = False  # ablation: select a random task instead of max-alpha
    mutation_temperature: float = 0.25  # softmax temperature for soft bottleneck sampling
    screening_factor: int = 1  # generate k*lambda offspring, surrogate-screen to lambda (1=off)
    guidance: str = "attention"  # Channel-A signal source: attention (bare HAN) | tape (fused)
    operator_selection: str = "random"  # Channel-B AOS source: random | attention | oracle
    operator_temperature: float = 1.0  # softmax tau for operator probs (exploitation knob)
    operator_speed_weight: float = _SPEED_WEIGHT  # speed op score (>=structural; anti-crowd-out)
    operator_granularity: str = "population"  # Channel-B bias scope: population | per_task
    aggregation_window: str = "front"  # bottleneck-type readout source: full | front | best
    seed: int = 0


@dataclass(frozen=True, slots=True)
class AttentionNSGA2Result:
    """Outcome of an attention-guided NSGA-II run."""

    front: tuple[Objectives, ...]  # non-dominated (makespan, energy), ascending
    schedules: tuple[Schedule, ...]  # schedule for each front point (aligned)
    front_history: tuple[tuple[Objectives, ...], ...]  # archive front per generation
    evaluations: int
    deadlocks_rejected: int  # swap mutations rejected by Kahn re-validation


def default_config(
    instance: Instance, *, generations: int = 100, seed: int = 0
) -> AttentionNSGA2Config:
    """Population P = 20N (matching the BRKGA baseline) for fair comparison."""
    return AttentionNSGA2Config(
        pop_size=20 * instance.num_tasks, generations=generations, seed=seed
    )


# --------------------------------------------------------------------------------------
# Attention bottleneck identification
# --------------------------------------------------------------------------------------
def attention_bottleneck_task(schedule: Schedule, instance: Instance, model: EHGATv2) -> int:
    """Return the task delivered by the maximum-attention AGV arc of schedule.

    This is the surrogate's learned bottleneck: the disjunctive resource arc whose
    criticality (semantic attention weight) is highest. Ties are broken by the lowest
    arc index for determinism.
    """
    data = build_hetero_graph(schedule, instance).to(next(model.parameters()).device)
    edge_index, alpha = model.attention(data)[AGV_EDGE[1]]
    if alpha.numel() == 0:
        return 0
    best = int(alpha.argmax().item())
    return int(edge_index[1, best].item())


def attention_task_probabilities(
    schedule: Schedule, instance: Instance, model: EHGATv2, *, temperature: float = 0.25
) -> np.ndarray:
    """Per-task mutation probabilities from the surrogate's AGV-arc attention.

    A temperature-scaled softmax over each task's incoming-arc attention. This is the
    soft alternative to attention_bottleneck_task: high-attention tasks are
    favoured, but every task keeps a non-zero probability, preserving search diversity
    (a hard argmax concentrates ~all mutations on 1-2 tasks and collapses exploration).
    """
    n = instance.num_tasks
    data = build_hetero_graph(schedule, instance).to(next(model.parameters()).device)
    edge_index, alpha = model.attention(data)[AGV_EDGE[1]]
    if alpha.numel() == 0:
        return np.full(n, 1.0 / n)
    scores = np.zeros(n)
    tasks = edge_index[1].cpu().numpy()
    scores[tasks] = alpha.cpu().numpy()
    logits = (scores - scores.max()) / max(temperature, 1e-6)
    probs = np.exp(logits)
    return np.asarray(probs / probs.sum())


def attention_bottleneck_type(
    schedule: Schedule, instance: Instance, model: EHGATv2
) -> tuple[np.ndarray, np.ndarray]:
    """Per-task semantic attention on the AGV vs QC relation: (w_agv, w_qc).

    The surrogate's HAN semantic weights classify, for each task, which resource chain
    gates it -- w_agv (AGV routing/sequencing) vs w_qc (QC serialisation). Because
    the weights are a per-task softmax over the two relations, w_agv[j] + w_qc[j] == 1
    (a task with no incoming QC arc is pure AGV-bound, w_qc[j] == 0). This is the
    bottleneck-type signal the Channel-B operator-selection controller consumes,
    distinct from the which-task signal of attention_task_probabilities.
    """
    n = instance.num_tasks
    attn = model.attention(build_hetero_graph(schedule, instance).to(next(model.parameters()).device))
    w_agv = np.zeros(n)
    agv_index, agv_alpha = attn[AGV_EDGE[1]]
    if agv_alpha.numel() > 0:
        w_agv[agv_index[1].cpu().numpy()] = agv_alpha.cpu().numpy()
    w_qc = np.zeros(n)
    qc_index, qc_alpha = attn[QC_EDGE[1]]
    if qc_alpha.numel() > 0:
        w_qc[qc_index[1].cpu().numpy()] = qc_alpha.cpu().numpy()
    return w_agv, w_qc


def _batch_attention_signals(
    schedules: list[Schedule], instance: Instance, model: EHGATv2, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched (task_probs, w_agv, w_qc) for many schedules in one forward pass.

    This is the batched equivalent of _attention_signals: it block-diagonal
    batches every schedule's graph, runs a single model.attention on the model's device
    (CUDA when placed there), and scatters the per-arc semantic weights back to per-task
    [G, N] arrays. The per-graph result is identical to calling
    _attention_signals on each schedule (same w_agv softmax), so only throughput
    changes -- one ~10k-graphs/s GPU launch instead of G serial ~10-graphs/s CPU forwards.
    """
    n = instance.num_tasks
    g = len(schedules)
    if g == 0:
        return np.empty((0, n)), np.empty((0, n)), np.empty((0, n))
    graphs = [build_hetero_graph(s, instance) for s in schedules]
    batch = Batch.from_data_list(graphs)
    device = next(model.parameters()).device
    attn = model.attention(batch.to(device))
    ptr = batch[NODE_TYPE].ptr.cpu().numpy()  # [G+1] node offset per graph
    node_batch = batch[NODE_TYPE].batch  # [G*N] node -> graph

    def _scatter(edge_alpha: tuple) -> np.ndarray:
        index, alpha = edge_alpha
        out = np.zeros((g, n))
        if alpha.numel() > 0:
            dst = index[1]
            gid = node_batch[dst].cpu().numpy()
            local = dst.cpu().numpy() - ptr[gid]
            out[gid, local] = alpha.cpu().numpy()
        return out

    w_agv = _scatter(attn[AGV_EDGE[1]])
    w_qc = _scatter(attn[QC_EDGE[1]])
    shifted = (w_agv - w_agv.max(axis=1, keepdims=True)) / max(temperature, 1e-6)
    exp = np.exp(shifted)
    task_probs = exp / exp.sum(axis=1, keepdims=True)
    return task_probs, w_agv, w_qc


def _attention_signals(
    schedule: Schedule, instance: Instance, model: EHGATv2, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused single-forward-pass readout: (task_probs, w_agv, w_qc).

    The which-task signal (attention_task_probabilities) and the bottleneck-type
    signal (attention_bottleneck_type) both consume the same HAN attention dict, so
    per-task Channel-B routing reuses one model.attention call -- it is cost-neutral
    versus the population-mode guided mutation, which already pays for one pass per child.
    """
    n = instance.num_tasks
    attn = model.attention(build_hetero_graph(schedule, instance).to(next(model.parameters()).device))
    agv_index, agv_alpha = attn[AGV_EDGE[1]]
    w_agv = np.zeros(n)
    if agv_alpha.numel() > 0:
        w_agv[agv_index[1].cpu().numpy()] = agv_alpha.cpu().numpy()
    w_qc = np.zeros(n)
    qc_index, qc_alpha = attn[QC_EDGE[1]]
    if qc_alpha.numel() > 0:
        w_qc[qc_index[1].cpu().numpy()] = qc_alpha.cpu().numpy()
    if agv_alpha.numel() == 0:
        task_probs = np.full(n, 1.0 / n)
    else:
        logits = (w_agv - w_agv.max()) / max(temperature, 1e-6)
        probs = np.exp(logits)
        task_probs = np.asarray(probs / probs.sum())
    return task_probs, w_agv, w_qc


# --------------------------------------------------------------------------------------
# Mutation operators (each returns a new Schedule, or None if it deadlocks)
# --------------------------------------------------------------------------------------
def mutate_speed(schedule: Schedule, task: int, rng: np.random.Generator) -> Schedule:
    """Nudge the empty- or loaded-leg speed level of task to a different value.

    A pure cost and timing change with no precedence effect; the result is always acyclic.
    Trades makespan against energy, the second axis of the bottleneck decision.
    """
    leg = "empty" if rng.random() < 0.5 else "loaded"
    current = schedule.empty_speed[task] if leg == "empty" else schedule.loaded_speed[task]
    choices = [lvl for lvl in _SPEED_LEVELS if lvl != current]
    new_level = _SPEED_LEVELS[0] if not choices else choices[int(rng.integers(len(choices)))]
    if leg == "empty":
        speeds = list(schedule.empty_speed)
        speeds[task] = new_level
        return replace(schedule, empty_speed=tuple(speeds))
    speeds = list(schedule.loaded_speed)
    speeds[task] = new_level
    return replace(schedule, loaded_speed=tuple(speeds))


def mutate_reassign_agv(
    schedule: Schedule, instance: Instance, task: int, rng: np.random.Generator
) -> Schedule:
    """Reassign task to a different AGV, re-projecting the existing global order.

    Because both AGV and QC sequences remain projections of one shared total order
    (global_order), the schedule stays acyclic by construction.
    """
    if instance.num_agvs < 2:
        return schedule
    current = schedule.assignment[task]
    others = [a for a in range(instance.num_agvs) if a != current]
    new_agv = others[int(rng.integers(len(others)))]
    assignment = list(schedule.assignment)
    assignment[task] = new_agv
    agv_sequences = tuple(
        tuple(t for t in schedule.global_order if assignment[t] == a)
        for a in range(instance.num_agvs)
    )
    return replace(schedule, assignment=tuple(assignment), agv_sequences=agv_sequences)


def mutate_swap_on_agv(schedule: Schedule, instance: Instance, task: int) -> Schedule | None:
    """Swap task with its immediate predecessor on its AGV chain.

    Reordering a single AGV chain independently of the QC chains may create an AGV/QC
    deadlock. The result is re-validated with Kahn's algorithm and None is returned on a
    cycle, in which case the caller keeps the parent. On success the now-consistent global
    order is refreshed from the Kahn topological order, under which the schedule round-trips
    through encode_canonical/decode.
    """
    agv = schedule.assignment[task]
    seq = list(schedule.agv_sequences[agv])
    pos = seq.index(task)
    if pos == 0:
        return None  # no predecessor on this AGV to swap with
    seq[pos - 1], seq[pos] = seq[pos], seq[pos - 1]
    agv_sequences = list(schedule.agv_sequences)
    agv_sequences[agv] = tuple(seq)

    n = instance.num_tasks
    try:
        _, _, topo = build_precedence(tuple(agv_sequences), schedule.qc_sequences, n)
    except ScheduleCycleError:
        return None  # deadlock -> reject the mutation
    return replace(schedule, agv_sequences=tuple(agv_sequences), global_order=topo)


def mutate_swap_on_qc(schedule: Schedule, instance: Instance, task: int) -> Schedule | None:
    """Swap task with its immediate predecessor on its QC serialisation chain.

    The QC analogue of mutate_swap_on_agv. Reordering a single QC chain independently of
    the AGV chains may create an AGV/QC deadlock, and the result is re-validated with
    Kahn's algorithm (None on a cycle, in which case the parent is kept). This is the
    operator that addresses a QC-bound bottleneck; the AGV operators (reassign, swap) cannot
    relieve a serialisation gated by the crane chain. On success the global order is
    refreshed from the Kahn topological order, under which the schedule round-trips through
    encode_canonical/decode.
    """
    qc_idx = instance.qcs.index(instance.tasks[task].qc)
    seq = list(schedule.qc_sequences[qc_idx])
    pos = seq.index(task)
    if pos == 0:
        return None  # no predecessor on this QC chain to swap with
    seq[pos - 1], seq[pos] = seq[pos], seq[pos - 1]
    qc_sequences = list(schedule.qc_sequences)
    qc_sequences[qc_idx] = tuple(seq)

    n = instance.num_tasks
    try:
        _, _, topo = build_precedence(schedule.agv_sequences, tuple(qc_sequences), n)
    except ScheduleCycleError:
        return None  # deadlock -> reject the mutation
    return replace(schedule, qc_sequences=tuple(qc_sequences), global_order=topo)


def _predict_objectives(
    schedules: list[Schedule], instance: Instance, model: EHGATv2
) -> list[Objectives]:
    """Batched surrogate prediction of (makespan, energy) for candidate schedules.

    Used by offspring screening: the surrogate's near-exact regression, not its attention,
    pre-filters a k-times larger candidate pool, restricting the exact evaluations to
    predicted-dominant offspring.
    """
    graphs = [build_hetero_graph(s, instance) for s in schedules]
    batch = Batch.from_data_list(graphs)
    device = next(model.parameters()).device
    preds = model.predict(batch.to(device))
    return [(float(m), float(e)) for m, e in preds.tolist()]


# --------------------------------------------------------------------------------------
# Channel-B: XAI-driven adaptive operator selection (AOS)
# --------------------------------------------------------------------------------------
def operator_probabilities(
    agv_bias: float, temperature: float, *, speed_weight: float = _SPEED_WEIGHT
) -> np.ndarray:
    """Operator distribution over _MUTATION_OPS from an AGV-vs-QC bottleneck bias.

    agv_bias in [0, 1] is the share of the bottleneck attributable to the AGV
    resource (vs the QC chain). It routes the structural operators (Channel B): high ->
    favour reassign/swap_agv; low -> favour swap_qc. speed carries a fixed
    speed_weight score because it is the universal makespan<->energy lever (and tunes
    AGV travel time -- the makespan driver when AGV-bound); keeping it >= the structural
    scores prevents the crowd-out that would otherwise, at high agv_bias, push speed
    below the uniform share and starved HV spread. temperature is the exploitation knob
    (low -> sharpen; high -> uniform, recovering random AOS).
    """
    b = float(np.clip(agv_bias, 0.0, 1.0))
    scores = np.array([speed_weight, b, b, 1.0 - b])  # speed, reassign, swap_agv, swap_qc
    logits = (scores - scores.max()) / max(temperature, 1e-6)
    probs = np.exp(logits)
    return np.asarray(probs / probs.sum())


def operator_reward(parent: Objectives, child: Objectives) -> float:
    """Pareto-dominance credit in [0, 1] for a mutation (child vs its primary parent).

    The bounded multi-objective analogue of the scalar fitness-improvement credit used by
    the canonical AOS literature (Thierens 2005; Da Costa, Fialho, Schoenauer & Sebag 2008):
    1.0 if the child dominates the parent (an unambiguous win), 0.5 if the two
    are mutually non-dominated (a new makespan<->energy trade-off, still valuable in a
    Pareto search), and 0.0 if the parent dominates the child or the two are equal (no
    progress -- including a deadlock-rejected swap, which returns the parent unchanged). It
    needs no extra exact evaluations: the parent objective is already known from the
    population and the child objective from the offspring evaluation.
    """
    pm, pe = parent
    cm, ce = child
    better = cm < pm or ce < pe
    worse = cm > pm or ce > pe
    if better and not worse:
        return 1.0  # child dominates parent
    if better and worse:
        return 0.5  # incomparable (new trade-off)
    return 0.0  # parent dominates, or equal


class AdaptivePursuit:
    """Adaptive Pursuit operator selection over _MUTATION_OPS (Thierens, GECCO 2005).

    The field-standard online AOS that the reward arm uses as the genuine
    operator-utility baseline/ceiling -- distinct from the bottleneck-identity oracle,
    which assumes the bottleneck-type -> operator map. It maintains an exponential-recency
    quality estimate q per operator (learning rate alpha) updated from the measured
    operator_reward, and a selection distribution p that pursues the current
    best operator at rate beta toward p_max = 1 - (K - 1) * p_min while every other
    operator decays toward the exploration floor p_min. Because each update is a convex
    move toward a point on the simplex, p stays a valid distribution with every mass in
    [p_min, p_max] (guaranteed exploration). The controller is stateful across
    generations within a single run and fully deterministic given the search seed.
    """

    __slots__ = ("k", "alpha", "beta", "p_min", "p_max", "q", "p")

    def __init__(self, k: int, *, alpha: float, beta: float, p_min: float) -> None:
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.p_min = p_min
        self.p_max = 1.0 - (k - 1) * p_min
        self.q = np.zeros(k)
        self.p = np.full(k, 1.0 / k)

    def probabilities(self) -> np.ndarray:
        """Current selection distribution over _MUTATION_OPS (a defensive copy)."""
        return self.p.copy()

    def update(self, rewards_per_op: list[list[float]]) -> None:
        """Fold one generation's per-operator rewards into q then pursue the best p.

        rewards_per_op[a] holds every operator_reward observed for operator
        a this generation. Operators that did not fire keep their previous quality (and
        so do not move the selection probabilities until they are sampled again).
        """
        fired = False
        for a, rewards in enumerate(rewards_per_op):
            if rewards:
                fired = True
                self.q[a] += self.alpha * (float(np.mean(rewards)) - self.q[a])
        if not fired:
            return
        best = int(np.argmax(self.q))  # ties -> lowest index (deterministic)
        for a in range(self.k):
            target = self.p_max if a == best else self.p_min
            self.p[a] += self.beta * (target - self.p[a])


def _aggregation_window(
    population: list[Schedule], front0: list[int], objectives: list[Objectives], mode: str
) -> list[Schedule]:
    """Schedules whose bottleneck signal feeds the operator controller (Signal-to-Noise)."""
    if mode == "full":
        return population
    if mode == "front":
        return [population[i] for i in front0]
    if mode == "best":  # the front member with the smallest makespan (the C_max-critical one)
        best = min(front0, key=lambda i: objectives[i][0])
        return [population[best]]
    raise ValueError(f"unknown aggregation_window {mode!r}; use one of {_AGGREGATION_WINDOWS}")


def _agv_bias(
    schedules: list[Schedule],
    instance: Instance,
    model: EHGATv2,
    *,
    source: str,
    temperature: float,
) -> float:
    """Aggregate AGV-vs-QC bottleneck bias over schedules for the operator controller.

    attention uses the surrogate's semantic w_agv weighted by its which-task
    attention (the learned readout); oracle uses the exact Max-Plus critical-path
    binding fraction (the upper-bound signal). Returns a bias in [0, 1].
    """
    if not schedules:
        return 0.5
    if source == "attention":
        # One batched GPU pass for all window schedules (was 2 serial forwards each).
        probs, w_agv, _ = _batch_attention_signals(schedules, instance, model, temperature)
        biases = [float(np.dot(probs[i], w_agv[i])) for i in range(len(schedules))]
        return float(np.mean(biases))
    if source == "oracle":
        biases = []
        for s in schedules:
            agv_bound, qc_bound = critical_path_binding(s, instance)
            total = len(agv_bound) + len(qc_bound)
            biases.append(len(agv_bound) / total if total else 0.5)
        return float(np.mean(biases))
    raise ValueError(f"unknown operator_selection {source!r}; use one of {_OPERATOR_SOURCES}")


def _operator_distribution(
    population: list[Schedule],
    front0: list[int],
    objectives: list[Objectives],
    instance: Instance,
    model: EHGATv2,
    config: AttentionNSGA2Config,
) -> np.ndarray | None:
    """Per-generation Channel-B operator distribution (None => uniform/random AOS)."""
    if config.operator_selection == "random":
        return None
    window = _aggregation_window(population, front0, objectives, config.aggregation_window)
    bias = _agv_bias(
        window,
        instance,
        model,
        source=config.operator_selection,
        temperature=config.mutation_temperature,
    )
    return operator_probabilities(
        bias, config.operator_temperature, speed_weight=config.operator_speed_weight
    )


def _per_task_bias(
    schedule: Schedule,
    instance: Instance,
    model: EHGATv2,
    task: int,
    source: str,
    *,
    w_agv: np.ndarray | None = None,
    w_qc: np.ndarray | None = None,
) -> float:
    """AGV-vs-QC bias for a single task (per-task Channel-B routing).

    attention reads the task's own semantic weights w_agv[task] / (w_agv+w_qc)
    (reusing the fused readout when available); oracle uses the exact critical-path
    membership (1.0 if AGV-bound, 0.0 if QC-bound, 0.5 if the task gates neither).
    """
    if source == "attention":
        if w_agv is None or w_qc is None:
            _, w_agv, w_qc = _attention_signals(schedule, instance, model, 1.0)
        denom = float(w_agv[task] + w_qc[task])
        return float(w_agv[task] / denom) if denom > 0.0 else 0.5
    agv_bound, qc_bound = critical_path_binding(schedule, instance)
    if task in agv_bound:
        return 1.0
    if task in qc_bound:
        return 0.0
    return 0.5


def _mutate(
    schedule: Schedule,
    instance: Instance,
    model: EHGATv2,
    rng: np.random.Generator,
    *,
    guided: bool = True,
    temperature: float = 0.25,
    op_probs: np.ndarray | None = None,
    per_task: tuple[str, float, float] | None = None,
    signals: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Schedule, bool, int]:
    """Apply one mutation operator. Returns (schedule, deadlock_rejected, op_index).

    guided controls Channel A (which task): the target is sampled from the
    surrogate's attention distribution, else uniformly at random (the ablation control).
    Channel B (which operator) has three modes: per_task ((source, op_tau,
    speed_weight)) routes the operator from the chosen task's own bottleneck type;
    else op_probs (a fixed per-generation distribution) is the population-mode bias
    (also how the reward controller injects its current policy); else None =>
    uniform operator choice (random AOS). op_index is the index into _MUTATION_OPS of the
    operator that fired, from which the caller assigns credit in the reward arm.

    signals are precomputed (task_probs, w_agv, w_qc) for this schedule from a
    batched GPU _batch_attention_signals pass; when given, no per-child surrogate
    forward is run (the batched path). The result is identical to recomputing them.
    """
    probs = w_agv = w_qc = None
    if signals is not None:
        probs, w_agv, w_qc = signals
    if guided:
        if probs is None:
            if per_task is not None and per_task[0] == "attention":
                probs, w_agv, w_qc = _attention_signals(schedule, instance, model, temperature)
            else:
                probs = attention_task_probabilities(
                    schedule, instance, model, temperature=temperature
                )
        task = int(rng.choice(instance.num_tasks, p=probs))
    else:
        task = int(rng.integers(instance.num_tasks))
    if per_task is not None:
        source, op_tau, speed_weight = per_task
        bias = _per_task_bias(schedule, instance, model, task, source, w_agv=w_agv, w_qc=w_qc)
        local = operator_probabilities(bias, op_tau, speed_weight=speed_weight)
        op_index = int(rng.choice(len(_MUTATION_OPS), p=local))
    elif op_probs is None:
        op_index = int(rng.integers(len(_MUTATION_OPS)))
    else:
        op_index = int(rng.choice(len(_MUTATION_OPS), p=op_probs))
    op = _MUTATION_OPS[op_index]
    if op == "speed":
        return mutate_speed(schedule, task, rng), False, op_index
    if op == "reassign":
        return mutate_reassign_agv(schedule, instance, task, rng), False, op_index
    if op == "swap_agv":
        mutated = mutate_swap_on_agv(schedule, instance, task)
    else:  # swap_qc
        mutated = mutate_swap_on_qc(schedule, instance, task)
    if mutated is None:
        return schedule, True, op_index
    return mutated, False, op_index


# --------------------------------------------------------------------------------------
# Crossover (random-key space) and selection
# --------------------------------------------------------------------------------------
def _crossover(
    parent_a: Schedule,
    parent_b: Schedule,
    instance: Instance,
    rng: np.random.Generator,
    inherit_prob: float,
) -> Schedule:
    """Biased uniform crossover in random-key space; the decoded child is acyclic."""
    keys_a = encode_canonical(parent_a, instance)
    keys_b = encode_canonical(parent_b, instance)
    take_a = rng.random(NUM_BLOCKS * instance.num_tasks) < inherit_prob
    child_keys = np.where(take_a, keys_a, keys_b)
    return decode(child_keys, instance)


def _tournament(rank: list[int], crowding: list[float], rng: np.random.Generator, k: int) -> int:
    """Return the index winning a k-way tournament (lower rank, then higher crowd)."""
    contenders = rng.integers(len(rank), size=k)
    best = int(contenders[0])
    for c in contenders[1:]:
        ci = int(c)
        if rank[ci] < rank[best] or (rank[ci] == rank[best] and crowding[ci] > crowding[best]):
            best = ci
    return best


def _rank_and_crowding(
    objectives: list[Objectives],
) -> tuple[list[list[int]], list[int], list[float]]:
    """Compute Pareto fronts plus per-individual rank and crowding distance."""
    fronts = fast_non_dominated_sort(objectives)
    n = len(objectives)
    rank = [0] * n
    crowd = [0.0] * n
    for r, front in enumerate(fronts):
        cd = crowding_distance(objectives, front)
        for i in front:
            rank[i] = r
            crowd[i] = cd[i]
    return fronts, rank, crowd


# --------------------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------------------
def _update_archive(
    archive_obj: list[Objectives],
    archive_sched: list[Schedule],
    new_obj: list[Objectives],
    new_sched: list[Schedule],
) -> tuple[list[Objectives], list[Schedule]]:
    """Merge new into the archive, keeping a deduplicated non-dominated set."""
    seen: set[tuple[float, float]] = set()
    merged_obj: list[Objectives] = []
    merged_sched: list[Schedule] = []
    for obj, sched in zip(archive_obj + new_obj, archive_sched + new_sched, strict=True):
        key = (round(obj[0], _ARCHIVE_ROUND), round(obj[1], _ARCHIVE_ROUND))
        if key in seen:
            continue
        seen.add(key)
        merged_obj.append(obj)
        merged_sched.append(sched)
    keep = fast_non_dominated_sort(merged_obj)[0]
    return [merged_obj[i] for i in keep], [merged_sched[i] for i in keep]


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def run_attention_nsga2(
    instance: Instance,
    model: EHGATv2 | None,
    config: AttentionNSGA2Config,
    *,
    fused_model: FusedEHGATv2 | None = None,
) -> AttentionNSGA2Result:
    """Run the attention-guided NSGA-II and return its non-dominated front.

    config.guidance selects the Channel-A signal: "attention" (default) uses the
    bare HAN readout of model; "tape" uses fused_model's native critical-path
    gradients (faithful by construction) for both the bottleneck task selection and the
    offspring screening, making the signal that explains a schedule the one that steers the
    search.
    fused_model is required when guidance == "tape"; model may then be None
    (the fused core is used for the unused-in-tape attention calls).
    """
    if config.guidance not in _GUIDANCE_SOURCES:
        raise ValueError(
            f"guidance must be one of {_GUIDANCE_SOURCES}, got {config.guidance!r}"
        )
    if config.guidance == "tape":
        if fused_model is None:
            raise ValueError("guidance='tape' requires a fused_model (FusedEHGATv2)")
        fused_model.eval()
        if model is None:
            model = fused_model.core
    if model is None:
        raise ValueError("model is required unless guidance='tape' supplies a fused_model")
    if config.pop_size < 2:
        raise ValueError(f"pop_size must be >= 2, got {config.pop_size}")
    if config.operator_selection not in _OPERATOR_SOURCES:
        raise ValueError(
            f"operator_selection must be one of {_OPERATOR_SOURCES}, "
            f"got {config.operator_selection!r}"
        )
    if config.aggregation_window not in _AGGREGATION_WINDOWS:
        raise ValueError(
            f"aggregation_window must be one of {_AGGREGATION_WINDOWS}, "
            f"got {config.aggregation_window!r}"
        )
    if config.operator_granularity not in _OPERATOR_GRANULARITIES:
        raise ValueError(
            f"operator_granularity must be one of {_OPERATOR_GRANULARITIES}, "
            f"got {config.operator_granularity!r}"
        )
    model.eval()
    rng = make_rng(config.seed)
    chrom_len = NUM_BLOCKS * instance.num_tasks

    population = [decode(rng.random(chrom_len), instance) for _ in range(config.pop_size)]
    objectives = [evaluate(s, instance).objectives for s in population]
    evaluations = len(population)

    archive_obj: list[Objectives] = []
    archive_sched: list[Schedule] = []
    history: list[tuple[Objectives, ...]] = []
    deadlocks_rejected = 0

    # Channel-B per-task routing replaces the per-generation population bias with the
    # individual's own bottleneck type, evaluated lazily inside _mutate. Only the
    # bottleneck-type sources (attention / oracle) support per-task scope; the reward
    # arm is an online controller with a single global policy (population scope by nature).
    if config.guidance == "tape":
        # TAPE drives Channel-B per-task too: the operator router consumes the provided
        # fused-TAPE (w_agv, w_qc) criticalities (source label "attention" only selects the
        # use-provided-signals branch of _per_task_bias; it never recomputes here because
        # every mutated child carries precomputed TAPE signals).
        per_task = ("attention", config.operator_temperature, config.operator_speed_weight)
    else:
        per_task = (
            (config.operator_selection, config.operator_temperature, config.operator_speed_weight)
            if config.operator_selection in ("attention", "oracle")
            and config.operator_granularity == "per_task"
            else None
        )
    # Channel-B reward arm: the field-standard online AOS (Adaptive Pursuit driven by a
    # measured fitness-improvement credit). Its policy replaces the structural-bias
    # op_probs and is updated from each generation's offspring credit below.
    controller = (
        AdaptivePursuit(len(_MUTATION_OPS), alpha=_AP_ALPHA, beta=_AP_BETA, p_min=_AP_PMIN)
        if config.operator_selection == "reward"
        else None
    )

    for gen in range(config.generations + 1):
        fronts, rank, crowd = _rank_and_crowding(objectives)
        front0 = fronts[0]
        archive_obj, archive_sched = _update_archive(
            archive_obj,
            archive_sched,
            [objectives[i] for i in front0],
            [population[i] for i in front0],
        )
        history.append(tuple(sorted(archive_obj)))

        if gen == config.generations:
            break

        if controller is not None:
            op_probs = controller.probabilities()
        elif per_task is not None:
            op_probs = None
        else:
            op_probs = _operator_distribution(
                population, front0, objectives, instance, model, config
            )

        # ---- create lambda offspring (optionally surrogate-screened from k*lambda) ----
        # Batched two-phase generation: (A) draw all base children (tournament +
        # crossover); (B) one batched attention pass on the model's device for every child
        # that will be guided-mutated; (C) apply each per-child mutation from the precomputed
        # signals. This replaces O(num_candidates) serial CPU forwards with a single batched
        # GPU launch -- ~1000x the surrogate throughput at N>=10 (see profile_gpu_inference).
        num_candidates = config.pop_size * max(1, config.screening_factor)
        need_signals = (not config.random_mutation) or (
            per_task is not None and per_task[0] == "attention"
        )
        base_children: list[Schedule] = []
        will_mutate: list[bool] = []
        parent_objs: list[Objectives] = []
        for _ in range(num_candidates):
            pa_idx = _tournament(rank, crowd, rng, config.tournament_size)
            pa = population[pa_idx]
            if rng.random() < config.crossover_prob:
                pb = population[_tournament(rank, crowd, rng, config.tournament_size)]
                child = _crossover(pa, pb, instance, rng, config.inherit_prob)
            else:
                child = pa
            base_children.append(child)
            will_mutate.append(rng.random() < config.mutation_prob)
            parent_objs.append(objectives[pa_idx])

        # Phase B: one batched (GPU) attention pass for all children that need signals.
        signals_by_idx: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if need_signals:
            mut_idx = [i for i, m in enumerate(will_mutate) if m]
            if mut_idx and config.guidance == "tape":
                # TAPE attribution per child (the tropical DP is per-graph).
                sig_list = tape_signals_batch(
                    fused_model,  # type: ignore[arg-type]  # not-None when guidance=="tape"
                    [base_children[i] for i in mut_idx],
                    instance,
                    config.mutation_temperature,
                )
                for k, i in enumerate(mut_idx):
                    signals_by_idx[i] = sig_list[k]
            elif mut_idx:
                pb_probs, pb_wagv, pb_wqc = _batch_attention_signals(
                    [base_children[i] for i in mut_idx],
                    instance,
                    model,
                    config.mutation_temperature,
                )
                for k, i in enumerate(mut_idx):
                    signals_by_idx[i] = (pb_probs[k], pb_wagv[k], pb_wqc[k])

        # Phase C: per-child task + operator selection and mutation.
        candidates: list[Schedule] = []
        cand_ops: list[int] = []  # operator fired per candidate (reward credit); -1 if none
        cand_parent_obj: list[Objectives] = []  # primary-parent objectives (reward baseline)
        for i, child in enumerate(base_children):
            op_index = -1
            if will_mutate[i]:
                child, rejected, op_index = _mutate(
                    child,
                    instance,
                    model,
                    rng,
                    guided=not config.random_mutation,
                    temperature=config.mutation_temperature,
                    op_probs=op_probs,
                    per_task=per_task,
                    signals=signals_by_idx.get(i),
                )
                deadlocks_rejected += int(rejected)
            candidates.append(child)
            if controller is not None:
                cand_ops.append(op_index)
                cand_parent_obj.append(parent_objs[i])
        if config.screening_factor > 1:
            predicted = (
                tape_predict_objectives(fused_model, candidates, instance)  # type: ignore[arg-type]
                if config.guidance == "tape"
                else _predict_objectives(candidates, instance, model)
            )
            pred_fronts = fast_non_dominated_sort(predicted)
            keep = order_by_rank_crowding(predicted, pred_fronts)[: config.pop_size]
            offspring = [candidates[i] for i in keep]
            sel_ops = [cand_ops[i] for i in keep] if controller is not None else []
            sel_parent_obj = [cand_parent_obj[i] for i in keep] if controller is not None else []
        else:
            offspring = candidates
            sel_ops, sel_parent_obj = cand_ops, cand_parent_obj
        off_obj = [evaluate(child, instance).objectives for child in offspring]
        evaluations += len(offspring)

        # ---- reward credit assignment: fold offspring utility into the AOS controller ----
        if controller is not None:
            rewards_per_op: list[list[float]] = [[] for _ in _MUTATION_OPS]
            for op_index, parent_obj, child_obj in zip(
                sel_ops, sel_parent_obj, off_obj, strict=True
            ):
                if op_index >= 0:
                    rewards_per_op[op_index].append(operator_reward(parent_obj, child_obj))
            controller.update(rewards_per_op)

        # ---- (mu + lambda) environmental selection ----
        combined = population + offspring
        combined_obj = objectives + off_obj
        sel_fronts = fast_non_dominated_sort(combined_obj)
        order = order_by_rank_crowding(combined_obj, sel_fronts)[: config.pop_size]
        population = [combined[i] for i in order]
        objectives = [combined_obj[i] for i in order]

    order = sorted(range(len(archive_obj)), key=lambda i: archive_obj[i])
    return AttentionNSGA2Result(
        front=tuple(archive_obj[i] for i in order),
        schedules=tuple(archive_sched[i] for i in order),
        front_history=tuple(history),
        evaluations=evaluations,
        deadlocks_rejected=deadlocks_rejected,
    )
