"""E-HGATv2: a max-plus heterogeneous GATv2 surrogate with edge-level attention.

The makespan of a disjunctive schedule is a max-plus longest path over the
precedence DAG (see evaluator.py). This model embeds that physics:

- GATv2 message passing is run per edge type (agv resource arcs, qc
  serialisation arcs) with aggr='max' -- the longest-path (max-plus) reduction
  over a node's same-resource predecessors -- rather than the usual attention-weighted
  sum.
- Cross-relation fusion uses semantic attention (HAN-style): a task's completion is
  max(agv_ready, qc_ready) + processing, and a learned attention over the AGV-chain
  and QC-chain messages selects which resource gates the task. In a resolved schedule
  each task has a single predecessor per resource, under which the per-relation GATv2
  softmax is degenerate (always 1); the non-degenerate attention is the cross-resource
  one.
- Edge attention alpha is extracted (detached) per arc as the destination's
  semantic weight on that relation -- i.e. per-arc criticality. The max-attention
  agv arc is the bottleneck the guided mutation targets; its faithfulness is validated
  against the Oracle's exact critical path.

Semantic-tensor assertions guard every entry point: malformed node/edge tensors crash
loudly (SemanticTensorError) instead of silently cross-contaminating physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool

from ehgat.surrogate.graph import (
    AGV_EDGE,
    NODE_TYPE,
    QC_EDGE,
    assert_graph_semantics,
)
from ehgat.utils.assertions import (
    EDGE_FEATURES,
    NODE_FEATURES,
    assert_edge_features,
    assert_node_features,
)

__all__ = ["AttentionMap", "EHGATv2", "EHGATv2Config"]

NODE_DIM = len(NODE_FEATURES)  # 4
EDGE_DIM = len(EDGE_FEATURES)  # 3

# Per-arc attention: maps each edge type to (edge_index [2, E], alpha [E]) head-averaged.
AttentionMap = dict[str, tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class EHGATv2Config:
    """Architecture hyper-parameters."""

    hidden: int = 64
    layers: int = 3
    heads: int = 4
    out_dim: int = 2  # (makespan, energy)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden % self.heads != 0:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by heads ({self.heads})")


_NEG_INF = -1.0e9


class _HeteroSemanticAttnLayer(nn.Module):
    """One heterogeneous layer: max-plus GATv2 within each resource relation, then a
    learned semantic attention across relations (HAN-style).

    In a resolved schedule every task has at most one AGV and one QC predecessor, under
    which the per-relation GATv2 softmax over a single incoming arc is degenerate
    (always 1). The
    informative, non-degenerate signal is which resource chain gates the task -- the
    max-plus max(agv_ready, qc_ready) argmax. That is captured here by a semantic
    attention over the two relation messages, whose per-node weight on a relation is
    exposed as the criticality of that incoming arc (used by the guided mutation).
    """

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        out_per_head = hidden // heads
        kwargs = {
            "heads": heads,
            "concat": True,
            "edge_dim": EDGE_DIM,
            "add_self_loops": False,
            "dropout": dropout,
            "aggr": "max",  # max-plus longest-path reduction within a resource chain
        }
        self.agv_conv = GATv2Conv(hidden, out_per_head, **kwargs)
        self.qc_conv = GATv2Conv(hidden, out_per_head, **kwargs)
        # Semantic-level attention scorer shared across relations (HAN, Wang et al. 2019).
        self.sem_proj = nn.Linear(hidden, hidden)
        self.sem_query = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.sem_query, std=0.1)

    def _semantic_score(self, message: Tensor) -> Tensor:
        """Per-node scalar relevance of a relation's message [N, hidden] -> [N]."""
        return (torch.tanh(self.sem_proj(message)) * self.sem_query).sum(dim=-1)

    def forward(
        self,
        x: Tensor,
        agv_index: Tensor,
        agv_attr: Tensor,
        qc_index: Tensor,
        qc_attr: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, AttentionMap | None]:
        n = x.shape[0]
        out_agv, _ = self.agv_conv(x, agv_index, agv_attr, return_attention_weights=True)

        if qc_index.numel() > 0:
            out_qc, _ = self.qc_conv(x, qc_index, qc_attr, return_attention_weights=True)
            has_qc = torch.zeros(n, dtype=torch.bool, device=x.device)
            has_qc[qc_index[1]] = True
            s_agv = self._semantic_score(out_agv)
            s_qc = torch.where(has_qc, self._semantic_score(out_qc), x.new_full((n,), _NEG_INF))
            weights = torch.softmax(torch.stack([s_agv, s_qc], dim=-1), dim=-1)  # [N, 2]
            w_agv, w_qc = weights[:, 0:1], weights[:, 1:2]
            out = w_agv * out_agv + w_qc * torch.nan_to_num(out_qc)
        else:
            w_agv = x.new_ones((n, 1))
            w_qc = x.new_zeros((n, 1))
            out = out_agv

        attention: AttentionMap | None = None
        if return_attention:
            # Per-arc criticality = the semantic weight its destination places on that
            # relation. AGV arcs the search targets; QC arcs are the serialisation chain.
            attention = {
                AGV_EDGE[1]: (agv_index.detach(), w_agv.detach().squeeze(-1)[agv_index[1]]),
            }
            if qc_index.numel() > 0:
                attention[QC_EDGE[1]] = (
                    qc_index.detach(),
                    w_qc.detach().squeeze(-1)[qc_index[1]],
                )
            else:
                empty_ei = torch.empty((2, 0), dtype=torch.long, device=x.device)
                attention[QC_EDGE[1]] = (empty_ei, torch.empty(0, device=x.device))
        return out, attention


class EHGATv2(nn.Module):
    """Heterogeneous max-plus GATv2 regressor for (C_max, E) with edge attention."""

    # Declared so the registered buffers type as Tensor (not nn.Module.__getattr__'s
    # Tensor | Module); the actual values are set by register_buffer below.
    node_mean: Tensor
    node_std: Tensor
    agv_mean: Tensor
    agv_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def __init__(self, config: EHGATv2Config | None = None) -> None:
        super().__init__()
        self.config = config or EHGATv2Config()
        hidden = self.config.hidden
        self.node_encoder = nn.Linear(NODE_DIM, hidden)
        self.layers = nn.ModuleList(
            _HeteroSemanticAttnLayer(hidden, self.config.heads, self.config.dropout)
            for _ in range(self.config.layers)
        )
        self.act = nn.ReLU()
        # Readout sees (1) mean + max pooled node embeddings from the max-plus message
        # passing -- the longest-path signal for the makespan -- and (2) a SUM over AGV
        # arc features, since energy is additive over arcs and would otherwise be lost
        # by the max aggregation. This dual readout matches both objectives' physics.
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden + EDGE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.out_dim),
        )

        # Standardisation baked in as (non-trainable) buffers so RAW physical graphs can
        # be fed at inference by the guided search while training stays well-conditioned. Features
        # are normalised on input; predictions are produced in normalised-target space
        # (balanced MSE across makespan/energy) and de-normalised by predict.
        self.register_buffer("node_mean", torch.zeros(NODE_DIM))
        self.register_buffer("node_std", torch.ones(NODE_DIM))
        self.register_buffer("agv_mean", torch.zeros(EDGE_DIM))
        self.register_buffer("agv_std", torch.ones(EDGE_DIM))
        self.register_buffer("target_mean", torch.zeros(self.config.out_dim))
        self.register_buffer("target_std", torch.ones(self.config.out_dim))

    def set_normalization(
        self,
        *,
        node_mean: Tensor,
        node_std: Tensor,
        agv_mean: Tensor,
        agv_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
    ) -> None:
        """Populate the standardisation buffers from training-set statistics."""
        eps = 1e-8
        self.node_mean.copy_(node_mean)
        self.node_std.copy_(node_std.clamp_min(eps))
        self.agv_mean.copy_(agv_mean)
        self.agv_std.copy_(agv_std.clamp_min(eps))
        self.target_mean.copy_(target_mean)
        self.target_std.copy_(target_std.clamp_min(eps))

    def forward(
        self, data: HeteroData, *, return_attention: bool = False
    ) -> tuple[Tensor, AttentionMap | None]:
        """Predict [num_graphs, out_dim]; optionally return last-layer attention.

        data may be a single HeteroData or a PyG mini-batch of them.
        """
        assert_graph_semantics(data)
        x = data[NODE_TYPE].x
        agv_index = data[AGV_EDGE].edge_index
        agv_attr = data[AGV_EDGE].edge_attr
        qc_index = data[QC_EDGE].edge_index
        qc_attr = data[QC_EDGE].edge_attr
        assert_node_features(x, name="forward.task.x")
        assert_edge_features(agv_attr, name="forward.agv.edge_attr")
        assert_edge_features(qc_attr, name="forward.qc.edge_attr")

        # Standardise inputs (QC arcs are structural zeros and are left untouched).
        x = (x - self.node_mean) / self.node_std
        agv_attr = (agv_attr - self.agv_mean) / self.agv_std

        h = self.act(self.node_encoder(x))
        attention: AttentionMap | None = None
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            message, attn = layer(
                h, agv_index, agv_attr, qc_index, qc_attr,
                return_attention=return_attention and last,
            )
            h = h + self.act(message)  # residual eases deep critical-path propagation
            if last:
                attention = attn

        batch = getattr(data[NODE_TYPE], "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = int(batch.max().item()) + 1
        # Additive energy branch: sum the (normalised) AGV arc features per graph.
        agv_edge_batch = batch[agv_index[1]]
        energy_pool = global_add_pool(agv_attr, agv_edge_batch, size=num_graphs)
        pooled = torch.cat(
            [
                global_mean_pool(h, batch, size=num_graphs),
                global_max_pool(h, batch, size=num_graphs),
                energy_pool,
            ],
            dim=-1,
        )
        out: Tensor = self.readout(pooled)
        return out, attention

    def encode(self, data: HeteroData) -> Tensor:
        """Return per-node embeddings h [N, hidden] after max-plus message passing.

        This exposes the frozen structural representation that the fused tropical head
        (ehgat.explain.fused_ehgat) projects into local physical attributes. It runs
        the same standardisation and residual GATv2 stack as forward but stops before the
        scalar readout, leaving the existing (C_max, E) pipeline unchanged.
        """
        assert_graph_semantics(data)
        x = data[NODE_TYPE].x
        agv_index = data[AGV_EDGE].edge_index
        agv_attr = data[AGV_EDGE].edge_attr
        qc_index = data[QC_EDGE].edge_index
        qc_attr = data[QC_EDGE].edge_attr
        assert_node_features(x, name="encode.task.x")
        assert_edge_features(agv_attr, name="encode.agv.edge_attr")
        assert_edge_features(qc_attr, name="encode.qc.edge_attr")

        x = (x - self.node_mean) / self.node_std
        agv_attr = (agv_attr - self.agv_mean) / self.agv_std

        h = self.act(self.node_encoder(x))
        for layer in self.layers:
            message, _ = layer(h, agv_index, agv_attr, qc_index, qc_attr, return_attention=False)
            h = h + self.act(message)
        return h

    @torch.no_grad()
    def predict(self, data: HeteroData) -> Tensor:
        """Return the physical [num_graphs, out_dim] (C_max, E) prediction."""
        self.eval()
        out, _ = self.forward(data)
        return out * self.target_std + self.target_mean

    @torch.no_grad()
    def attention(self, data: HeteroData) -> AttentionMap:
        """Return the detached last-layer per-arc attention map for a single graph."""
        self.eval()
        _, attn = self.forward(data, return_attention=True)
        assert attn is not None  # return_attention=True always populates it
        return attn
