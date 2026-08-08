"""Batched, layer-vectorised max-plus longest path (same semantics as tropical_dp).

The per-graph ehgat.explain.tropical_dp.TropicalMaxPlus runs a Python loop over
every node — fine for one small DAG, but the dominant cost when training over thousands of
graphs for many epochs. This module computes the identical longest path and tropical
subgradient, but:

- batches an arbitrary number of DAGs as one block-diagonal graph (their longest paths are
  independent because the components are disconnected), and
- processes nodes by longest-path rank/layer: every edge strictly increases rank, so all
  nodes at a given rank are resolved in one vectorised scatter_reduce(amax) step.

The Python loop length therefore drops from sum_k N_k to max_k depth_k — independent of
the batch size — and every step is a tensor op that runs on GPU. The static schedule (ranks,
per-layer node/edge groupings) depends on graph structure alone and is built once and
reused across epochs; the leg-time and wait values change, the topology does not.

Tie-breaking matches the reference: the smallest edge index among equal maxima carries the
subgradient (binary 0/1 critical-path indicator).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ehgat.explain.tropical_dp import _topological_order

__all__ = ["BatchSchedule", "build_batch_schedule", "batched_longest_path"]

_NEG_INF = -1.0e30


@dataclass(frozen=True)
class BatchSchedule:
    """Pre-computed, structure-only schedule for a block-diagonal batch of DAGs.

    layers is in ascending rank (rank 0 = source nodes, already equal to their node
    weight). Each entry is (nodes, edges, dst_compact): the global node ids resolved at
    that rank, the global incoming-edge ids feeding them, and the compacted (0..M-1) index of
    each edge's destination within nodes.
    """

    edge_index: Tensor
    num_nodes: int
    layers: tuple[tuple[Tensor, Tensor, Tensor], ...]


def build_batch_schedule(edge_index: Tensor, num_nodes: int) -> BatchSchedule:
    """Compute longest-path ranks and per-rank node/edge groupings for edge_index."""
    edge_index = edge_index.long()
    order = _topological_order(num_nodes, edge_index)  # validates the DAG
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    preds: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src_list, dst_list, strict=True):
        preds[v].append(u)

    rank = [0] * num_nodes
    for v_t in order:
        v = int(v_t)
        if preds[v]:
            rank[v] = max(rank[u] + 1 for u in preds[v])
    max_rank = max(rank) if num_nodes else 0

    # Group edges by their destination's rank (every edge lands in dst's layer).
    edges_by_rank: list[list[int]] = [[] for _ in range(max_rank + 1)]
    for e, v in enumerate(dst_list):
        edges_by_rank[rank[v]].append(e)
    nodes_by_rank: list[list[int]] = [[] for _ in range(max_rank + 1)]
    for v in range(num_nodes):
        nodes_by_rank[rank[v]].append(v)

    dev = edge_index.device
    dst = edge_index[1]
    layers: list[tuple[Tensor, Tensor, Tensor]] = []
    for r in range(1, max_rank + 1):
        nodes_L = torch.tensor(sorted(nodes_by_rank[r]), dtype=torch.long, device=dev)
        edges_L = torch.tensor(edges_by_rank[r], dtype=torch.long, device=dev)
        if edges_L.numel() == 0:
            continue
        # Compact each edge's destination into 0..len(nodes_L)-1 (nodes_L is sorted).
        dst_compact = torch.searchsorted(nodes_L, dst[edges_L])
        layers.append((nodes_L, edges_L, dst_compact))
    return BatchSchedule(edge_index=edge_index, num_nodes=num_nodes, layers=tuple(layers))


class _BatchedTropical(torch.autograd.Function):
    """Vectorised max-plus longest path with the reference's binary subgradient."""

    @staticmethod
    def forward(ctx, node_weights: Tensor, edge_weights: Tensor, schedule: BatchSchedule) -> Tensor:
        edge_index = schedule.edge_index
        x = node_weights.clone()
        arg_edge = torch.full((schedule.num_nodes,), -1, dtype=torch.long, device=x.device)
        for nodes_L, edges_L, dst_compact in schedule.layers:
            src = edge_index[0, edges_L]
            cand = x[src] + edge_weights[edges_L]
            m = nodes_L.numel()
            maxval = cand.new_full((m,), _NEG_INF)
            maxval.scatter_reduce_(0, dst_compact, cand, reduce="amax", include_self=True)
            # Smallest edge index among equal maxima -> matches reference first-argmax tie-break.
            is_max = cand >= maxval[dst_compact] - 1e-9
            local = torch.arange(edges_L.numel(), device=x.device)
            arg_local = local.new_full((m,), edges_L.numel())
            arg_local.scatter_reduce_(
                0, dst_compact[is_max], local[is_max], reduce="amin", include_self=True
            )
            x[nodes_L] = node_weights[nodes_L] + maxval
            arg_edge[nodes_L] = edges_L[arg_local]
        ctx.save_for_backward(edge_index, arg_edge)
        ctx.schedule = schedule
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        edge_index, arg_edge = ctx.saved_tensors
        schedule: BatchSchedule = ctx.schedule
        grad_nodes = grad_output.clone()
        grad_edges = torch.zeros(
            edge_index.shape[1], dtype=grad_output.dtype, device=grad_output.device
        )
        for nodes_L, _edges_L, _dst in reversed(schedule.layers):
            e = arg_edge[nodes_L]
            valid = e >= 0
            if not bool(valid.any()):
                continue
            nodes_v = nodes_L[valid]
            e_v = e[valid]
            g = grad_nodes[nodes_v]
            grad_edges.index_add_(0, e_v, g)
            grad_nodes.index_add_(0, edge_index[0, e_v], g)
        return grad_nodes, grad_edges, None


def batched_longest_path(
    node_weights: Tensor, edge_weights: Tensor, schedule: BatchSchedule
) -> Tensor:
    """Completion values for every node of a block-diagonal batch of max-plus DAGs."""
    return _BatchedTropical.apply(node_weights, edge_weights, schedule)
