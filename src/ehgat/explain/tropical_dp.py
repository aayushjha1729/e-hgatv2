"""Differentiable max-plus dynamic programming for DAG bottleneck attribution.

TropicalMaxPlus computes the longest-path value at each node of a DAG:

    x[v] = node_weight[v] + max_{u -> v}(x[u] + edge_weight[u, v])

The custom backward routes the subgradient only through the physical argmax edge of
each max-plus recurrence. Ties follow PyTorch's deterministic argmax convention:
the first maximum in edge order receives the subgradient.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["TropicalMaxPlus", "tropical_longest_path", "tropical_makespan"]

_NEG_INF = -1.0e30


def _topological_order(num_nodes: int, edge_index: Tensor) -> Tensor:
    """Return a deterministic Kahn topological order for edge_index."""
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    indeg = [0] * num_nodes
    succ: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst, strict=True):
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge ({u}, {v}) outside 0..{num_nodes - 1}")
        indeg[v] += 1
        succ[u].append(v)
    ready = [i for i, deg in enumerate(indeg) if deg == 0]
    order: list[int] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if len(order) != num_nodes:
        raise ValueError("edge_index is not a DAG; tropical DP requires acyclic precedence")
    return torch.tensor(order, dtype=torch.long, device=edge_index.device)


class TropicalMaxPlus(torch.autograd.Function):
    """Custom autograd for max-plus longest paths on a DAG."""

    @staticmethod
    def forward(ctx, node_weights: Tensor, edge_index: Tensor, edge_weights: Tensor) -> Tensor:
        if node_weights.ndim != 1:
            raise ValueError(f"node_weights must be [N], got {tuple(node_weights.shape)}")
        if edge_weights.ndim != 1:
            raise ValueError(f"edge_weights must be [E], got {tuple(edge_weights.shape)}")
        if edge_index.shape != (2, edge_weights.numel()):
            raise ValueError("edge_index and edge_weights disagree on edge count")

        n = int(node_weights.numel())
        order = _topological_order(n, edge_index)
        x = torch.empty_like(node_weights)
        arg_edge = torch.full((n,), -1, dtype=torch.long, device=edge_index.device)
        incoming = [[] for _ in range(n)]
        for e, v in enumerate(edge_index[1].detach().cpu().tolist()):
            incoming[v].append(e)

        for v_t in order:
            v = int(v_t.item())
            if not incoming[v]:
                x[v] = node_weights[v]
                continue
            inc = torch.tensor(incoming[v], dtype=torch.long, device=edge_index.device)
            src = edge_index[0, inc]
            vals = x[src] + edge_weights[inc]
            best = int(torch.argmax(vals).item())
            arg_edge[v] = inc[best]
            x[v] = node_weights[v] + vals[best]

        ctx.save_for_backward(edge_index, arg_edge, order)
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, Tensor]:
        edge_index, arg_edge, order = ctx.saved_tensors
        grad_nodes = grad_output.clone()
        grad_node_weights = grad_output.clone()
        grad_edges = torch.zeros(edge_index.shape[1], dtype=grad_output.dtype, device=grad_output.device)

        for v_t in torch.flip(order, dims=(0,)):
            v = int(v_t.item())
            e = int(arg_edge[v].item())
            if e < 0:
                continue
            g = grad_nodes[v]
            grad_edges[e] += g
            grad_nodes[int(edge_index[0, e].item())] += g

        grad_node_weights += grad_nodes - grad_output
        return grad_node_weights, None, grad_edges


def tropical_longest_path(node_weights: Tensor, edge_index: Tensor, edge_weights: Tensor) -> Tensor:
    """Completion values for every node of a max-plus DAG."""
    return TropicalMaxPlus.apply(node_weights, edge_index.long(), edge_weights)


def tropical_makespan(node_weights: Tensor, edge_index: Tensor, edge_weights: Tensor) -> Tensor:
    """Scalar max over terminal completion values, with tropical subgradients."""
    return tropical_longest_path(node_weights, edge_index, edge_weights).max()
