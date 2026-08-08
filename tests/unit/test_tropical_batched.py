"""The batched layer-vectorised tropical DP must equal the per-graph reference exactly."""

from __future__ import annotations

import torch

from ehgat.explain.tropical_dp import tropical_longest_path
from ehgat.explain.tropical_dp_batched import batched_longest_path, build_batch_schedule


def _random_dag(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random DAG on n nodes (edges only i->j with i<j) + random weights."""
    src, dst = [], []
    for j in range(1, n):
        # each node draws a few predecessors from earlier nodes
        k = int(torch.randint(1, min(j, 3) + 1, (1,), generator=gen).item())
        preds = torch.randperm(j, generator=gen)[:k].tolist()
        for u in preds:
            src.append(u)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    node_w = torch.rand(n, generator=gen, dtype=torch.float64) * 10
    edge_w = torch.rand(len(src), generator=gen, dtype=torch.float64) * 5
    return node_w, edge_index, edge_w


def test_batched_matches_reference_values_and_grads() -> None:
    gen = torch.Generator().manual_seed(0)
    sizes = [4, 7, 5, 9, 6]
    node_ws, edge_ws, local_eis, offset_eis = [], [], [], []
    offset = 0
    for n in sizes:
        nw, ei, ew = _random_dag(n, gen)
        node_ws.append(nw)
        edge_ws.append(ew)
        local_eis.append(ei)
        offset_eis.append(ei + offset)
        offset += n

    # Per-graph reference: run each DAG independently (LOCAL indices), collect values + grads.
    ref_x, ref_gn, ref_ge = [], [], []
    for nw, ei, ew in zip(node_ws, local_eis, edge_ws, strict=True):
        nwv = nw.clone().requires_grad_(True)
        ewv = ew.clone().requires_grad_(True)
        x = tropical_longest_path(nwv, ei, ewv)
        x.sum().backward()
        ref_x.append(x.detach())
        ref_gn.append(nwv.grad)
        ref_ge.append(ewv.grad)
    ref_x = torch.cat(ref_x)
    ref_gn = torch.cat(ref_gn)
    ref_ge = torch.cat(ref_ge)

    # Batched: one block-diagonal graph (OFFSET indices).
    bnw = torch.cat(node_ws).clone().requires_grad_(True)
    bew = torch.cat(edge_ws).clone().requires_grad_(True)
    bei = torch.cat(offset_eis, dim=1)
    sched = build_batch_schedule(bei, bnw.numel())
    bx = batched_longest_path(bnw, bew, sched)
    bx.sum().backward()

    assert torch.allclose(bx.detach(), ref_x, atol=1e-9), "batched values diverge from reference"
    assert torch.allclose(bnw.grad, ref_gn, atol=1e-9), "node-weight grads diverge"
    assert torch.allclose(bew.grad, ref_ge, atol=1e-9), "edge-weight grads diverge"
    # Node-weight subgradients are binary counts (number of critical descendants); >=1 reachable.
    assert torch.all(bnw.grad >= 0)
