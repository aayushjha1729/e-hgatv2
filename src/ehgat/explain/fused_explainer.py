"""Pure-gradient extraction for the fused E-HGATv2 (the model-native TAPE).

Because ehgat.explain.fused_ehgat.FusedEHGATv2 routes C_max through the exact
max-plus DP, dC_max/d(leg time) is a binary critical-path indicator with no
numerical smearing -- the network's own gradients are the explanation. This module:

- runs the fused model and extracts dC_max / dE w.r.t. the predicted local leg
  attributes, packaged as a ehgat.explain.tape_explainer.TapeExplanation (so the
  shared ehgat.explain.tcs_calculator.tradeoff_criticality_scores works unchanged), and
- validates faithfulness by comparing the fused model's critical path against the exact
  simulator-based TAPE oracle on the same schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ehgat.environment.decoder import Schedule
from ehgat.environment.instance import Instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.explain.tcs_calculator import ParetoPoint, tradeoff_criticality_scores
from ehgat.explain.tape_explainer import (
    TapeExplanation,
    explain_schedule,
    explain_schedule_coupled,
)
from ehgat.surrogate.graph import AGV_EDGE, NODE_TYPE, build_hetero_graph

__all__ = [
    "FaithfulnessReport",
    "explain_fused",
    "explain_fused_batch",
    "explain_fused_schedules",
    "faithfulness_report",
    "fused_tradeoff_criticality_scores",
]


def _grad_tuple(t: torch.Tensor) -> tuple[float, ...]:
    if t.grad is None:
        return tuple(0.0 for _ in range(t.numel()))
    return tuple(float(v) for v in t.grad.detach().cpu())


def explain_fused(
    model: FusedEHGATv2, schedule: Schedule, instance: Instance
) -> TapeExplanation:
    """Extract the fused model's native tropical gradients as a TapeExplanation.

    The returned empty_time_grad / loaded_time_grad / event_edge_grad are
    dC_max/d(predicted leg) -- exact binary critical-path indicators -- and the
    energy gradients are dE/d(predicted leg energy) = 1 by additive construction.
    """
    model.eval()
    device = next(model.parameters()).device
    data = build_hetero_graph(schedule, instance).to(device)
    # Leg energies are read straight from the input arc features; make them differentiable
    # so dE/d(leg energy) propagates (it is exactly 1 by additive construction). Set the flag
    # AFTER moving to device so edge_attr stays a leaf tensor on the model's device.
    data[AGV_EDGE].edge_attr.requires_grad_(True)
    out = model(data)

    for leaf in (out.empty_t, out.loaded_t, out.empty_e, out.loaded_e, out.node_delay):
        leaf.retain_grad()
    # In the coupled DAG leg times are NODE weights and the edges are zero-weight structural
    # arcs (requires_grad=False) -- the critical-path signal lives in the leg-time grads, so
    # only retain edge grads when they carry gradient (uncoupled leg-on-edge DAG).
    if out.dag.edge_weights.requires_grad:
        out.dag.edge_weights.retain_grad()

    out.makespan.backward(retain_graph=True)
    node_grad = _grad_tuple(out.node_delay)
    empty_time_grad = _grad_tuple(out.empty_t)
    loaded_time_grad = _grad_tuple(out.loaded_t)
    event_edge_grad = _grad_tuple(out.dag.edge_weights)

    # Leg energies feed both the makespan (via the leg-time prior) and the energy, so clear
    # their accumulated makespan grads before the energy pass, isolating dE/d(leg)=1.
    for leaf in (out.empty_e, out.loaded_e):
        leaf.grad = None
    out.energy.backward()
    return TapeExplanation(
        makespan=float(out.makespan.detach()),
        energy=float(out.energy.detach()),
        node_grad=node_grad,
        empty_time_grad=empty_time_grad,
        loaded_time_grad=loaded_time_grad,
        empty_energy_grad=_grad_tuple(out.empty_e),
        loaded_energy_grad=_grad_tuple(out.loaded_e),
        event_edges=tuple(out.dag.meta),
        event_edge_grad=event_edge_grad,
        completion_nodes=tuple(int(v) for v in out.dag.completion_nodes.tolist()),
        surrogate_grad=None,
    )


def explain_fused_batch(
    model: FusedEHGATv2,
    schedules: list[Schedule],
    instance: Instance,
    *,
    chunk_size: int = 128,
) -> list[TapeExplanation]:
    """Batched TAPE for many schedules -- identical result to per-schedule explain_fused,
    ~O(K)x faster on the GNN pass.

    The dominant cost in explain_fused is the frozen-core message-passing encode; the
    tropical DP is comparatively cheap. A whole batch of graphs is encoded in one forward
    (model.core.encode), the per-graph physics head and DP then run on each graph's
    precomputed embedding slice (the h= fast path of FusedEHGATv2.forward), and a single
    summed-makespan backward is taken. The batched graphs are disjoint, under which the
    gradient of sum_i C_max^{(i)} with respect to graph i's legs equals dC_max^{(i)}/d(leg_i)
    exactly,
    with no cross terms; the per-graph explanations are bit-identical to the serial path at
    one GNN forward and one backward per chunk instead of K. Chunking bounds autograd
    memory.
    """
    from torch_geometric.data import Batch

    model.eval()
    device = next(model.parameters()).device
    results: list[TapeExplanation] = []
    for start in range(0, len(schedules), chunk_size):
        chunk = schedules[start:start + chunk_size]
        graphs = [build_hetero_graph(s, instance).to(device) for s in chunk]
        for g in graphs:
            g[AGV_EDGE].edge_attr.requires_grad_(True)  # after .to(): stays a leaf on device
        batch = Batch.from_data_list(graphs)
        h_all = model.core.encode(batch)               # one batched message-passing forward
        ptr = batch[NODE_TYPE].ptr.tolist()            # node offsets per graph
        outs = [model(g, h=h_all[ptr[i]:ptr[i + 1]]) for i, g in enumerate(graphs)]
        for out in outs:
            for leaf in (out.empty_t, out.loaded_t, out.empty_e, out.loaded_e, out.node_delay):
                leaf.retain_grad()
            if out.dag.edge_weights.requires_grad:
                out.dag.edge_weights.retain_grad()

        # One backward over the summed makespan -> per-graph dC_max/d(leg) (disjoint => exact).
        sum(o.makespan for o in outs).backward(retain_graph=True)
        m_grads = [(_grad_tuple(o.node_delay), _grad_tuple(o.empty_t),
                    _grad_tuple(o.loaded_t), _grad_tuple(o.dag.edge_weights)) for o in outs]
        # Clear leg-energy makespan grads, then one energy backward for dE/d(leg)=1.
        for o in outs:
            for leaf in (o.empty_e, o.loaded_e):
                leaf.grad = None
        sum(o.energy for o in outs).backward()
        for o, (node_grad, empty_time_grad, loaded_time_grad, event_edge_grad) in zip(outs, m_grads):
            results.append(TapeExplanation(
                makespan=float(o.makespan.detach()), energy=float(o.energy.detach()),
                node_grad=node_grad, empty_time_grad=empty_time_grad,
                loaded_time_grad=loaded_time_grad,
                empty_energy_grad=_grad_tuple(o.empty_e), loaded_energy_grad=_grad_tuple(o.loaded_e),
                event_edges=tuple(o.dag.meta), event_edge_grad=event_edge_grad,
                completion_nodes=tuple(int(v) for v in o.dag.completion_nodes.tolist()),
                surrogate_grad=None,
            ))
    return results


def explain_fused_schedules(
    model: FusedEHGATv2, schedules: list[Schedule], instance: Instance
) -> list[TapeExplanation]:
    """Run the fused explainer over a Pareto set / final-generation schedule list."""
    return [explain_fused(model, s, instance) for s in schedules]


def fused_tradeoff_criticality_scores(
    model: FusedEHGATv2, schedules: list[Schedule], instance: Instance
) -> list[dict[str, Any]]:
    """Compute Trade-off Criticality Scores from the fused model's native gradients."""
    points = [
        ParetoPoint(str(i), ex.makespan, ex.energy, ex)
        for i, ex in enumerate(explain_fused_schedules(model, schedules, instance))
    ]
    return tradeoff_criticality_scores(points)


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Agreement of the fused model's critical path with the exact TAPE oracle."""

    makespan_abs_error: float
    energy_abs_error: float
    leg_critical_jaccard: float
    arc_critical_jaccard: float

    def to_dict(self) -> dict[str, float]:
        return {
            "makespan_abs_error": self.makespan_abs_error,
            "energy_abs_error": self.energy_abs_error,
            "leg_critical_jaccard": self.leg_critical_jaccard,
            "arc_critical_jaccard": self.arc_critical_jaccard,
        }


def _critical_set(grads: tuple[float, ...], *, threshold: float = 0.5) -> set[int]:
    return {i for i, g in enumerate(grads) if g > threshold}


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def faithfulness_report(
    model: FusedEHGATv2, schedule: Schedule, instance: Instance
) -> FaithfulnessReport:
    """Compare the fused model's critical path to the exact simulator-based TAPE oracle.

    A leg_critical_jaccard near 1 indicates that the fused (anchored) model recovers the
    same binding legs the exact max-plus oracle identifies, the faithfulness that follows by
    construction once the local physical attributes are anchored.
    """
    fused = explain_fused(model, schedule, instance)
    # Under coupling the true critical path runs over the coupled activity DAG (leg+wait
    # effective weights); compare against the matching coupled oracle, not the uncoupled one.
    if instance.peak_power is not None:
        exact = explain_schedule_coupled(schedule, instance)
    else:
        exact = explain_schedule(schedule, instance)

    fused_legs = _critical_set(fused.empty_time_grad) | {
        i + len(fused.empty_time_grad) for i in _critical_set(fused.loaded_time_grad)
    }
    exact_legs = _critical_set(exact.empty_time_grad) | {
        i + len(exact.empty_time_grad) for i in _critical_set(exact.loaded_time_grad)
    }
    arc_fused = _critical_set(fused.event_edge_grad)
    arc_exact = _critical_set(exact.event_edge_grad)

    return FaithfulnessReport(
        makespan_abs_error=abs(fused.makespan - exact.makespan),
        energy_abs_error=abs(fused.energy - exact.energy),
        leg_critical_jaccard=_jaccard(fused_legs, exact_legs),
        arc_critical_jaccard=_jaccard(arc_fused, arc_exact),
    )
