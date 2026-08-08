"""TAPE extraction engine: exact tropical gradients for resolved schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ehgat.environment.decoder import Schedule
from ehgat.environment.evaluator import build_precedence, evaluate
from ehgat.environment.instance import Instance, TaskKind
from ehgat.environment.physics import leg_energy, travel_time
from ehgat.explain.event_dag import assemble_coupled_event_dag, assemble_event_dag
from ehgat.explain.tropical_dp import tropical_longest_path
from ehgat.surrogate.graph import AGV_EDGE, NODE_TYPE, QC_EDGE, build_hetero_graph

__all__ = [
    "TapeExplanation",
    "explain_schedule",
    "explain_schedule_coupled",
    "explain_schedules",
    "surrogate_feature_gradients",
]


@dataclass(frozen=True, slots=True)
class TapeExplanation:
    """Gradients of exact tropical makespan and additive energy for one schedule."""

    makespan: float
    energy: float
    node_grad: tuple[float, ...]
    empty_time_grad: tuple[float, ...]
    loaded_time_grad: tuple[float, ...]
    empty_energy_grad: tuple[float, ...]
    loaded_energy_grad: tuple[float, ...]
    event_edges: tuple[dict[str, Any], ...]
    event_edge_grad: tuple[float, ...]
    completion_nodes: tuple[int, ...]
    surrogate_grad: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": {"makespan": self.makespan, "energy": self.energy},
            "tropical": {
                "node_grad": list(self.node_grad),
                "empty_time_grad": list(self.empty_time_grad),
                "loaded_time_grad": list(self.loaded_time_grad),
                "event_edge_grad": list(self.event_edge_grad),
                "event_edges": list(self.event_edges),
                "completion_nodes": list(self.completion_nodes),
            },
            "energy": {
                "empty_energy_grad": list(self.empty_energy_grad),
                "loaded_energy_grad": list(self.loaded_energy_grad),
            },
            "surrogate_grad": self.surrogate_grad,
        }


def _leg_tensors(schedule: Schedule, instance: Instance, *, dtype: torch.dtype) -> tuple[Tensor, ...]:
    empty_t, loaded_t, empty_e, loaded_e = [], [], [], []
    agv_prev, _qc_prev, _ = build_precedence(schedule.agv_sequences, schedule.qc_sequences, instance.num_tasks)
    for j, task in enumerate(instance.tasks):
        origin = instance.agv_start if agv_prev[j] < 0 else instance.tasks[agv_prev[j]].dropoff
        empty_d = instance.distance.distance(origin, task.pickup)
        loaded_d = instance.loaded_distance(task)
        empty_t.append(travel_time(empty_d, schedule.empty_speed[j], loaded=False))
        loaded_t.append(travel_time(loaded_d, schedule.loaded_speed[j], loaded=True))
        empty_e.append(leg_energy(empty_d, schedule.empty_speed[j], loaded=False))
        loaded_e.append(leg_energy(loaded_d, schedule.loaded_speed[j], loaded=True))
    leaves = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in (empty_t, loaded_t, empty_e, loaded_e)]
    return tuple(leaves)  # type: ignore[return-value]


def _tau_tensor(instance: Instance, *, dtype: torch.dtype) -> Tensor:
    """Per-task quay-crane handling delay tau as a differentiable leaf [N]."""
    handling = [float(task.handling_time) for task in instance.tasks]
    return torch.tensor(handling, dtype=dtype, requires_grad=True)


def explain_schedule(
    schedule: Schedule, instance: Instance, model: nn.Module | None = None, *, dtype: torch.dtype = torch.float64
) -> TapeExplanation:
    """Extract exact TAPE gradients; optionally attach frozen-surrogate feature gradients.

    The makespan path is the exact max-plus longest path; its subgradients w.r.t. the
    per-task leg times (empty_time_grad/loaded_time_grad), node handling
    (node_grad = dC/dtau) and event arcs are binary critical-path indicators. The
    energy is exact-additive, so its leg gradients are all 1.
    """
    empty_t, loaded_t, empty_e, loaded_e = _leg_tensors(schedule, instance, dtype=dtype)
    tau = _tau_tensor(instance, dtype=dtype)
    agv_prev, qc_prev, _ = build_precedence(schedule.agv_sequences, schedule.qc_sequences, instance.num_tasks)
    is_load = torch.tensor(
        [task.kind is TaskKind.LOAD for task in instance.tasks], dtype=torch.bool
    )
    dag = assemble_event_dag(is_load, agv_prev, qc_prev, empty_t, loaded_t, tau)

    x = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
    makespan = x[dag.completion_nodes].max()
    energy = (empty_e + loaded_e).sum()

    dag.edge_weights.retain_grad()
    makespan.backward(retain_graph=True)
    node_grad = tuple(float(v) for v in tau.grad.detach())
    edge_grad = tuple(float(v) for v in dag.edge_weights.grad.detach())
    empty_time_grad = tuple(float(v) for v in empty_t.grad.detach())
    loaded_time_grad = tuple(float(v) for v in loaded_t.grad.detach())

    energy.backward()
    return TapeExplanation(
        makespan=float(makespan.detach()),
        energy=float(energy.detach()),
        node_grad=node_grad,
        empty_time_grad=empty_time_grad,
        loaded_time_grad=loaded_time_grad,
        empty_energy_grad=tuple(float(v) for v in empty_e.grad.detach()),
        loaded_energy_grad=tuple(float(v) for v in loaded_e.grad.detach()),
        event_edges=tuple(dag.meta),
        event_edge_grad=edge_grad,
        completion_nodes=tuple(int(v) for v in dag.completion_nodes.tolist()),
        surrogate_grad=surrogate_feature_gradients(model, schedule, instance) if model is not None else None,
    )


def explain_schedule_coupled(
    schedule: Schedule, instance: Instance, *, dtype: torch.dtype = torch.float64
) -> TapeExplanation:
    """Exact TAPE oracle under peak-power coupling.

    The simulator's per-leg power waits are folded into each leg's effective duration
    (leg_time + wait) so the precedence-only coupled activity DAG reproduces the coupled
    makespan exactly; its subgradients w.r.t. the leg times are the true coupled
    critical-path indicators. tau (QC handling) stays exact and differentiable.
    """
    ev = evaluate(schedule, instance)
    empty_t, loaded_t, empty_e, loaded_e = _leg_tensors(schedule, instance, dtype=dtype)
    tau = _tau_tensor(instance, dtype=dtype)
    agv_prev, qc_prev, _ = build_precedence(
        schedule.agv_sequences, schedule.qc_sequences, instance.num_tasks
    )
    is_load = torch.tensor(
        [task.kind is TaskKind.LOAD for task in instance.tasks], dtype=torch.bool
    )
    wait_e = torch.tensor(ev.wait_empty, dtype=dtype)
    wait_l = torch.tensor(ev.wait_loaded, dtype=dtype)
    dag = assemble_coupled_event_dag(
        is_load, agv_prev, qc_prev, empty_t + wait_e, loaded_t + wait_l, tau, []
    )

    x = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
    makespan = x[dag.completion_nodes].max()
    energy = (empty_e + loaded_e).sum()

    # Coupled DAG: leg times are node weights; structural edges carry no gradient.
    if dag.edge_weights.requires_grad:
        dag.edge_weights.retain_grad()
    makespan.backward(retain_graph=True)
    node_grad = tuple(float(v) for v in tau.grad.detach())
    edge_grad = (
        tuple(float(v) for v in dag.edge_weights.grad.detach())
        if dag.edge_weights.grad is not None
        else tuple(0.0 for _ in range(dag.edge_weights.numel()))
    )
    empty_time_grad = tuple(float(v) for v in empty_t.grad.detach())
    loaded_time_grad = tuple(float(v) for v in loaded_t.grad.detach())

    energy.backward()
    return TapeExplanation(
        makespan=float(makespan.detach()),
        energy=float(energy.detach()),
        node_grad=node_grad,
        empty_time_grad=empty_time_grad,
        loaded_time_grad=loaded_time_grad,
        empty_energy_grad=tuple(float(v) for v in empty_e.grad.detach()),
        loaded_energy_grad=tuple(float(v) for v in loaded_e.grad.detach()),
        event_edges=tuple(dag.meta),
        event_edge_grad=edge_grad,
        completion_nodes=tuple(int(v) for v in dag.completion_nodes.tolist()),
        surrogate_grad=None,
    )


def explain_schedules(
    schedules: list[Schedule], instance: Instance, model: nn.Module | None = None
) -> list[TapeExplanation]:
    """Run TAPE on a Pareto set or final-generation schedule list."""
    return [explain_schedule(s, instance, model) for s in schedules]


def surrogate_feature_gradients(model: nn.Module, schedule: Schedule, instance: Instance) -> dict[str, Any]:
    """Frozen E-HGATv2 input-feature gradients (model-faithfulness, not the TAPE oracle)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    data = build_hetero_graph(schedule, instance)
    data[NODE_TYPE].x.requires_grad_(True)
    data[AGV_EDGE].edge_attr.requires_grad_(True)
    data[QC_EDGE].edge_attr.requires_grad_(True)
    out, _ = model(data)
    pred = out * model.target_std + model.target_mean  # physical units
    pred[0, 0].backward(retain_graph=True)
    c_node = data[NODE_TYPE].x.grad.detach().clone()
    c_agv = data[AGV_EDGE].edge_attr.grad.detach().clone()
    data[NODE_TYPE].x.grad.zero_(); data[AGV_EDGE].edge_attr.grad.zero_()
    if data[QC_EDGE].edge_attr.grad is not None:
        data[QC_EDGE].edge_attr.grad.zero_()
    pred[0, 1].backward()
    return {
        "makespan_node_grad": c_node.tolist(),
        "makespan_agv_edge_grad": c_agv.tolist(),
        "energy_node_grad": data[NODE_TYPE].x.grad.detach().tolist(),
        "energy_agv_edge_grad": data[AGV_EDGE].edge_attr.grad.detach().tolist(),
    }
