"""The coupled event DAG + tropical longest path reproduces the simulator exactly.

This is the faithfulness backbone for the nonlinear regime: given the simulator's
resolved power-resolution arcs, the max-plus longest path over the activity-end DAG
equals the coupled makespan to floating-point tolerance, and its subgradient is the
true coupled critical path.
"""

from __future__ import annotations

import dataclasses

import torch

from ehgat.environment.decoder import decode
from ehgat.environment.evaluator import build_precedence, evaluate
from ehgat.environment.instance import build_toy_instance, TaskKind
from ehgat.environment.physics import leg_energy, travel_time
from ehgat.explain.event_dag import assemble_coupled_event_dag
from ehgat.explain.tropical_dp import tropical_longest_path
from ehgat.utils.seeding import make_rng


def _legs_tau(schedule, instance):
    n = instance.num_tasks
    agv_prev, _qc, _ = build_precedence(schedule.agv_sequences, schedule.qc_sequences, n)
    empty_t, loaded_t = [], []
    for j, task in enumerate(instance.tasks):
        origin = instance.agv_start if agv_prev[j] < 0 else instance.tasks[agv_prev[j]].dropoff
        ed = instance.distance.distance(origin, task.pickup)
        ld = instance.loaded_distance(task)
        empty_t.append(travel_time(ed, schedule.empty_speed[j], loaded=False))
        loaded_t.append(travel_time(ld, schedule.loaded_speed[j], loaded=True))
    tau = [float(t.handling_time) for t in instance.tasks]
    return (
        torch.tensor(empty_t, dtype=torch.float64),
        torch.tensor(loaded_t, dtype=torch.float64),
        torch.tensor(tau, dtype=torch.float64),
    )


def test_coupled_dag_reproduces_simulator_makespan() -> None:
    base = build_toy_instance(num_tasks=8)
    for budget in (25.0, 30.0, 45.0):
        inst = dataclasses.replace(base, peak_power=budget)
        rng = make_rng(int(budget))  # vary schedules per budget
        for _ in range(30):
            sched = decode(rng.random(4 * inst.num_tasks), inst)
            ev = evaluate(sched, inst)
            agv_prev, qc_prev, _ = build_precedence(
                sched.agv_sequences, sched.qc_sequences, inst.num_tasks
            )
            is_load = torch.tensor(
                [t.kind is TaskKind.LOAD for t in inst.tasks], dtype=torch.bool
            )
            empty_t, loaded_t, tau = _legs_tau(sched, inst)
            dag = assemble_coupled_event_dag(
                is_load, agv_prev, qc_prev, empty_t, loaded_t, tau, list(ev.power_arcs)
            )
            x = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
            makespan = float(x[dag.completion_nodes].max())
            assert abs(makespan - ev.makespan) < 1e-6, (
                f"budget={budget}: dag {makespan} vs sim {ev.makespan}"
            )


def test_effective_leg_waits_reproduce_makespan_without_arcs() -> None:
    """leg_time + power_wait as the leg duration reproduces makespan over precedence only.

    This is the learnable formulation: the surrogate predicts a continuous per-leg wait
    instead of a discrete resolution-arc set, yet the max-plus longest path is still exact.
    """
    base = build_toy_instance(num_tasks=8)
    for budget in (25.0, 30.0, 45.0):
        inst = dataclasses.replace(base, peak_power=budget)
        rng = make_rng(int(budget) + 100)
        for _ in range(30):
            sched = decode(rng.random(4 * inst.num_tasks), inst)
            ev = evaluate(sched, inst)
            agv_prev, qc_prev, _ = build_precedence(
                sched.agv_sequences, sched.qc_sequences, inst.num_tasks
            )
            is_load = torch.tensor(
                [t.kind is TaskKind.LOAD for t in inst.tasks], dtype=torch.bool
            )
            empty_t, loaded_t, tau = _legs_tau(sched, inst)
            empty_eff = empty_t + torch.tensor(ev.wait_empty, dtype=torch.float64)
            loaded_eff = loaded_t + torch.tensor(ev.wait_loaded, dtype=torch.float64)
            dag = assemble_coupled_event_dag(
                is_load, agv_prev, qc_prev, empty_eff, loaded_eff, tau, []  # no power arcs
            )
            x = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
            makespan = float(x[dag.completion_nodes].max())
            assert abs(makespan - ev.makespan) < 1e-6, f"budget={budget}: {makespan} vs {ev.makespan}"


def test_coupled_dag_gradient_is_binary_critical_path() -> None:
    """dC_max/d(leg/tau) is 1 on the coupled critical path, 0 off it (faithful)."""
    base = build_toy_instance(num_tasks=8)
    inst = dataclasses.replace(base, peak_power=30.0)
    rng = make_rng(7)
    sched = decode(rng.random(4 * inst.num_tasks), inst)
    ev = evaluate(sched, inst)
    agv_prev, qc_prev, _ = build_precedence(
        sched.agv_sequences, sched.qc_sequences, inst.num_tasks
    )
    is_load = torch.tensor([t.kind is TaskKind.LOAD for t in inst.tasks], dtype=torch.bool)
    empty_t, loaded_t, tau = _legs_tau(sched, inst)
    for leaf in (empty_t, loaded_t, tau):
        leaf.requires_grad_(True)
    dag = assemble_coupled_event_dag(
        is_load, agv_prev, qc_prev, empty_t, loaded_t, tau, list(ev.power_arcs)
    )
    x = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)
    x[dag.completion_nodes].max().backward()
    grads = torch.cat([empty_t.grad, loaded_t.grad, tau.grad])
    # Every gradient is exactly 0 or 1 (max-plus subgradient routes along the argmax path).
    assert torch.all((grads.abs() < 1e-9) | ((grads - 1.0).abs() < 1e-9))
    assert grads.sum() > 0  # the critical path is non-empty
