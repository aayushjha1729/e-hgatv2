"""Tests for the Tropical-Algebraic Pareto Explainer."""

from __future__ import annotations

import torch

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.evaluator import evaluate
from ehgat.environment.instance import build_toy_instance
from ehgat.explain.tcs_calculator import ParetoPoint, tradeoff_criticality_scores
from ehgat.explain.tape_explainer import explain_schedule
from ehgat.explain.tropical_dp import tropical_longest_path
from ehgat.utils.seeding import make_rng


def test_tropical_backward_routes_only_argmax_path() -> None:
    node_w = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    edge_w = torch.tensor([2.0, 5.0, 10.0, 1.0], dtype=torch.float64, requires_grad=True)
    x = tropical_longest_path(node_w, edge_index, edge_w)
    x[3].backward()
    # Node 3 chooses path 0->1->3 (12), not 0->2->3 (6).
    assert edge_w.grad.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert node_w.grad.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_tape_matches_exact_evaluator_objectives() -> None:
    inst = build_toy_instance(num_tasks=6)
    schedule = decode(make_rng(0).random(NUM_BLOCKS * inst.num_tasks), inst)
    exact = evaluate(schedule, inst)
    ex = explain_schedule(schedule, inst)
    assert abs(ex.makespan - exact.makespan) < 1e-9
    assert abs(ex.energy - exact.energy) < 1e-9
    assert len(ex.empty_time_grad) == inst.num_tasks
    assert len(ex.loaded_time_grad) == inst.num_tasks
    assert max(ex.event_edge_grad) == 1.0


def test_critical_path_durations_sum_to_makespan() -> None:
    # The makespan is the max-plus longest path, under which the on-path activity
    # durations (empty and loaded legs plus QC handling) sum exactly to C_max. This is the
    # additive quantification used by scripts/run_critical_path_demo.py.
    rng = make_rng(3)
    for num_tasks in (5, 8, 10):
        inst = build_toy_instance(num_tasks=num_tasks)
        for _ in range(5):
            schedule = decode(rng.random(NUM_BLOCKS * inst.num_tasks), inst)
            ev = evaluate(schedule, inst)
            ex = explain_schedule(schedule, inst)
            total = 0.0
            for j in range(inst.num_tasks):
                if ex.empty_time_grad[j] > 0.5:
                    total += ev.empty_time[j]
                if ex.loaded_time_grad[j] > 0.5:
                    total += ev.loaded_time[j]
                if ex.node_grad[j] > 0.5:
                    total += inst.tasks[j].handling_time
            assert abs(total - ex.makespan) <= 1e-6 * max(1.0, abs(ex.makespan)), (
                f"N={num_tasks}: on-path duration sum {total} != C_max {ex.makespan}"
            )


def test_tcs_output_is_json_shaped() -> None:
    inst = build_toy_instance(num_tasks=4)
    rng = make_rng(1)
    pts = []
    for i in range(2):
        schedule = decode(rng.random(NUM_BLOCKS * inst.num_tasks), inst)
        ex = explain_schedule(schedule, inst)
        pts.append(ParetoPoint(str(i), ex.makespan, ex.energy, ex))
    out = tradeoff_criticality_scores(pts)
    assert len(out) == 2
    assert "lambda" in out[0]
    assert out[0]["tasks"]
    assert out[0]["event_arcs"]
