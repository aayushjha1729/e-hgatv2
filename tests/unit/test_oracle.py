"""Unit tests for the exact Pareto-front oracle.

The principal test cross-validates the smart speed Pareto DP against a full
3^(2N) brute force on a tiny instance, proving the DP is exact.
"""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

from ehgat.environment.decoder import Schedule
from ehgat.environment.evaluator import build_precedence, evaluate
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance
from ehgat.environment.oracle import (
    OracleTooLargeError,
    enumerate_structures,
    evaluate_speeds,
    exact_pareto_front,
    pareto_front_2d,
    save_front,
    structure_speed_front,
)
from ehgat.environment.physics import SpeedLevel

# A tiny instance: 3 tasks, 2 QCs (so QC1 holds 2 tasks -> exercises QC precedence),
# 2 AGVs. Small enough to brute-force all 3^(2N) speed assignments.
TINY = build_toy_instance(num_tasks=3, qcs=("QC1", "QC2"))

GOLDEN_FRONT_N5 = Path(__file__).resolve().parents[1] / "data" / "golden" / "exact_front_n5.json"


def _load_golden() -> dict:
    return json.loads(GOLDEN_FRONT_N5.read_text())


def _brute_force_front_exact(instance) -> tuple[tuple[Fraction, Fraction], ...]:
    """Exact PF* by enumerating all 3^(2N) speed assignments in rational arithmetic."""
    n = instance.num_tasks
    levels = list(SpeedLevel)
    points: list[tuple[Fraction, Fraction]] = []
    for struct in enumerate_structures(instance):
        for empty_combo in product(levels, repeat=n):
            for loaded_combo in product(levels, repeat=n):
                points.append(evaluate_speeds(struct, empty_combo, loaded_combo, instance))
    return pareto_front_2d(points)


def _to_float_front(
    front: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(m), 6), round(float(e), 6)) for m, e in front)


def test_smart_oracle_matches_brute_force() -> None:
    smart = exact_pareto_front(TINY).front
    brute = _to_float_front(_brute_force_front_exact(TINY))
    assert smart == brute


def test_structure_speed_front_matches_brute_force_per_structure() -> None:
    n = TINY.num_tasks
    levels = list(SpeedLevel)
    for struct in enumerate_structures(TINY):
        brute_points: list[tuple[Fraction, Fraction]] = []
        for empty_combo in product(levels, repeat=n):
            for loaded_combo in product(levels, repeat=n):
                brute_points.append(evaluate_speeds(struct, empty_combo, loaded_combo, TINY))
        assert structure_speed_front(struct, TINY) == pareto_front_2d(brute_points)


def test_evaluate_speeds_matches_float_evaluator() -> None:
    # Ties the oracle's rational recurrence to the hand-validated float evaluator.
    n = TINY.num_tasks
    empty = (SpeedLevel.NOMINAL,) * n
    loaded = (SpeedLevel.LOWER,) * n
    for struct in enumerate_structures(TINY):
        _agv_prev, _qc_prev, order = build_precedence(
            struct.agv_sequences, struct.qc_sequences, n
        )
        makespan, energy = evaluate_speeds(struct, empty, loaded, TINY)
        sched = Schedule(
            global_order=order,
            assignment=struct.assignment,
            agv_sequences=struct.agv_sequences,
            qc_sequences=struct.qc_sequences,
            empty_speed=empty,
            loaded_speed=loaded,
        )
        ev = evaluate(sched, TINY)
        assert abs(float(makespan) - ev.makespan) < 1e-9
        assert abs(float(energy) - ev.energy) < 1e-9


def test_pareto_front_2d_basic() -> None:
    pts = [(10.0, 5.0), (10.0, 7.0), (8.0, 9.0), (12.0, 3.0), (9.0, 6.0)]
    # Non-dominated: (8,9), (9,6), (10,5), (12,3); (10,7) dominated by (10,5)/(9,6).
    assert pareto_front_2d(pts) == ((8.0, 9.0), (9.0, 6.0), (10.0, 5.0), (12.0, 3.0))


def test_front_is_sorted_and_non_dominated() -> None:
    front = exact_pareto_front(TINY).front
    makespans = [m for m, _ in front]
    energies = [e for _, e in front]
    assert makespans == sorted(makespans)
    assert energies == sorted(energies, reverse=True)  # strict trade-off
    for i, (m_i, e_i) in enumerate(front):
        for j, (m_j, e_j) in enumerate(front):
            if i != j:
                assert not (m_j <= m_i and e_j <= e_i)  # no point dominates another


def test_enumerate_structures_are_unique() -> None:
    structs = list(enumerate_structures(TINY))
    keys = {(s.agv_sequences, s.qc_sequences) for s in structs}
    assert len(keys) == len(structs)


def test_guard_blocks_intractable_instance() -> None:
    big = build_toy_instance()  # N=10 default -> ~3.7e9 structure iterations
    with pytest.raises(OracleTooLargeError, match="exceeds the guard"):
        exact_pareto_front(big)


def test_exact_front_is_deterministic() -> None:
    assert exact_pareto_front(TINY).front == exact_pareto_front(TINY).front


def test_save_front_round_trips(tmp_path) -> None:
    result = exact_pareto_front(TINY)
    path = tmp_path / "front.json"
    save_front(result, path)
    payload = json.loads(path.read_text())
    assert payload["num_tasks"] == TINY.num_tasks
    assert payload["front_size"] == len(result.front)
    assert [tuple(p) for p in payload["front"]] == list(result.front)


def test_golden_front_n5_is_valid() -> None:
    # Fast regression: the frozen N=5 exact front loads and is a valid Pareto set.
    payload = _load_golden()
    assert payload["num_tasks"] == EXACT_TOY_TASKS == 5
    front = [tuple(p) for p in payload["front"]]
    assert payload["front_size"] == len(front)
    makespans = [m for m, _ in front]
    energies = [e for _, e in front]
    assert makespans == sorted(makespans)
    assert energies == sorted(energies, reverse=True)
    for i, (m_i, e_i) in enumerate(front):
        for j, (m_j, e_j) in enumerate(front):
            if i != j:
                assert not (m_j <= m_i and e_j <= e_i)


@pytest.mark.slow
def test_exact_front_matches_golden_n5() -> None:
    # Compute-once recomputation must reproduce the frozen golden front exactly.
    result = exact_pareto_front(build_toy_instance(num_tasks=EXACT_TOY_TASKS))
    payload = _load_golden()
    assert result.num_tasks == payload["num_tasks"]
    assert result.num_structures == payload["num_structures"]
    assert [list(p) for p in result.front] == payload["front"]
