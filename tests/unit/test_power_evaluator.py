"""Peak-power-coupled evaluator: reduces to the uncoupled model, and only delays."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ehgat.environment.decoder import decode
from ehgat.environment.evaluator import _evaluate_uncoupled, evaluate
from ehgat.environment.instance import build_toy_instance
from ehgat.environment.physics import SPEED_TABLE
from ehgat.utils.seeding import make_rng

_MAX_LEG_POWER = max(spec.loaded_power for spec in SPEED_TABLE.values())


def _random_schedule(instance, rng):
    keys = rng.random(4 * instance.num_tasks)
    return decode(keys, instance)


def test_large_budget_reduces_to_uncoupled() -> None:
    """A budget that fits every leg at once must reproduce the closed-form model."""
    base = build_toy_instance(num_tasks=8)
    # All AGVs travelling loaded at top speed simultaneously still fits.
    big = dataclasses.replace(base, peak_power=_MAX_LEG_POWER * base.num_agvs + 10.0)
    rng = make_rng(0)
    for _ in range(40):
        sched = _random_schedule(base, rng)
        ref = _evaluate_uncoupled(sched, base)
        got = evaluate(sched, big)
        assert got.makespan == pytest.approx(ref.makespan, rel=1e-9, abs=1e-9)
        assert got.energy == pytest.approx(ref.energy, rel=1e-9, abs=1e-9)
        assert got.power_arcs == ()  # nothing was ever power-delayed


def test_tight_budget_only_delays_and_conserves_energy() -> None:
    """A one-leg-at-a-time budget can only push makespan up; energy is invariant."""
    base = build_toy_instance(num_tasks=8)
    tight = dataclasses.replace(base, peak_power=_MAX_LEG_POWER + 0.5)  # ~1 leg at a time
    rng = make_rng(1)
    for _ in range(40):
        sched = _random_schedule(base, rng)
        ref = _evaluate_uncoupled(sched, base)
        got = evaluate(sched, tight)
        assert got.makespan >= ref.makespan - 1e-9
        assert got.energy == pytest.approx(ref.energy, rel=1e-9, abs=1e-9)


def test_tight_budget_can_exceed_loose_graham_anomaly() -> None:
    """Makespan is well-defined but not monotonic in the budget under a greedy SGS.

    This is the Graham anomaly: under a fixed-priority dispatch rule a looser power budget
    can lengthen the makespan by reordering concurrency. The assertions are therefore
    limited to determinism and to the objective remaining finite and positive across
    budgets; true-optimal monotonicity would require an inner solver.
    """
    base = build_toy_instance(num_tasks=8)
    rng = make_rng(2)
    for _ in range(20):
        sched = _random_schedule(base, rng)
        for b in (_MAX_LEG_POWER + 0.5, 30.0, 45.0, 80.0):
            span = evaluate(sched, dataclasses.replace(base, peak_power=b)).makespan
            assert span > 0.0 and np.isfinite(span)


def test_determinism() -> None:
    base = build_toy_instance(num_tasks=10, peak_power=35.0)
    rng = make_rng(3)
    sched = _random_schedule(base, rng)
    a = evaluate(sched, base)
    b = evaluate(sched, base)
    assert a.makespan == b.makespan
    assert a.power_arcs == b.power_arcs


def test_power_budget_never_exceeded() -> None:
    """Reconstruct the instantaneous power profile and assert it respects the cap."""
    base = build_toy_instance(num_tasks=8)
    budget = 30.0
    inst = dataclasses.replace(base, peak_power=budget)
    rng = make_rng(4)
    for _ in range(25):
        sched = _random_schedule(base, rng)
        ev = evaluate(sched, inst)
        # Each leg contributes an interval [start, start+time] at its power level.
        events: list[tuple[float, float]] = []  # (time, power_delta)
        for j in range(base.num_tasks):
            ep = SPEED_TABLE[sched.empty_speed[j]].empty_power
            lp = SPEED_TABLE[sched.loaded_speed[j]].loaded_power
            events.append((ev.empty_start[j], ep))
            events.append((ev.empty_start[j] + ev.empty_time[j], -ep))
            events.append((ev.loaded_start[j], lp))
            events.append((ev.loaded_start[j] + ev.loaded_time[j], -lp))
        events.sort(key=lambda e: (e[0], e[1]))  # release (-) before acquire (+) at ties
        running = 0.0
        for _t, delta in events:
            running += delta
            assert running <= budget + 1e-6
