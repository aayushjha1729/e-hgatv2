"""Unit tests for the multi-objective BRKGA baseline.

Soundness is anchored to the exact Oracle: every BRKGA solution must be weakly
dominated by some point of the frozen golden Pareto front PF* (if not, either
BRKGA produced an infeasible objective or the Oracle is not exact).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ehgat.baselines.brkga import BRKGAConfig, default_config, run_brkga
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance

GOLDEN_FRONT_N5 = Path(__file__).resolve().parents[1] / "data" / "golden" / "exact_front_n5.json"


def _golden_front() -> list[tuple[float, float]]:
    return [tuple(p) for p in json.loads(GOLDEN_FRONT_N5.read_text())["front"]]


def test_run_brkga_is_deterministic() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    cfg = BRKGAConfig(pop_size=60, generations=30, seed=7)
    a = run_brkga(inst, cfg)
    b = run_brkga(inst, cfg)
    assert a.front == b.front
    assert a.evaluations == b.evaluations


def test_evaluation_count() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    cfg = BRKGAConfig(pop_size=40, generations=20, seed=0)
    res = run_brkga(inst, cfg)
    assert res.evaluations == cfg.pop_size * (cfg.generations + 1)
    assert len(res.front_history) == cfg.generations + 1
    assert res.front_history[-1] == res.front


def test_front_is_mutually_non_dominated() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_brkga(inst, BRKGAConfig(pop_size=60, generations=40, seed=1))
    front = res.front
    assert len(front) >= 1
    for i, (m_i, e_i) in enumerate(front):
        for j, (m_j, e_j) in enumerate(front):
            if i != j:
                assert not (m_j <= m_i and e_j <= e_i)
    # chromosomes align with front and have the right length.
    assert len(res.chromosomes) == len(front)
    assert all(c.shape == (4 * inst.num_tasks,) for c in res.chromosomes)


def test_brkga_is_bounded_by_oracle() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_brkga(inst, BRKGAConfig(pop_size=60, generations=40, seed=2))
    golden = _golden_front()
    tol = 1e-4
    for p in res.front:
        assert any(g[0] <= p[0] + tol and g[1] <= p[1] + tol for g in golden), p


def test_brkga_recovers_extremes() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_brkga(inst, default_config(inst, generations=150, seed=0))
    golden = _golden_front()
    assert min(p[0] for p in res.front) == pytest.approx(min(g[0] for g in golden), abs=1e-3)
    assert min(p[1] for p in res.front) == pytest.approx(min(g[1] for g in golden), abs=1e-3)
    # On this exact toy, BRKGA should recover a substantial share of PF* (the dense
    # corrected front has 123 points, so exact 1e-3 hits on a third is already strong;
    # the boundary solutions above are recovered exactly).
    recovered = sum(
        1
        for g in golden
        if any(abs(g[0] - p[0]) <= 1e-3 and abs(g[1] - p[1]) <= 1e-3 for p in res.front)
    )
    assert recovered >= len(golden) // 3


def test_pop_size_too_small_raises() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    with pytest.raises(ValueError, match="too small"):
        run_brkga(inst, BRKGAConfig(pop_size=1, generations=5, seed=0))


def test_chromosomes_decode_within_unit_interval() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_brkga(inst, BRKGAConfig(pop_size=40, generations=10, seed=3))
    for chrom in res.chromosomes:
        assert np.all(chrom >= 0.0) and np.all(chrom <= 1.0)
