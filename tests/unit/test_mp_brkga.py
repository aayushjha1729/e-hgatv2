"""Unit tests for the multi-population BRKGA (mp-BRKGA) baseline.

These pin the procedural faithfulness to Fontes & Homayouni (2022) Sec. 4 (the part most
at risk of being mis-implemented):

- determinism + correct evaluation accounting ((Omega+Pi) * P per generation);
- every solution is weakly dominated by the exact Oracle PF* (soundness);
- the front is mutually non-dominated and chromosomes decode to [0, 1];
- biased crossover limits (p_e -> 1 copies the elite parent; p_e -> 0 copies the other);
- the multi-population scheme recovers the objective extremes at least as well as the
  single-population BRKGA, the qualitative behaviour asserted in the paper's Fig. 6.

The decisive quantitative gate (reproducing the paper's Table 2 GD+/Delta on the real DS
instances) requires the published per-instance data, which is not part of this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ehgat.baselines.brkga import BRKGAConfig, run_brkga
from ehgat.baselines.mp_brkga import (
    MpBRKGAConfig,
    _biased_crossover,
    default_mp_config,
    run_mp_brkga,
)
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance
from ehgat.utils.seeding import make_rng

GOLDEN_FRONT_N5 = Path(__file__).resolve().parents[1] / "data" / "golden" / "exact_front_n5.json"


def _golden_front() -> list[tuple[float, float]]:
    return [tuple(p) for p in json.loads(GOLDEN_FRONT_N5.read_text())["front"]]


def test_mp_brkga_is_deterministic() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    cfg = MpBRKGAConfig(pop_size=40, generations=20, seed=7)
    a = run_mp_brkga(inst, cfg)
    b = run_mp_brkga(inst, cfg)
    assert a.front == b.front
    assert a.evaluations == b.evaluations


def test_evaluation_count_accounts_for_all_populations() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    cfg = MpBRKGAConfig(pop_size=30, generations=15, num_objectives=2, num_multi=2, seed=0)
    res = run_mp_brkga(inst, cfg)
    pops = cfg.num_objectives + cfg.num_multi  # Omega + Pi
    assert res.evaluations == pops * cfg.pop_size * (cfg.generations + 1)
    assert len(res.front_history) == cfg.generations + 1
    assert res.front_history[-1] == res.front


def test_front_is_mutually_non_dominated() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_mp_brkga(inst, MpBRKGAConfig(pop_size=40, generations=30, seed=1))
    front = res.front
    assert len(front) >= 1
    for i, (m_i, e_i) in enumerate(front):
        for j, (m_j, e_j) in enumerate(front):
            if i != j:
                assert not (m_j <= m_i and e_j <= e_i)
    assert len(res.chromosomes) == len(front)
    assert all(c.shape == (4 * inst.num_tasks,) for c in res.chromosomes)


def test_chromosomes_decode_within_unit_interval() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_mp_brkga(inst, MpBRKGAConfig(pop_size=30, generations=10, seed=3))
    for chrom in res.chromosomes:
        assert np.all(chrom >= 0.0) and np.all(chrom <= 1.0)


def test_mp_brkga_is_bounded_by_oracle() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_mp_brkga(inst, MpBRKGAConfig(pop_size=60, generations=40, seed=2))
    golden = _golden_front()
    tol = 1e-4
    for p in res.front:
        assert any(g[0] <= p[0] + tol and g[1] <= p[1] + tol for g in golden), p


def test_mp_brkga_recovers_extremes() -> None:
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_mp_brkga(inst, default_mp_config(inst, generations=120, seed=0))
    golden = _golden_front()
    assert min(p[0] for p in res.front) == pytest.approx(min(g[0] for g in golden), abs=1e-3)
    assert min(p[1] for p in res.front) == pytest.approx(min(g[1] for g in golden), abs=1e-3)


def test_mp_matches_or_beats_single_pop_on_boundaries() -> None:
    """Paper Fig. 6: mp-BRKGA should be no worse than single-pop on the objective extremes
    at a matched per-population budget (the boundary specialists are the whole point)."""
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    gens, pop, seed = 60, 40, 5
    mp = run_mp_brkga(inst, MpBRKGAConfig(pop_size=pop, generations=gens, seed=seed))
    sp = run_brkga(inst, BRKGAConfig(pop_size=pop, generations=gens, seed=seed))
    mp_best_mk = min(p[0] for p in mp.front)
    mp_best_e = min(p[1] for p in mp.front)
    sp_best_mk = min(p[0] for p in sp.front)
    sp_best_e = min(p[1] for p in sp.front)
    assert mp_best_mk <= sp_best_mk + 1e-6
    assert mp_best_e <= sp_best_e + 1e-6


def test_biased_crossover_inheritance_limits() -> None:
    rng = make_rng(0)
    elite = np.full(16, 0.9)
    other = np.full(16, 0.1)
    all_elite = _biased_crossover(rng, elite, other, inherit_prob=1.0)
    all_other = _biased_crossover(rng, elite, other, inherit_prob=0.0)
    assert np.allclose(all_elite, elite)
    assert np.allclose(all_other, other)


def test_omega_zero_is_pure_multipopulation() -> None:
    """Omega=0 (no single-objective pops) must still run and stay oracle-bounded."""
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    res = run_mp_brkga(
        inst, MpBRKGAConfig(pop_size=40, generations=30, num_objectives=0, num_multi=2, seed=4)
    )
    assert res.evaluations == 2 * 40 * 31
    golden = _golden_front()
    for p in res.front:
        assert any(g[0] <= p[0] + 1e-4 and g[1] <= p[1] + 1e-4 for g in golden)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="Omega"):
        MpBRKGAConfig(pop_size=40, num_objectives=3)
    with pytest.raises(ValueError, match="Pi"):
        MpBRKGAConfig(pop_size=40, num_multi=0)
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    with pytest.raises(ValueError, match="too small"):
        run_mp_brkga(inst, MpBRKGAConfig(pop_size=1, generations=3, seed=0))
