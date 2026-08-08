"""Tests for the AOS statistics module (benchmark/stats.py).

Pure numpy/scipy -- no Torch -- so these run fast everywhere and pin down the
significance/effect-size machinery independently of any actual ablation run.
"""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.benchmark.stats import (
    analyse_ablation,
    analyse_metric,
    friedman,
    holm_correction,
    load_metric_matrix,
    rank_biserial,
    wilcoxon_paired,
)

_METRICS = ("final_hv", "hv_auc", "igd_plus", "gd_plus", "spread")


def _synthetic_ablation(n: int = 24, seed: int = 0) -> dict:
    """Build an aos_ablation.json-like dict where oracle > attention > random.

    Higher-is-better metrics increase across arms; lower-is-better metrics decrease.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.80, 0.02, n)
    shift = {"random": 0.0, "attention": 0.03, "oracle": 0.05}
    arms: dict[str, list[dict]] = {}
    for arm, s in shift.items():
        noise = rng.normal(0.0, 0.004, n)
        hv = base + s + noise
        seeds = []
        for i in range(n):
            seeds.append(
                {
                    "seed": i,
                    "final_hv": float(hv[i]),
                    "hv_auc": float(hv[i] * 0.9),
                    "igd_plus": float(0.20 - s + noise[i]),  # lower better
                    "gd_plus": float(0.15 - s + noise[i]),
                    "spread": float(0.50 - s + noise[i]),
                }
            )
        arms[arm] = seeds
    return {"experiment": "aos_ablation", "arms": {a: {"seeds": s} for a, s in arms.items()}}


# --------------------------------------------------------------------------------------
# holm_correction
# --------------------------------------------------------------------------------------
def test_holm_correction_known_values() -> None:
    adj = holm_correction([0.01, 0.02, 0.04])
    assert adj == pytest.approx([0.03, 0.04, 0.04])


def test_holm_correction_is_monotone_and_geq_raw() -> None:
    raw = [0.001, 0.2, 0.04, 0.5]
    adj = holm_correction(raw)
    assert all(a >= r for a, r in zip(adj, raw, strict=True))
    assert all(a <= 1.0 for a in adj)


def test_holm_empty() -> None:
    assert holm_correction([]) == []


# --------------------------------------------------------------------------------------
# rank_biserial
# --------------------------------------------------------------------------------------
def test_rank_biserial_all_positive_is_one() -> None:
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    r, n = rank_biserial(a, b)
    assert r == pytest.approx(1.0)
    assert n == 4


def test_rank_biserial_identical_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    r, n = rank_biserial(a, a)
    assert r == 0.0
    assert n == 0


def test_rank_biserial_sign_flips() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([5.0, 6.0, 7.0])
    r, _ = rank_biserial(a, b)
    assert r == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------
# wilcoxon_paired
# --------------------------------------------------------------------------------------
def test_wilcoxon_detects_clear_difference() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(0.8, 0.02, 30)
    a = b + 0.05  # a strictly better
    res = wilcoxon_paired(a, b, ("attention", "random"), rng=rng)
    assert res.pvalue < 0.001
    assert res.rank_biserial == pytest.approx(1.0)
    assert res.median_diff == pytest.approx(0.05, abs=1e-6)
    assert res.ci_lo <= res.median_diff <= res.ci_hi


def test_wilcoxon_identical_samples_no_evidence() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.8])
    res = wilcoxon_paired(a, a.copy(), ("attention", "random"))
    assert res.pvalue == 1.0
    assert res.rank_biserial == 0.0
    assert res.n_nonzero == 0


# --------------------------------------------------------------------------------------
# friedman
# --------------------------------------------------------------------------------------
def test_friedman_detects_separation() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0.8, 0.02, 25)
    arms = {"random": base, "attention": base + 0.03, "oracle": base + 0.05}
    res = friedman(arms, ["random", "attention", "oracle"])
    assert res.k == 3
    assert res.n == 25
    assert res.pvalue < 0.001


def test_friedman_identical_arms_null() -> None:
    base = np.linspace(0.7, 0.9, 10)
    arms = {"random": base, "attention": base.copy(), "oracle": base.copy()}
    res = friedman(arms, ["random", "attention", "oracle"])
    assert res.pvalue == 1.0


# --------------------------------------------------------------------------------------
# analyse_metric / analyse_ablation
# --------------------------------------------------------------------------------------
def test_load_metric_matrix_sorts_by_seed() -> None:
    seeds = [
        {"seed": 2, "final_hv": 3.0},
        {"seed": 0, "final_hv": 1.0},
        {"seed": 1, "final_hv": 2.0},
    ]
    data = {"arms": {"random": {"seeds": seeds}}}
    mat = load_metric_matrix(data, "final_hv")
    assert list(mat["random"]) == [1.0, 2.0, 3.0]


def test_analyse_metric_holm_geq_raw() -> None:
    data = _synthetic_ablation()
    mat = load_metric_matrix(data, "final_hv")
    ms = analyse_metric(mat, "final_hv")
    assert ms.higher_is_better is True
    for w in ms.pairwise:
        assert w.pvalue_holm >= w.pvalue - 1e-12


def test_analyse_ablation_attention_beats_random_on_hv() -> None:
    data = _synthetic_ablation()
    stats = analyse_ablation(data, resamples=1000)
    assert stats.arms == ("random", "attention", "oracle")
    hv = stats.metrics["final_hv"]
    assert hv.friedman.pvalue < 0.01
    att_vs_rnd = next(w for w in hv.pairwise if w.pair == ("attention", "random"))
    assert att_vs_rnd.median_diff > 0
    assert att_vs_rnd.rank_biserial > 0.5
    assert att_vs_rnd.pvalue_holm < 0.05


def test_analyse_ablation_covers_all_metrics() -> None:
    stats = analyse_ablation(_synthetic_ablation(), resamples=500)
    assert set(stats.metrics) == set(_METRICS)
    assert stats.metrics["igd_plus"].higher_is_better is False
