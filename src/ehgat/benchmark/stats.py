"""Significance and effect-size statistics for the operator-selection ablation.

Consumes the aos_ablation.json artifact, or its in-memory dict, and computes per metric:

- the Friedman omnibus across the arms, paired by seed, as a non-parametric omnibus
  test preceding any pairwise comparison;
- post-hoc pairwise Wilcoxon signed-rank tests (attention-vs-random, oracle-vs-random,
  reward-vs-random, attention-vs-oracle, attention-vs-reward) under Holm-Bonferroni
  correction within each metric's family; pairs absent from the artifact are skipped;
- the matched-pairs rank-biserial effect size (Kerby 2014) for each pairwise test,
  giving magnitude alongside significance;
- a bootstrap confidence interval of the paired median difference, in the arms'
  native units.

The module depends only on numpy and scipy, and is therefore testable without the
learning stack.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sps

__all__ = [
    "AblationStats",
    "FriedmanResult",
    "MetricStats",
    "WilcoxonResult",
    "analyse_ablation",
    "holm_correction",
    "load_metric_matrix",
    "rank_biserial",
    "to_json_dict",
    "wilcoxon_paired",
]

# Metric name -> whether larger values are better. Drives direction interpretation.
METRIC_ORIENTATION: dict[str, bool] = {
    "final_hv": True,
    "hv_auc": True,
    "igd_plus": False,
    "gd_plus": False,
    "spread": False,
}
# Canonical arm order (null -> method -> structural ceiling -> learned-utility ceiling).
ARM_ORDER: tuple[str, ...] = ("random", "attention", "oracle", "reward")
# Post-hoc pairs: (a, b) tested as a - b. (attention, random) is the primary pair;
# (reward, random) establishes whether any operator-selection scheme improves on the null;
# (attention, reward) compares the attribution signal against the learned-utility ceiling.
# Pairs whose arms are absent from the data are skipped, leaving 3-arm artifacts analysable.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("attention", "random"),
    ("oracle", "random"),
    ("reward", "random"),
    ("attention", "oracle"),
    ("attention", "reward"),
)


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    statistic: float
    pvalue: float
    k: int  # number of arms (treatments)
    n: int  # number of paired blocks (seeds)


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    pair: tuple[str, str]  # (a, b); tested as a - b
    n: int  # paired sample size
    n_nonzero: int  # pairs with a non-zero difference (entered the test)
    statistic: float  # scipy Wilcoxon W (min of signed-rank sums)
    pvalue: float  # raw two-sided p-value
    pvalue_holm: float  # Holm-corrected within the metric's family
    rank_biserial: float  # matched-pairs effect size in [-1, 1]; >0 => a > b
    median_diff: float  # median(a - b) in native units
    ci_lo: float  # bootstrap CI of the median difference
    ci_hi: float


@dataclass(frozen=True, slots=True)
class MetricStats:
    metric: str
    higher_is_better: bool
    arm_means: dict[str, float]
    arm_medians: dict[str, float]
    friedman: FriedmanResult
    pairwise: tuple[WilcoxonResult, ...]


@dataclass(frozen=True, slots=True)
class AblationStats:
    arms: tuple[str, ...]
    n_seeds: int
    metrics: dict[str, MetricStats]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_metric_matrix(
    data: dict[str, object] | str | Path, metric: str
) -> dict[str, np.ndarray]:
    """Return {arm: per-seed values} for metric, aligned (sorted) by seed.

    Accepts the aos_ablation.json path, its parsed dict, or an already-loaded dict.
    Alignment by seed is what makes the downstream tests correctly paired.
    """
    if isinstance(data, str | Path):
        data = json.loads(Path(data).read_text())
    assert isinstance(data, dict)
    arms_obj = data["arms"]
    assert isinstance(arms_obj, dict)
    out: dict[str, np.ndarray] = {}
    for arm, payload in arms_obj.items():
        seeds = payload["seeds"]
        ordered = sorted(seeds, key=lambda r: r["seed"])
        out[arm] = np.asarray([r[metric] for r in ordered], dtype=float)
    return out


# --------------------------------------------------------------------------------------
# Effect size + corrections
# --------------------------------------------------------------------------------------
def rank_biserial(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Matched-pairs rank-biserial correlation for a - b (Kerby 2014).

    r = (W+ - W-) / (W+ + W-) where W+/W- are the summed ranks of the positive
    and negative absolute differences. Zero-differences are dropped. r in [-1, 1];
    r > 0 means a tends to exceed b. Returns (r, n_nonzero).
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = diff[diff != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return 0.0, 0
    ranks = sps.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total = w_pos + w_neg
    if total == 0.0:
        return 0.0, n
    return (w_pos - w_neg) / total, n


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values (order preserved)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, min(val, 1.0))  # enforce monotonic non-decreasing
        adjusted[idx] = running
    return adjusted


def _bootstrap_median_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    resamples: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the paired median difference median(a - b)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n == 0:
        return 0.0, 0.0
    idx = rng.integers(0, n, size=(resamples, n))
    boot = np.median(diff[idx], axis=1)
    lo = float(np.quantile(boot, (1.0 - ci) / 2.0))
    hi = float(np.quantile(boot, 1.0 - (1.0 - ci) / 2.0))
    return lo, hi


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def wilcoxon_paired(
    a: np.ndarray,
    b: np.ndarray,
    pair: tuple[str, str],
    *,
    resamples: int = 5000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank for a - b with effect size + bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = rng or np.random.default_rng(0)
    diff = a - b
    n = int(diff.size)
    r, n_nonzero = rank_biserial(a, b)
    median_diff = float(np.median(diff))
    lo, hi = _bootstrap_median_diff_ci(a, b, resamples=resamples, ci=ci, rng=rng)

    if n_nonzero == 0:  # identical samples: Wilcoxon is undefined -> no evidence
        return WilcoxonResult(pair, n, 0, 0.0, 1.0, 1.0, 0.0, median_diff, lo, hi)
    res = sps.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return WilcoxonResult(
        pair=pair,
        n=n,
        n_nonzero=n_nonzero,
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        pvalue_holm=float(res.pvalue),  # filled in by analyse_metric across the family
        rank_biserial=float(r),
        median_diff=median_diff,
        ci_lo=lo,
        ci_hi=hi,
    )


def friedman(arms_values: dict[str, np.ndarray], arms: Sequence[str]) -> FriedmanResult:
    """Friedman omnibus across arms (paired by seed)."""
    samples = [np.asarray(arms_values[a], dtype=float) for a in arms]
    n = int(samples[0].size)
    if len({s.size for s in samples}) != 1:
        raise ValueError("all arms must have the same number of paired seeds")
    # Friedman needs >=3 treatments and >=2 blocks; if degenerate, return a null result.
    if len(samples) < 3 or n < 2 or all(np.allclose(s, samples[0]) for s in samples):
        return FriedmanResult(statistic=0.0, pvalue=1.0, k=len(samples), n=n)
    stat, pval = sps.friedmanchisquare(*samples)
    return FriedmanResult(statistic=float(stat), pvalue=float(pval), k=len(samples), n=n)


def analyse_metric(
    arms_values: dict[str, np.ndarray],
    metric: str,
    *,
    arms: Sequence[str] = ARM_ORDER,
    pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS,
    resamples: int = 5000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> MetricStats:
    """Friedman + Holm-corrected pairwise Wilcoxon for one metric."""
    rng = rng or np.random.default_rng(0)
    present = [a for a in arms if a in arms_values]
    fried = friedman(arms_values, present)

    raw = [
        wilcoxon_paired(arms_values[x], arms_values[y], (x, y), resamples=resamples, ci=ci, rng=rng)
        for x, y in pairs
        if x in arms_values and y in arms_values
    ]
    holm = holm_correction([w.pvalue for w in raw])
    pairwise = tuple(
        WilcoxonResult(
            pair=w.pair,
            n=w.n,
            n_nonzero=w.n_nonzero,
            statistic=w.statistic,
            pvalue=w.pvalue,
            pvalue_holm=holm[i],
            rank_biserial=w.rank_biserial,
            median_diff=w.median_diff,
            ci_lo=w.ci_lo,
            ci_hi=w.ci_hi,
        )
        for i, w in enumerate(raw)
    )
    return MetricStats(
        metric=metric,
        higher_is_better=METRIC_ORIENTATION.get(metric, True),
        arm_means={a: float(np.mean(arms_values[a])) for a in present},
        arm_medians={a: float(np.median(arms_values[a])) for a in present},
        friedman=fried,
        pairwise=pairwise,
    )


def analyse_ablation(
    data: dict[str, object] | str | Path,
    *,
    metrics: Sequence[str] = tuple(METRIC_ORIENTATION),
    arms: Sequence[str] = ARM_ORDER,
    pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS,
    resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> AblationStats:
    """Run the full stats suite over an aos_ablation.json artifact."""
    if isinstance(data, str | Path):
        data = json.loads(Path(data).read_text())
    assert isinstance(data, dict)
    rng = np.random.default_rng(seed)
    out: dict[str, MetricStats] = {}
    n_seeds = 0
    available: set[str] = set()
    for metric in metrics:
        arms_values = load_metric_matrix(data, metric)
        available.update(arms_values)
        n_seeds = max(n_seeds, max((v.size for v in arms_values.values()), default=0))
        out[metric] = analyse_metric(
            arms_values, metric, arms=arms, pairs=pairs, resamples=resamples, ci=ci, rng=rng
        )
    present = tuple(a for a in arms if a in available)
    return AblationStats(arms=present, n_seeds=int(n_seeds), metrics=out)


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------
def _wilcoxon_dict(w: WilcoxonResult) -> dict[str, object]:
    return {
        "pair": list(w.pair),
        "n": w.n,
        "n_nonzero": w.n_nonzero,
        "statistic": w.statistic,
        "pvalue": w.pvalue,
        "pvalue_holm": w.pvalue_holm,
        "rank_biserial": w.rank_biserial,
        "median_diff": w.median_diff,
        "ci_lo": w.ci_lo,
        "ci_hi": w.ci_hi,
    }


def to_json_dict(stats: AblationStats) -> dict[str, object]:
    return {
        "arms": list(stats.arms),
        "n_seeds": stats.n_seeds,
        "metrics": {
            name: {
                "metric": m.metric,
                "higher_is_better": m.higher_is_better,
                "arm_means": m.arm_means,
                "arm_medians": m.arm_medians,
                "friedman": {
                    "statistic": m.friedman.statistic,
                    "pvalue": m.friedman.pvalue,
                    "k": m.friedman.k,
                    "n": m.friedman.n,
                },
                "pairwise": [_wilcoxon_dict(w) for w in m.pairwise],
            }
            for name, m in stats.metrics.items()
        },
    }
