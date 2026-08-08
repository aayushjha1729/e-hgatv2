"""Cross-instance significance tests for the faithful-guidance study.

Reads every experiments/fused_tape_guided/tape_bench_*.json and computes, in the
Demsar (2006) "compare methods over multiple datasets" framework:

- per-instance mean HV ratio / IGD+ / GD+ per method (one block per instance);
- a Friedman omnibus test across the 5 methods over all instances;
- average ranks (Nemenyi critical difference at alpha=0.05);
- pairwise Wilcoxon signed-rank tests (TAPE vs each baseline) across instances with
  Holm-Bonferroni correction AND Cliff's-delta distribution-free effect sizes;
- per-instance paired Wilcoxon for the GNN-vs-mp-BRKGA HV gap, with a percentile bootstrap
  95% CI on the gap;
- percentile bootstrap (10^4 resamples) 95% CIs on HV ratio per method per instance.

Prints a compact report and writes experiments/fused_tape_guided/paper_stats.json.
Usage: python scripts/compute_paper_stats.py.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sps

OUT = Path(__file__).resolve().parents[1] / "experiments" / "fused_tape_guided"
METHODS = [
    "E-HGATv2-TAPE",
    "E-HGATv2-attn",
    "NSGA-II (random)",
    "mp-BRKGA",
    "single-pop BRKGA",
]


def _load() -> list[dict]:
    recs = []
    for f in sorted(OUT.glob("tape_bench_*.json")):
        recs.append(json.loads(f.read_text()))
    return recs


def _nemenyi_cd(k: int, n: int, q_alpha: float) -> float:
    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))


# Studentized range / sqrt(2) critical values at alpha=0.05 (Demsar Table 5).
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def _bootstrap_ci(
    vals: np.ndarray, *, reps: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap (lo, mean, hi) of the mean of vals."""
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(reps, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(vals.mean()), float(hi)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1]: P(a>b) - P(a<b) (distribution-free)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff).sum()) / (a.size * b.size))


def _cliffs_label(d: float) -> str:
    """Romano et al. (2006) magnitude thresholds for |delta|."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def main() -> None:
    recs = _load()
    insts = [r["instance"] + ("/coupled" if r["coupled"] else "") for r in recs]
    print(f"instances ({len(recs)}): {', '.join(insts)}\n")

    report: dict = {"instances": insts, "n_instances": len(recs), "per_metric": {},
                    "bootstrap_ci_hv_ratio": {}, "per_instance_tape_vs_mpbrkga": [],
                    "tape_vs_attn_hv": {}}

    for metric, lower_better in [("hv_ratio", False), ("igd_plus", True), ("gd_plus", True)]:
        # block matrix: rows = instances, cols = methods (per-instance mean over seeds)
        M = np.array([[np.mean(r["raw"][m][metric]) for m in METHODS] for r in recs])
        # ranks per instance (1 = best)
        if lower_better:
            ranks = np.array([sps.rankdata(row) for row in M])
        else:
            ranks = np.array([sps.rankdata(-row) for row in M])
        avg_rank = ranks.mean(axis=0)
        chi2, p = sps.friedmanchisquare(*[M[:, j] for j in range(len(METHODS))])
        cd = _nemenyi_cd(len(METHODS), len(recs), Q05[len(METHODS)])
        print(f"=== {metric} ({'lower' if lower_better else 'higher'} better) ===")
        print(f"Friedman chi2={chi2:.3f}, p={p:.4g}  | Nemenyi CD(alpha=.05)={cd:.3f}")
        order = np.argsort(avg_rank)
        for j in order:
            print(f"   avg rank {avg_rank[j]:.3f}  {METHODS[j]}")
        # pairwise Wilcoxon TAPE vs each (across instance means), Holm corrected
        tape = M[:, 0]
        raw_p = {}
        stat_w = {}
        delta = {}
        for j in range(1, len(METHODS)):
            try:
                wj, pj = sps.wilcoxon(tape, M[:, j])
            except ValueError:
                wj, pj = float("nan"), float("nan")
            raw_p[METHODS[j]] = pj
            stat_w[METHODS[j]] = float(wj)
            delta[METHODS[j]] = _cliffs_delta(tape, M[:, j])
        # Holm
        items = sorted(raw_p.items(), key=lambda kv: (math.inf if math.isnan(kv[1]) else kv[1]))
        m = len(items)
        print("   TAPE vs baseline (Wilcoxon across instances, Holm-adj):")
        holm = {}
        for i, (name, pj) in enumerate(items):
            adj = min(1.0, pj * (m - i)) if not math.isnan(pj) else float("nan")
            holm[name] = adj
            star = "*" if (not math.isnan(adj) and adj < 0.05) else " "
            print(f"     {star} {name:20s} p={pj:.4g}  p_holm={adj:.4g}  "
                  f"cliffs_delta={delta[name]:+.3f} ({_cliffs_label(delta[name])})")
        print()
        report["per_metric"][metric] = {
            "lower_better": lower_better,
            "friedman_chi2": float(chi2), "friedman_p": float(p),
            "nemenyi_cd": float(cd),
            "avg_rank": {METHODS[j]: float(avg_rank[j]) for j in range(len(METHODS))},
            "tape_vs": {name: {"wilcoxon_stat": stat_w[name], "p": raw_p[name],
                               "p_holm": holm[name], "cliffs_delta": delta[name],
                               "cliffs_magnitude": _cliffs_label(delta[name])}
                        for name in raw_p},
        }

    # bootstrap 95% CIs on HV ratio per method per instance (from raw seeds)
    for r, lab in zip(recs, insts):
        report["bootstrap_ci_hv_ratio"][lab] = {
            mth: list(_bootstrap_ci(np.asarray(r["raw"][mth]["hv_ratio"], float)))
            for mth in METHODS
        }

    # per-instance GNN(TAPE) vs mp-BRKGA HV, 5-seed paired Wilcoxon + bootstrap CI of the gap
    print("=== per-instance HV: TAPE vs mp-BRKGA (paired Wilcoxon, bootstrap CI of gap) ===")
    nwin = 0
    for r, lab in zip(recs, insts):
        a = np.array(r["raw"]["E-HGATv2-TAPE"]["hv_ratio"], float)
        b = np.array(r["raw"]["mp-BRKGA"]["hv_ratio"], float)
        try:
            wj, pj = sps.wilcoxon(a, b)
        except ValueError:
            wj, pj = float("nan"), float("nan")
        d_lo, d_mean, d_hi = _bootstrap_ci(a - b)
        win = a.mean() > b.mean()
        nwin += int(win)
        report["per_instance_tape_vs_mpbrkga"].append({
            "instance": lab, "tape_mean": float(a.mean()), "mp_mean": float(b.mean()),
            "wilcoxon_stat": float(wj), "wilcoxon_p": float(pj),
            "gap_mean": d_mean, "gap_ci95": [d_lo, d_hi],
            "cliffs_delta": _cliffs_delta(a, b), "win": bool(win),
        })
        print(f"   {lab:16s} TAPE={a.mean():.3f} mp={b.mean():.3f} "
              f"{'WIN' if win else 'loss'} W={wj:.1f} p={pj:.3f} "
              f"gapCI=[{d_lo:+.3f},{d_hi:+.3f}]")
    print(f"   TAPE > mp-BRKGA on {nwin}/{len(recs)} instances")

    # TAPE vs attn HV across instances (equivalence-ish)
    print("\n=== HV: TAPE vs attn (across-instance means) ===")
    a = np.array([np.mean(r["raw"]["E-HGATv2-TAPE"]["hv_ratio"]) for r in recs])
    b = np.array([np.mean(r["raw"]["E-HGATv2-attn"]["hv_ratio"]) for r in recs])
    wta, pta = sps.wilcoxon(a, b)
    report["tape_vs_attn_hv"] = {
        "tape_mean": float(a.mean()), "attn_mean": float(b.mean()),
        "wilcoxon_stat": float(wta), "wilcoxon_p": float(pta),
        "cliffs_delta": _cliffs_delta(a, b), "tape_wins": int((a > b).sum()),
    }
    print(f"   mean TAPE={a.mean():.3f}  attn={b.mean():.3f}  Wilcoxon p={pta:.3f} "
          f"(TAPE wins {int((a>b).sum())}/{len(recs)})")

    out_path = OUT / "paper_stats.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
