"""Generate publication figures for the faithful-guidance paper.

Reads experiments/fused_tape_guided/tape_bench_*.json and writes vector PDFs into
paper/figs/:

- fig_faithfulness.pdf : per-instance TAPE leg-critical Jaccard vs |attention Spearman|
  (+ random precision@1 baseline) -- the Req-2 result.
- fig_hv_box.pdf       : distribution of HV/HV* per method across all instances.
- fig_avgrank.pdf      : average-rank chart for HV with the Nemenyi critical difference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "experiments" / "fused_tape_guided"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

METHODS = ["E-HGATv2-TAPE", "E-HGATv2-attn", "NSGA-II (random)", "mp-BRKGA", "single-pop BRKGA"]
SHORT = {"E-HGATv2-TAPE": "TAPE (ours)", "E-HGATv2-attn": "Attention",
         "NSGA-II (random)": "Random", "mp-BRKGA": "mp-BRKGA", "single-pop BRKGA": "sp-BRKGA"}
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}


def _label(rec: dict) -> str:
    s = rec["instance"]
    if s.startswith("toy:"):
        s = "SD-" + s.split(":")[1]
    if rec["coupled"]:
        s += "-C"
    return s


def load():
    recs = [json.loads(f.read_text()) for f in sorted(IN.glob("tape_bench_*.json"))]
    recs.sort(key=lambda r: (r["coupled"], r["instance"].startswith("L"), r["n"]))
    return recs


def fig_faithfulness(recs):
    labels = [_label(r) for r in recs]
    jac = [r["faithfulness"]["tape_leg_critical_jaccard"] for r in recs]
    rho = [abs(r["faithfulness"]["attention_spearman_rho"]) for r in recs]
    x = np.arange(len(recs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    ax.bar(x - w / 2, jac, w, label="TAPE: critical-path Jaccard vs oracle", color="#2a6f97")
    ax.bar(x + w / 2, rho, w, label=r"Attention: $|\rho_{\mathrm{Spearman}}|$ vs true levers",
           color="#bc4749")
    ax.axhline(1.0, ls=":", c="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("faithfulness")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax.set_title("Guidance-signal faithfulness (higher = more faithful)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_faithfulness.pdf")
    plt.close(fig)


def fig_hv_box(recs):
    data = [[np.mean(r["raw"][m]["hv_ratio"]) for r in recs] for m in METHODS]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black"))
    colors = ["#2a6f97", "#468faf", "#a9d6e5", "#e9c46a", "#f4a261"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"HV / HV$^\star$ (higher = better)")
    ax.axhline(1.0, ls=":", c="grey", lw=0.8)
    ax.set_title("Hypervolume ratio across all instances", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_hv_box.pdf")
    plt.close(fig)


def fig_avgrank(recs):
    M = np.array([[np.mean(r["raw"][m]["hv_ratio"]) for m in METHODS] for r in recs])
    ranks = np.array([sps.rankdata(-row) for row in M])  # higher HV = rank 1
    avg = ranks.mean(axis=0)
    n = len(recs)
    cd = Q05[len(METHODS)] * math.sqrt(len(METHODS) * (len(METHODS) + 1) / (6.0 * n))
    order = np.argsort(avg)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(METHODS))[::-1]
    ax.barh(y, avg[order], color="#2a6f97", height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[METHODS[i]] for i in order], fontsize=9)
    ax.set_xlabel("average rank (1 = best, lower = better)")
    for yi, i in zip(y, order):
        ax.text(avg[i] + 0.05, yi, f"{avg[i]:.2f}", va="center", fontsize=8)
    best = avg[order][0]
    ax.axvline(best, c="grey", ls="--", lw=0.8)
    ax.axvspan(best, best + cd, color="grey", alpha=0.15)
    ax.set_title(f"Average HV rank over {n} instances (Nemenyi CD$={cd:.2f}$)", fontsize=10)
    ax.set_xlim(0, max(avg) + 0.6)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_avgrank.pdf")
    plt.close(fig)


def main():
    recs = load()
    fig_faithfulness(recs)
    fig_hv_box(recs)
    fig_avgrank(recs)
    print(f"wrote {len(list(FIGS.glob('*.pdf')))} figures to {FIGS}")


if __name__ == "__main__":
    main()
