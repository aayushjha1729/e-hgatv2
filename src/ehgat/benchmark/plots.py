"""Benchmark charts: hypervolume convergence and the Pareto trade-off.

Two figures back the effectiveness story:

1. Hypervolume vs generations -- mean curve + 95% CI band per method, with the exact
   HV* of the golden PF* as a dashed ceiling (the convergence-velocity plot).
2. Pareto trade-off -- the best front each method achieves (non-dominated union over
   seeds) against the solid black Oracle front: BRKGA in red, E-HGATv2-NSGA-II in green,
   the random ablation in grey.

A non-interactive (Agg) backend is selected so the charts render headless (CI / tests).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ehgat.benchmark.runner import BenchmarkResult, MethodResult
from ehgat.search.nsga2 import non_dominated_indices

__all__ = ["plot_hv_convergence", "plot_pareto_tradeoff", "save_benchmark_figures"]

_COLORS = {
    "BRKGA": "tab:red",
    "E-HGATv2-NSGA-II": "tab:green",
    "NSGA-II (random)": "tab:gray",
}


def _combined_front(method: MethodResult) -> list[tuple[float, float]]:
    """Non-dominated union of a method's per-seed final fronts."""
    points: list[tuple[float, float]] = [p for front in method.final_fronts for p in front]
    if not points:
        return []
    keep = non_dominated_indices(points)
    return sorted(points[i] for i in keep)


def plot_hv_convergence(result: BenchmarkResult) -> Figure:
    """Hypervolume-vs-generations curves (mean + 95% CI) with the golden HV* ceiling."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    generations = range(result.generations + 1)
    for name, method in result.methods.items():
        color = _COLORS.get(name)
        ax.plot(generations, method.hv_curve_mean, label=name, color=color, linewidth=2)
        ax.fill_between(
            generations, method.hv_curve_lo, method.hv_curve_hi, color=color, alpha=0.18
        )
    ax.axhline(
        result.golden_hv, linestyle="--", color="black", linewidth=1.2, label="Oracle HV*"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    n_seeds = len(result.config.seeds)
    ax.set_title(f"Hypervolume convergence (N={result.num_tasks}, {n_seeds} seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pareto_tradeoff(result: BenchmarkResult) -> Figure:
    """Best achieved front per method vs the exact Oracle front."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    golden = sorted(result.golden_front)
    gx = [p[0] for p in golden]
    gy = [p[1] for p in golden]
    ax.step(gx, gy, where="post", color="black", linewidth=1.6, label="Oracle PF*", zorder=1)
    ax.scatter(gx, gy, color="black", s=14, zorder=2)

    for name, method in result.methods.items():
        front = _combined_front(method)
        if not front:
            continue
        ax.scatter(
            [p[0] for p in front],
            [p[1] for p in front],
            color=_COLORS.get(name),
            s=30,
            alpha=0.8,
            label=name,
            zorder=3,
        )
    ax.set_xlabel("Makespan  $C_{max}$  (s)")
    ax.set_ylabel("Total AGV energy  $E$  (kJ)")
    ax.set_title(f"Pareto trade-off vs Oracle (N={result.num_tasks})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_benchmark_figures(result: BenchmarkResult, out_dir: str | Path) -> tuple[Path, Path]:
    """Render and save both charts as PNGs; returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hv_path = out / "hypervolume_convergence.png"
    pf_path = out / "pareto_tradeoff.png"
    fig_hv = plot_hv_convergence(result)
    fig_hv.savefig(hv_path, dpi=150)
    plt.close(fig_hv)
    fig_pf = plot_pareto_tradeoff(result)
    fig_pf.savefig(pf_path, dpi=150)
    plt.close(fig_pf)
    return hv_path, pf_path
