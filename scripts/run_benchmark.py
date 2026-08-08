"""scripts/run_benchmark.py -- benchmark entrypoint.

Runs the full multi-seed effectiveness benchmark (BRKGA vs E-HGATv2-NSGA-II vs random
ablation) and writes:

- experiments/hypervolume_convergence.png -- convergence curves
- experiments/pareto_tradeoff.png          -- Pareto trade-off vs Oracle PF*
- experiments/benchmark_results.json        -- all raw numbers

Usage::

    python scripts/run_benchmark.py                     # quick (5 seeds, 40 gens)
    python scripts/run_benchmark.py --seeds 30 --gens 100  # full paper config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from ehgat.benchmark.plots import save_benchmark_figures
from ehgat.benchmark.runner import BenchmarkConfig, BenchmarkResult, run_benchmark

console = Console()
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments"


def _fmt(stat: object) -> str:
    """Format a Stat as 'mean [lo, hi]'."""
    from ehgat.benchmark.runner import Stat

    if isinstance(stat, Stat):
        return f"{stat.mean:.4f}  [{stat.lo:.4f}, {stat.hi:.4f}]"
    return str(stat)


def _print_summary(result: BenchmarkResult) -> None:
    console.rule("[bold white]E-HGATv2 Benchmark Results")
    console.print(
        f"  Instance: N={result.num_tasks}  A={result.num_agvs}  QC={result.num_qcs}\n"
        f"  Pop={result.pop_size}  Gens={result.generations}  "
        f"Seeds={len(result.config.seeds)}  Evals={result.pop_size*(result.generations+1)}\n"
        f"  Oracle |PF*|={len(result.golden_front)}  HV*={result.golden_hv:.4f}  "
        f"ref={tuple(round(x,1) for x in result.reference_point)}"
    )

    # ── convergence table ──────────────────────────────────────────────────────
    tbl = Table(title="Final metrics (mean  [95 % CI])", show_header=True, header_style="bold cyan")
    tbl.add_column("Method", style="bold", min_width=22)
    tbl.add_column("HV  (↑)", min_width=26)
    tbl.add_column("IGD+  (↓)", min_width=26)
    tbl.add_column("GD+  (↓)", min_width=26)
    tbl.add_column("Spread  (↓)", min_width=26)
    tbl.add_column("HV / HV*", min_width=10)

    for name, m in result.methods.items():
        tbl.add_row(
            name,
            _fmt(m.final_hv),
            _fmt(m.final_igd_plus),
            _fmt(m.final_gd_plus),
            _fmt(m.final_spread),
            f"{m.final_hv.mean / result.golden_hv:.1%}",
        )
    console.print(tbl)

    # ── faithfulness ──────────────────────────────────────────────────────────
    if result.faithfulness:
        ftbl = Table(title="H3 -- Attention faithfulness", header_style="bold cyan")
        ftbl.add_column("Method")
        ftbl.add_column("Precision@1  (↑)")
        ftbl.add_column("Spearman rho  (up)")
        ftbl.add_column("Random P@1 (foil)")
        for name, faith in result.faithfulness.items():
            ftbl.add_row(
                name,
                f"{faith.precision_at_1:.3f}",
                f"{faith.spearman_rho:.3f}",
                f"{result.random_precision_at_1:.3f}",
            )
        console.print(ftbl)

    # ── HV gap to Oracle ──────────────────────────────────────────────────────
    guided_hv = result.methods.get("E-HGATv2-NSGA-II")
    brkga_hv = result.methods.get("BRKGA")
    if guided_hv and brkga_hv:
        improvement = (guided_hv.final_hv.mean - brkga_hv.final_hv.mean) / brkga_hv.final_hv.mean
        console.print(
            f"\n[bold green]Ablation summary:[/bold green]  E-HGATv2-NSGA-II HV vs BRKGA  "
            f"[bold]{improvement:+.1%}[/bold]"
        )


def _save_json(result: BenchmarkResult, out: Path) -> Path:
    data: dict = {
        "config": {
            "num_tasks": result.num_tasks,
            "num_agvs": result.num_agvs,
            "pop_size": result.pop_size,
            "generations": result.generations,
            "seeds": list(result.config.seeds),
            "oracle": result.config.oracle,
            "oracle_seeds": result.config.oracle_seeds,
            "oracle_generations": result.config.oracle_generations,
            "oracle_pop_size": result.config.oracle_pop_size,
            "oracle_workers": result.config.oracle_workers,
            "surrogate_samples": result.config.surrogate_samples,
            "surrogate_epochs": result.config.surrogate_epochs,
        },
        "golden_hv": result.golden_hv,
        "reference_point": list(result.reference_point),
        "methods": {
            name: {
                "final_hv": {"mean": m.final_hv.mean, "lo": m.final_hv.lo, "hi": m.final_hv.hi},
                "final_igd_plus": {
                    "mean": m.final_igd_plus.mean,
                    "lo": m.final_igd_plus.lo,
                    "hi": m.final_igd_plus.hi,
                },
                "final_gd_plus": {
                    "mean": m.final_gd_plus.mean,
                    "lo": m.final_gd_plus.lo,
                    "hi": m.final_gd_plus.hi,
                },
                "final_spread": {
                    "mean": m.final_spread.mean,
                    "lo": m.final_spread.lo,
                    "hi": m.final_spread.hi,
                },
                "hv_curve_mean": m.hv_curve_mean.tolist(),
                "hv_curve_lo": m.hv_curve_lo.tolist(),
                "hv_curve_hi": m.hv_curve_hi.tolist(),
            }
            for name, m in result.methods.items()
        },
        "faithfulness": {
            name: {
                "precision_at_1": f.precision_at_1,
                "spearman_rho": f.spearman_rho,
                "num_schedules": f.num_schedules,
            }
            for name, f in result.faithfulness.items()
        },
        "random_precision_at_1": result.random_precision_at_1,
    }
    path = out / "benchmark_results.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E-HGATv2 effectiveness benchmark")
    parser.add_argument("--tasks", type=int, default=5, help="number of tasks/nodes")
    parser.add_argument(
        "--agvs", type=int, default=None,
        help="AGV fleet size (default: scaled_fleet(N), ~1 AGV per 12 tasks)",
    )
    parser.add_argument(
        "--qcs", type=int, default=None,
        help="number of quay cranes, max 6 (default: scaled_fleet(N), 3..6)",
    )
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (default 5)")
    parser.add_argument("--gens", type=int, default=40, help="generations (default 40)")
    parser.add_argument("--pop", type=int, default=None, help="population size (default 20N)")
    parser.add_argument(
        "--oracle",
        choices=("auto", "exact", "approx"),
        default="auto",
        help="reference front strategy: exact for N<=5, approx otherwise by default",
    )
    parser.add_argument("--oracle-seeds", type=int, default=100)
    parser.add_argument("--oracle-gens", type=int, default=100)
    parser.add_argument("--oracle-pop", type=int, default=None)
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel processes for the search runs (3 methods x N seeds); set to vCPU count",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads (keep at 1: tiny graphs thrash with many threads)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="softmax temperature for soft attention bottleneck sampling",
    )
    parser.add_argument(
        "--screening",
        type=int,
        default=4,
        help="surrogate offspring screening factor for the guided method (1=off)",
    )
    parser.add_argument("--surrogate-samples", type=int, default=500)
    parser.add_argument("--surrogate-epochs", type=int, default=30)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    ns = parser.parse_args(args)

    config = BenchmarkConfig(
        num_tasks=ns.tasks,
        num_agvs=ns.agvs,
        num_qcs=ns.qcs,
        num_seeds=ns.seeds,
        generations=ns.gens,
        pop_size=ns.pop,
        oracle=ns.oracle,
        oracle_seeds=ns.oracle_seeds,
        oracle_generations=ns.oracle_gens,
        oracle_pop_size=ns.oracle_pop,
        oracle_workers=ns.oracle_workers,
        search_workers=ns.workers,
        torch_threads=ns.torch_threads,
        mutation_temperature=ns.temperature,
        screening_factor=ns.screening,
        surrogate_samples=ns.surrogate_samples,
        surrogate_epochs=ns.surrogate_epochs,
    )

    console.print(
        f"[bold]Running benchmark[/bold]: N={ns.tasks}, {ns.seeds} seeds x {ns.gens} gens  "
        f"(surrogate: {ns.surrogate_samples} samples / {ns.surrogate_epochs} epochs, "
        f"oracle={ns.oracle})"
    )
    with console.status("Training surrogate + running all seeds..."):
        result = run_benchmark(config)

    _print_summary(result)

    out = Path(ns.out)
    hv_png, pf_png = save_benchmark_figures(result, out)
    json_path = _save_json(result, out)
    console.print(
        f"\n[bold]Saved:[/bold]\n  {hv_png}\n  {pf_png}\n  {json_path}"
    )


if __name__ == "__main__":
    main()
