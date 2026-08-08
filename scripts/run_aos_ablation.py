"""scripts/run_aos_ablation.py -- operator-selection ablation entrypoint.

Runs the three-arm Channel-B operator-selection ablation -- random (null),
attention (the method), oracle (upper bound) -- with an identical NSGA-II
skeleton, and writes a structured JSON artifact the stats module consumes::

    experiments/aos_ablation.json

Usage::

    python scripts/run_aos_ablation.py                         # quick (N=5, 5 seeds)
    python scripts/run_aos_ablation.py --tasks 10 --seeds 30 \\
        --gens 60 --workers 30 --surrogate-samples 1000 --surrogate-epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from ehgat.benchmark.aos_ablation import (
    AOS_ARMS,
    AOSAblationConfig,
    AOSAblationResult,
    run_aos_ablation,
    to_json_dict,
)

console = Console()
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments"


def _fmt_stat(stat: object) -> str:
    from ehgat.benchmark.runner import Stat

    if isinstance(stat, Stat):
        return f"{stat.mean:.4f}  [{stat.lo:.4f}, {stat.hi:.4f}]"
    return str(stat)


def _median_evals(result: AOSAblationResult, arm: str) -> str:
    records = result.arms[arm].records
    vals = [r.evals_to_threshold for r in records if r.evals_to_threshold is not None]
    n = len(records)
    if not vals:
        return f"-- (0/{n})"
    vals.sort()
    median = vals[len(vals) // 2]
    return f"{median}  ({len(vals)}/{n})"


def _print_summary(result: AOSAblationResult) -> None:
    console.rule("[bold white]AOS Ablation -- Channel B (operator selection)")
    console.print(
        f"  Instance: N={result.num_tasks}  A={result.num_agvs}  QC={result.num_qcs}\n"
        f"  Pop={result.pop_size}  Gens={result.generations}  "
        f"Seeds={len(result.config.seeds)}\n"
        f"  HV*={result.golden_hv:.4f}  threshold={result.config.threshold_fraction:.0%}*HV*"
        f"={result.threshold_hv:.4f}  ref={tuple(round(x,1) for x in result.reference_point)}"
    )

    tbl = Table(title="Per-arm metrics (mean  [95% CI])", header_style="bold cyan")
    tbl.add_column("Arm", style="bold", min_width=12)
    tbl.add_column("HV  (up)", min_width=26)
    tbl.add_column("HV-AUC  (up)", min_width=24)
    tbl.add_column("IGD+  (down)", min_width=24)
    tbl.add_column("Median evals->thr", min_width=18)

    for arm in AOS_ARMS:
        a = result.arms[arm]
        tbl.add_row(
            arm,
            _fmt_stat(a.final_hv),
            _fmt_stat(a.hv_auc),
            _fmt_stat(a.igd_plus),
            _median_evals(result, arm),
        )
    console.print(tbl)

    rnd = result.arms["random"].final_hv.mean
    att = result.arms["attention"].final_hv.mean
    orc = result.arms["oracle"].final_hv.mean
    gap = (orc - rnd)
    closed = (att - rnd) / gap if abs(gap) > 1e-12 else 0.0
    console.print(
        f"\n[bold green]Ablation summary:[/bold green]  attention vs random HV "
        f"[bold]{(att - rnd) / rnd:+.1%}[/bold]; closes "
        f"[bold]{closed:.0%}[/bold] of the random->oracle gap."
    )


def _save_json(result: AOSAblationResult, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "aos_ablation.json"
    path.write_text(json.dumps(to_json_dict(result), indent=2))
    return path


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E-HGATv2 AOS Channel-B ablation")
    parser.add_argument("--tasks", type=int, default=5, help="number of tasks/nodes")
    parser.add_argument("--seeds", type=int, default=5, help="number of search seeds")
    parser.add_argument("--gens", type=int, default=40, help="generations")
    parser.add_argument("--pop", type=int, default=None, help="population size (default 20N)")
    parser.add_argument(
        "--random-task",
        action="store_true",
        help="ablate Channel A too (random task selection); default keeps attention task selection",
    )
    parser.add_argument("--mutation-temp", type=float, default=0.25, help="Channel-A softmax tau")
    parser.add_argument("--operator-temp", type=float, default=0.5, help="Channel-B operator tau")
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=1.0,
        help="`speed` operator score (>=structural; 1.0 fixes the crowd-out, 0.5 = old behaviour)",
    )
    parser.add_argument(
        "--operator-granularity",
        choices=("population", "per_task"),
        default="population",
        help="Channel-B bias scope: population-averaged per generation, or per-task routing",
    )
    parser.add_argument(
        "--aggregation-window",
        choices=("full", "front", "best"),
        default="front",
        help="bottleneck-type readout window (population mode only)",
    )
    parser.add_argument(
        "--screening",
        type=int,
        default=1,
        help="surrogate screening factor (held equal across arms)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95, help="HV fraction for evals-to-threshold"
    )
    parser.add_argument(
        "--oracle", choices=("auto", "exact", "approx"), default="auto", help="reference front"
    )
    parser.add_argument("--oracle-seeds", type=int, default=200)
    parser.add_argument("--oracle-gens", type=int, default=200)
    parser.add_argument("--oracle-pop", type=int, default=None)
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument("--surrogate-samples", type=int, default=500)
    parser.add_argument("--surrogate-epochs", type=int, default=30)
    parser.add_argument("--surrogate-seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="parallel processes over (arm x seed) runs"
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap CI resamples")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    ns = parser.parse_args(args)

    config = AOSAblationConfig(
        num_tasks=ns.tasks,
        generations=ns.gens,
        pop_size=ns.pop,
        num_seeds=ns.seeds,
        random_task=ns.random_task,
        mutation_temperature=ns.mutation_temp,
        operator_temperature=ns.operator_temp,
        operator_speed_weight=ns.speed_weight,
        operator_granularity=ns.operator_granularity,
        aggregation_window=ns.aggregation_window,
        screening_factor=ns.screening,
        threshold_fraction=ns.threshold,
        oracle=ns.oracle,
        oracle_seeds=ns.oracle_seeds,
        oracle_generations=ns.oracle_gens,
        oracle_pop_size=ns.oracle_pop,
        oracle_workers=ns.oracle_workers,
        surrogate_samples=ns.surrogate_samples,
        surrogate_epochs=ns.surrogate_epochs,
        surrogate_seed=ns.surrogate_seed,
        search_workers=ns.workers,
        torch_threads=ns.torch_threads,
        bootstrap_resamples=ns.bootstrap,
    )

    console.print(
        f"[bold]Running AOS ablation[/bold]: N={ns.tasks}, {ns.seeds} seeds x {ns.gens} gens x "
        f"3 arms (surrogate: {ns.surrogate_samples}/{ns.surrogate_epochs}, oracle={ns.oracle})"
    )
    result = run_aos_ablation(config)

    _print_summary(result)
    json_path = _save_json(result, Path(ns.out))
    console.print(f"\n[bold]Saved:[/bold]\n  {json_path}")


if __name__ == "__main__":
    main()
