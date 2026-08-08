"""scripts/run_fused_tape.py -- physics-fused TAPE validation.

Trains the anchored fused E-HGATv2 head (tropical Max-Plus DP makespan + exact additive
energy) on a frozen surrogate core, then validates:

1. Calibration -- held-out R^2 for (C_max, E) restores to >= 0.99 once the heads snap
   onto the physics layer.
2. Faithfulness-by-construction -- the fused model's native gradients (its critical
   path) agree with the exact, simulator-based TAPE oracle (leg/arc critical Jaccard ~ 1).
3. Trade-off Criticality Scores -- per-task/edge TCS over a sampled near-front, the explainer
   output for the Homayouni_XAI+MOO landscape.

Writes a JSON artifact per instance size to experiments/fused_tape/fused_tape_n{N}.json.

Usage::

    python scripts/run_fused_tape.py                                   # quick (N=6)
    python scripts/run_fused_tape.py --tasks 6 10 20 \\
        --core-samples 1500 --core-epochs 60 --fused-samples 800 --fused-epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import build_toy_instance
from ehgat.explain.fused_explainer import faithfulness_report, fused_tradeoff_criticality_scores
from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
from ehgat.utils.seeding import make_rng

console = Console()
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "fused_tape"


def _summary(num_tasks: int, args: argparse.Namespace) -> dict[str, object]:
    instance = build_toy_instance(num_tasks=num_tasks, peak_power=args.peak_power)
    core = build_core(
        instance, seed=args.seed, num_samples=args.core_samples, epochs=args.core_epochs
    )
    result = train_fused(
        instance,
        core,
        FusedTrainConfig(
            num_samples=args.fused_samples,
            epochs=args.fused_epochs,
            seed=args.seed,
            use_physics_prior=args.physics_prior,
        ),
    )

    rng = make_rng(args.seed + 7)
    schedules = [
        decode(rng.random(NUM_BLOCKS * instance.num_tasks), instance)
        for _ in range(args.explain_samples)
    ]
    reports = [faithfulness_report(result.model, s, instance) for s in schedules]
    pts = fused_tradeoff_criticality_scores(result.model, schedules, instance)

    n = len(reports)
    faith = {
        "mean_leg_critical_jaccard": sum(r.leg_critical_jaccard for r in reports) / n,
        "mean_arc_critical_jaccard": sum(r.arc_critical_jaccard for r in reports) / n,
        "mean_makespan_abs_error": sum(r.makespan_abs_error for r in reports) / n,
        "mean_energy_abs_error": sum(r.energy_abs_error for r in reports) / n,
    }
    return {
        "instance": {
            "num_tasks": instance.num_tasks,
            "num_agvs": instance.num_agvs,
            "num_qcs": len(instance.qcs),
        },
        "mode": "physics_prior" if args.physics_prior else "gnn_predicts_legs",
        "peak_power": args.peak_power,
        "calibration": result.metrics,
        "history": result.history,
        "faithfulness": faith,
        "tradeoff_criticality_scores": pts,
    }


def _print(payload: dict[str, object]) -> None:
    inst = payload["instance"]  # type: ignore[index]
    console.rule(f"[bold white]Fused TAPE -- N={inst['num_tasks']} A={inst['num_agvs']} QC={inst['num_qcs']}")

    cal = payload["calibration"]  # type: ignore[index]
    tbl = Table(title="Calibration (held-out, physical units)", header_style="bold cyan")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    for key in ("r2_makespan", "r2_energy", "mae_makespan", "mae_energy"):
        if key in cal:  # type: ignore[operator]
            tbl.add_row(key, f"{cal[key]:.4f}")  # type: ignore[index]
    console.print(tbl)

    faith = payload["faithfulness"]  # type: ignore[index]
    ftbl = Table(title="Faithfulness vs exact TAPE oracle", header_style="bold magenta")
    ftbl.add_column("Metric", style="bold")
    ftbl.add_column("Value", justify="right")
    for key, val in faith.items():  # type: ignore[union-attr]
        ftbl.add_row(key, f"{val:.4f}")
    console.print(ftbl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics-fused TAPE validation.")
    parser.add_argument("--tasks", type=int, nargs="+", default=[6])
    parser.add_argument("--core-samples", type=int, default=1500)
    parser.add_argument("--core-epochs", type=int, default=60)
    parser.add_argument("--fused-samples", type=int, default=800)
    parser.add_argument("--fused-epochs", type=int, default=30)
    parser.add_argument("--explain-samples", type=int, default=32)
    parser.add_argument(
        "--physics-prior",
        action="store_true",
        help="Use the exact closed-form leg prior (faithful baseline) instead of letting "
        "the GNN predict the leg times itself (default).",
    )
    parser.add_argument(
        "--peak-power",
        type=float,
        default=None,
        help="Fleet-wide instantaneous power budget (kW). Omit for the uncoupled "
        "(closed-form) physics; set it to engage the nonlinear power-coupled simulator.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for num_tasks in args.tasks:
        payload = _summary(num_tasks, args)
        _print(payload)
        tag = f"_pp{args.peak_power:g}" if args.peak_power is not None else ""
        out_path = OUT_DIR / f"fused_tape_n{num_tasks}{tag}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]wrote[/green] {out_path}")


if __name__ == "__main__":
    main()
