"""scripts/run_fused_eval.py -- publishable multi-seed evaluation of the physics-fused HGAT.

Runs the coupled (or uncoupled) fused-TAPE pipeline over a grid of instance sizes x seeds
and aggregates each metric to a mean with a 95% confidence interval (Student-t over seeds),
the reporting convention used throughout the results. Per (N, seed) it:

  1. trains the frozen E-HGATv2 core,
  2. fine-tunes the anchored fused tropical head (coupled when --peak-power is set),
  3. measures held-out calibration R^2 / MAE and faithfulness vs the exact TAPE oracle
     (leg/arc critical-path Jaccard) on a fresh held-out explain set.

Seeds run in parallel across CPU worker processes (the per-sample fused fine-tune is small
-graph CPU work), each pinned to one thread to avoid BLAS oversubscription.

Usage::

    python scripts/run_fused_eval.py --tasks 6 10 20 --seeds 10 --peak-power 30 \\
        --core-samples 3000 --core-epochs 80 --fused-samples 1500 --fused-epochs 40 \\
        --explain-samples 256 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats as sps

console = Console()
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "fused_eval"

# Headline metrics: (key, label, decimals, higher_is_better).
_METRICS = (
    ("r2_makespan", "R2 makespan", 4, True),
    ("r2_energy", "R2 energy", 4, True),
    ("mae_makespan", "MAE makespan (s)", 3, False),
    ("mae_energy", "MAE energy", 4, False),
    ("mean_leg_critical_jaccard", "Leg critical Jaccard", 4, True),
    ("mean_arc_critical_jaccard", "Arc critical Jaccard", 4, True),
    ("mean_makespan_abs_error", "TAPE makespan |err| (s)", 4, False),
    ("mean_energy_abs_error", "TAPE energy |err|", 6, False),
)


@dataclass(frozen=True)
class Job:
    num_tasks: int
    seed: int


def _run_one(job: Job, args_dict: dict) -> dict:
    """Worker: one (N, seed) fused-TAPE run -> flat metric dict.

    Threads per worker come from THREADS_PER_WORKER (default 1) so a many-core box can
    trade a little per-worker BLAS parallelism against running every (N, seed) concurrently.
    """
    nthr = int(os.environ.get("THREADS_PER_WORKER", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(nthr))
    os.environ.setdefault("MKL_NUM_THREADS", str(nthr))
    import torch

    torch.set_num_threads(nthr)

    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.environment.instance import build_toy_instance
    from ehgat.explain.fused_explainer import faithfulness_report
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.explain.train_fused_batched import train_fused_batched
    from ehgat.utils.seeding import make_rng

    t0 = time.perf_counter()
    instance = build_toy_instance(num_tasks=job.num_tasks, peak_power=args_dict["peak_power"])
    core = build_core(
        instance,
        seed=job.seed,
        num_samples=args_dict["core_samples"],
        epochs=args_dict["core_epochs"],
        batch_size=args_dict["core_batch"],
        lr=args_dict["core_lr"],
        device=args_dict["device"],
    )
    fused_cfg = FusedTrainConfig(
        num_samples=args_dict["fused_samples"],
        epochs=args_dict["fused_epochs"],
        batch_size=args_dict["fused_batch"],
        lr=args_dict["fused_lr"],
        seed=job.seed,
        use_physics_prior=args_dict["physics_prior"],
        alpha_makespan=args_dict["alpha_makespan"],
        unroll_steps=args_dict["unroll_steps"],
    )
    # Batched (vectorised, GPU-ready) fused fine-tune replaces the per-sample Python loop;
    # the loss is identical for fixed N (validated in test_fused_batched.py), only faster.
    if args_dict["batched"]:
        result = train_fused_batched(instance, core, fused_cfg, device=args_dict["device"])
    else:
        result = train_fused(instance, core, fused_cfg)
    rng = make_rng(job.seed + 7)
    schedules = [
        decode(rng.random(NUM_BLOCKS * instance.num_tasks), instance)
        for _ in range(args_dict["explain_samples"])
    ]
    reports = [faithfulness_report(result.model, s, instance) for s in schedules]
    n = len(reports)
    cal = result.metrics
    out = {
        "num_tasks": job.num_tasks,
        "seed": job.seed,
        "r2_makespan": float(cal.get("r2_makespan", float("nan"))),
        "r2_energy": float(cal.get("r2_energy", float("nan"))),
        "mae_makespan": float(cal.get("mae_makespan", float("nan"))),
        "mae_energy": float(cal.get("mae_energy", float("nan"))),
        "mean_leg_critical_jaccard": sum(r.leg_critical_jaccard for r in reports) / n,
        "mean_arc_critical_jaccard": sum(r.arc_critical_jaccard for r in reports) / n,
        "mean_makespan_abs_error": sum(r.makespan_abs_error for r in reports) / n,
        "mean_energy_abs_error": sum(r.energy_abs_error for r in reports) / n,
        "wall_s": time.perf_counter() - t0,
    }
    return out


def _ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_lo, ci_hi) with a Student-t 95% CI over seeds."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    k = arr.size
    if k == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if k == 1:
        return mean, 0.0, mean, mean
    sd = float(arr.std(ddof=1))
    half = float(sps.t.ppf(0.975, k - 1) * sd / np.sqrt(k))
    return mean, sd, mean - half, mean + half


def _emit_tables(
    agg: dict, tasks: list[int], seeds: int, coupling: str, mode: str, base: Path
) -> None:
    """Write paper-ready Markdown + LaTeX (booktabs) tables: rows = N, cols = metric mean +/- CI."""
    cols = [*_METRICS, ("wall_s", "Train wall (s)", 1, False)]

    def cell(n: int, key: str, dec: int) -> str:
        a = agg[str(n)][key]
        half = (a["ci_hi"] - a["ci_lo"]) / 2.0
        return f"{a['mean']:.{dec}f} \u00b1 {half:.{dec}f}"

    # --- Markdown ---
    md = [f"# Physics-coupled fused-HGAT evaluation ({coupling}, {mode})",
          f"\n_{seeds} seeds per cell; cells are mean \u00b1 95% Student-t CI._\n",
          "| Metric | " + " | ".join(f"N={n}" for n in tasks) + " |",
          "|---|" + "|".join(["---"] * len(tasks)) + "|"]
    for key, label, dec, _hib in cols:
        md.append(f"| {label} | " + " | ".join(cell(n, key, dec) for n in tasks) + " |")
    base.with_suffix(".md").write_text("\n".join(md) + "\n")

    # --- LaTeX (booktabs) ---
    tex = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{Physics-coupled fused E-HGATv2: calibration, faithfulness, and cost "
        f"({coupling}, {mode}). Mean $\\pm$ 95\\% CI over {seeds} seeds.}}",
        "\\label{tab:fused-coupled}",
        "\\begin{tabular}{l" + "r" * len(tasks) + "}",
        "\\toprule",
        "Metric & " + " & ".join(f"$N={n}$" for n in tasks) + " \\\\",
        "\\midrule",
    ]
    for key, label, dec, _hib in cols:
        cells = " & ".join(
            cell(n, key, dec).replace("\u00b1", "$\\pm$") for n in tasks
        )
        tex.append(f"{label} & {cells} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    base.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    console.print(f"[green]wrote tables[/green] {base.with_suffix('.md')} , {base.with_suffix('.tex')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed publishable fused-TAPE evaluation.")
    p.add_argument("--tasks", type=int, nargs="+", default=[6, 10, 20])
    p.add_argument("--seeds", type=int, default=10, help="number of seeds (0..S-1)")
    p.add_argument("--peak-power", type=float, default=30.0,
                   help="fleet power budget (engages coupling). Use a sentinel <0 for uncoupled.")
    p.add_argument("--core-samples", type=int, default=3000)
    p.add_argument("--core-epochs", type=int, default=80)
    p.add_argument("--core-batch", type=int, default=32,
                   help="core mini-batch (large=fewer GPU kernel launches; scale --core-lr with it)")
    p.add_argument("--core-lr", type=float, default=1e-3)
    p.add_argument("--fused-samples", type=int, default=1500)
    p.add_argument("--fused-epochs", type=int, default=40)
    p.add_argument("--fused-batch", type=int, default=32,
                   help="fused mini-batch (large=fewer GPU kernel launches; scale --fused-lr with it)")
    p.add_argument("--fused-lr", type=float, default=1e-3)
    p.add_argument("--explain-samples", type=int, default=256)
    p.add_argument("--physics-prior", action="store_true")
    p.add_argument("--alpha-makespan", type=float, default=1.0,
                   help="weight on the makespan loss term (push the heads to nail the critical path)")
    p.add_argument("--unroll-steps", type=int, default=0,
                   help="physics-unrolled coupled refinement steps (0 = static wait head)")
    p.add_argument("--per-sample", action="store_true",
                   help="use the slow per-sample fused trainer instead of the batched one")
    p.add_argument("--device", type=str, default="cpu",
                   help="device for the batched fused fine-tune (cpu keeps 60-way parallelism)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    peak_power = None if args.peak_power is not None and args.peak_power < 0 else args.peak_power
    args_dict = {
        "peak_power": peak_power,
        "core_samples": args.core_samples,
        "core_epochs": args.core_epochs,
        "core_batch": args.core_batch,
        "core_lr": args.core_lr,
        "fused_samples": args.fused_samples,
        "fused_epochs": args.fused_epochs,
        "fused_batch": args.fused_batch,
        "fused_lr": args.fused_lr,
        "explain_samples": args.explain_samples,
        "physics_prior": args.physics_prior,
        "alpha_makespan": args.alpha_makespan,
        "unroll_steps": args.unroll_steps,
        "batched": not args.per_sample,
        "device": args.device,
    }
    jobs = [Job(n, s) for n in args.tasks for s in range(args.seeds)]
    mode = "physics_prior" if args.physics_prior else "gnn_predicts_legs"
    coupling = "uncoupled" if peak_power is None else f"coupled(pp={peak_power:g})"
    console.rule(f"[bold]Fused-TAPE eval | {coupling} | {mode} | "
                 f"N={args.tasks} | {args.seeds} seeds | {len(jobs)} runs | {args.workers} workers")

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, j, args_dict): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            j = futs[fut]
            r = fut.result()
            rows.append(r)
            done += 1
            console.print(f"[green]done[/green] N={j.num_tasks} seed={j.seed} "
                          f"R2_mk={r['r2_makespan']:.3f} legJacc={r['mean_leg_critical_jaccard']:.3f} "
                          f"({r['wall_s']:.0f}s)  [{done}/{len(jobs)}]")

    # Aggregate per N.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "config": {**args_dict, "tasks": args.tasks, "seeds": args.seeds, "mode": mode},
        "per_seed": rows,
        "aggregate": {},
    }
    agg = summary["aggregate"]
    assert isinstance(agg, dict)
    for n in args.tasks:
        vals = {k: [r[k] for r in rows if r["num_tasks"] == n] for k, *_ in _METRICS}
        agg[str(n)] = {k: dict(zip(("mean", "std", "ci_lo", "ci_hi"), _ci95(v), strict=True))
                       for k, v in vals.items()}

    # Aggregate training wall-time per N as a cost column.
    for n in args.tasks:
        wv = [r["wall_s"] for r in rows if r["num_tasks"] == n]
        agg[str(n)]["wall_s"] = dict(
            zip(("mean", "std", "ci_lo", "ci_hi"), _ci95(wv), strict=True)
        )

    tag = args.tag or (coupling.replace("(", "_").replace(")", "").replace("=", "") )
    out_path = OUT_DIR / f"fused_eval_{tag}_{mode}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    _emit_tables(agg, args.tasks, args.seeds, coupling, mode, out_path.with_suffix(""))

    # Print one table per metric: rows = N, value = mean +/- half-CI.
    for key, label, dec, hib in _METRICS:
        tbl = Table(title=f"{label}  (mean \u00b1 95% CI, {args.seeds} seeds)", header_style="bold cyan")
        tbl.add_column("N", justify="right")
        tbl.add_column("mean \u00b1 CI", justify="right")
        for n in args.tasks:
            a = agg[str(n)][key]
            half = (a["ci_hi"] - a["ci_lo"]) / 2.0
            tbl.add_row(str(n), f"{a['mean']:.{dec}f} \u00b1 {half:.{dec}f}")
        console.print(tbl)

    console.print(f"\n[bold green]wrote[/bold green] {out_path}   "
                  f"total wall {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
