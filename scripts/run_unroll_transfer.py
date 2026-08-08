"""Size-transfer evaluation for the physics-unrolled GNN+TAPE (the scaling claim).

Train a fused E-HGATv2 (frozen core + tropical head) on a small N with exact event-driven
simulator labels, then predict the coupled (C_max, E) solver-free on larger, unseen N
-- no retraining, no simulator at test time. Compares the static wait head (K=0) against
the physics-unrolled refinement (K>0) to show the unroll both lifts in-distribution
accuracy and transfers to scale, while remaining solver-free.

Both the GNN (which predicts the nonlinear waits) and TAPE (max-plus timing and
composition) act at every unroll step. Emits R2 and MAE per test-N together with a
compute-saved column (exact-simulator wall time against the model's batched forward).

    python scripts/run_unroll_transfer.py --train-tasks 8 --test-tasks 12 20 30 \
        --seeds 8 --peak-power 30 --unroll-steps 2 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "unroll_transfer"


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or 1e-12
    return 1.0 - ss_res / ss_tot


def _train_and_transfer(task: tuple[int, int, dict]) -> dict:
    """Worker: train on small N (one seed, one K) -> zero-shot eval on every test N."""
    seed, k, a = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)

    from ehgat.environment.instance import build_toy_instance
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, build_samples
    from ehgat.explain.train_fused_batched import _forward_batch, build_batch, train_fused_batched

    train_inst = build_toy_instance(num_tasks=a["train_tasks"], peak_power=a["peak_power"])
    core = build_core(train_inst, seed=seed, num_samples=a["core_samples"], epochs=a["core_epochs"])
    cfg = FusedTrainConfig(
        num_samples=a["fused_samples"], epochs=a["fused_epochs"], seed=seed, unroll_steps=k
    )
    res = train_fused_batched(train_inst, core, cfg, device="cpu")
    model = res.model
    model.eval()

    out: dict = {
        "seed": seed,
        "k": k,
        "in_r2_mk": float(res.metrics.get("r2_makespan", float("nan"))),
        "per_n": {},
    }
    for nt in a["test_tasks"]:
        ti = build_toy_instance(num_tasks=nt, peak_power=a["peak_power"])
        # build_batch runs the exact simulator to attach mk_true/energy_true labels; its
        # time is the solver cost being amortised. The model forward is the solver-free
        # prediction.
        t_sim = time.perf_counter()
        samples = build_samples(ti, a["eval_samples"], seed=10_000 + seed)
        b = build_batch(model, samples, ti)
        sim_s = time.perf_counter() - t_sim
        with torch.no_grad():
            t_fwd = time.perf_counter()
            _, _, _, mk, en, _ = _forward_batch(model, b, coupled=True)
            fwd_s = time.perf_counter() - t_fwd
        mk = mk.cpu().numpy(); en = en.cpu().numpy()
        mk_t = b.mk_true.cpu().numpy(); en_t = b.energy_true.cpu().numpy()
        out["per_n"][str(nt)] = {
            "r2_mk": _r2(mk, mk_t),
            "r2_e": _r2(en, en_t),
            "mae_mk": float(np.mean(np.abs(mk - mk_t))),
            "sim_s": sim_s,
            "fwd_s": fwd_s,
            "n_eval": len(samples),
        }
    return out


def _ci95(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-tasks", type=int, default=8)
    p.add_argument("--test-tasks", type=int, nargs="+", default=[8, 12, 20, 30])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--peak-power", type=float, default=30.0)
    p.add_argument("--unroll-steps", type=int, default=2, help="K for the unrolled arm (0 arm always run)")
    p.add_argument("--core-samples", type=int, default=1500)
    p.add_argument("--core-epochs", type=int, default=50)
    p.add_argument("--fused-samples", type=int, default=1500)
    p.add_argument("--fused-epochs", type=int, default=60)
    p.add_argument("--eval-samples", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    console = Console()
    ks = sorted({0, args.unroll_steps})
    a = vars(args)
    tasks = [(s, k, a) for k in ks for s in range(args.seeds)]
    console.rule(f"[bold]Unroll size-transfer | train N={args.train_tasks} -> test {args.test_tasks} | "
                 f"coupled(pp={args.peak_power:g}) | K in {ks} | {args.seeds} seeds")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_train_and_transfer, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            tn = args.test_tasks[-1]
            console.print(f"[green]done[/green] K={r['k']} seed={r['seed']} "
                          f"inR2={r['in_r2_mk']:.3f}  transfer N={tn} R2mk={r['per_n'][str(tn)]['r2_mk']:.3f}  "
                          f"[{done}/{len(tasks)}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"config": a, "per_seed": rows}
    tag = args.tag or f"train{args.train_tasks}_pp{args.peak_power:g}"
    out_path = OUT_DIR / f"transfer_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Table: rows = test N, columns = K arms, R2 makespan (mean +/- CI).
    for metric, label, dec in [("r2_mk", "Transfer R2 makespan", 3),
                               ("r2_e", "Transfer R2 energy", 3),
                               ("mae_mk", "Transfer MAE makespan (s)", 3)]:
        tbl = Table(title=f"{label}  (mean \u00b1 95% CI, {args.seeds} seeds)", header_style="bold cyan")
        tbl.add_column("test N", justify="right")
        for k in ks:
            tbl.add_column(f"K={k}", justify="right")
        for nt in args.test_tasks:
            cells = [str(nt)]
            for k in ks:
                vals = [r["per_n"][str(nt)][metric] for r in rows if r["k"] == k]
                m, h = _ci95(vals)
                cells.append(f"{m:.{dec}f} \u00b1 {h:.{dec}f}")
            tbl.add_row(*cells)
        console.print(tbl)

    # Compute-saved: exact-sim wall vs model forward over the eval set (mean over seeds, K arms).
    cs = Table(title="Compute saved (exact-sim vs solver-free forward, eval set)", header_style="bold magenta")
    cs.add_column("test N", justify="right"); cs.add_column("sim s", justify="right")
    cs.add_column("forward s", justify="right"); cs.add_column("speedup", justify="right")
    for nt in args.test_tasks:
        sim = np.mean([r["per_n"][str(nt)]["sim_s"] for r in rows])
        fwd = np.mean([r["per_n"][str(nt)]["fwd_s"] for r in rows])
        cs.add_row(str(nt), f"{sim:.2f}", f"{fwd:.3f}", f"{sim / max(fwd, 1e-9):.0f}x")
    console.print(cs)
    console.print(f"\n[bold green]wrote[/bold green] {out_path}")


if __name__ == "__main__":
    main()
