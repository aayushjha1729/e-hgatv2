"""Train-small / evaluate-large size-generalization study.

The surrogate is inductive (shared-weight message passing): the core and physics-fused
head are trained once on a small instance and inference is then run on a ladder of much
larger instances, with no retraining at large N. Under test are size generalization of the
surrogate and the persistence of the attention/TAPE faithfulness picture.

Per eval-N (mean over K random schedules, then over training seeds), reported:

- core_r2_makespan  -- frozen black-box global-readout head. A global readout cannot
  extrapolate makespan magnitude as C_max grows with N.
- fused_r2_makespan / fused_r2_energy -- physics-fused head. It emits per-leg times, which
  are local and size-invariant, and the tropical max-plus DP aggregates them into a
  correct-magnitude makespan at any N; energy is exactly additive.
- tape_leg_jaccard -- fused critical path against the exact O(N) TAPE oracle, near 1 by
  construction.
- attn_rho / attn_prec1 -- attention against the true per-task makespan sensitivity,
  measuring the dissociation between search guidance and faithfulness.

Cost note: everything is O(N) per schedule EXCEPT the attention ground truth
marginal_makespan_speedup (N leave-one-out evals, hence O(N^2)). Beyond --rho-max-tasks
that many task indices are subsampled for the rho computation, keeping the attention metric
feasible at N>=5000; the subsample flag is recorded per row.

Usage::

    python scripts/run_scaling_generalization.py \\
        --train-n 16 --train-seeds 0 1 2 \\
        --eval-tasks 16 25 50 100 250 500 1000 2000 5000 10000 \\
        --eval-schedules 24 --rho-max-tasks 256 --device cuda \\
        > experiments/scaling/generalization.log

Writes experiments/scaling/generalization{tag}.json (per-(N,seed) rows + per-N means).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "scaling"


def _build_instance(n: int, peak_power):
    from ehgat.environment.instance import build_toy_instance, scaled_fleet

    num_agvs, num_qcs = scaled_fleet(n)
    qcs = tuple(f"QC{i + 1}" for i in range(num_qcs))
    return build_toy_instance(num_tasks=n, qcs=qcs, num_agvs=num_agvs, peak_power=peak_power)


def _random_schedules(instance, k: int, seed: int):
    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.utils.seeding import make_rng

    rng = make_rng(seed)
    n = instance.num_tasks
    return [decode(rng.random(NUM_BLOCKS * n), instance) for _ in range(k)]


def _subsampled_speedup(schedule, instance, idx):
    """Leave-one-fastest makespan speedup for the task indices in idx only (O(|idx|*N))."""
    from ehgat.benchmark.faithfulness import _FASTEST
    from ehgat.environment.evaluator import evaluate

    base = evaluate(schedule, instance).makespan
    empty = list(schedule.empty_speed)
    loaded = list(schedule.loaded_speed)
    out = []
    for j in idx:
        e0, l0 = empty[j], loaded[j]
        empty[j] = loaded[j] = _FASTEST
        faster = replace(schedule, empty_speed=tuple(empty), loaded_speed=tuple(loaded))
        out.append(base - evaluate(faster, instance).makespan)
        empty[j], loaded[j] = e0, l0
    return out


def _attention_faithfulness(schedules, instance, core, *, rho_max_tasks: int, seed: int):
    """Attention precision@1 + Spearman rho vs the true per-task makespan sensitivity."""
    import numpy as np
    from scipy.stats import spearmanr

    from ehgat.benchmark.faithfulness import (
        attention_per_task,
        critical_agv_arcs,
        marginal_makespan_speedup,
    )
    from ehgat.utils.seeding import make_rng

    n = instance.num_tasks
    subsample = n > rho_max_tasks
    if subsample:
        rng = make_rng(seed + 31)
        idx = np.sort(rng.choice(n, size=rho_max_tasks, replace=False))
    else:
        idx = np.arange(n)

    hits, rhos = 0, []
    for sched in schedules:
        alpha = attention_per_task(sched, instance, core)
        critical = critical_agv_arcs(sched, instance)
        if int(np.argmax(alpha)) in critical:
            hits += 1
        if subsample:
            speedup = np.asarray(_subsampled_speedup(sched, instance, idx), dtype=float)
            a_sub = alpha[idx]
        else:
            speedup = np.asarray(marginal_makespan_speedup(sched, instance), dtype=float)
            a_sub = alpha
        if np.std(a_sub) > 0 and np.std(speedup) > 0:
            rho = spearmanr(a_sub, speedup).correlation
            if not np.isnan(rho):
                rhos.append(float(rho))

    prec1 = hits / len(schedules)
    rho_mean = float(np.mean(rhos)) if rhos else float("nan")
    return prec1, rho_mean, bool(subsample)


def _eval_at_n(core, fused, n, peak_power, k, seed, rho_max_tasks):
    """Inference-only evaluation of the small-trained surrogate at scale n."""
    import torch

    from ehgat.environment.evaluator import evaluate
    from ehgat.explain.fused_explainer import faithfulness_report
    from ehgat.surrogate.graph import build_hetero_graph
    from ehgat.surrogate.train import regression_metrics

    instance = _build_instance(n, peak_power)
    scheds = _random_schedules(instance, k, seed + 7)

    core.eval()
    fused.eval()
    core_pred, fused_pred, truth = [], [], []
    with torch.no_grad():
        for s in scheds:
            g = build_hetero_graph(s, instance)
            ev = evaluate(s, instance)
            out, _ = core(g)
            core_pred.append((out * core.target_std + core.target_mean)[0].detach())
            fp = fused(g)
            fused_pred.append(torch.stack([fp.makespan, fp.energy]).detach())
            truth.append(torch.tensor([ev.makespan, ev.energy]))
    truth_t = torch.stack(truth)
    core_m = regression_metrics(torch.stack(core_pred), truth_t)
    fused_m = regression_metrics(torch.stack(fused_pred), truth_t)

    reps = [faithfulness_report(fused, s, instance) for s in scheds]
    prec1, rho, subsampled = _attention_faithfulness(
        scheds, instance, core, rho_max_tasks=rho_max_tasks, seed=seed
    )
    return {
        "num_tasks": n,
        "seed": seed,
        "core_r2_makespan": round(core_m["r2_makespan"], 4),
        "fused_r2_makespan": round(fused_m["r2_makespan"], 4),
        "fused_r2_energy": round(fused_m["r2_energy"], 4),
        "tape_leg_jaccard": round(st.mean(r.leg_critical_jaccard for r in reps), 4),
        "fused_cmax_abs_err": round(st.mean(r.makespan_abs_error for r in reps), 4),
        "attn_prec1": round(prec1, 4),
        "attn_rho": (round(rho, 4) if rho == rho else None),  # None when NaN
        "attn_rho_subsampled": subsampled,
    }


def _train_once(train_n, seed, peak_power, core_samples, core_epochs, fused_samples,
                fused_epochs, device, unroll):
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused

    instance = _build_instance(train_n, peak_power)
    core = build_core(instance, seed=seed, num_samples=core_samples,
                      epochs=core_epochs, device=device)
    res = train_fused(instance, core, FusedTrainConfig(
        num_samples=fused_samples, epochs=fused_epochs,
        unroll_steps=(unroll if peak_power is not None else 0), seed=seed))
    return core.cpu(), res.model.cpu()


def _train_curriculum(train_ns, seed, peak_power, core_samples, core_epochs, fused_samples,
                      fused_epochs, device, unroll):
    """Train once on a pooled set of samples drawn across several instance sizes.

    Addresses the single-N head's inability to observe how the coupled power-contention
    density grows with N. Pooling samples across train_ns exposes a range of contention
    regimes, under which the
    fused head can interpolate/extrapolate the per-leg power-wait it must predict at larger N.
    core_samples/fused_samples are the TOTAL pooled budgets, split evenly across sizes.
    """
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, build_samples, train_fused
    from ehgat.surrogate.dataset import generate_graphs

    per_core = max(1, core_samples // len(train_ns))
    per_fused = max(1, fused_samples // len(train_ns))
    core_graphs, fused_pool, ref = [], [], None
    for k, tn in enumerate(train_ns):
        inst = _build_instance(tn, peak_power)
        if ref is None:
            ref = inst
        core_graphs += generate_graphs(inst, per_core, seed=seed * 100 + k)
        fused_pool += build_samples(inst, per_fused, seed=seed * 100 + k)
    core = build_core(ref, seed=seed, epochs=core_epochs, device=device, graphs=core_graphs)
    res = train_fused(ref, core, FusedTrainConfig(
        epochs=fused_epochs, unroll_steps=(unroll if peak_power is not None else 0),
        seed=seed), samples=fused_pool)
    return core.cpu(), res.model.cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train-small / eval-large generalization study.")
    ap.add_argument("--train-n", type=int, default=16, help="instance size the surrogate is fit on")
    ap.add_argument("--train-ns", type=int, nargs="+", default=None,
                    help="if set, train on a POOLED range of sizes (size-generalisation curriculum) "
                         "instead of a single --train-n; e.g. --train-ns 12 20 28 40")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-tasks", type=int, nargs="+",
                    default=[16, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--eval-schedules", type=int, default=24)
    ap.add_argument("--rho-max-tasks", type=int, default=256,
                    help="cap tasks used for the O(N^2) attention-rho ground truth")
    ap.add_argument("--peak-power", type=float, default=None,
                    help="None = uncoupled (clean regime); a kW budget = coupled")
    ap.add_argument("--unroll", type=int, default=2,
                    help="coupled fused unroll steps (ignored when uncoupled)")
    ap.add_argument("--core-samples", type=int, default=2400)
    ap.add_argument("--core-epochs", type=int, default=60)
    ap.add_argument("--fused-samples", type=int, default=1600)
    ap.add_argument("--fused-epochs", type=int, default=60)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print(f"[gen] train-n={args.train_n} seeds={args.train_seeds} "
          f"ladder={args.eval_tasks} K={args.eval_schedules} "
          f"peak_power={args.peak_power} rho_max_tasks={args.rho_max_tasks}", flush=True)

    rows = []
    for seed in args.train_seeds:
        if args.train_ns:
            print(f"[gen] training surrogate on POOLED sizes {args.train_ns} seed={seed} ...", flush=True)
            core, fused = _train_curriculum(
                args.train_ns, seed, args.peak_power, args.core_samples, args.core_epochs,
                args.fused_samples, args.fused_epochs, args.device, args.unroll)
        else:
            print(f"[gen] training surrogate at N={args.train_n} seed={seed} ...", flush=True)
            core, fused = _train_once(
                args.train_n, seed, args.peak_power, args.core_samples, args.core_epochs,
                args.fused_samples, args.fused_epochs, args.device, args.unroll)
        for n in args.eval_tasks:
            row = _eval_at_n(core, fused, n, args.peak_power, args.eval_schedules,
                             seed, args.rho_max_tasks)
            rows.append(row)
            print(f"[gen] seed={seed} N={n:6d} core_r2={row['core_r2_makespan']:+.3f} "
                  f"fused_r2={row['fused_r2_makespan']:+.3f} "
                  f"jac={row['tape_leg_jaccard']:.3f} rho={row['attn_rho']} "
                  f"prec1={row['attn_prec1']:.3f}", flush=True)

    per_n = {}
    for n in args.eval_tasks:
        sub = [r for r in rows if r["num_tasks"] == n]
        rho_vals = [r["attn_rho"] for r in sub if r["attn_rho"] is not None]
        per_n[n] = {
            "core_r2_makespan": round(st.mean(r["core_r2_makespan"] for r in sub), 4),
            "fused_r2_makespan": round(st.mean(r["fused_r2_makespan"] for r in sub), 4),
            "fused_r2_energy": round(st.mean(r["fused_r2_energy"] for r in sub), 4),
            "tape_leg_jaccard": round(st.mean(r["tape_leg_jaccard"] for r in sub), 4),
            "attn_prec1": round(st.mean(r["attn_prec1"] for r in sub), 4),
            "attn_rho": (round(st.mean(rho_vals), 4) if rho_vals else None),
            "attn_rho_subsampled": any(r["attn_rho_subsampled"] for r in sub),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"_pp{args.peak_power:g}" if args.peak_power is not None else "_uncoupled")
    if args.train_ns:
        tag += "_curriculum"
    out = OUT_DIR / f"generalization{tag}.json"
    out.write_text(json.dumps({
        "train_n": args.train_n, "train_seeds": args.train_seeds,
        "eval_tasks": args.eval_tasks, "eval_schedules": args.eval_schedules,
        "peak_power": args.peak_power, "rho_max_tasks": args.rho_max_tasks,
        "rows": rows, "per_n_mean": per_n,
    }, indent=2))

    print("\n[gen] per-N means (trained once at N="
          f"{args.train_n}, evaluated at scale):", flush=True)
    print(f"{'N':>6} {'core_r2':>9} {'fused_r2':>9} {'energy_r2':>10} "
          f"{'tape_jac':>9} {'attn_rho':>9} {'prec@1':>7}", flush=True)
    for n in args.eval_tasks:
        m = per_n[n]
        rho_s = f"{m['attn_rho']:+.3f}" if m["attn_rho"] is not None else "   n/a"
        print(f"{n:>6} {m['core_r2_makespan']:>+9.3f} {m['fused_r2_makespan']:>+9.3f} "
              f"{m['fused_r2_energy']:>+10.3f} {m['tape_leg_jaccard']:>9.3f} "
              f"{rho_s:>9} {m['attn_prec1']:>7.3f}", flush=True)
    print(f"[gen] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
