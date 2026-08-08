"""Attention vs TAPE alignment with the exact critical path, at coarse->fine resolution.

In natural-language settings the attention-as-explanation question admits no ground
truth. The scheduling problem is physically grounded: the makespan is an exact max-plus
longest path, so every schedule has a known critical path and a known per-task marginal
makespan-sensitivity. The alignment of the surrogate's attention, and of TAPE's exact
tropical gradient, with that structure is therefore measurable at several granularities:

  L1 coarse  precision@1 : whether the single highest-signal task is on the critical path
  L2 medium  ROC-AUC     : whether the signal ranks on-critical-path tasks above off-path
  L3 fine    Spearman rho: whether the signal ranks tasks by graded makespan-sensitivity
  L4 effect  separation  : mean signal on critical tasks minus mean on off-critical

The predicted pattern is a degradation of attention from L1 to L3, capturing coarse
objective-relevant structure (which tasks matter) but not the exact bottleneck ranking,
with TAPE remaining high at every level. Both are referred to the random baseline
(precision@1 equal to the critical fraction; AUC 0.5).

Usage:
    python scripts/attention_ladder.py --instances toy:5 toy:10 L07 L15 ... \
        --samples 60 --out experiments/fused_tape_guided/attention_ladder.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def _auc(signal: np.ndarray, label: np.ndarray) -> float:
    """ROC-AUC of signal predicting binary label (Mann-Whitney). NaN if degenerate."""
    pos = label > 0.5
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(signal, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(signal) + 1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _load_instance(spec: str):
    from ehgat.environment.dsdl import load_tables_4_5
    from ehgat.environment.instance import build_toy_instance
    if spec.startswith("toy:"):
        return build_toy_instance(num_tasks=int(spec.split(":", 1)[1])), spec
    data = Path(__file__).resolve().parents[1] / "data" / "tables_4_5.json"
    return load_tables_4_5(data, only=[spec])[0].instance, spec


def _ladder_for_instance(spec: str, args) -> dict:
    import numpy as np
    from scipy.stats import spearmanr

    from ehgat.benchmark.faithfulness import (
        attention_per_task,
        critical_path_binding,
        marginal_makespan_speedup,
    )
    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.explain.fused_explainer import explain_fused
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.utils.seeding import make_rng

    instance, label = _load_instance(spec)
    n = instance.num_tasks
    print(f"  [{spec}] training core+fused (N={n}) ...", flush=True)
    core = build_core(instance, seed=0, num_samples=args.core_samples,
                      epochs=args.core_epochs, device="cpu")
    fused = train_fused(instance, core, FusedTrainConfig(
        num_samples=args.fused_samples, epochs=args.fused_epochs, unroll_steps=0, seed=0)
    ).model.cpu()

    rng = make_rng(7)
    rows = {sig: {"p1": [], "auc": [], "rho": [], "sep": []} for sig in ("attn", "tape")}
    base_rates = []
    for _ in range(args.samples):
        sched = decode(rng.random(NUM_BLOCKS * n), instance)
        agv_bound, qc_bound = critical_path_binding(sched, instance)
        crit = agv_bound | qc_bound
        y = np.zeros(n)
        for j in crit:
            y[j] = 1.0
        if y.sum() == 0 or y.sum() == n:
            continue
        base_rates.append(float(y.mean()))
        g = marginal_makespan_speedup(sched, instance)  # graded exact criticality

        attn = attention_per_task(sched, instance, core)
        ex = explain_fused(fused, sched, instance)
        tape = np.maximum.reduce([
            np.asarray(ex.node_grad), np.asarray(ex.empty_time_grad),
            np.asarray(ex.loaded_time_grad)])  # per-task exact tropical criticality

        for sig_name, s in (("attn", attn), ("tape", tape)):
            rows[sig_name]["p1"].append(float(y[int(np.argmax(s))]))
            auc = _auc(s, y)
            if not np.isnan(auc):
                rows[sig_name]["auc"].append(auc)
            if np.std(s) > 0 and np.std(g) > 0:
                rho = spearmanr(s, g).correlation
                if not np.isnan(rho):
                    rows[sig_name]["rho"].append(float(rho))
            sep = (s[y > 0.5].mean() - s[y < 0.5].mean()) if (s.std() > 0) else 0.0
            # normalise separation by signal scale for cross-instance comparability
            scale = s.std() if s.std() > 0 else 1.0
            rows[sig_name]["sep"].append(float(sep / scale))

    def agg(sig):
        d = rows[sig]
        return {
            "precision_at_1": float(np.mean(d["p1"])) if d["p1"] else float("nan"),
            "auc_on_path": float(np.nanmean(d["auc"])) if d["auc"] else float("nan"),
            "spearman_graded": float(np.mean(d["rho"])) if d["rho"] else float("nan"),
            "separation_z": float(np.mean(d["sep"])) if d["sep"] else float("nan"),
            "n": len(d["p1"]),
        }

    base = float(np.mean(base_rates)) if base_rates else float("nan")
    out = {"instance": label, "n_tasks": n, "critical_fraction_baseline": base,
           "attention": agg("attn"), "tape": agg("tape")}
    a, t = out["attention"], out["tape"]
    print(f"  [{spec}] attn: p@1={a['precision_at_1']:.2f} auc={a['auc_on_path']:.2f} "
          f"rho={a['spearman_graded']:+.2f} | tape: p@1={t['precision_at_1']:.2f} "
          f"auc={t['auc_on_path']:.2f} rho={t['spearman_graded']:+.2f} | base p@1={base:.2f}",
          flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--core-samples", type=int, default=800)
    ap.add_argument("--core-epochs", type=int, default=50)
    ap.add_argument("--fused-samples", type=int, default=700)
    ap.add_argument("--fused-epochs", type=int, default=50)
    ap.add_argument("--out", default="experiments/fused_tape_guided/attention_ladder.json")
    args = ap.parse_args()

    per = [_ladder_for_instance(s, args) for s in args.instances]

    def pool(path):
        vals = [d[path[0]][path[1]] for d in per
                if not np.isnan(d[path[0]][path[1]])]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {
        "n_instances": len(per),
        "attention": {k: pool(("attention", k)) for k in
                      ("precision_at_1", "auc_on_path", "spearman_graded", "separation_z")},
        "tape": {k: pool(("tape", k)) for k in
                 ("precision_at_1", "auc_on_path", "spearman_graded", "separation_z")},
        "critical_fraction_baseline": float(np.nanmean([d["critical_fraction_baseline"] for d in per])),
    }
    Path(args.out).write_text(json.dumps({"aggregate": agg, "per_instance": per}, indent=2))
    a, t = agg["attention"], agg["tape"]
    print("\n=== GRANULARITY LADDER (pooled) ===")
    print(f"  level                | attention | TAPE | random")
    print(f"  L1 precision@1       |   {a['precision_at_1']:.2f}    | {t['precision_at_1']:.2f} | {agg['critical_fraction_baseline']:.2f}")
    print(f"  L2 AUC (on-path)     |   {a['auc_on_path']:.2f}    | {t['auc_on_path']:.2f} | 0.50")
    print(f"  L3 Spearman (graded) |  {a['spearman_graded']:+.2f}    | {t['spearman_graded']:+.2f} | 0.00")
    print(f"  L4 separation (z)    |  {a['separation_z']:+.2f}    | {t['separation_z']:+.2f} | 0.00")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()