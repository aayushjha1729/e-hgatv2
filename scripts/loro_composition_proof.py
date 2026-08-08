"""Amortization test on a composition-diverse instance set.

The benchmark instances are transport-saturated (transport_frac ~0.88 throughout), which
leaves no variance for the amortized predictor to recover and makes a constant predictor
hard to beat. This script evaluates the predictor on a synthetic set spanning the full
composition spectrum (QC-bound to transport-bound, via num_agvs and num_qcs), where the
constant baseline is weak. Recovery of composition from structural features together with
an improvement over the constant baseline demonstrates amortization.

Leave-one-instance-out. Reports, per held-out instance and in aggregate:
  - model MAE against the constant-mean MAE;
  - between-instance composition recovery: corr(predicted instance-mean, true
    instance-mean) across held-out instances.

Usage:
    python scripts/loro_composition_proof.py --instances toy:30:2:10 ... \
        --cache-dir experiments/front_learning/cache --out .../loro_composition.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from run_front_learning import _evaluate_predictor, _feature_row, _load_cached, _train_predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--cache-dir", default="experiments/front_learning/cache")
    ap.add_argument("--out", default="experiments/front_learning/loro_composition.json")
    args = ap.parse_args()

    cdir = Path(args.cache_dir)
    pool = {}
    for s in args.instances:
        try:
            dps = _load_cached(cdir, s)
            if dps:
                pool[s] = dps
        except FileNotFoundError:
            print(f"  (skip {s}: no cache)")
    print(f"Loaded {len(pool)} instances, {sum(len(v) for v in pool.values())} front points")

    per = {}
    for held in sorted(pool):
        test = pool[held]
        train = [dp for s, dps in pool.items() if s != held for dp in dps]
        model, mu, sd = _train_predictor(train)
        m = _evaluate_predictor(model, mu, sd, test)

        tf_true = np.array([d.transport_frac for d in test])
        # model's predicted transport_frac on held-out points
        X = torch.tensor(np.array([_feature_row(d) for d in test]), dtype=torch.float32)
        with torch.no_grad():
            pred = model((X - mu) / sd).numpy()[:, 0]
        train_mean = float(np.mean([d.transport_frac for d in train]))
        naive_mae = float(np.mean(np.abs(tf_true - train_mean)))

        per[held] = {
            "true_mean": float(tf_true.mean()),
            "pred_mean": float(pred.mean()),
            "model_mae": m["mae_transport_frac"],
            "naive_const_mae": naive_mae,
            "corr_within_front": m["corr_transport"],
            "n": len(test),
        }
        print(f"  [{held:>14}] true={tf_true.mean():.3f} pred={pred.mean():.3f} | "
              f"MAE model={m['mae_transport_frac']:.3f} vs naive={naive_mae:.3f} | "
              f"within-front corr={m['corr_transport']:+.2f}", flush=True)

    true_means = np.array([v["true_mean"] for v in per.values()])
    pred_means = np.array([v["pred_mean"] for v in per.values()])
    model_mae = np.array([v["model_mae"] for v in per.values()])
    naive_mae = np.array([v["naive_const_mae"] for v in per.values()])
    between_corr = float(np.corrcoef(pred_means, true_means)[0, 1])

    agg = {
        "n": len(per),
        "model_mae_mean": float(model_mae.mean()),
        "naive_const_mae_mean": float(naive_mae.mean()),
        "mae_improvement_over_naive": float(naive_mae.mean() - model_mae.mean()),
        "model_beats_naive_count": int(np.sum(model_mae < naive_mae - 0.005)),
        "between_instance_composition_corr": between_corr,
        "true_mean_spread_std": float(true_means.std()),
        "true_mean_range": [float(true_means.min()), float(true_means.max())],
    }
    Path(args.out).write_text(json.dumps({"aggregate": agg, "per_instance": per}, indent=2))

    print("\n=== COMPOSITION-DIVERSE LORO — proof of amortization ===")
    print(f"  composition spread across set: std={agg['true_mean_spread_std']:.3f} "
          f"range=[{agg['true_mean_range'][0]:.2f},{agg['true_mean_range'][1]:.2f}]")
    print(f"  MODEL MAE      = {agg['model_mae_mean']:.3f}")
    print(f"  naive-const MAE= {agg['naive_const_mae_mean']:.3f}  "
          f"(model improvement = {agg['mae_improvement_over_naive']:+.3f})")
    print(f"  model beats naive on {agg['model_beats_naive_count']}/{agg['n']} instances")
    print(f"  BETWEEN-INSTANCE composition recovery corr(pred_mean,true_mean) = "
          f"{between_corr:+.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()