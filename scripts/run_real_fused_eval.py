"""scripts/run_real_fused_eval.py -- fused-TAPE evaluation on the published instances.

Runs the physics-fused E-HGATv2 (max-plus DP makespan head + additive energy head) on the
published Homayouni & Fontes (2022) container-terminal instances in data/tables_4_5.json
(Table 4 geometry + Table 5 L01-L35 loading task sets). Per (instance, seed) it trains the
frozen core, fine-tunes the anchored fused head, and measures held-out calibration
(R2/MAE) and faithfulness vs the exact TAPE oracle (leg/arc critical-path Jaccard).

Results are aggregated per instance to mean +/- 95% Student-t CI over seeds and written as
JSON + Markdown to experiments/fused_eval/.

Usage::

    python scripts/run_real_fused_eval.py --seeds 3 --workers 6
    python scripts/run_real_fused_eval.py --instances L07 L15 L29 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import stats as sps

DATA = Path(__file__).resolve().parents[1] / "data" / "tables_4_5.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "fused_eval"

_METRICS = (
    ("r2_makespan", "R2 makespan", 4),
    ("r2_energy", "R2 energy", 4),
    ("mae_makespan", "MAE makespan (s)", 3),
    ("mae_energy", "MAE energy", 4),
    ("leg_jaccard", "Leg critical Jaccard", 4),
    ("arc_jaccard", "Arc critical Jaccard", 4),
    ("wall_s", "Train wall (s)", 1),
)


def _run_one(inst_id: str, seed: int, cfg: dict) -> dict:
    nthr = int(os.environ.get("THREADS_PER_WORKER", "1"))
    os.environ["OMP_NUM_THREADS"] = str(nthr)
    os.environ["MKL_NUM_THREADS"] = str(nthr)
    import torch

    torch.set_num_threads(nthr)

    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.environment.dsdl import load_tables_4_5
    from ehgat.explain.fused_explainer import faithfulness_report
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.explain.train_fused_batched import train_fused_batched
    from ehgat.utils.seeding import make_rng

    t0 = time.perf_counter()
    inst = load_tables_4_5(DATA, only=[inst_id])[0].instance
    core = build_core(inst, seed=seed, num_samples=cfg["core_samples"],
                      epochs=cfg["core_epochs"], batch_size=cfg["core_batch"],
                      lr=cfg["core_lr"], device="cpu")
    fcfg = FusedTrainConfig(num_samples=cfg["fused_samples"], epochs=cfg["fused_epochs"],
                            batch_size=cfg["fused_batch"], lr=cfg["fused_lr"], seed=seed)
    try:
        result = train_fused_batched(inst, core, fcfg, device="cpu")
    except Exception:
        result = train_fused(inst, core, fcfg)
    rng = make_rng(seed + 7)
    schedules = [decode(rng.random(NUM_BLOCKS * inst.num_tasks), inst)
                 for _ in range(cfg["explain_samples"])]
    reps = [faithfulness_report(result.model, s, inst) for s in schedules]
    n = len(reps)
    m = result.metrics
    return {
        "instance": inst_id,
        "num_tasks": inst.num_tasks,
        "num_qcs": len(inst.qcs),
        "seed": seed,
        "r2_makespan": float(m.get("r2_makespan", float("nan"))),
        "r2_energy": float(m.get("r2_energy", float("nan"))),
        "mae_makespan": float(m.get("mae_makespan", float("nan"))),
        "mae_energy": float(m.get("mae_energy", float("nan"))),
        "leg_jaccard": sum(r.leg_critical_jaccard for r in reps) / n,
        "arc_jaccard": sum(r.arc_critical_jaccard for r in reps) / n,
        "wall_s": time.perf_counter() - t0,
    }


def _ci95(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    k = arr.size
    if k == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    mean = float(arr.mean())
    if k == 1:
        return {"mean": mean, "ci_lo": mean, "ci_hi": mean}
    sd = float(arr.std(ddof=1))
    half = float(sps.t.ppf(0.975, k - 1) * sd / np.sqrt(k))
    return {"mean": mean, "ci_lo": mean - half, "ci_hi": mean + half}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=None, help="default = all L01..L35")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--core-samples", type=int, default=2000)
    p.add_argument("--core-epochs", type=int, default=80)
    p.add_argument("--core-batch", type=int, default=64)
    p.add_argument("--core-lr", type=float, default=2e-3)
    p.add_argument("--fused-samples", type=int, default=1200)
    p.add_argument("--fused-epochs", type=int, default=40)
    p.add_argument("--fused-batch", type=int, default=64)
    p.add_argument("--fused-lr", type=float, default=2e-3)
    p.add_argument("--explain-samples", type=int, default=128)
    p.add_argument("--tag", type=str, default="real_Lset")
    args = p.parse_args()

    payload = json.loads(DATA.read_text())
    all_ids = list(payload["table5_loading_instances"]["instances"].keys())
    inst_ids = args.instances or all_ids
    cfg = {k: getattr(args, k) for k in (
        "core_samples", "core_epochs", "core_batch", "core_lr",
        "fused_samples", "fused_epochs", "fused_batch", "fused_lr", "explain_samples")}

    jobs = [(iid, s) for iid in inst_ids for s in range(args.seeds)]
    print(f"real fused eval | {len(inst_ids)} instances x {args.seeds} seeds = {len(jobs)} runs "
          f"| {args.workers} workers", flush=True)

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, iid, s, cfg): (iid, s) for iid, s in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            print(f"done {r['instance']} seed={r['seed']} N={r['num_tasks']} "
                  f"R2mk={r['r2_makespan']:.3f} legJ={r['leg_jaccard']:.3f} "
                  f"({r['wall_s']:.0f}s) [{done}/{len(jobs)}]", flush=True)

    agg: dict[str, dict] = {}
    for iid in inst_ids:
        agg[iid] = {"num_tasks": next(r["num_tasks"] for r in rows if r["instance"] == iid),
                    "num_qcs": next(r["num_qcs"] for r in rows if r["instance"] == iid)}
        for key, *_ in _METRICS:
            agg[iid][key] = _ci95([r[key] for r in rows if r["instance"] == iid])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fused_eval_{args.tag}.json"
    out.write_text(json.dumps({"config": cfg, "seeds": args.seeds,
                               "per_seed": rows, "aggregate": agg}, indent=2))

    md = [f"# Fused E-HGATv2 on REAL instances ({args.tag})",
          f"\n_{args.seeds} seeds/instance; cells = mean (95% CI). Data: data/tables_4_5.json "
          f"(real Table 4 geometry + Table 5 L01-L35)._\n",
          "| Instance | N | QCs | " + " | ".join(lbl for _, lbl, _ in _METRICS) + " |",
          "|---|---|---|" + "|".join(["---"] * len(_METRICS)) + "|"]
    for iid in inst_ids:
        a = agg[iid]
        cells = []
        for key, _lbl, dec in _METRICS:
            c = a[key]
            cells.append(f"{c['mean']:.{dec}f}")
        md.append(f"| {iid} | {a['num_tasks']} | {a['num_qcs']} | " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out} and {out.with_suffix('.md')}  (total {time.perf_counter()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
