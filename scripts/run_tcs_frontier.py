"""Pareto-front behaviour analysis via TAPE Trade-off Criticality Scores (TCS).

Extracts the explanation knowledge in a form that characterises the Pareto front's
behaviour. Per instance:

1. train the core E-HGATv2 and the physics-fused TAPE head;
2. run the TAPE-guided NSGA-II to obtain the approximated Pareto set (schedules);
3. compute, for every Pareto point, the local trade-off weight lambda from the front
   tangent and the lambda-weighted TAPE attribution (TCS) per task;
4. quantify the migration of the binding-bottleneck structure along the front:
   - lambda range across the front (the trade-off sweep covered),
   - top-bottleneck task at the makespan-optimal vs energy-optimal extreme,
   - bottleneck migration = 1 - Jaccard(top-k tasks at the two extremes),
   - TCS concentration = share of total TCS carried by the top-k tasks (sparsity =>
     interpretable, few tasks explain each point).

Writes experiments/fused_tape_guided/tcs_frontier_<tag>.{json,md} and a combined
tcs_frontier_summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "experiments" / "fused_tape_guided"
TOPK = 3


def _load_instance(spec: str, peak_power: float | None):
    from ehgat.environment.dsdl import load_tables_4_5
    from ehgat.environment.instance import build_toy_instance

    if spec.startswith("toy:"):
        n = int(spec.split(":", 1)[1])
        return build_toy_instance(num_tasks=n, peak_power=peak_power)
    data = Path(__file__).resolve().parents[1] / "data" / "tables_4_5.json"
    return load_tables_4_5(data, peak_power=peak_power, only=[spec])[0].instance


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def analyse(spec: str, peak_power: float | None, args) -> dict:
    from ehgat.explain.fused_explainer import explain_fused_schedules
    from ehgat.explain.tcs_calculator import ParetoPoint, tradeoff_criticality_scores
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.search.attention_nsga2 import AttentionNSGA2Config, run_attention_nsga2

    instance = _load_instance(spec, peak_power)
    n = instance.num_tasks
    coupled = instance.peak_power is not None

    core = build_core(instance, seed=0, num_samples=args.core_samples,
                      epochs=args.core_epochs, device=args.device)
    fused_res = train_fused(instance, core, FusedTrainConfig(
        num_samples=args.fused_samples, epochs=args.fused_epochs,
        unroll_steps=(args.unroll if coupled else 0), seed=0))
    fused = fused_res.model.cpu()

    res = run_attention_nsga2(
        instance, None,
        AttentionNSGA2Config(args.pop, args.gens, seed=0, guidance="tape",
                             screening_factor=args.screening),
        fused_model=fused)
    scheds = list(res.schedules)
    explanations = explain_fused_schedules(fused, scheds, instance)
    points = [ParetoPoint(str(i), ex.makespan, ex.energy, ex)
              for i, ex in enumerate(explanations)]
    rows = tradeoff_criticality_scores(points)  # sorted by makespan ascending

    # Per point: lambda + ranked tasks by tcs_total.
    lambdas = [r["lambda"] for r in rows]
    top_tasks = []
    concentration = []
    for r in rows:
        tasks = sorted(r["tasks"], key=lambda t: t["tcs_total"], reverse=True)
        ranked = [t["task"] for t in tasks]
        top_tasks.append(ranked)
        tot = sum(t["tcs_total"] for t in r["tasks"]) or 1.0
        concentration.append(sum(t["tcs_total"] for t in tasks[:TOPK]) / tot)

    # Front extremes: row 0 = makespan-optimal end, row -1 = energy-optimal end.
    mk_end = set(top_tasks[0][:TOPK]) if top_tasks else set()
    en_end = set(top_tasks[-1][:TOPK]) if top_tasks else set()
    migration = 1.0 - _jaccard(mk_end, en_end)

    result = {
        "instance": spec, "n": n, "coupled": coupled, "peak_power": peak_power,
        "n_front": len(rows),
        "lambda_min": float(min(lambdas)) if lambdas else None,
        "lambda_max": float(max(lambdas)) if lambdas else None,
        "lambda_mean": float(np.mean(lambdas)) if lambdas else None,
        "top_tasks_makespan_end": top_tasks[0][:TOPK] if top_tasks else [],
        "top_tasks_energy_end": top_tasks[-1][:TOPK] if top_tasks else [],
        "bottleneck_migration": migration,
        "tcs_concentration_topk": float(np.mean(concentration)) if concentration else None,
        "topk": TOPK,
        "fused_r2_makespan": float(fused_res.metrics.get("r2_makespan", float("nan"))),
    }
    print(f"  {spec}: front={result['n_front']} lambda[{result['lambda_min']:.2f},"
          f"{result['lambda_max']:.2f}] migration={migration:.2f} "
          f"conc@{TOPK}={result['tcs_concentration_topk']:.2f}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+",
                   default=["toy:5", "toy:8", "toy:10", "L07", "L21"])
    p.add_argument("--peak-power", type=float, default=None)
    p.add_argument("--pop", type=int, default=120)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--screening", type=int, default=3)
    p.add_argument("--unroll", type=int, default=2)
    p.add_argument("--core-samples", type=int, default=1200)
    p.add_argument("--core-epochs", type=int, default=60)
    p.add_argument("--fused-samples", type=int, default=1000)
    p.add_argument("--fused-epochs", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None,
                   help="output filename suffix (default: 'uncoupled'/'coupled'); set a "
                        "unique value per process when running instances in parallel")
    args = p.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.instances:
        print(f"instance {spec} ...", flush=True)
        results.append(analyse(spec, args.peak_power, args))

    tag = args.tag or ("coupled" if args.peak_power is not None else "uncoupled")
    (OUT / f"tcs_frontier_{tag}.json").write_text(json.dumps(results, indent=2))

    md = [f"# Pareto-front behaviour via TAPE TCS ({tag})\n",
          "_Per instance: lambda sweep across the front, the top bottleneck tasks at the "
          "makespan-optimal vs energy-optimal extreme, the bottleneck migration "
          "(1 - Jaccard of top-3 at the two extremes), and the TCS concentration "
          "(share of total criticality on the top-3 tasks)._\n",
          "| Instance | N | front | lambda range | top-3 (Cmax end) | top-3 (E end) | "
          "migration | TCS conc.@3 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(
            f"| {r['instance']} | {r['n']} | {r['n_front']} | "
            f"[{r['lambda_min']:.2f}, {r['lambda_max']:.2f}] | "
            f"{r['top_tasks_makespan_end']} | {r['top_tasks_energy_end']} | "
            f"{r['bottleneck_migration']:.2f} | {r['tcs_concentration_topk']:.2f} |")
    (OUT / f"tcs_frontier_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
