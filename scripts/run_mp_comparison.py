"""scripts/run_mp_comparison.py -- E-HGATv2-NSGA-II vs mp-BRKGA vs single-pop BRKGA.

Head-to-head at a matched total evaluation budget (the only fair currency, since
mp-BRKGA evaluates (Omega+Pi) populations per generation). For Omega=Pi=2 the multi-
population run processes 4*P individuals per generation; the single-population BRKGA and
the attention-guided search are therefore given pop_size = 4*P to equalise true evaluations
per generation, and all methods use the same number of generations.

Reference front (PF* proxy): exact Oracle for tiny toy instances, otherwise the non-
dominated union of long runs of all three methods (standard EMO practice when the true
front is unknown -- avoids biasing the reference toward any single algorithm).

Usage::

    python scripts/run_mp_comparison.py --instance toy:10 --seeds 5 --gens 60
    python scripts/run_mp_comparison.py --instance L15 --seeds 5 --gens 60 --screening 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import stats as sps

OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "mp_comparison"
Front = tuple[tuple[float, float], ...]


def _load_instance(spec: str):
    from ehgat.environment.dsdl import load_tables_4_5
    from ehgat.environment.instance import build_toy_instance

    if spec.startswith("toy:"):
        return build_toy_instance(num_tasks=int(spec.split(":", 1)[1])), spec
    data = Path(__file__).resolve().parents[1] / "data" / "tables_4_5.json"
    return load_tables_4_5(data, only=[spec])[0].instance, spec


def _pareto(points: list[tuple[float, float]]) -> Front:
    seen: set[tuple[float, float]] = set()
    uniq: list[tuple[float, float]] = []
    for m, e in points:
        k = (round(float(m), 6), round(float(e), 6))
        if k not in seen:
            seen.add(k)
            uniq.append((float(m), float(e)))
    front: list[tuple[float, float]] = []
    best_e: float | None = None
    for m, e in sorted(uniq):
        if best_e is None or e < best_e:
            front.append((m, e))
            best_e = e
    return tuple(front)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="toy:10", help="'toy:N' or an L-id like 'L15'")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--ref-gens", type=int, default=200, help="generations for the PF* proxy runs")
    p.add_argument("--screening", type=int, default=4, help="GAT surrogate screening factor")
    p.add_argument("--surrogate-samples", type=int, default=2000)
    p.add_argument("--surrogate-epochs", type=int, default=100)
    p.add_argument("--mutation-temperature", type=float, default=0.25)
    args = p.parse_args()

    import torch

    torch.set_num_threads(1)

    from ehgat.baselines.brkga import BRKGAConfig, run_brkga
    from ehgat.baselines.mp_brkga import MpBRKGAConfig, run_mp_brkga
    from ehgat.environment.instance import EXACT_TOY_TASKS
    from ehgat.environment.oracle import exact_pareto_front
    from ehgat.metrics import gd_plus, hypervolume, igd_plus, nadir_reference, spread
    from ehgat.search.attention_nsga2 import AttentionNSGA2Config, run_attention_nsga2
    from ehgat.surrogate.train import TrainConfig, train_surrogate

    instance, label = _load_instance(args.instance)
    n = instance.num_tasks
    base_pop = 20 * n           # mp per-population size P (paper)
    matched_pop = 4 * base_pop  # single-pop / GAT pop so total true-evals/gen match mp (Omega+Pi=4)
    G = args.gens

    print(f"instance={label} N={n} QCs={len(instance.qcs)} | mp pop={base_pop}/pop x4 pops, "
          f"single/GAT pop={matched_pop} | gens={G} | matched evals/gen={matched_pop}", flush=True)

    print("training surrogate for GAT ...", flush=True)
    model = train_surrogate(
        instance, TrainConfig(num_samples=args.surrogate_samples,
                              epochs=args.surrogate_epochs, seed=0)).model

    def run_mp(seed: int):
        r = run_mp_brkga(instance, MpBRKGAConfig(pop_size=base_pop, generations=G, seed=seed))
        return r.front, r.evaluations

    def run_sp(seed: int):
        r = run_brkga(instance, BRKGAConfig(pop_size=matched_pop, generations=G, seed=seed))
        return r.front, r.evaluations

    def run_gat(seed: int):
        r = run_attention_nsga2(instance, model, AttentionNSGA2Config(
            matched_pop, G, seed=seed, screening_factor=args.screening,
            mutation_temperature=args.mutation_temperature))
        return r.front, r.evaluations

    methods = {"E-HGATv2-NSGA-II": run_gat, "mp-BRKGA": run_mp, "single-pop BRKGA": run_sp}

    # ---- reference PF* proxy ----
    if label.startswith("toy:") and n <= EXACT_TOY_TASKS:
        reference = tuple((float(m), float(e)) for m, e in exact_pareto_front(instance).front)
        ref_kind = "exact Oracle"
    else:
        pool: list[tuple[float, float]] = []
        rg = args.ref_gens
        pool += list(run_mp_brkga(instance, MpBRKGAConfig(base_pop, rg, seed=1000)).front)
        pool += list(run_brkga(instance, BRKGAConfig(matched_pop, rg, seed=1000)).front)
        pool += list(run_attention_nsga2(instance, model, AttentionNSGA2Config(
            matched_pop, rg, seed=1000, screening_factor=args.screening)).front)
        reference = _pareto(pool)
        ref_kind = f"non-dominated union of 3 methods @ {rg} gens"
    ref_point = nadir_reference(reference, margin=0.1)
    ref_hv = hypervolume(reference, ref_point)
    print(f"reference: {ref_kind} | {len(reference)} pts | HV*={ref_hv:.1f}", flush=True)

    # ---- runs ----
    raw: dict[str, dict[str, list[float]]] = {
        m: {"gd_plus": [], "igd_plus": [], "spread": [], "hv": [], "hv_ratio": [], "evals": []}
        for m in methods}
    t0 = time.perf_counter()
    for name, fn in methods.items():
        for seed in range(args.seeds):
            front, evals = fn(seed)
            raw[name]["gd_plus"].append(gd_plus(front, reference))
            raw[name]["igd_plus"].append(igd_plus(front, reference))
            raw[name]["spread"].append(spread(front, reference))
            hv = hypervolume(front, ref_point)
            raw[name]["hv"].append(hv)
            raw[name]["hv_ratio"].append(hv / ref_hv if ref_hv > 0 else float("nan"))
            raw[name]["evals"].append(float(evals))
        print(f"  {name}: done {args.seeds} seeds", flush=True)

    def ci(vals: list[float]) -> tuple[float, float]:
        a = np.asarray(vals, float)
        if a.size < 2:
            return float(a.mean()), 0.0
        half = float(sps.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))
        return float(a.mean()), half

    metrics = [("hv_ratio", "HV / HV*", 4), ("gd_plus", "GD+", 4),
               ("igd_plus", "IGD+", 4), ("spread", "Spread", 4), ("evals", "true evals", 0)]
    agg = {m: {k: ci(raw[m][k]) for k, *_ in metrics} for m in methods}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = label.replace(":", "")
    (OUT_DIR / f"mp_comparison_{tag}.json").write_text(json.dumps(
        {"instance": label, "n": n, "gens": G, "seeds": args.seeds,
         "matched_pop": matched_pop, "ref_kind": ref_kind, "ref_hv": ref_hv,
         "raw": raw, "aggregate": {m: {k: list(v) for k, v in agg[m].items()} for m in methods}},
        indent=2))

    md = [f"# mp-BRKGA vs E-HGATv2 vs single-pop BRKGA -- {label} (N={n})",
          f"\n_{args.seeds} seeds, {G} gens, matched true-eval budget (mp {base_pop}x4 = "
          f"single/GAT {matched_pop} per gen). Reference: {ref_kind}. Cells = mean (95% CI)._\n",
          "| Method | " + " | ".join(lbl for _, lbl, _ in metrics) + " |",
          "|---|" + "|".join(["---"] * len(metrics)) + "|"]
    for m in methods:
        cells = []
        for k, _lbl, dec in metrics:
            mean, half = agg[m][k]
            cells.append(f"{mean:.{dec}f} ± {half:.{dec}f}" if dec else f"{mean:.0f}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md_text = "\n".join(md) + "\n"
    (OUT_DIR / f"mp_comparison_{tag}.md").write_text(md_text)
    print("\n" + md_text, flush=True)
    print(f"wrote experiments/mp_comparison/mp_comparison_{tag}.* "
          f"(total {time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
