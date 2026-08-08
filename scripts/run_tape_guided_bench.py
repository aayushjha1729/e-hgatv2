"""scripts/run_tape_guided_bench.py -- faithful explanations as a search signal.

The experiment joins the explanation and the guidance (optimization) results into a single
evidenced claim. Per instance:

1. trains the core E-HGATv2 + the physics-fused TAPE head (train_fused);
2. runs NSGA-II under three guidance signals at a matched exact-evaluation budget:
   - E-HGATv2-TAPE  -- fused-model TAPE attribution (faithful by construction) drives
     both the bottleneck task selection AND the offspring screening;
   - E-HGATv2-attn  -- the bare HAN attention readout (the unfaithful comparison);
   - NSGA-II (random) -- no guidance (the null);
   plus the published baselines mp-BRKGA (multi-population) and single-pop BRKGA;
3. measures the faithfulness of each guidance signal on a fixed schedule sample:
   attention precision@1 / Spearman (expected ~random) vs TAPE leg-critical Jaccard vs the
   exact max-plus oracle (expected ~1.0).

TAPE is the only signal that is simultaneously faithful and competitive in the search.
Attention can assist the search while remaining unfaithful, and therefore does not support
the result that the explanation steers the search.

Budget matching: mp-BRKGA evaluates (Omega+Pi)*P = 4P chromosomes per generation, and the
GAT and BRKGA arms accordingly use pop = 4P to equalise exact evaluations per generation,
all at the same --gens. The surrogate screening (screening_factor) screens k*pop candidates
at inference cost while still spending only pop exact evaluations per generation.

Usage::

    python scripts/run_tape_guided_bench.py --instance toy:10 --seeds 5 --gens 60
    python scripts/run_tape_guided_bench.py --instance toy:10 --peak-power 30 --seeds 5 --gens 60
    python scripts/run_tape_guided_bench.py --instance L15 --seeds 5 --gens 60

Writes experiments/fused_tape_guided/tape_bench_<tag>.{json,md}.
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

OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "fused_tape_guided"
Front = tuple[tuple[float, float], ...]


def _load_instance(spec: str, peak_power: float | None):
    """Return (instance, label). spec is 'toy:N' or an L-id like 'L15'."""
    from ehgat.environment.dsdl import load_tables_4_5
    from ehgat.environment.instance import build_toy_instance

    if spec.startswith("toy:"):
        n = int(spec.split(":", 1)[1])
        return build_toy_instance(num_tasks=n, peak_power=peak_power), spec
    data = Path(__file__).resolve().parents[1] / "data" / "tables_4_5.json"
    return load_tables_4_5(data, peak_power=peak_power, only=[spec])[0].instance, spec


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


def _ci(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, float)
    if a.size < 2:
        return float(a.mean()) if a.size else float("nan"), 0.0
    half = float(sps.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))
    return float(a.mean()), half


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="toy:10", help="'toy:N' or an L-id like 'L15'")
    p.add_argument("--peak-power", type=float, default=None, help="kW budget => coupled regime")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0,
                   help="first seed index (for sharding seeds across parallel processes)")
    p.add_argument("--out-tag", default=None,
                   help="override output filename tag (default derived from instance)")
    p.add_argument("--out-dir", default=None,
                   help="override output directory (e.g. a shards/ subdir)")
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--ref-gens", type=int, default=200, help="generations for the PF* proxy runs")
    p.add_argument("--screening", type=int, default=4, help="surrogate screening factor for GAT arms")
    p.add_argument("--mp-screen", action="store_true",
                   help="add a GNN-screened mp-BRKGA ablation arm (same surrogate + screening_factor "
                        "as the guided arms) -- tests whether the GNN enhances the mp-BRKGA backbone too")
    p.add_argument("--p-mult", type=int, default=20, help="mp-BRKGA per-population size P = p_mult*N "
                   "(paper: 20; lower for compute tractability -- all methods stay budget-matched)")
    p.add_argument("--base-pop", type=int, default=None,
                   help="fixed mp per-population size P independent of N (overrides p_mult*N); "
                        "use for a fixed-budget scaling sweep so per-instance cost is O(N)")
    p.add_argument("--unroll", type=int, default=2, help="coupled fused unroll steps")
    p.add_argument("--core-samples", type=int, default=2000)
    p.add_argument("--core-epochs", type=int, default=80)
    p.add_argument("--fused-samples", type=int, default=1500)
    p.add_argument("--fused-epochs", type=int, default=80)
    p.add_argument("--faith-samples", type=int, default=40)
    p.add_argument("--mutation-temperature", type=float, default=0.25)
    p.add_argument("--device", default="cpu", help="training device (cuda|cpu)")
    p.add_argument("--search-device", default="cpu",
                   help="device for the surrogate DURING search (cuda batches the guided arms' "
                        "GNN screening onto the GPU; exact-eval arms stay on CPU regardless)")
    args = p.parse_args()

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    from ehgat.baselines.brkga import BRKGAConfig, run_brkga
    from ehgat.baselines.mp_brkga import MpBRKGAConfig, run_mp_brkga
    from ehgat.benchmark.faithfulness import evaluate_faithfulness
    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.environment.instance import EXACT_TOY_TASKS
    from ehgat.environment.oracle import exact_pareto_front
    from ehgat.explain.fused_explainer import faithfulness_report
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.metrics import gd_plus, hypervolume, igd_plus, nadir_reference, spread
    from ehgat.search.attention_nsga2 import (
        AttentionNSGA2Config, _predict_objectives, run_attention_nsga2)
    from ehgat.utils.seeding import make_rng

    instance, label = _load_instance(args.instance, args.peak_power)
    n = instance.num_tasks
    coupled = instance.peak_power is not None
    # base_pop = mp per-population size P. Default P = p_mult*N (paper: 20*N). --base-pop fixes
    # P independent of N for a FIXED-budget scaling study (so cost is O(N), not O(N^2), and the
    # comparison reveals which methods stall as N grows under a constant evaluation budget).
    base_pop = args.base_pop if args.base_pop else args.p_mult * n
    matched_pop = 4 * base_pop   # GAT/BRKGA pop so exact-evals/gen match mp (Omega+Pi=4)
    G = args.gens
    tag = args.out_tag or (label.replace(":", "") + ("_pp%g" % args.peak_power if coupled else "_unc"))
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    print(f"instance={label} N={n} coupled={coupled} | mp P={base_pop}x4, GAT/BRKGA pop={matched_pop} "
          f"| gens={G} | matched evals/gen={matched_pop}", flush=True)

    # ---- train core + fused TAPE head (the engine behind the TAPE arm) ----
    print("training core + fused TAPE head ...", flush=True)
    core = build_core(instance, seed=0, num_samples=args.core_samples,
                      epochs=args.core_epochs, device=args.device)
    fused_res = train_fused(instance, core, FusedTrainConfig(
        num_samples=args.fused_samples, epochs=args.fused_epochs,
        unroll_steps=(args.unroll if coupled else 0), seed=0))
    # Keep the surrogate on the search device: the guided arms' offspring screening and
    # attention/TAPE signals are single batched forwards on the model's device, so CUDA moves
    # that cost to the GPU. The exact-eval arms (mp/random/sp) are CPU-only regardless.
    fused = fused_res.model.to(args.search_device)
    core = core.to(args.search_device)
    print(f"  fused R2_makespan={fused_res.metrics.get('r2_makespan'):.4f} "
          f"R2_energy={fused_res.metrics.get('r2_energy'):.4f}", flush=True)

    # ---- arms (all at matched exact-eval budget) ----
    def run_tape(seed: int):
        r = run_attention_nsga2(instance, None, AttentionNSGA2Config(
            matched_pop, G, seed=seed, guidance="tape", screening_factor=args.screening,
            mutation_temperature=args.mutation_temperature), fused_model=fused)
        return r.front, r.evaluations

    def run_attn(seed: int):
        r = run_attention_nsga2(instance, core, AttentionNSGA2Config(
            matched_pop, G, seed=seed, guidance="attention", screening_factor=args.screening,
            mutation_temperature=args.mutation_temperature))
        return r.front, r.evaluations

    def run_rand(seed: int):
        r = run_attention_nsga2(instance, core, AttentionNSGA2Config(
            matched_pop, G, seed=seed, random_mutation=True, screening_factor=1))
        return r.front, r.evaluations

    def run_mp(seed: int):
        r = run_mp_brkga(instance, MpBRKGAConfig(pop_size=base_pop, generations=G, seed=seed))
        return r.front, r.evaluations

    def run_sp(seed: int):
        r = run_brkga(instance, BRKGAConfig(pop_size=matched_pop, generations=G, seed=seed))
        return r.front, r.evaluations

    def run_mp_screen(seed: int):
        # Same GNN surrogate + screening_factor as the guided arms, ported to the mp-BRKGA
        # backbone: over-produce k*No offspring per population, surrogate-rank, keep the best
        # No. Budget-neutral in exact evals (only the kept P are exact-evaluated next gen).
        def screen(chroms):
            return _predict_objectives([decode(c, instance) for c in chroms], instance, core)
        r = run_mp_brkga(instance, MpBRKGAConfig(
            pop_size=base_pop, generations=G, seed=seed, screening_factor=args.screening),
            screen_fn=screen)
        return r.front, r.evaluations

    methods = {
        "E-HGATv2-TAPE": run_tape,
        "E-HGATv2-attn": run_attn,
        "NSGA-II (random)": run_rand,
        "mp-BRKGA": run_mp,
        "single-pop BRKGA": run_sp,
    }
    if args.mp_screen:
        methods["mp-BRKGA+GNN-screen"] = run_mp_screen

    # ---- optimisation runs (collect every front FIRST so PF* can dominate them) ----
    t0 = time.perf_counter()
    fronts: dict[str, list[tuple[tuple[float, float], ...]]] = {m: [] for m in methods}
    evals_by: dict[str, list[float]] = {m: [] for m in methods}
    for name, fn in methods.items():
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            front, evals = fn(seed)
            fronts[name].append(tuple((float(m), float(e)) for m, e in front))
            evals_by[name].append(float(evals))
        print(f"  {name}: done {args.seeds} seeds", flush=True)

    # ---- reference PF* ----
    # Exact oracle where enumerable; otherwise the non-dominated union of a high-budget
    # reference pool AND every evaluated front. Folding in the evaluated fronts guarantees
    # PF* weakly dominates all of them, so no method's HV can exceed HV* (no hv_ratio > 1)
    # by construction -- a residual ratio > 1 would then signal a real numerical bug.
    if label.startswith("toy:") and n <= EXACT_TOY_TASKS and not coupled:
        reference = tuple((float(m), float(e)) for m, e in exact_pareto_front(instance).front)
        ref_kind = "exact Oracle"
    else:
        pool: list[tuple[float, float]] = []
        rg = args.ref_gens
        pool += list(run_mp_brkga(instance, MpBRKGAConfig(base_pop, rg, seed=1000)).front)
        pool += list(run_brkga(instance, BRKGAConfig(matched_pop, rg, seed=1000)).front)
        pool += list(run_attention_nsga2(instance, None, AttentionNSGA2Config(
            matched_pop, rg, seed=1000, guidance="tape", screening_factor=args.screening),
            fused_model=fused).front)
        for name in methods:
            for fr in fronts[name]:
                pool.extend(fr)
        reference = _pareto(pool)
        ref_kind = (f"non-dominated union of high-budget mp-BRKGA + BRKGA + TAPE @ {rg} gens "
                    "and all evaluated fronts")
    ref_point = nadir_reference(reference, margin=0.1)
    ref_hv = hypervolume(reference, ref_point)
    print(f"reference: {ref_kind} | {len(reference)} pts | HV*={ref_hv:.1f}", flush=True)

    # ---- metrics from the stored fronts (against the folded reference) ----
    raw: dict[str, dict[str, list[float]]] = {
        m: {"gd_plus": [], "igd_plus": [], "spread": [], "hv": [], "hv_ratio": [], "evals": []}
        for m in methods}
    for name in methods:
        for front, evals in zip(fronts[name], evals_by[name]):
            raw[name]["gd_plus"].append(gd_plus(front, reference))
            raw[name]["igd_plus"].append(igd_plus(front, reference))
            raw[name]["spread"].append(spread(front, reference))
            hv = hypervolume(front, ref_point)
            raw[name]["hv"].append(hv)
            raw[name]["hv_ratio"].append(hv / ref_hv if ref_hv > 0 else float("nan"))
            raw[name]["evals"].append(evals)

    # ---- faithfulness head-to-head on a fixed schedule sample ----
    # Move the surrogate back to CPU: the faithfulness helpers build CPU graphs per schedule.
    core = core.cpu()
    fused = fused.cpu()
    rng = make_rng(123)
    faith_scheds = [decode(rng.random(NUM_BLOCKS * n), instance) for _ in range(args.faith_samples)]
    attn_faith = evaluate_faithfulness(faith_scheds, instance, core)
    tape_reports = [faithfulness_report(fused, s, instance) for s in faith_scheds]
    tape_jaccard = float(np.mean([r.leg_critical_jaccard for r in tape_reports]))
    tape_cmax_err = float(np.mean([r.makespan_abs_error for r in tape_reports]))
    faithfulness = {
        "attention_precision_at_1": attn_faith.precision_at_1,
        "attention_spearman_rho": attn_faith.spearman_rho,
        "tape_leg_critical_jaccard": tape_jaccard,
        "tape_makespan_abs_error": tape_cmax_err,
        "random_precision_at_1_baseline": 1.0 / n,
    }

    metrics = [("hv_ratio", "HV / HV*", 4), ("gd_plus", "GD+", 4),
               ("igd_plus", "IGD+", 4), ("spread", "Spread", 4), ("evals", "true evals", 0)]
    agg = {m: {k: _ci(raw[m][k]) for k, *_ in metrics} for m in methods}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"tape_bench_{tag}.json").write_text(json.dumps(
        {"instance": label, "n": n, "coupled": coupled, "peak_power": args.peak_power,
         "gens": G, "seeds": args.seeds, "p_mult": args.p_mult, "base_pop": base_pop,
         "matched_pop": matched_pop, "screening": args.screening,
         "ref_kind": ref_kind, "ref_hv": ref_hv, "faithfulness": faithfulness, "raw": raw,
         "aggregate": {m: {k: list(v) for k, v in agg[m].items()} for m in methods}}, indent=2))

    md = [f"# Faithful-guidance study -- {label} (N={n}, {'coupled' if coupled else 'uncoupled'})",
          f"\n_{args.seeds} seeds, {G} gens, matched exact-eval budget (mp {base_pop}x4 = "
          f"GAT/BRKGA {matched_pop}/gen). Reference: {ref_kind}. Cells = mean (95% CI)._\n",
          "## Optimisation\n",
          "| Method | " + " | ".join(lbl for _, lbl, _ in metrics) + " |",
          "|---|" + "|".join(["---"] * len(metrics)) + "|"]
    for m in methods:
        cells = []
        for k, _lbl, dec in metrics:
            mean, half = agg[m][k]
            cells.append(f"{mean:.{dec}f} ± {half:.{dec}f}" if dec else f"{mean:.0f}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md += ["\n## Guidance-signal faithfulness\n",
           "| Signal | precision@1 | Spearman rho | leg-critical Jaccard vs oracle |",
           "|---|---|---|---|",
           f"| attention | {attn_faith.precision_at_1:.3f} | "
           f"{attn_faith.spearman_rho:.3f} | n/a |",
           f"| **TAPE** | n/a | n/a | **{tape_jaccard:.3f}** |",
           f"| random baseline | {1.0 / n:.3f} | 0.000 | n/a |",
           f"\n_TAPE makespan abs-error vs oracle: {tape_cmax_err:.3f}. "
           f"A faithful signal that also tops the optimisation table supports both the explanation and the guidance claim._\n"]
    md_text = "\n".join(md) + "\n"
    (out_dir / f"tape_bench_{tag}.md").write_text(md_text)
    print("\n" + md_text, flush=True)
    print(f"wrote experiments/fused_tape_guided/tape_bench_{tag}.* "
          f"(total {time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
