"""Attribution of the guided-search benefit: mutation targeting against shared screening.

The attention and TAPE arms share the surrogate's offspring screening and differ only in
the task that mutation targets. The granularity ladder shows that attention does not
localise the critical path (AUC ~= 0.5) while guiding as well as TAPE. This ablation
separates the two causes by adding the missing cell, random mutation with screening:

  A  attn  + screen   guidance=attention, screening=K
  R  rand  + screen   random_mutation=True, screening=K   <-- the missing control
  N  rand  + noscreen random_mutation=True, screening=1   (the published null)
  T  tape  + screen   guidance=tape, screening=K           (context)

HV(A) ~= HV(R) indicates that mutation targeting, and hence attention's bottleneck signal,
contributes nothing beyond screening, attributing the benefit to the shared
objective-regression screening and accounting for attention guiding well while carrying no
critical-path information. HV(R) > HV(N) attributes the benefit to screening.

Usage:
    python scripts/screening_ablation.py --instance toy:10 --seeds 8 --screening 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import scipy.stats as sps

OUT = Path(__file__).resolve().parents[1] / "experiments" / "fused_tape_guided"


def _ci(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, float)
    if a.size < 2:
        return (float(a.mean()) if a.size else float("nan")), 0.0
    return float(a.mean()), float(sps.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))


def _pareto(points):
    seen, uniq = set(), []
    for m, e in points:
        k = (round(float(m), 6), round(float(e), 6))
        if k not in seen:
            seen.add(k); uniq.append((float(m), float(e)))
    front, best_e = [], None
    for m, e in sorted(uniq):
        if best_e is None or e < best_e:
            best_e = e; front.append((m, e))
    return tuple(front)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="toy:10")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--screening", type=int, default=2)
    p.add_argument("--p-mult", type=int, default=5)
    p.add_argument("--ref-gens", type=int, default=60)
    p.add_argument("--core-samples", type=int, default=800)
    p.add_argument("--core-epochs", type=int, default=50)
    p.add_argument("--fused-samples", type=int, default=700)
    p.add_argument("--fused-epochs", type=int, default=50)
    p.add_argument("--out-tag", default=None)
    p.add_argument("--out-dir", default=str(OUT / "ablation"))
    args = p.parse_args()

    from ehgat.baselines.brkga import BRKGAConfig, run_brkga
    from ehgat.baselines.mp_brkga import MpBRKGAConfig, run_mp_brkga
    from ehgat.environment.dsdl import load_tables_4_5
    from ehgat.environment.instance import build_toy_instance
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.metrics import hypervolume, nadir_reference
    from ehgat.search.attention_nsga2 import AttentionNSGA2Config, run_attention_nsga2

    spec = args.instance
    if spec.startswith("toy:"):
        instance = build_toy_instance(num_tasks=int(spec.split(":", 1)[1]))
    else:
        data = Path(__file__).resolve().parents[1] / "data" / "tables_4_5.json"
        instance = load_tables_4_5(data, only=[spec])[0].instance
    n = instance.num_tasks
    base_pop, matched_pop, G, K = args.p_mult * n, 4 * args.p_mult * n, args.gens, args.screening

    print(f"[{spec}] N={n} screening K={K} | training core+fused ...", flush=True)
    core = build_core(instance, seed=0, num_samples=args.core_samples,
                      epochs=args.core_epochs, device="cpu")
    fused = train_fused(instance, core, FusedTrainConfig(
        num_samples=args.fused_samples, epochs=args.fused_epochs, unroll_steps=0, seed=0)
    ).model.cpu()
    core = core.cpu()

    arms = {
        "attn+screen":   lambda s: run_attention_nsga2(instance, core, AttentionNSGA2Config(
            matched_pop, G, seed=s, guidance="attention", screening_factor=K)),
        "rand+screen":   lambda s: run_attention_nsga2(instance, core, AttentionNSGA2Config(
            matched_pop, G, seed=s, random_mutation=True, screening_factor=K)),
        "rand+noscreen": lambda s: run_attention_nsga2(instance, core, AttentionNSGA2Config(
            matched_pop, G, seed=s, random_mutation=True, screening_factor=1)),
        "tape+screen":   lambda s: run_attention_nsga2(instance, None, AttentionNSGA2Config(
            matched_pop, G, seed=s, guidance="tape", screening_factor=K), fused_model=fused),
    }

    # reference PF* proxy: high-budget union
    pool = []
    rg = args.ref_gens
    pool += list(run_mp_brkga(instance, MpBRKGAConfig(base_pop, rg, seed=1000)).front)
    pool += list(run_brkga(instance, BRKGAConfig(matched_pop, rg, seed=1000)).front)
    pool += list(run_attention_nsga2(instance, None, AttentionNSGA2Config(
        matched_pop, rg, seed=1000, guidance="tape", screening_factor=K), fused_model=fused).front)
    reference = _pareto(pool)
    ref_point = nadir_reference(reference, margin=0.1)
    ref_hv = hypervolume(reference, ref_point)

    raw = {a: [] for a in arms}
    for a, fn in arms.items():
        for s in range(args.seeds):
            front = fn(s).front
            raw[a].append(hypervolume(front, ref_point) / ref_hv if ref_hv > 0 else float("nan"))
        print(f"  {a:14s}: HV/HV* = {np.mean(raw[a]):.4f}", flush=True)

    A, R, N = np.array(raw["attn+screen"]), np.array(raw["rand+screen"]), np.array(raw["rand+noscreen"])
    # paired difference attention-targeting effect (A - R) over common seeds
    dAR = A - R
    res = {
        "instance": spec, "n": n, "seeds": args.seeds, "screening": K,
        "hv_ratio": {a: list(_ci(raw[a])) for a in arms},
        "attn_minus_randscreen_mean": float(dAR.mean()),
        "attn_minus_randscreen_ci95": list(_ci(list(dAR))),
        "randscreen_minus_noscreen_mean": float((R - N).mean()),
        "raw": raw,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or spec.replace(":", "")
    (Path(args.out_dir) / f"ablation_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"  attention-targeting effect (attn+screen - rand+screen) = "
          f"{dAR.mean():+.4f} (95% CI {res['attn_minus_randscreen_ci95'][0]:+.4f} "
          f"+/- {res['attn_minus_randscreen_ci95'][1]:.4f})", flush=True)
    print(f"  screening effect (rand+screen - rand+noscreen) = "
          f"{(R - N).mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()