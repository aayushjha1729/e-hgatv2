"""Direct measurement of surrogate-screening fidelity, without a full multi-seed HV ablation.

Screening (attention_nsga2, screening_factor=k) generates k*P offspring, ranks them with the
fused surrogate, and spends exact evaluations only on the predicted-best P. Its value is
determined by a single quantity: the extent to which top-P-by-surrogate recovers the
true-best-P inside a realistic near-Pareto offspring pool. Global held-out R2 is an
optimistic upper bound on that quantity, the pool being narrow and harder to rank. This
probe measures it directly.

For a shared pool of k_max*P offspring drawn around a warmed TAPE-guided population, it
reports, per screening factor k in {2,4,8,16}:

  - precision@P  -- |surrogate-selected-P  n  oracle-selected-P| / P
  - Spearman(pred,true) on makespan and energy over the pool
  - HV-recovery  -- (HV_surrogate - HV_random) / (HV_oracle - HV_random), all true HV:
        ~1.0 => screening selects as well as an oracle
        ~0.0 => screening is no better than random selection
        < 0  => screening is worse than random selection

All HVs are true (exact-evaluated) objectives against a common reference point, which makes
the comparison across k and across selection rules a controlled one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--coupled", action="store_true", help="peak-power-coupled regime (pp=30)")
    ap.add_argument("--peak-power", type=float, default=30.0)
    ap.add_argument("--pop", type=int, default=50, help="P (selection size); pool = k_max*P")
    ap.add_argument("--warm-gens", type=int, default=25, help="TAPE-guided warm-up generations")
    ap.add_argument("--mut-rate", type=float, default=0.15, help="per-gene offspring mutation rate")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    # surrogate training (defaults track the sweep so fidelity is representative)
    ap.add_argument("--core-samples", type=int, default=800)
    ap.add_argument("--core-epochs", type=int, default=40)
    ap.add_argument("--fused-samples", type=int, default=600)
    ap.add_argument("--fused-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ehgat.environment.decoder import NUM_BLOCKS, decode, encode_canonical
    from ehgat.environment.evaluator import evaluate
    from ehgat.environment.instance import build_toy_instance, scaled_fleet
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.metrics import hypervolume, nadir_reference
    from ehgat.search.attention_nsga2 import (
        AttentionNSGA2Config, run_attention_nsga2,
    )
    from ehgat.search.nsga2 import fast_non_dominated_sort, order_by_rank_crowding
    from ehgat.search.tape_guidance import tape_predict_objectives
    from ehgat.utils.seeding import make_rng

    n = args.n
    P = args.pop
    ks = sorted(args.ks)
    k_max = ks[-1]
    pool_size = k_max * P
    pp = args.peak_power if args.coupled else None

    a, q = scaled_fleet(n)
    inst = build_toy_instance(num_tasks=n, qcs=tuple(f"QC{i+1}" for i in range(q)),
                              num_agvs=a, peak_power=pp)

    print(f"== screening-fidelity probe ==  N={n}  coupled={bool(pp)}  P={P}  "
          f"pool={pool_size}  warm_gens={args.warm_gens}", flush=True)

    # ---- train surrogate (same recipe as the sweep) ----
    core = build_core(inst, seed=args.seed, num_samples=args.core_samples,
                      epochs=args.core_epochs, device=args.device)
    fused_res = train_fused(inst, core, FusedTrainConfig(
        num_samples=args.fused_samples, epochs=args.fused_epochs,
        unroll_steps=(2 if pp else 0), seed=args.seed))
    fused = fused_res.model.to(args.device)
    print(f"  fused R2_makespan={fused_res.metrics.get('r2_makespan'):.4f} "
          f"R2_energy={fused_res.metrics.get('r2_energy'):.4f}  (global held-out)", flush=True)

    # ---- warm up a realistic near-Pareto population with TAPE-guided search ----
    warm = run_attention_nsga2(inst, None, AttentionNSGA2Config(
        pop_size=P, generations=args.warm_gens, guidance="tape",
        screening_factor=1, seed=args.seed), fused_model=fused)
    parents = [encode_canonical(s, inst) for s in warm.schedules]
    if len(parents) < 2:  # degenerate archive -> seed from random keys
        rng0 = make_rng(args.seed + 1)
        parents = [rng0.random(NUM_BLOCKS * n) for _ in range(P)]
    print(f"  warm-up archive size={len(parents)} (parents for offspring pool)", flush=True)

    # ---- build one shared offspring pool (crossover + mutation around the archive) ----
    rng = make_rng(args.seed + 7)
    L = NUM_BLOCKS * n
    pool: list = []
    guard = 0
    while len(pool) < pool_size and guard < pool_size * 20:
        guard += 1
        i, j = rng.integers(0, len(parents), size=2)
        mask = rng.random(L) < 0.5
        child = np.where(mask, parents[i], parents[j]).astype(float).copy()
        mut = rng.random(L) < args.mut_rate
        if mut.any():
            child[mut] = rng.random(int(mut.sum()))
        try:
            pool.append(decode(child, inst))
        except Exception:
            continue
    if len(pool) < pool_size:
        print(f"  WARN only built {len(pool)}/{pool_size} valid offspring", flush=True)
        pool_size = len(pool)

    # ---- exact + surrogate objectives for the whole pool ----
    true_obj = [tuple(evaluate(s, inst).objectives) for s in pool]
    pred_obj = [tuple(p) for p in tape_predict_objectives(fused, pool, inst)]
    true_arr = np.asarray(true_obj, dtype=float)
    pred_arr = np.asarray(pred_obj, dtype=float)

    rho_mk = _spearman(pred_arr[:, 0], true_arr[:, 0])
    rho_en = _spearman(pred_arr[:, 1], true_arr[:, 1])
    print(f"  pool Spearman(pred,true): makespan={rho_mk:.3f}  energy={rho_en:.3f}", flush=True)

    # common reference point over the full pool's true objectives (fair HV across selections)
    ref = nadir_reference(true_obj, margin=0.1)

    def hv_of(idx: list[int]) -> float:
        pts = [true_obj[i] for i in idx]
        fronts = fast_non_dominated_sort(pts)
        nd = [pts[i] for i in fronts[0]]  # HV depends only on the non-dominated subset
        return hypervolume(nd, ref)

    def select_by(objs: list[tuple[float, float]], cand: list[int], take: int) -> list[int]:
        sub = [objs[i] for i in cand]
        order = order_by_rank_crowding(sub, fast_non_dominated_sort(sub))[:take]
        return [cand[i] for i in order]

    rrng = make_rng(args.seed + 99)
    print("\n   k | precision@P | HV_surr  HV_orac  HV_rand |  HV-recovery", flush=True)
    print("  ---+-------------+--------------------------+-------------", flush=True)
    for k in ks:
        m = k * P
        if m > pool_size:
            break
        cand = list(range(m))  # first k*P of the shared pool
        s_sel = select_by(pred_obj, cand, P)   # surrogate screening
        o_sel = select_by(true_obj, cand, P)   # oracle screening (upper bound)
        r_sel = list(rrng.permutation(m)[:P])  # no screening (random P of the k*P)

        prec = len(set(s_sel) & set(o_sel)) / P
        hv_s, hv_o, hv_r = hv_of(s_sel), hv_of(o_sel), hv_of(r_sel)
        denom = hv_o - hv_r
        recov = (hv_s - hv_r) / denom if abs(denom) > 1e-9 else float("nan")
        print(f"   {k:2d}|    {prec:5.3f}    | {hv_s:8.1f} {hv_o:8.1f} {hv_r:8.1f} |   {recov:+.3f}",
              flush=True)

    print("\n  verdict guide: HV-recovery ~1 => strong lever, run the ablation;"
          "  ~0 => dead weight, skip it.", flush=True)


if __name__ == "__main__":
    main()
