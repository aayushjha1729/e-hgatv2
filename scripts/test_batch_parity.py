"""Parity + timing test for batched TAPE (explain_fused_batch / tape_signals_batch /
tape_predict_objectives) vs the per-schedule path. The batched path must match the serial path exactly."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--coupled", action="store_true")
    args = ap.parse_args()

    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.environment.instance import build_toy_instance, scaled_fleet
    from ehgat.explain.fused_explainer import explain_fused, explain_fused_batch
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused
    from ehgat.search.tape_guidance import (
        tape_predict_objectives, tape_signals, tape_signals_batch,
    )
    from ehgat.utils.seeding import make_rng

    n = args.n
    a, q = scaled_fleet(n)
    pp = 30.0 if args.coupled else None
    inst = build_toy_instance(num_tasks=n, qcs=tuple(f"QC{i+1}" for i in range(q)),
                              num_agvs=a, peak_power=pp)
    core = build_core(inst, seed=0, num_samples=200, epochs=5, device="cpu")
    fused = train_fused(inst, core, FusedTrainConfig(
        num_samples=150, epochs=5, unroll_steps=(2 if pp else 0), seed=0)).model.cpu()

    rng = make_rng(1)
    scheds = [decode(rng.random(NUM_BLOCKS * n), inst) for _ in range(args.k)]

    # --- parity: explain_fused_batch vs serial ---
    t0 = time.perf_counter()
    serial = [explain_fused(fused, s, inst) for s in scheds]
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    batched = explain_fused_batch(fused, scheds, inst)
    t_batch = time.perf_counter() - t0

    max_makespan_err = max(abs(a.makespan - b.makespan) for a, b in zip(serial, batched))
    max_grad_err = 0.0
    for a, b in zip(serial, batched):
        for ga, gb in ((a.empty_time_grad, b.empty_time_grad),
                       (a.loaded_time_grad, b.loaded_time_grad),
                       (a.node_grad, b.node_grad)):
            if ga:
                max_grad_err = max(max_grad_err, float(np.max(np.abs(np.array(ga) - np.array(gb)))))
    # critical-set (Jaccard-relevant) agreement
    def crit(g): return {i for i, v in enumerate(g) if v > 0.5}
    mism = sum(crit(a.empty_time_grad) != crit(b.empty_time_grad)
               or crit(a.loaded_time_grad) != crit(b.loaded_time_grad)
               or crit(a.node_grad) != crit(b.node_grad)
               for a, b in zip(serial, batched))

    print(f"N={n} K={args.k} coupled={bool(pp)}")
    print(f"[explain_fused]  max|makespan diff|={max_makespan_err:.3e}  max|grad diff|={max_grad_err:.3e}"
          f"  critical-set mismatches={mism}/{args.k}")
    print(f"[timing] serial={t_serial:.2f}s  batched={t_batch:.2f}s  speedup={t_serial/max(t_batch,1e-9):.1f}x")

    # --- parity: tape_signals_batch vs serial ---
    sser = [tape_signals(fused, s, inst, 0.25) for s in scheds]
    sbat = tape_signals_batch(fused, scheds, inst, 0.25)
    sig_err = max(float(np.max(np.abs(np.array(x[0]) - np.array(y[0])))) for x, y in zip(sser, sbat))
    print(f"[tape_signals]   max|task_prob diff|={sig_err:.3e}")

    # --- parity: tape_predict_objectives (already batched internally) is deterministic ---
    preds = tape_predict_objectives(fused, scheds, inst)
    print(f"[tape_predict]   {len(preds)} predictions, sample={preds[0]}")

    ok = max_makespan_err < 1e-4 and max_grad_err < 1e-3 and mism == 0 and sig_err < 1e-4
    print("PARITY: " + ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
