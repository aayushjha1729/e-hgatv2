"""The batched offspring-screening predictor must equal the per-graph reference.

tape_predict_objectives (public, batched: one batched_longest_path per chunk) must
return the same (makespan, energy) as _tape_predict_pergraph (the per-node .item()
DP loop) for the same model weights, in both the uncoupled and coupled regimes. The batched
path is 68-83% cheaper on the guided search; this pins it to the reference so the speedup is
free of any behaviour change (screening ranks unchanged).
"""

from __future__ import annotations

import numpy as np

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import build_toy_instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.explain.train_fused import build_core
from ehgat.search.tape_guidance import (
    _batched_makespan_grads,
    _tape_predict_pergraph,
    _tape_signals_pergraph,
    tape_predict_objectives,
    tape_signals_batch,
)


def _schedules(inst, n: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    return [decode(rng.random(NUM_BLOCKS * n), inst) for _ in range(count)]


def _model(peak_power, unroll: int):
    n = 9
    inst = build_toy_instance(num_tasks=n, peak_power=peak_power)
    core = build_core(inst, seed=0, num_samples=120, epochs=3)
    model = FusedEHGATv2(
        core, coupled=peak_power is not None, unroll_steps=unroll, peak_power=peak_power
    )
    model.freeze_core()
    return inst, model, n


def _check(peak_power, unroll: int) -> None:
    n = 9
    inst = build_toy_instance(num_tasks=n, peak_power=peak_power)
    core = build_core(inst, seed=0, num_samples=120, epochs=3)
    model = FusedEHGATv2(
        core, coupled=peak_power is not None, unroll_steps=unroll, peak_power=peak_power
    )
    model.freeze_core()

    scheds = _schedules(inst, n, 24, seed=1)
    ref = _tape_predict_pergraph(model, scheds, inst)
    got = tape_predict_objectives(model, scheds, inst)

    assert len(ref) == len(got) == len(scheds)
    for i, ((m1, e1), (m2, e2)) in enumerate(zip(ref, got)):
        assert abs(m1 - m2) < 1e-3, f"makespan mismatch @ {i}: {m1} vs {m2}"
        assert abs(e1 - e2) <= 1e-4 * max(1.0, abs(e1)), f"energy mismatch @ {i}: {e1} vs {e2}"


def test_screening_parity_uncoupled() -> None:
    _check(None, unroll=0)


def test_screening_parity_coupled() -> None:
    _check(30.0, unroll=2)


# --- guidance path (tape_signals_batch) --------------------------------------------------

def _guidance_batch_invariance(peak_power, unroll: int) -> None:
    """The block-diagonal batch must give each graph the same per-task critical-path grads as
    running that graph alone -- i.e. no cross-graph leakage in sum(makespan).backward()."""
    inst, model, n = _model(peak_power, unroll)
    scheds = _schedules(inst, n, 12, seed=3)
    e_all, l_all, t_all = _batched_makespan_grads(model, scheds, inst)
    e1, l1, t1 = [], [], []
    for s in scheds:
        e, l, t = _batched_makespan_grads(model, [s], inst)
        e1.append(e); l1.append(l); t1.append(t)
    assert np.allclose(e_all, np.concatenate(e1), atol=1e-5), "empty_t grad batch-variance"
    assert np.allclose(l_all, np.concatenate(l1), atol=1e-5), "loaded_t grad batch-variance"
    assert np.allclose(t_all, np.concatenate(t1), atol=1e-5), "tau grad batch-variance"


def test_guidance_batch_invariance_uncoupled() -> None:
    _guidance_batch_invariance(None, unroll=0)


def test_guidance_batch_invariance_coupled() -> None:
    _guidance_batch_invariance(30.0, unroll=2)


def test_guidance_signal_matches_pergraph_coupled() -> None:
    """Coupled: batched and per-graph share one coupled event DAG, under which the guidance
    signal matches the per-graph reference task for task (identical first-argmax
    tie-break)."""
    inst, model, n = _model(30.0, unroll=2)
    scheds = _schedules(inst, n, 16, seed=5)
    ref = _tape_signals_pergraph(model, scheds, inst, temperature=0.25)
    got = tape_signals_batch(model, scheds, inst, temperature=0.25)
    for i, ((p1, a1, q1), (p2, a2, q2)) in enumerate(zip(ref, got)):
        assert np.allclose(a1, a2, atol=1e-4), f"w_agv mismatch @ {i}"
        assert np.allclose(q1, q2, atol=1e-4), f"w_qc mismatch @ {i}"
        assert np.allclose(p1, p2, atol=1e-4), f"task_probs mismatch @ {i}"
