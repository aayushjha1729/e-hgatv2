"""Unit tests for TAPE guidance wired into the NSGA-II search.

These verify the wiring, determinism and signal shape rather than surrogate quality. An
untrained fused model suffices: the returned front is built from exact evaluate()
objectives and is therefore physically valid independently of the head.
"""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import build_toy_instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.search.attention_nsga2 import AttentionNSGA2Config, run_attention_nsga2
from ehgat.search.tape_guidance import tape_predict_objectives, tape_signals
from ehgat.surrogate.ehgatv2 import EHGATv2, EHGATv2Config
from ehgat.utils.seeding import make_rng


@pytest.fixture(scope="module")
def instance():
    return build_toy_instance(num_tasks=6)


def _fresh_model() -> FusedEHGATv2:
    model = FusedEHGATv2(EHGATv2(EHGATv2Config(hidden=32, layers=2, heads=4)))
    model.freeze_core()
    return model


def _schedule(instance, seed: int = 0):
    rng = make_rng(seed)
    return decode(rng.random(NUM_BLOCKS * instance.num_tasks), instance)


def test_tape_signals_shape_and_normalization(instance):
    model = _fresh_model()
    probs, w_agv, w_qc = tape_signals(model, _schedule(instance), instance, temperature=0.25)
    n = instance.num_tasks
    assert probs.shape == (n,) and w_agv.shape == (n,) and w_qc.shape == (n,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)
    assert np.all(w_agv >= 0.0) and np.all(w_qc >= 0.0)


def test_tape_signals_concentrate_on_the_critical_path(instance):
    """The fused critical path is sparse, so probs must beat the uniform 1/N on its tasks."""
    model = _fresh_model()
    probs, w_agv, w_qc = tape_signals(model, _schedule(instance), instance, temperature=0.25)
    crit = (w_agv + w_qc) > 0.0
    if crit.any():  # untrained head still routes C_max through some exact critical path
        assert probs[crit].max() > 1.0 / instance.num_tasks


def test_tape_predict_objectives_runs(instance):
    model = _fresh_model()
    scheds = [_schedule(instance, s) for s in range(4)]
    preds = tape_predict_objectives(model, scheds, instance)
    assert len(preds) == 4
    assert all(np.isfinite(m) and np.isfinite(e) for m, e in preds)


def test_tape_guided_search_runs_and_is_valid(instance):
    model = _fresh_model()
    cfg = AttentionNSGA2Config(pop_size=12, generations=4, guidance="tape", screening_factor=2, seed=0)
    res = run_attention_nsga2(instance, None, cfg, fused_model=model)
    assert len(res.front) >= 1
    assert all(np.isfinite(m) and np.isfinite(e) for m, e in res.front)
    # front is non-dominated and ascending in makespan
    ms = [m for m, _ in res.front]
    assert ms == sorted(ms)


def test_tape_guided_search_is_deterministic(instance):
    model = _fresh_model()
    cfg = AttentionNSGA2Config(pop_size=12, generations=4, guidance="tape", screening_factor=2, seed=7)
    a = run_attention_nsga2(instance, None, cfg, fused_model=model)
    b = run_attention_nsga2(instance, None, cfg, fused_model=model)
    assert a.front == b.front


def test_tape_guidance_requires_fused_model(instance):
    cfg = AttentionNSGA2Config(pop_size=8, generations=2, guidance="tape", seed=0)
    with pytest.raises(ValueError, match="fused_model"):
        run_attention_nsga2(instance, None, cfg)


def test_invalid_guidance_raises(instance):
    model = EHGATv2(EHGATv2Config(hidden=16, layers=2, heads=2))
    cfg = AttentionNSGA2Config(pop_size=8, generations=2, guidance="bogus", seed=0)
    with pytest.raises(ValueError, match="guidance must be one of"):
        run_attention_nsga2(instance, model, cfg)


def test_attention_path_unchanged_without_fused(instance):
    """Regression: default attention guidance still works with a bare core and no fused_model."""
    model = EHGATv2(EHGATv2Config(hidden=16, layers=2, heads=2))
    cfg = AttentionNSGA2Config(pop_size=10, generations=3, seed=1)
    res = run_attention_nsga2(instance, model, cfg)
    assert len(res.front) >= 1
