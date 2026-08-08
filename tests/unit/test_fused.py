"""Tests for the physics-fused E-HGATv2 (tropical makespan + additive energy head)."""

from __future__ import annotations

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import build_toy_instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.explain.fused_explainer import (
    explain_fused,
    faithfulness_report,
    fused_tradeoff_criticality_scores,
)
from ehgat.surrogate.ehgatv2 import EHGATv2, EHGATv2Config
from ehgat.utils.seeding import make_rng


def _fresh_model() -> FusedEHGATv2:
    core = EHGATv2(EHGATv2Config(hidden=32, layers=2, heads=4))
    model = FusedEHGATv2(core)
    model.freeze_core()
    return model


def _is_integer(value: float, tol: float = 1e-6) -> bool:
    return abs(value - round(value)) < tol


def test_fused_makespan_gradients_are_binary_critical_path() -> None:
    # Seed locally so the random head induces a unique critical path independent of test
    # order (with ties the max-plus subgradient legitimately splits into fractions; this test
    # checks the no-smearing/binary property, which requires a tie-free critical path).
    from ehgat.utils.seeding import seed_everything

    seed_everything(0)
    inst = build_toy_instance(num_tasks=6)
    schedule = decode(make_rng(0).random(NUM_BLOCKS * inst.num_tasks), inst)
    ex = explain_fused(_fresh_model(), schedule, inst)

    # The tropical DP head routes the makespan subgradient only along the critical path,
    # so leg/arc gradients are exact integers (no smearing) regardless of head weights.
    for g in ex.empty_time_grad + ex.loaded_time_grad:
        assert _is_integer(g) and g >= 0.0
    assert max(ex.event_edge_grad) == 1.0
    assert max(ex.empty_time_grad + ex.loaded_time_grad) >= 1.0

    # The additive energy head is strictly linear: dE/d(leg energy) == 1 everywhere.
    assert all(abs(g - 1.0) < 1e-6 for g in ex.empty_energy_grad)
    assert all(abs(g - 1.0) < 1e-6 for g in ex.loaded_energy_grad)


def test_leg_time_prior_is_exact() -> None:
    # The closed-form empty/loaded split (constant loaded/empty power ratio) must reproduce
    # the exact physical leg times from the input arc features -- the basis for C_max ~ 0.99.
    import torch

    from ehgat.explain.fused_ehgat import _leg_time_prior
    from ehgat.explain.train_fused import _exact_legs
    from ehgat.surrogate.graph import AGV_EDGE, build_hetero_graph

    inst = build_toy_instance(num_tasks=8)
    schedule = decode(make_rng(5).random(NUM_BLOCKS * inst.num_tasks), inst)
    legs, _ = _exact_legs(schedule, inst)  # [N, 4] exact (empty_t, loaded_t, empty_e, loaded_e)
    data = build_hetero_graph(schedule, inst)
    arc = data[AGV_EDGE].edge_attr
    order = torch.argsort(data[AGV_EDGE].edge_index[1])
    arc = arc[order]
    empty_p, loaded_p = _leg_time_prior(arc[:, 0], arc[:, 1], arc[:, 2])
    assert torch.allclose(empty_p, legs[:, 0], atol=1e-4)
    assert torch.allclose(loaded_p, legs[:, 1], atol=1e-4)


def test_fused_tcs_is_json_shaped() -> None:
    inst = build_toy_instance(num_tasks=4)
    rng = make_rng(1)
    schedules = [decode(rng.random(NUM_BLOCKS * inst.num_tasks), inst) for _ in range(3)]
    out = fused_tradeoff_criticality_scores(_fresh_model(), schedules, inst)
    assert len(out) == 3
    assert "lambda" in out[0]
    assert out[0]["tasks"]
    assert out[0]["event_arcs"]


def test_fused_objectives_are_finite_and_positive() -> None:
    from ehgat.surrogate.graph import build_hetero_graph

    inst = build_toy_instance(num_tasks=5)
    schedule = decode(make_rng(2).random(NUM_BLOCKS * inst.num_tasks), inst)
    out = _fresh_model()(build_hetero_graph(schedule, inst))
    assert float(out.makespan) > 0.0
    assert float(out.energy) > 0.0


def test_fused_training_recovers_physics_and_is_faithful() -> None:
    # Integration test: after anchoring, the fused model snaps onto the physics
    # (high R^2) and its native critical path agrees with the exact TAPE oracle.
    from ehgat.explain.train_fused import FusedTrainConfig, build_core, train_fused

    # Faithful-by-construction baseline (exact leg prior): deterministic high R^2 for CI.
    inst = build_toy_instance(num_tasks=6)
    core = build_core(inst, seed=0, num_samples=700, epochs=50)
    result = train_fused(
        inst, core, FusedTrainConfig(num_samples=400, epochs=30, seed=0, use_physics_prior=True)
    )

    assert result.metrics["r2_makespan"] >= 0.98
    assert result.metrics["r2_energy"] >= 0.99

    rng = make_rng(99)
    schedule = decode(rng.random(NUM_BLOCKS * inst.num_tasks), inst)
    report = faithfulness_report(result.model, schedule, inst)
    assert report.leg_critical_jaccard >= 0.5
