"""Unit tests for the GNN/TAPE-derived feature-importance landscape.

These exercise the model-native, scalable landscape: the same aggregator over TAPE
explanations from either the exact oracle (exact_landscape) or the fused model
(gnn_landscape), the per-family profile / Pareto-contrast structure, and the
exact-vs-GNN rank-agreement validation primitive.
"""

from __future__ import annotations

import pytest

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import build_toy_instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.explain.gnn_landscape import (
    FEATURE_FAMILIES,
    aggregate_landscape,
    exact_landscape,
    gnn_landscape,
    landscape_rank_agreement,
)
from ehgat.surrogate.ehgatv2 import EHGATv2, EHGATv2Config
from ehgat.utils.seeding import make_rng


@pytest.fixture(scope="module")
def instance():
    return build_toy_instance(num_tasks=6)


def _schedules(instance, n: int, seed: int = 0):
    rng = make_rng(seed)
    return [decode(rng.random(NUM_BLOCKS * instance.num_tasks), instance) for _ in range(n)]


def _fresh_model() -> FusedEHGATv2:
    model = FusedEHGATv2(EHGATv2(EHGATv2Config(hidden=32, layers=2, heads=4)))
    model.freeze_core()
    return model


def test_exact_landscape_profile_is_a_distribution(instance):
    res = exact_landscape(instance, _schedules(instance, 24))
    assert res.source == "exact"
    assert res.num_samples == 24
    assert set(res.makespan_importance) == set(FEATURE_FAMILIES)
    # Normalised profile sums to 1 (it is a comparable importance distribution).
    assert abs(sum(res.makespan_importance.values()) - 1.0) < 1e-6
    assert all(v >= 0.0 for v in res.makespan_importance.values())
    assert 0.0 <= res.agv_share <= 1.0


def test_exact_landscape_is_deterministic(instance):
    schedules = _schedules(instance, 16, seed=3)
    a = exact_landscape(instance, schedules)
    b = exact_landscape(instance, schedules)
    assert a.makespan_importance == b.makespan_importance
    assert a.agv_share == b.agv_share


def test_agv_is_the_dominant_bottleneck(instance):
    """Headline physical truth: AGV routing carries most critical-path mass (2 AGVs vs 3 QCs)."""
    res = exact_landscape(instance, _schedules(instance, 48))
    assert res.agv_share > 0.5


def test_pareto_contrast_keys_and_bounds(instance):
    res = exact_landscape(instance, _schedules(instance, 48))
    assert set(res.pareto_importance) == set(FEATURE_FAMILIES)
    assert set(res.dominated_importance) == set(FEATURE_FAMILIES)
    for fam in FEATURE_FAMILIES:
        assert res.family_contrast[fam] == pytest.approx(
            res.pareto_importance[fam] - res.dominated_importance[fam]
        )


def test_gnn_landscape_has_same_shape_as_exact(instance):
    schedules = _schedules(instance, 16, seed=1)
    g = gnn_landscape(_fresh_model(), instance, schedules)
    e = exact_landscape(instance, schedules)
    assert g.source == "gnn" and e.source == "exact"
    assert set(g.makespan_importance) == set(e.makespan_importance) == set(FEATURE_FAMILIES)
    assert abs(sum(g.makespan_importance.values()) - 1.0) < 1e-6
    # Rank agreement is a well-formed correlation in [-1, 1].
    rho = landscape_rank_agreement(g, e)
    assert -1.0 <= rho <= 1.0


def test_rank_agreement_self_is_perfect(instance):
    e = exact_landscape(instance, _schedules(instance, 24))
    assert landscape_rank_agreement(e, e) == pytest.approx(1.0)


def test_aggregate_rejects_length_mismatch(instance):
    schedules = _schedules(instance, 4)
    from ehgat.explain.tape_explainer import explain_schedule

    exps = [explain_schedule(s, instance) for s in schedules]
    with pytest.raises(ValueError):
        aggregate_landscape(exps, [(1.0, 1.0)], source="exact", num_tasks=instance.num_tasks)
