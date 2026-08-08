"""Unit tests for the exact landscape analysis.

These exercise the core (Torch-free) landscape primitives against the exact evaluator:
grouped Sobol' indices, critical-path cascade attribution, and the Pareto contrast. The
physical checks encode ground truth that any correct decomposition must satisfy, for
instance the dominance of the structural families in the energy variance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ehgat.benchmark.landscape import (
    FEATURE_FAMILIES,
    OBJECTIVES,
    critical_path_landscape,
    grouped_sobol,
    pareto_contrast,
    run_landscape,
    to_json_dict,
)
from ehgat.environment.instance import build_toy_instance


@pytest.fixture(scope="module")
def instance():
    return build_toy_instance(num_tasks=6)


# --------------------------------------------------------------------------------------
# Grouped Sobol
# --------------------------------------------------------------------------------------
def test_sobol_structure_and_bounds(instance):
    res = grouped_sobol(instance, base_samples=128, seed=0)
    assert set(res.indices) == set(OBJECTIVES)
    for obj in OBJECTIVES:
        assert set(res.indices[obj]) == set(FEATURE_FAMILIES)
        for fam in FEATURE_FAMILIES:
            idx = res.indices[obj][fam]
            # First-order is clipped to a probability; total-order is a non-negative
            # variance ratio (bounded by 1 in expectation, finite on a finite sample).
            assert 0.0 <= idx.first_order <= 1.0
            assert 0.0 <= idx.total_order <= 1.5
    # Saltelli A/B + one re-sample per family.
    assert res.total_evaluations == res.base_samples * (2 + len(FEATURE_FAMILIES))


def test_sobol_is_deterministic(instance):
    a = grouped_sobol(instance, base_samples=64, seed=7)
    b = grouped_sobol(instance, base_samples=64, seed=7)
    for obj in OBJECTIVES:
        for fam in FEATURE_FAMILIES:
            assert a.indices[obj][fam].first_order == b.indices[obj][fam].first_order
            assert a.indices[obj][fam].total_order == b.indices[obj][fam].total_order


def test_sobol_energy_is_routing_dominated(instance):
    """Physical truth: energy is dominated by routing, not speed.

    Loaded-leg distance is fixed per task, which makes the speed lever on that distance
    (loaded_speed) second-order, while empty_speed modulates the empty leg by a few percent.
    Energy variance is driven instead by which empty (repositioning) legs are travelled, a
    topological function of sequence and AGV assignment. The structural families therefore
    dominate the energy decomposition, against the kinematic reading a flat attributor
    would encode.
    """
    res = grouped_sobol(instance, base_samples=256, seed=0)
    energy = res.indices["energy"]
    speed_total = energy["empty_speed"].total_order + energy["loaded_speed"].total_order
    struct_total = energy["sequence"].total_order + energy["assignment"].total_order
    assert struct_total > speed_total


def test_sobol_makespan_is_structure_dominated(instance):
    """Topology dominates makespan: sequence/assignment outweigh the speed families.

    Makespan is a max-plus longest path, under which the routing and sequencing decisions
    that shape the critical path carry far more total-order variance than the per-leg
    speeds. This cascade coupling lies outside the expressive range of a flat additive
    attributor.
    """
    res = grouped_sobol(instance, base_samples=256, seed=0)
    mk = res.indices["makespan"]
    struct_total = mk["sequence"].total_order + mk["assignment"].total_order
    speed_total = mk["empty_speed"].total_order + mk["loaded_speed"].total_order
    assert struct_total > speed_total
    assert struct_total > 0.1


# --------------------------------------------------------------------------------------
# Critical-path cascade
# --------------------------------------------------------------------------------------
def test_critical_path_landscape_shapes_and_bounds(instance):
    res = critical_path_landscape(instance, num_samples=32, seed=0)
    n = instance.num_tasks
    for arr in (res.agv_bound_freq, res.qc_bound_freq, res.cascade_size,
                res.marginal_makespan, res.marginal_energy):
        assert len(arr) == n
    assert all(0.0 <= f <= 1.0 for f in res.agv_bound_freq)
    assert all(0.0 <= f <= 1.0 for f in res.qc_bound_freq)
    assert 0.0 <= res.agv_share <= 1.0
    # Accelerating a task can only shorten (never lengthen) the makespan.
    assert all(m >= -1e-9 for m in res.marginal_makespan)
    # A task always cascades to at least itself when it is on the critical path; >= 0 always.
    assert all(c >= 0.0 for c in res.cascade_size)


def test_critical_path_landscape_is_deterministic(instance):
    a = critical_path_landscape(instance, num_samples=24, seed=3)
    b = critical_path_landscape(instance, num_samples=24, seed=3)
    assert a.marginal_makespan == b.marginal_makespan
    assert a.agv_bound_freq == b.agv_bound_freq


def test_cascade_size_at_least_self_when_accelerated(instance):
    """At least one task whose completion shifts should appear when speeds change.

    Accelerating any task changes its own travel time, hence its own completion time, so
    averaged cascade size over samples must be strictly positive for at least one task.
    """
    res = critical_path_landscape(instance, num_samples=32, seed=1)
    assert max(res.cascade_size) > 0.0


# --------------------------------------------------------------------------------------
# Pareto contrast
# --------------------------------------------------------------------------------------
def test_pareto_contrast_structure(instance):
    res = pareto_contrast(instance, num_samples=128, seed=0)
    assert res.num_pareto >= 1
    assert res.num_pareto <= res.num_samples
    for name in res.descriptors:
        assert -1.0 <= res.cliffs_delta[name] <= 1.0
        assert name in res.pareto_mean
        assert name in res.dominated_mean


def test_pareto_contrast_is_deterministic(instance):
    a = pareto_contrast(instance, num_samples=128, seed=5)
    b = pareto_contrast(instance, num_samples=128, seed=5)
    assert a.cliffs_delta == b.cliffs_delta
    assert a.num_pareto == b.num_pareto


# --------------------------------------------------------------------------------------
# Bundle + serialisation
# --------------------------------------------------------------------------------------
def test_run_landscape_and_json_roundtrip(instance):
    res = run_landscape(
        instance, sobol_base_samples=32, cascade_samples=16, contrast_samples=64, seed=0
    )
    payload = to_json_dict(res)
    # Must be JSON-serialisable.
    text = json.dumps(payload)
    again = json.loads(text)
    assert again["instance"]["num_tasks"] == instance.num_tasks
    assert set(again["sobol"]["indices"]) == set(OBJECTIVES)
    assert len(again["cascade"]["marginal_makespan"]) == instance.num_tasks
    assert again["pareto_contrast"]["num_pareto"] >= 1


def test_zero_variance_objective_is_safe():
    """A single-task instance has degenerate structure; indices must stay finite."""
    inst = build_toy_instance(num_tasks=1)
    res = grouped_sobol(inst, base_samples=16, seed=0)
    for obj in OBJECTIVES:
        for fam in FEATURE_FAMILIES:
            assert np.isfinite(res.indices[obj][fam].first_order)
            assert np.isfinite(res.indices[obj][fam].total_order)
