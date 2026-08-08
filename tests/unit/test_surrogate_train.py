"""Tests for surrogate training and the XGBoost/SHAP baseline.

Held-out accuracy thresholds, training determinism, and the logged GNN-vs-XGBoost
comparison that motivates the physics-aware encoding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance  # noqa: E402
from ehgat.surrogate.train import (  # noqa: E402
    TrainConfig,
    regression_metrics,
    train_surrogate,
)

pytestmark = pytest.mark.learn


def _instance():
    return build_toy_instance(num_tasks=EXACT_TOY_TASKS)


def test_regression_metrics_perfect_prediction() -> None:
    y = torch.tensor([[100.0, 2000.0], [200.0, 3000.0], [150.0, 2500.0]])
    m = regression_metrics(y.clone(), y.clone())
    assert m["r2_makespan"] == pytest.approx(1.0)
    assert m["r2_energy"] == pytest.approx(1.0)
    assert m["mae_overall"] == pytest.approx(0.0)


def test_train_deterministic_small() -> None:
    inst = _instance()
    cfg = TrainConfig(num_samples=400, epochs=12, seed=5)
    a = train_surrogate(inst, cfg).metrics
    b = train_surrogate(inst, cfg).metrics
    assert a == b


@pytest.mark.slow
def test_train_surrogate_meets_accuracy_threshold() -> None:
    inst = _instance()
    result = train_surrogate(inst, TrainConfig(num_samples=1000, epochs=50, seed=0))
    m = result.metrics
    assert m["r2_energy"] >= 0.95, m
    assert m["r2_makespan"] >= 0.78, m


@pytest.mark.slow
def test_gnn_outperforms_xgb_on_energy() -> None:
    xgb_mod = pytest.importorskip("ehgat.surrogate.explainer_xgb")
    inst = _instance()

    gnn = train_surrogate(inst, TrainConfig(num_samples=1500, epochs=60, seed=0)).metrics
    xgb = xgb_mod.train_xgb(inst, xgb_mod.XGBConfig(num_samples=1500, seed=0)).metrics

    # Logged comparison (visible with -s / on failure).
    print(
        f"\n[surrogate accuracy] GNN  R2_mk={gnn['r2_makespan']:.3f} R2_E={gnn['r2_energy']:.3f}"
        f"\n[surrogate accuracy] XGB  R2_mk={xgb['r2_makespan']:.3f} R2_E={xgb['r2_energy']:.3f}"
    )
    assert gnn["r2_energy"] > xgb["r2_energy"] + 0.1
