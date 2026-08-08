"""Tests for the XGBoost + TreeSHAP baseline surrogate."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance
from ehgat.surrogate.explainer_xgb import (
    OBJECTIVES,
    XGBConfig,
    feature_names,
    schedule_features,
    shap_importance,
    train_xgb,
)

pytestmark = pytest.mark.learn


def _instance():
    return build_toy_instance(num_tasks=EXACT_TOY_TASKS)


def test_feature_vector_and_names_length() -> None:
    inst = _instance()
    schedule = decode(np.random.default_rng(0).random(NUM_BLOCKS * inst.num_tasks), inst)
    feats = schedule_features(schedule, inst)
    assert feats.shape == (4 * inst.num_tasks,)
    assert len(feature_names(inst)) == 4 * inst.num_tasks


def test_xgb_trains_and_reports_finite_metrics() -> None:
    inst = _instance()
    result = train_xgb(inst, XGBConfig(num_samples=600, seed=0))
    assert set(result.models) == set(OBJECTIVES)
    for key in ("r2_makespan", "r2_energy", "mae_overall"):
        assert np.isfinite(result.metrics[key])


def test_shap_importance_shape_and_sign() -> None:
    inst = _instance()
    result = train_xgb(inst, XGBConfig(num_samples=600, seed=0))
    importance = shap_importance(result, inst, num_samples=128, seed=1)
    assert set(importance) == set(OBJECTIVES)
    for vals in importance.values():
        assert vals.shape == (4 * inst.num_tasks,)
        assert np.all(vals >= 0.0)  # mean |SHAP| is non-negative


def test_xgb_deterministic() -> None:
    inst = _instance()
    cfg = XGBConfig(num_samples=500, seed=3)
    assert train_xgb(inst, cfg).metrics == train_xgb(inst, cfg).metrics
