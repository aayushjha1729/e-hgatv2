"""XGBoost and TreeSHAP baseline surrogate (Torch-free).

This is the comparison baseline to the E-HGATv2 GNN. It learns (C_max, E) from a
flat, interpretable decision-variable vector -- per task: sequence position, AGV
assignment, empty-leg speed level, loaded-leg speed level (the 4N chromosome semantics)
-- and attributes predictions with TreeSHAP (exact for trees), giving the sequencing and
speed attribution over the same decision variables the GNN attributes structurally.

It depends only on NumPy, XGBoost and SHAP (no Torch or PyG), and therefore runs
independently of the GNN stack while sharing the same random-schedule sampling protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shap
import xgboost as xgb

from ehgat.environment.decoder import NUM_BLOCKS, Schedule, decode
from ehgat.environment.evaluator import evaluate
from ehgat.environment.instance import Instance
from ehgat.utils.seeding import make_rng

__all__ = [
    "OBJECTIVES",
    "XGBConfig",
    "XGBResult",
    "feature_names",
    "schedule_features",
    "shap_importance",
    "train_xgb",
]

OBJECTIVES = ("makespan", "energy")


@dataclass(frozen=True)
class XGBConfig:
    """XGBoost hyper-parameters (one regressor per objective)."""

    num_samples: int = 1000
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.9
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 0


@dataclass
class XGBResult:
    """Trained per-objective regressors, feature names and held-out metrics."""

    models: dict[str, xgb.XGBRegressor]
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


def feature_names(instance: Instance) -> list[str]:
    """Ordered, human-readable names of the 4N decision-variable features."""
    names: list[str] = []
    for j in range(instance.num_tasks):
        names += [f"seq_pos_t{j}", f"agv_t{j}", f"empty_spd_t{j}", f"loaded_spd_t{j}"]
    return names


def schedule_features(schedule: Schedule, instance: Instance) -> np.ndarray:
    """Flatten a schedule into its 4N decision-variable vector (float64)."""
    n = instance.num_tasks
    position = [0] * n
    for rank, task_id in enumerate(schedule.global_order):
        position[task_id] = rank
    rows: list[float] = []
    for j in range(n):
        rows += [
            float(position[j]),
            float(schedule.assignment[j]),
            float(int(schedule.empty_speed[j])),
            float(int(schedule.loaded_speed[j])),
        ]
    return np.asarray(rows, dtype=np.float64)


def _build_dataset(
    instance: Instance, num_samples: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = make_rng(seed)
    n = instance.num_tasks
    x_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    for _ in range(num_samples):
        schedule = decode(rng.random(NUM_BLOCKS * n), instance)
        evaluation = evaluate(schedule, instance)
        x_rows.append(schedule_features(schedule, instance))
        y_rows.append([evaluation.makespan, evaluation.energy])
    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.float64)


def _split(
    x: np.ndarray, y: np.ndarray, *, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, ...]:
    rng = make_rng(seed)
    order = rng.permutation(x.shape[0])
    x, y = x[order], y[order]
    n = x.shape[0]
    n_test = round(test_frac * n)
    n_val = round(val_frac * n)
    x_test, y_test = x[:n_test], y[:n_test]
    x_val, y_val = x[n_test : n_test + n_val], y[n_test : n_test + n_val]
    x_train, y_train = x[n_test + n_val :], y[n_test + n_val :]
    return x_train, y_train, x_val, y_val, x_test, y_test


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    abs_err = np.abs(pred - true)
    for i, name in enumerate(OBJECTIVES):
        y = true[:, i]
        ss_res = float(np.sum((y - pred[:, i]) ** 2))
        ss_tot = float(max(np.sum((y - y.mean()) ** 2), 1e-12))
        out[f"r2_{name}"] = 1.0 - ss_res / ss_tot
        out[f"mae_{name}"] = float(abs_err[:, i].mean())
    out["mae_overall"] = float(abs_err.mean())
    return out


def train_xgb(instance: Instance, config: XGBConfig | None = None) -> XGBResult:
    """Train one XGBoost regressor per objective and report held-out R^2 / MAE."""
    config = config or XGBConfig()
    x, y = _build_dataset(instance, config.num_samples, seed=config.seed)
    x_train, y_train, _x_val, _y_val, x_test, y_test = _split(
        x, y, val_frac=config.val_frac, test_frac=config.test_frac, seed=config.seed
    )

    models: dict[str, xgb.XGBRegressor] = {}
    preds = np.zeros((x_test.shape[0], len(OBJECTIVES)), dtype=np.float64)
    for i, name in enumerate(OBJECTIVES):
        model = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            random_state=config.seed,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(x_train, y_train[:, i])
        models[name] = model
        preds[:, i] = model.predict(x_test)

    return XGBResult(
        models=models,
        feature_names=feature_names(instance),
        metrics=_metrics(preds, y_test),
    )


def shap_importance(
    result: XGBResult, instance: Instance, *, num_samples: int = 256, seed: int = 0
) -> dict[str, np.ndarray]:
    """Global TreeSHAP importance (mean |SHAP| per feature) for each objective.

    Returns a dict objective -> array[4N] aligned with result.feature_names.
    """
    x, _ = _build_dataset(instance, num_samples, seed=seed)
    importance: dict[str, np.ndarray] = {}
    for name, model in result.models.items():
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(x))
        importance[name] = np.abs(values).mean(axis=0)
    return importance
