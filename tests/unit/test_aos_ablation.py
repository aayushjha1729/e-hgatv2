"""Tests for the AOS Channel-B ablation harness (benchmark/aos_ablation.py).

Two layers: (1) fast, model-free unit tests of the metric helpers (_hv_auc and the
evals_to_threshold / HV-AUC logic in _seed_record); (2) a learn-marked
end-to-end smoke test that trains a tiny surrogate, runs all AOS arms on the exact
N=5 oracle, and checks structure, oracle-bound soundness, and JSON round-tripping.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ehgat.benchmark.aos_ablation import (  # noqa: E402
    AOS_ARMS,
    AOSAblationConfig,
    _hv_auc,
    _seed_record,
    run_aos_ablation,
    to_json_dict,
)
from ehgat.environment.instance import EXACT_TOY_TASKS  # noqa: E402
from ehgat.metrics import hypervolume, nadir_reference  # noqa: E402

pytestmark = pytest.mark.learn


# --------------------------------------------------------------------------------------
# _hv_auc -- model-free
# --------------------------------------------------------------------------------------
def test_hv_auc_constant_at_optimum_is_one() -> None:
    assert _hv_auc([5.0, 5.0, 5.0], 5.0) == pytest.approx(1.0)


def test_hv_auc_linear_ramp_is_half() -> None:
    # trapezoid of [0, 5] over one unit interval = 2.5; normalised by 1*5 -> 0.5
    assert _hv_auc([0.0, 5.0], 5.0) == pytest.approx(0.5)


def test_hv_auc_single_point_uses_ratio() -> None:
    assert _hv_auc([3.0], 6.0) == pytest.approx(0.5)


def test_hv_auc_degenerate_inputs_are_zero() -> None:
    assert _hv_auc([], 5.0) == 0.0
    assert _hv_auc([5.0, 5.0], 0.0) == 0.0


# --------------------------------------------------------------------------------------
# _seed_record -- model-free (synthetic fronts)
# --------------------------------------------------------------------------------------
_GOLDEN = [(1.0, 4.0), (4.0, 1.0)]
_REFERENCE = nadir_reference(_GOLDEN, margin=0.1)
_GOLDEN_HV = hypervolume(_GOLDEN, _REFERENCE)


def test_seed_record_reaches_threshold_at_first_optimal_gen() -> None:
    # Every generation already sits on PF* -> threshold met at gen 0.
    history = [_GOLDEN, _GOLDEN, _GOLDEN]
    record = _seed_record(
        7,
        history,
        _GOLDEN,
        evaluations=123,
        deadlocks_rejected=2,
        wall_clock_s=0.5,
        golden=_GOLDEN,
        reference=_REFERENCE,
        golden_hv=_GOLDEN_HV,
        pop_size=10,
        threshold_hv=0.95 * _GOLDEN_HV,
    )
    assert record.seed == 7
    assert record.evaluations == 123
    assert record.deadlocks_rejected == 2
    assert record.evals_to_threshold == 10  # pop_size * (gen 0 + 1)
    assert record.hv_auc == pytest.approx(1.0)
    assert record.final_hv == pytest.approx(_GOLDEN_HV)
    assert len(record.hv_curve) == 3


def test_seed_record_never_reaching_threshold_is_none() -> None:
    # A single near-reference point has tiny HV, well below 0.95*HV*.
    history = [[(3.9, 3.9)]]
    record = _seed_record(
        0,
        history,
        [(3.9, 3.9)],
        evaluations=10,
        deadlocks_rejected=0,
        wall_clock_s=0.1,
        golden=_GOLDEN,
        reference=_REFERENCE,
        golden_hv=_GOLDEN_HV,
        pop_size=10,
        threshold_hv=0.95 * _GOLDEN_HV,
    )
    assert record.evals_to_threshold is None
    assert record.final_hv < 0.95 * _GOLDEN_HV


# --------------------------------------------------------------------------------------
# End-to-end smoke test (tiny config, exact oracle)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ablation_result():
    config = AOSAblationConfig(
        num_tasks=EXACT_TOY_TASKS,
        generations=3,
        num_seeds=2,
        screening_factor=1,
        oracle="exact",
        surrogate_samples=300,
        surrogate_epochs=5,
        bootstrap_resamples=200,
        search_workers=1,
    )
    return run_aos_ablation(config)


def test_ablation_has_all_arms(ablation_result) -> None:
    assert set(ablation_result.arms) == set(AOS_ARMS)


def test_ablation_record_count_matches_seeds(ablation_result) -> None:
    for arm in AOS_ARMS:
        assert len(ablation_result.arms[arm].records) == 2


def test_ablation_metrics_are_finite(ablation_result) -> None:
    for arm in AOS_ARMS:
        a = ablation_result.arms[arm]
        for stat in (a.final_hv, a.hv_auc, a.igd_plus, a.gd_plus, a.spread):
            assert np.isfinite([stat.mean, stat.lo, stat.hi]).all()
            assert stat.lo <= stat.mean <= stat.hi
        assert a.hv_curve_mean.shape == (ablation_result.generations + 1,)


def test_ablation_respects_oracle_bound(ablation_result) -> None:
    # Exact evaluator => every front is weakly dominated by PF*, so HV <= HV*.
    tol = 1e-9 * max(1.0, ablation_result.golden_hv)
    for arm in AOS_ARMS:
        for record in ablation_result.arms[arm].records:
            assert record.final_hv <= ablation_result.golden_hv + tol


def test_ablation_json_round_trips(ablation_result) -> None:
    payload = to_json_dict(ablation_result)
    restored = json.loads(json.dumps(payload))
    assert restored["experiment"] == "aos_ablation"
    assert set(restored["arms"]) == set(AOS_ARMS)
    for arm in AOS_ARMS:
        assert len(restored["arms"][arm]["seeds"]) == 2
        assert "final_hv" in restored["arms"][arm]
