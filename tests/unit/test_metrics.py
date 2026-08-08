"""Tests for the quality indicators (metrics/).

Closed-form cases pin the exact values; randomised cases cross-validate hypervolume and
IGD+ against pymoo. Pareto-compliance (a dominating front never scores worse) is
checked explicitly because it is the property the comparisons rely on.
"""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.metrics import (
    gd_plus,
    hypervolume,
    igd_plus,
    nadir_reference,
    spacing,
    spread,
)


# --------------------------------------------------------------------------------------
# Hypervolume
# --------------------------------------------------------------------------------------
def test_hypervolume_single_point() -> None:
    assert hypervolume([(1.0, 1.0)], reference=(2.0, 2.0)) == pytest.approx(1.0)


def test_hypervolume_staircase() -> None:
    # Two points, ref (4,4): strips (1,3)->(3-1)*(4-3)=2 ; (3,1)->(4-3)*(4-1)=3 ; total 5.
    front = [(1.0, 3.0), (3.0, 1.0)]
    assert hypervolume(front, reference=(4.0, 4.0)) == pytest.approx(5.0)


def test_hypervolume_ignores_dominated_and_out_of_bounds() -> None:
    # (2,3.5) is dominated by (1,3); (5,0.5) has x >= ref_x -> both contribute nothing.
    front = [(1.0, 3.0), (3.0, 1.0), (2.0, 3.5), (5.0, 0.5)]
    assert hypervolume(front, reference=(4.0, 4.0)) == pytest.approx(5.0)


def test_hypervolume_empty_is_zero() -> None:
    assert hypervolume([], reference=(4.0, 4.0)) == 0.0
    assert hypervolume([(5.0, 5.0)], reference=(4.0, 4.0)) == 0.0


def test_hypervolume_monotone_under_domination() -> None:
    ref = (10.0, 10.0)
    worse = [(4.0, 5.0), (5.0, 4.0)]
    better = [(2.0, 3.0), (3.0, 2.0)]  # dominates worse
    assert hypervolume(better, ref) > hypervolume(worse, ref)


# --------------------------------------------------------------------------------------
# GD+ / IGD+
# --------------------------------------------------------------------------------------
def test_gd_igd_zero_on_identical_front() -> None:
    pf = [(1.0, 4.0), (2.0, 3.0), (4.0, 1.0)]
    assert gd_plus(pf, pf) == pytest.approx(0.0)
    assert igd_plus(pf, pf) == pytest.approx(0.0)


def test_gd_plus_only_penalises_worse_objectives() -> None:
    # A solution that dominates its reference point is at d+ = 0 (no objective is worse).
    assert gd_plus([(1.0, 1.0)], [(2.0, 2.0)]) == pytest.approx(0.0)
    # A solution worse in one objective by 3 -> d+ = 3.
    assert gd_plus([(5.0, 1.0)], [(2.0, 2.0)]) == pytest.approx(3.0)


def test_igd_plus_monotone_under_domination() -> None:
    pf = [(1.0, 4.0), (2.0, 3.0), (4.0, 1.0)]
    worse = [(5.0, 6.0)]
    better = [(2.0, 3.0), (4.0, 1.0)]
    assert igd_plus(better, pf) < igd_plus(worse, pf)


def test_empty_front_is_infinite() -> None:
    assert gd_plus([], [(1.0, 1.0)]) == float("inf")
    assert igd_plus([], [(1.0, 1.0)]) == float("inf")


# --------------------------------------------------------------------------------------
# Spread / spacing
# --------------------------------------------------------------------------------------
def test_uniform_front_has_low_spacing() -> None:
    uniform = [(0.0, 4.0), (1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (4.0, 0.0)]
    clustered = [(0.0, 4.0), (0.1, 3.9), (0.2, 3.8), (3.0, 1.0), (4.0, 0.0)]
    assert spacing(uniform) < spacing(clustered)
    assert spacing(uniform) == pytest.approx(0.0)


def test_spread_small_for_uniform_reaching_extremes() -> None:
    pf = [(0.0, 4.0), (1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (4.0, 0.0)]
    uniform = pf
    gappy = [(0.0, 4.0), (0.2, 3.8), (4.0, 0.0)]
    assert spread(uniform, reference=pf) < spread(gappy, reference=pf)


def test_spread_and_spacing_degenerate() -> None:
    assert spread([(1.0, 1.0)]) == 0.0
    assert spacing([(1.0, 1.0)]) == 0.0


# --------------------------------------------------------------------------------------
# Reference point
# --------------------------------------------------------------------------------------
def test_nadir_reference_dominated_by_all_points() -> None:
    front = [(1.0, 4.0), (2.0, 3.0), (4.0, 1.0)]
    ref = nadir_reference(front, margin=0.1)
    assert ref[0] > 4.0 and ref[1] > 4.0  # strictly worse than every point's objectives
    for m, e in front:
        assert m < ref[0] and e < ref[1]


# --------------------------------------------------------------------------------------
# Cross-validation against pymoo
# --------------------------------------------------------------------------------------
def test_hypervolume_matches_pymoo() -> None:
    pytest.importorskip("pymoo")
    from pymoo.indicators.hv import HV

    rng = np.random.default_rng(0)
    ref = np.array([1.0, 1.0])
    for _ in range(20):
        pts = rng.random((rng.integers(1, 12), 2))
        pts = pts[(pts < 1.0).all(axis=1)]
        if pts.shape[0] == 0:
            continue
        expected = HV(ref_point=ref)(pts)
        assert hypervolume(pts, reference=(1.0, 1.0)) == pytest.approx(expected, abs=1e-9)


def test_igd_plus_matches_pymoo() -> None:
    pytest.importorskip("pymoo")
    from pymoo.indicators.igd_plus import IGDPlus

    rng = np.random.default_rng(1)
    pf = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    for _ in range(20):
        a = rng.random((rng.integers(1, 8), 2)) * 5.0
        expected = IGDPlus(pf)(a)
        assert igd_plus(a, pf) == pytest.approx(expected, abs=1e-9)
