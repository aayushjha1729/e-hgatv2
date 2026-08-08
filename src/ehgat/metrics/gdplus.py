"""GD+ and IGD+ -- the Pareto-compliant convergence/coverage indicators.

The modified distance of Ishibuchi et al. (2015) counts, between a solution a and a
reference point z (both minimised), only the objectives in which a is worse::

    d+(a, z) = sqrt( sum_i max(a_i - z_i, 0)^2 )

This makes GD+ and IGD+ weakly Pareto compliant (unlike plain GD/IGD): a front that
dominates another can never score worse. Both indicators are zero iff the candidate
matches the reference front, and smaller is better.

- GD+ (convergence): average over candidate points of the d+ to the nearest
  reference point -- "how far is the candidate from PF*".
- IGD+ (coverage + convergence): average over reference points of the d+ from the
  nearest candidate -- "how well does the candidate cover PF*". IGD+ is the primary
  quality indicator, penalising both poor convergence and poor spread.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["gd_plus", "igd_plus"]

Point = Sequence[float]


def _as_array(points: Sequence[Point] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 0)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array [k, M], got shape {arr.shape}")
    return arr


def _plus_distance_matrix(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """D[i, j] = d+(a_i, z_j) -- penalise only objectives where a_i exceeds z_j."""
    diff = a[:, None, :] - z[None, :, :]  # (|A|, |Z|, M)
    excess = np.clip(diff, a_min=0.0, a_max=None)
    return np.sqrt(np.sum(excess**2, axis=2))  # type: ignore[no-any-return]


def gd_plus(
    front: Sequence[Point] | np.ndarray, reference: Sequence[Point] | np.ndarray
) -> float:
    """GD+ of front against the reference front (minimisation; lower is better)."""
    a = _as_array(front, name="front")
    z = _as_array(reference, name="reference")
    if a.shape[0] == 0:
        return float("inf")
    dist = _plus_distance_matrix(a, z)  # d+(a_i, z_j)
    return float(np.mean(dist.min(axis=1)))


def igd_plus(
    front: Sequence[Point] | np.ndarray, reference: Sequence[Point] | np.ndarray
) -> float:
    """IGD+ of front against the reference front (minimisation; lower is better)."""
    a = _as_array(front, name="front")
    z = _as_array(reference, name="reference")
    if a.shape[0] == 0:
        return float("inf")
    # d+ from each reference point z_j to candidate a_i: penalise where a_i exceeds z_j.
    dist = _plus_distance_matrix(a, z)  # (|A|, |Z|); column j -> distances to z_j
    return float(np.mean(dist.min(axis=0)))
