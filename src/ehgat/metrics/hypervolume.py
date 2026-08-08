"""Exact 2-D hypervolume and a shared reference point for bi-objective fronts.

Hypervolume (HV) is the area of objective space dominated by a front and bounded by a
reference point r. For minimisation r must be weakly dominated by every point, that is,
no smaller than the worst values; a larger HV then corresponds to a front that is both
closer to the true front and better spread, and is the scalar used for the convergence
curve.

The bi-objective case admits an exact O(k log k) sweep, so no Monte-Carlo estimate
arises. For a fair comparison across methods the same reference point must be used;
nadir_reference derives it from the nadir of one or more fronts (typically the
exact golden PF*) plus a relative margin.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["hypervolume", "nadir_reference"]

Point = Sequence[float]


def _as_points(front: Sequence[Point] | np.ndarray) -> np.ndarray:
    arr = np.asarray(front, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected a 2-D front of shape [k, 2], got {arr.shape}")
    return arr


def nadir_reference(
    *fronts: Sequence[Point] | np.ndarray, margin: float = 0.1
) -> tuple[float, float]:
    """Reference point = component-wise nadir of all fronts inflated by margin.

    margin extends the reference beyond the worst objective values by a fraction of
    each objective's spread across the supplied fronts, so boundary solutions still
    contribute positive volume. Pass the golden front (and optionally the candidate
    fronts) to fix a single comparable reference.
    """
    stacked = np.vstack([_as_points(f) for f in fronts if len(f) > 0])
    if stacked.size == 0:
        raise ValueError("at least one non-empty front is required")
    worst = stacked.max(axis=0)
    best = stacked.min(axis=0)
    span = np.where(worst > best, worst - best, np.abs(worst) + 1.0)
    ref = worst + margin * span
    return float(ref[0]), float(ref[1])


def _non_dominated_min(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset (minimisation), sorted ascending by objective 0."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]
    keep: list[np.ndarray] = []
    best_y = np.inf
    for row in ordered:
        if row[1] < best_y:  # strictly improves objective 1 as objective 0 grows
            keep.append(row)
            best_y = float(row[1])
    return np.asarray(keep, dtype=float)


def hypervolume(
    front: Sequence[Point] | np.ndarray, reference: tuple[float, float]
) -> float:
    """Exact 2-D hypervolume of front w.r.t. reference (minimisation).

    Points not strictly better than reference in both objectives contribute nothing.
    Returns 0.0 for an empty (or fully dominated-by-reference) front.
    """
    pts = _as_points(front)
    ref_x, ref_y = reference
    mask = (pts[:, 0] < ref_x) & (pts[:, 1] < ref_y)
    pts = pts[mask]
    if pts.shape[0] == 0:
        return 0.0
    nd = _non_dominated_min(pts)  # ascending x, strictly descending y
    xs = nd[:, 0]
    ys = nd[:, 1]
    x_next = np.empty_like(xs)
    x_next[:-1] = xs[1:]
    x_next[-1] = ref_x
    # Vertical strips: width (x_next - x) x height (ref_y - y), summed left to right.
    return float(np.sum((x_next - xs) * (ref_y - ys)))
