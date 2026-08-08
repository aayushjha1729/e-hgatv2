"""Diversity indicators: Deb's spread Delta and Schott's spacing.

These measure how uniformly a front's points are distributed (orthogonal to the
convergence captured by HV/IGD+). Both are bi-objective formulations.

- Spread Delta (Deb et al., 2002): combines the gap to the true extreme points
  with the variance of consecutive gaps. Delta = 0 is a perfectly uniform front that
  also reaches both extremes; lower is better.
- Spacing (Schott, 1995): standard deviation of nearest-neighbour distances;
  independent of the extremes, lower is better.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["spacing", "spread"]

Point = Sequence[float]


def _as_points(front: Sequence[Point] | np.ndarray) -> np.ndarray:
    arr = np.asarray(front, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected a 2-D front of shape [k, 2], got {arr.shape}")
    return arr


def spread(
    front: Sequence[Point] | np.ndarray,
    reference: Sequence[Point] | np.ndarray | None = None,
) -> float:
    """Deb's Delta spread of front (lower = more uniform; reaches extremes).

    Consecutive distances are taken along the front sorted by objective 0. d_f/d_l
    are the gaps from the front's boundary solutions to the reference front's extreme
    points (the golden PF* extremes if reference is given, else the front's own
    extremes -> 0). Returns 0.0 for fewer than two points.
    """
    pts = _as_points(front)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    ordered = pts[np.argsort(pts[:, 0])]
    consecutive = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    d_mean = float(consecutive.mean())

    if reference is not None and len(reference) > 0:
        ref = _as_points(reference)
        ref_sorted = ref[np.argsort(ref[:, 0])]
        d_f = float(np.linalg.norm(ordered[0] - ref_sorted[0]))
        d_l = float(np.linalg.norm(ordered[-1] - ref_sorted[-1]))
    else:
        d_f = d_l = 0.0

    numerator = d_f + d_l + float(np.sum(np.abs(consecutive - d_mean)))
    denominator = d_f + d_l + (n - 1) * d_mean
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def spacing(front: Sequence[Point] | np.ndarray) -> float:
    """Schott's spacing: std-dev of nearest-neighbour L1 distances (lower = more uniform).

    Returns 0.0 for fewer than two points.
    """
    pts = _as_points(front)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    l1 = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)  # (n, n)
    np.fill_diagonal(l1, np.inf)
    nearest = l1.min(axis=1)
    d_bar = float(nearest.mean())
    return float(np.sqrt(np.mean((nearest - d_bar) ** 2)))
