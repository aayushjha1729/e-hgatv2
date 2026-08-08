"""NSGA-II multi-objective ranking primitives (Deb et al., 2002).

These are the shared building blocks for both the BRKGA baseline and the
attention-guided search: fast non-dominated sorting partitions a set of
objective vectors into Pareto fronts, and crowding distance measures diversity
within a front. All objectives are minimised.

The functions operate on plain objective tuples ((makespan, energy, ...)) so they
are decoupled from the scheduling environment and trivially unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "crowding_distance",
    "dominates",
    "fast_non_dominated_sort",
    "non_dominated_indices",
    "nsga2_sort",
    "order_by_rank_crowding",
]

Objective = Sequence[float]

# Above this population size the vectorised dominance peel beats Deb's Python double loop
# by ~100x (it was the NSGA-II inner-loop bottleneck: ~2.9 s at N=2000). The N^2 boolean
# matrix is only used past this threshold so tiny fronts keep the exact reference path.
_VECTORISED_MIN_N = 64


def dominates(p: Objective, q: Objective) -> bool:
    """Return True if p Pareto-dominates q (minimisation).

    p dominates q iff it is no worse in every objective and strictly better in
    at least one.
    """
    strictly_better = False
    for a, b in zip(p, q, strict=True):
        if a > b:
            return False
        if a < b:
            strictly_better = True
    return strictly_better


def _fast_non_dominated_sort_py(objectives: Sequence[Objective]) -> list[list[int]]:
    """Deb's O(M N^2) reference algorithm (kept for small N and as the semantic oracle)."""
    n = len(objectives)
    dominated_by: list[list[int]] = [[] for _ in range(n)]  # solutions each p dominates
    domination_count = [0] * n  # how many dominate p
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # trailing empty front
    return fronts


def fast_non_dominated_sort(objectives: Sequence[Objective]) -> list[list[int]]:
    """Partition indices into Pareto fronts (front 0 is non-dominated).

    Returns a list of fronts, each a list of indices into objectives; earlier fronts
    dominate later ones. Small N uses Deb's reference loop directly; large N
    vectorises only the O(M N^2) dominance matrix / domination counts (the part that
    was the dominant per-generation cost of the (mu + lambda) selection at N >= 1000
    -- ~2.9 s at N=2000) and then runs the identical reference peel over them. The output
    is therefore bit-identical to _fast_non_dominated_sort_py (same fronts, same
    within-front order), so search trajectories are unchanged -- only ~20x faster.
    """
    n = len(objectives)
    if n == 0:
        return []
    if n < _VECTORISED_MIN_N:
        return _fast_non_dominated_sort_py(objectives)

    obj = np.asarray(objectives, dtype=float)  # [N, M]
    # dom[p, q] := p dominates q  (no worse in every objective, strictly better in one).
    no_worse = (obj[:, None, :] <= obj[None, :, :]).all(axis=2)
    strictly = (obj[:, None, :] < obj[None, :, :]).any(axis=2)
    dom = no_worse & strictly
    np.fill_diagonal(dom, False)

    domination_count = dom.sum(axis=0).tolist()  # how many dominate each index
    dominated_by = [np.flatnonzero(row).tolist() for row in dom]  # each in ascending q
    fronts: list[list[int]] = [[p for p in range(n) if domination_count[p] == 0]]
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # trailing empty front
    return fronts


def crowding_distance(objectives: Sequence[Objective], front: Sequence[int]) -> dict[int, float]:
    """Crowding distance for each index in front (boundary points get inf).

    Distances are summed over objectives, each normalised by that objective's range across
    the front, which renders them comparable across objectives of different scales.
    """
    distance = dict.fromkeys(front, 0.0)
    size = len(front)
    if size == 0:
        return distance
    num_obj = len(objectives[front[0]])
    for m in range(num_obj):
        ordered = sorted(front, key=lambda i: objectives[i][m])
        f_min = objectives[ordered[0]][m]
        f_max = objectives[ordered[-1]][m]
        distance[ordered[0]] = math.inf
        distance[ordered[-1]] = math.inf
        if f_max == f_min:
            continue
        span = f_max - f_min
        for k in range(1, size - 1):
            nxt = objectives[ordered[k + 1]][m]
            prv = objectives[ordered[k - 1]][m]
            distance[ordered[k]] += (nxt - prv) / span
    return distance


def order_by_rank_crowding(
    objectives: Sequence[Objective], fronts: Sequence[Sequence[int]]
) -> list[int]:
    """Order indices by ascending front rank, then descending crowding distance."""
    order: list[int] = []
    for front in fronts:
        crowding = crowding_distance(objectives, front)
        order.extend(sorted(front, key=lambda i: -crowding[i]))
    return order


def nsga2_sort(objectives: Sequence[Objective]) -> list[int]:
    """Full NSGA-II ordering: best (rank 0, most crowded-out) first."""
    return order_by_rank_crowding(objectives, fast_non_dominated_sort(objectives))


def non_dominated_indices(objectives: Sequence[Objective]) -> list[int]:
    """Indices of the non-dominated (Pareto-optimal) objective vectors."""
    if not objectives:
        return []
    return fast_non_dominated_sort(objectives)[0]
