"""Unit tests for NSGA-II ranking primitives."""

from __future__ import annotations

import math

from ehgat.search.nsga2 import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    non_dominated_indices,
    nsga2_sort,
)


def test_dominates_basic() -> None:
    assert dominates((1.0, 2.0), (2.0, 3.0))  # better in both
    assert dominates((1.0, 2.0), (1.0, 3.0))  # equal in one, better in other
    assert not dominates((1.0, 2.0), (1.0, 2.0))  # identical -> no domination
    assert not dominates((1.0, 3.0), (2.0, 2.0))  # trade-off -> neither dominates
    assert not dominates((2.0, 3.0), (1.0, 2.0))


def test_fast_non_dominated_sort_two_fronts() -> None:
    # Front 0: (1,4),(2,2),(4,1). Front 1: (3,3) dominated by (2,2); (5,5) dominated by all.
    objs = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0), (5.0, 5.0)]
    fronts = fast_non_dominated_sort(objs)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert sorted(fronts[1]) == [3]
    assert sorted(fronts[2]) == [4]


def test_non_dominated_indices_matches_front0() -> None:
    objs = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0)]
    assert sorted(non_dominated_indices(objs)) == [0, 1, 2]
    assert non_dominated_indices([]) == []


def test_crowding_distance_boundaries_are_infinite() -> None:
    objs = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    front = [0, 1, 2]
    cd = crowding_distance(objs, front)
    # Extremes in makespan (0 and 2) are boundary -> infinite.
    assert cd[0] == math.inf
    assert cd[2] == math.inf
    assert cd[1] < math.inf  # interior point is finite


def test_crowding_distance_single_and_pair() -> None:
    assert crowding_distance([(1.0, 1.0)], [0]) == {0: math.inf}
    cd = crowding_distance([(1.0, 2.0), (2.0, 1.0)], [0, 1])
    assert cd[0] == math.inf and cd[1] == math.inf


def test_nsga2_sort_orders_by_rank_then_crowding() -> None:
    objs = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0), (5.0, 5.0)]
    order = nsga2_sort(objs)
    # All front-0 indices precede front-1, which precedes front-2.
    assert set(order[:3]) == {0, 1, 2}
    assert order[3] == 3
    assert order[4] == 4
    # Within front 0, boundary points (0 and 2) precede the interior point (1).
    assert order.index(1) > min(order.index(0), order.index(2))
