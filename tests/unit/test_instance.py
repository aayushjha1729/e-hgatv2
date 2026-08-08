"""Unit tests for the deterministic 10-task toy instance builder."""

from __future__ import annotations

import pytest

from ehgat.environment.distance import load_default_distance_matrix
from ehgat.environment.instance import (
    AVAILABLE_QCS,
    Instance,
    Task,
    TaskKind,
    build_toy_instance,
    scaled_fleet,
)


def test_toy_instance_topology() -> None:
    inst = build_toy_instance()
    assert inst.num_tasks == 10
    assert inst.qcs == ("QC1", "QC2", "QC3")
    assert inst.num_agvs == 2
    assert inst.agv_start == "LU1"


def test_toy_instance_is_deterministic() -> None:
    a = build_toy_instance(seed=0)
    b = build_toy_instance(seed=0)
    assert a.tasks == b.tasks


def test_toy_instance_seed_changes_handling_times() -> None:
    a = build_toy_instance(seed=0)
    b = build_toy_instance(seed=1)
    # Topology (QC/LU/kind) is fixed; only handling times depend on the seed.
    assert [t.qc for t in a.tasks] == [t.qc for t in b.tasks]
    assert [t.handling_time for t in a.tasks] != [t.handling_time for t in b.tasks]


def test_handling_times_in_paper_range() -> None:
    inst = build_toy_instance()
    assert all(30.0 <= t.handling_time <= 80.0 for t in inst.tasks)


def test_dual_cycling_has_both_kinds() -> None:
    inst = build_toy_instance()
    kinds = {t.kind for t in inst.tasks}
    assert kinds == {TaskKind.LOAD, TaskKind.UNLOAD}


def test_pickup_dropoff_semantics() -> None:
    load = Task(task_id=0, qc="QC1", lu="LU2", kind=TaskKind.LOAD, handling_time=40.0)
    unload = Task(task_id=1, qc="QC1", lu="LU2", kind=TaskKind.UNLOAD, handling_time=40.0)
    # LOAD: yard -> ship, so loaded leg LU -> QC.
    assert (load.pickup, load.dropoff) == ("LU2", "QC1")
    # UNLOAD: ship -> yard, so loaded leg QC -> LU.
    assert (unload.pickup, unload.dropoff) == ("QC1", "LU2")


def test_loaded_distance_matches_table() -> None:
    inst = build_toy_instance()
    dm = load_default_distance_matrix()
    load = Task(task_id=0, qc="QC1", lu="LU3", kind=TaskKind.LOAD, handling_time=40.0)
    assert inst.loaded_distance(load) == dm.distance("LU3", "QC1")


def test_tasks_of_qc_partitions_all_tasks() -> None:
    inst = build_toy_instance()
    grouped = [t.task_id for qc in inst.qcs for t in inst.tasks_of_qc(qc)]
    assert sorted(grouped) == list(range(inst.num_tasks))


def test_rejects_unknown_qc() -> None:
    dm = load_default_distance_matrix()
    bad = Task(task_id=0, qc="QC9", lu="LU1", kind=TaskKind.LOAD, handling_time=40.0)
    with pytest.raises(ValueError, match="unknown QC"):
        Instance(tasks=(bad,), qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)


def test_rejects_noncontiguous_task_ids() -> None:
    dm = load_default_distance_matrix()
    t = Task(task_id=5, qc="QC1", lu="LU1", kind=TaskKind.LOAD, handling_time=40.0)
    with pytest.raises(ValueError, match="contiguous"):
        Instance(tasks=(t,), qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)


def test_rejects_nonpositive_handling_time() -> None:
    dm = load_default_distance_matrix()
    t = Task(task_id=0, qc="QC1", lu="LU1", kind=TaskKind.LOAD, handling_time=0.0)
    with pytest.raises(ValueError, match="handling_time"):
        Instance(tasks=(t,), qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)


def test_scaled_fleet_is_monotone_and_bounded() -> None:
    # AGVs grow with N (never below 2); QCs stay within the 3..6 available cranes.
    prev_agvs = 0
    for n in (5, 10, 20, 50, 100, 160, 200):
        agvs, qcs = scaled_fleet(n)
        assert agvs >= 2
        assert agvs >= prev_agvs  # non-decreasing in N
        assert 3 <= qcs <= len(AVAILABLE_QCS)
        prev_agvs = agvs
    # Small instances reproduce the historical (2 AGV, 3 QC) toy for comparability.
    assert scaled_fleet(10) == (2, 3)
    assert scaled_fleet(20) == (2, 3)


def test_scaled_fleet_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="num_tasks"):
        scaled_fleet(0)


def test_large_scaled_instance_builds_and_validates() -> None:
    # N=200 must build a valid instance: fleet from scaled_fleet, QCs within the matrix.
    agvs, qcs = scaled_fleet(200)
    inst = build_toy_instance(num_tasks=200, num_agvs=agvs, qcs=AVAILABLE_QCS[:qcs])
    assert inst.num_tasks == 200
    assert inst.num_agvs == agvs
    assert len(inst.qcs) == qcs
    # Every task references a QC that exists in the distance matrix (no validation error).
    grouped = [t.task_id for qc in inst.qcs for t in inst.tasks_of_qc(qc)]
    assert sorted(grouped) == list(range(200))
