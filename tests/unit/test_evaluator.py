"""Unit tests for the makespan/energy evaluator and its timing model.

The custom mini-instances use placeholder distances of 999 on unused legs so a
mis-routed evaluation fails loudly instead of silently using the wrong edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.environment.decoder import NUM_BLOCKS, Schedule, decode
from ehgat.environment.distance import DistanceMatrix
from ehgat.environment.evaluator import ScheduleCycleError, evaluate
from ehgat.environment.instance import Instance, Task, TaskKind
from ehgat.environment.physics import SpeedLevel
from ehgat.utils.seeding import make_rng

NOM = SpeedLevel.NOMINAL


def _matrix(nodes: list[str], rows: list[list[float]]) -> DistanceMatrix:
    return DistanceMatrix(nodes, np.array(rows, dtype=float))


def test_single_load_task_timing_and_energy() -> None:
    # empty LU1->LU2 = 60 m, loaded LU2->QC1 = 90 m, tau = 40 s, nominal speeds.
    dm = _matrix(
        ["QC1", "LU1", "LU2"],
        [
            [0, 999, 999],
            [999, 0, 60],
            [90, 999, 0],
        ],
    )
    task = Task(task_id=0, qc="QC1", lu="LU2", kind=TaskKind.LOAD, handling_time=40.0)
    inst = Instance(tasks=(task,), qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)
    sched = Schedule(
        global_order=(0,),
        assignment=(0,),
        agv_sequences=((0,),),
        qc_sequences=((0,),),
        empty_speed=(NOM,),
        loaded_speed=(NOM,),
    )
    ev = evaluate(sched, inst)
    # empty: 60/6 = 10 s, 10 kW -> 100; loaded: 90/3 = 30 s, 15 kW -> 450.
    assert ev.arr_pickup[0] == pytest.approx(10.0)
    assert ev.arr_dropoff[0] == pytest.approx(40.0)
    assert ev.qc_finish[0] == pytest.approx(80.0)  # delivery 40 + tau 40
    assert ev.makespan == pytest.approx(80.0)
    assert ev.energy == pytest.approx(550.0)
    assert ev.objectives == (ev.makespan, ev.energy)


def test_single_unload_task_timing_and_energy() -> None:
    # empty LU1->QC1 = 120 m, loaded QC1->LU2 = 90 m, tau = 40 s, nominal speeds.
    dm = _matrix(
        ["QC1", "LU1", "LU2"],
        [
            [0, 999, 90],
            [120, 0, 999],
            [999, 999, 0],
        ],
    )
    task = Task(task_id=0, qc="QC1", lu="LU2", kind=TaskKind.UNLOAD, handling_time=40.0)
    inst = Instance(tasks=(task,), qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)
    sched = Schedule(
        global_order=(0,),
        assignment=(0,),
        agv_sequences=((0,),),
        qc_sequences=((0,),),
        empty_speed=(NOM,),
        loaded_speed=(NOM,),
    )
    ev = evaluate(sched, inst)
    # empty: 120/6 = 20 s, 200. First task on QC1, so Eq (15) gives c_j = arrival = 20 with
    # NO tau (Eq 10 charges tau only between consecutive crane tasks); loaded 90/3 = 30 ->
    # delivery r_j = 50.
    assert ev.arr_pickup[0] == pytest.approx(20.0)
    assert ev.qc_finish[0] == pytest.approx(20.0)  # first crane task: no tau floor (Eq 15)
    assert ev.arr_dropoff[0] == pytest.approx(50.0)
    assert ev.makespan == pytest.approx(50.0)  # Cmax >= r_j (delivery) for unload
    assert ev.energy == pytest.approx(650.0)


def test_qc_serialization_forces_wait() -> None:
    # Two LOAD tasks on QC1, one per AGV. Both delivered early, but QC1 is a serial
    # resource so task 1 must wait for task 0's QC op to finish.
    dm = _matrix(
        ["QC1", "LU1", "LU2"],
        [
            [0, 999, 999],
            [60, 0, 30],
            [90, 999, 0],
        ],
    )
    tasks = (
        Task(task_id=0, qc="QC1", lu="LU1", kind=TaskKind.LOAD, handling_time=40.0),
        Task(task_id=1, qc="QC1", lu="LU2", kind=TaskKind.LOAD, handling_time=40.0),
    )
    inst = Instance(tasks=tasks, qcs=("QC1",), num_agvs=2, agv_start="LU1", distance=dm)
    sched = Schedule(
        global_order=(0, 1),
        assignment=(0, 1),
        agv_sequences=((0,), (1,)),
        qc_sequences=((0, 1),),
        empty_speed=(NOM, NOM),
        loaded_speed=(NOM, NOM),
    )
    ev = evaluate(sched, inst)
    # task0: delivery 0+0+20 = 20 -> finish 60. task1: delivery 0+5+30 = 35, but QC
    # busy until 60 -> finish max(35,60)+40 = 100.
    assert ev.completion[0] == pytest.approx(60.0)
    assert ev.completion[1] == pytest.approx(100.0)
    assert ev.makespan == pytest.approx(100.0)
    assert ev.energy == pytest.approx(800.0)


def test_deadlock_raises_cycle_error() -> None:
    # AGV order says 1 before 0; QC order says 0 before 1 -> precedence cycle.
    dm = _matrix(
        ["QC1", "LU1"],
        [
            [0, 50],
            [50, 0],
        ],
    )
    tasks = (
        Task(task_id=0, qc="QC1", lu="LU1", kind=TaskKind.LOAD, handling_time=30.0),
        Task(task_id=1, qc="QC1", lu="LU1", kind=TaskKind.LOAD, handling_time=30.0),
    )
    inst = Instance(tasks=tasks, qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)
    deadlocked = Schedule(
        global_order=(0, 1),
        assignment=(0, 0),
        agv_sequences=((1, 0),),
        qc_sequences=((0, 1),),
        empty_speed=(NOM, NOM),
        loaded_speed=(NOM, NOM),
    )
    with pytest.raises(ScheduleCycleError, match="deadlock"):
        evaluate(deadlocked, inst)


def test_lower_speed_trades_energy_for_time() -> None:
    dm = _matrix(
        ["QC1", "LU1", "LU2"],
        [
            [0, 999, 999],
            [999, 0, 60],
            [90, 999, 0],
        ],
    )
    task = Task(task_id=0, qc="QC1", lu="LU2", kind=TaskKind.LOAD, handling_time=40.0)
    inst = Instance(tasks=(task,), qcs=("QC1",), num_agvs=1, agv_start="LU1", distance=dm)

    def sched_with(loaded: SpeedLevel) -> Schedule:
        return Schedule(
            global_order=(0,),
            assignment=(0,),
            agv_sequences=((0,),),
            qc_sequences=((0,),),
            empty_speed=(NOM,),
            loaded_speed=(loaded,),
        )

    low = evaluate(sched_with(SpeedLevel.LOWER), inst)
    high = evaluate(sched_with(SpeedLevel.HIGHER), inst)
    assert low.energy < high.energy  # slower loaded leg uses less energy
    assert low.makespan > high.makespan  # but takes longer


def test_objectives_match_returned_arrays_on_toy() -> None:
    from ehgat.environment.instance import build_toy_instance

    inst = build_toy_instance()
    keys = make_rng(42).random(NUM_BLOCKS * inst.num_tasks)
    ev = evaluate(decode(keys, inst), inst)
    assert ev.makespan == max(ev.completion)
    assert ev.energy == pytest.approx(sum(ev.empty_energy) + sum(ev.loaded_energy))


def test_decoded_toy_schedules_never_deadlock() -> None:
    from ehgat.environment.instance import build_toy_instance

    inst = build_toy_instance()
    for seed in range(25):
        keys = make_rng(seed).random(NUM_BLOCKS * inst.num_tasks)
        ev = evaluate(decode(keys, inst), inst)
        assert ev.makespan > 0.0
        assert ev.energy > 0.0
        assert len(ev.topo_order) == inst.num_tasks


def test_evaluation_is_deterministic_on_toy() -> None:
    from ehgat.environment.instance import build_toy_instance

    inst = build_toy_instance()
    keys = make_rng(1).random(NUM_BLOCKS * inst.num_tasks)
    sched = decode(keys, inst)
    assert evaluate(sched, inst).objectives == evaluate(sched, inst).objectives
