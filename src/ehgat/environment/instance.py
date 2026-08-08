"""Deterministic dual-cycling container-terminal instance.

Builds the canonical 10-task instance used throughout the study. Grounded in
Homayouni & Fontes (2022), Section 5 and Table 5:

- Tasks belong to Quay Cranes (QCs). Each task is either a LOAD (export:
  yard -> ship) or an UNLOAD (import: ship -> yard) operation.
- A LOAD moves a container from an LU (storage) station to the QC, where the QC
  lifts it onto the ship. The AGV's loaded leg is therefore LU -> QC.
- An UNLOAD moves a container the QC lifts off the ship onto a waiting AGV, which
  carries it to an LU station. The loaded leg is therefore QC -> LU.
- Each QC handles its containers one at a time; handling time tau ~ U(30, 80)
  seconds (seeded -> deterministic).
- AGVs start parked at LU station 1.

The instance is small and fully reproducible, within reach of the brute-force Oracle
(oracle.py) enumerating the exact Pareto front.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ehgat.environment.distance import DistanceMatrix, load_default_distance_matrix
from ehgat.environment.physics import SPEED_TABLE
from ehgat.utils.seeding import make_rng

# Largest power any single travel leg can draw (loaded leg at the highest speed level).
# A peak-power budget below this renders some legs individually infeasible; it is the
# hard lower bound for a valid Instance.peak_power.
_MAX_LEG_POWER = max(spec.loaded_power for spec in SPEED_TABLE.values())

__all__ = [
    "AVAILABLE_QCS",
    "EXACT_TOY_TASKS",
    "Instance",
    "Task",
    "TaskKind",
    "build_toy_instance",
    "scaled_fleet",
]

# Quay cranes available in the packaged Table-4 distance matrix (QC1..QC6). A QC label
# must be a node in the distance matrix, which bounds the usable fleet to these.
AVAILABLE_QCS: tuple[str, ...] = ("QC1", "QC2", "QC3", "QC4", "QC5", "QC6")

# Task count for the canonical exact instance. Even with the smart speed Pareto DP
# (which removes the 3^(2N) speed factor), the structure space grows as ~ (N+1)!. The
# oracle is therefore a compute-once artifact whose front is frozen as a golden JSON.
# N=5 runs in
# well under a minute on commodity hardware; the N=10 instance is for scaling studies.
EXACT_TOY_TASKS = 5


class TaskKind(IntEnum):
    """Container move type in a dual-cycling terminal."""

    LOAD = 0  # export: yard (LU) -> ship; loaded leg LU -> QC
    UNLOAD = 1  # import: ship -> yard (LU); loaded leg QC -> LU


@dataclass(frozen=True, slots=True)
class Task:
    """A single container move bound to one QC and one LU station.

    pickup/dropoff are the origin/destination of the AGV's loaded leg and
    are derived from TaskKind.
    """

    task_id: int
    qc: str
    lu: str
    kind: TaskKind
    handling_time: float  # tau (s): QC operation time for this container

    @property
    def pickup(self) -> str:
        """Node where the AGV acquires the container (start of the loaded leg)."""
        return self.lu if self.kind is TaskKind.LOAD else self.qc

    @property
    def dropoff(self) -> str:
        """Node where the AGV releases the container (end of the loaded leg)."""
        return self.qc if self.kind is TaskKind.LOAD else self.lu


@dataclass(frozen=True, slots=True)
class Instance:
    """An immutable, validated dual-cycling scheduling instance."""

    tasks: tuple[Task, ...]
    qcs: tuple[str, ...]
    num_agvs: int
    agv_start: str
    distance: DistanceMatrix
    # Fleet-wide instantaneous power budget (kW): the sum of the powers of all AGV
    # travel legs running at any instant may not exceed this. None (default) means
    # uncoupled physics -- the original max-plus longest-path model. When set, makespan
    # is resolved by the power-coupled event simulator (see evaluator.py), which has
    # no closed form; the differentiable surrogate is used in place of one.
    peak_power: float | None = None

    def __post_init__(self) -> None:
        if self.num_agvs < 1:
            raise ValueError(f"num_agvs must be >= 1, got {self.num_agvs}")
        if not self.tasks:
            raise ValueError("instance must contain at least one task")
        if self.peak_power is not None and self.peak_power < _MAX_LEG_POWER:
            raise ValueError(
                f"peak_power {self.peak_power} kW is below the max single-leg power "
                f"{_MAX_LEG_POWER} kW; some legs would be individually infeasible"
            )

        nodes = set(self.distance.nodes)
        if self.agv_start not in nodes:
            raise ValueError(f"agv_start {self.agv_start!r} is not a known node")
        for qc in self.qcs:
            if qc not in nodes:
                raise ValueError(f"QC {qc!r} is not a known node")

        qc_set = set(self.qcs)
        seen_ids: set[int] = set()
        for t in self.tasks:
            if t.task_id in seen_ids:
                raise ValueError(f"duplicate task_id {t.task_id}")
            seen_ids.add(t.task_id)
            if t.qc not in qc_set:
                raise ValueError(f"task {t.task_id} references unknown QC {t.qc!r}")
            if t.lu not in nodes:
                raise ValueError(f"task {t.task_id} references unknown LU {t.lu!r}")
            if t.handling_time <= 0.0:
                raise ValueError(f"task {t.task_id} handling_time must be > 0")
        # task_ids must be a contiguous 0..N-1 range so they can index tuples/arrays.
        if seen_ids != set(range(len(self.tasks))):
            raise ValueError("task_ids must be a contiguous range 0..N-1")

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def task(self, task_id: int) -> Task:
        return self.tasks[task_id]

    def tasks_of_qc(self, qc: str) -> tuple[Task, ...]:
        """Tasks bound to qc in ascending task_id order."""
        return tuple(t for t in self.tasks if t.qc == qc)

    def loaded_distance(self, task: Task) -> float:
        """Distance (m) of the loaded leg pickup -> dropoff for task."""
        return self.distance.distance(task.pickup, task.dropoff)

    def empty_distance(self, origin: str, task: Task) -> float:
        """Distance (m) of the empty leg origin -> task.pickup."""
        return self.distance.distance(origin, task.pickup)


def build_toy_instance(
    seed: int = 0,
    *,
    num_tasks: int = 10,
    qcs: tuple[str, ...] = ("QC1", "QC2", "QC3"),
    num_agvs: int = 2,
    peak_power: float | None = None,
) -> Instance:
    """Construct a deterministic small dual-cycling instance.

    Defaults give the N=10 scaling instance (3 QCs served by 2 SA-AGVs starting
    at LU1). Pass num_tasks=EXACT_TOY_TASKS for the small instance whose exact
    Pareto front the oracle can enumerate.

    Tasks alternate LOAD/UNLOAD for genuine dual-cycling; QC bindings and LU stations
    follow a fixed round-robin pattern; handling times are drawn ~ U(30, 80) seconds from
    a seeded RNG, which makes the entire instance reproducible from the seed alone.
    """
    rng = make_rng(seed)
    distance = load_default_distance_matrix()
    lu_stations = ("LU1", "LU2", "LU3", "LU4", "LU5", "LU6")

    tasks: list[Task] = []
    for task_id in range(num_tasks):
        qc = qcs[task_id % len(qcs)]
        lu = lu_stations[task_id % len(lu_stations)]
        kind = TaskKind.LOAD if task_id % 2 == 0 else TaskKind.UNLOAD
        handling_time = float(int(rng.integers(30, 81)))  # U[30, 80] inclusive seconds
        tasks.append(
            Task(task_id=task_id, qc=qc, lu=lu, kind=kind, handling_time=handling_time)
        )

    return Instance(
        tasks=tuple(tasks),
        qcs=qcs,
        num_agvs=num_agvs,
        agv_start="LU1",
        distance=distance,
        peak_power=peak_power,
    )


def scaled_fleet(num_tasks: int) -> tuple[int, int]:
    """Deterministic (num_agvs, num_qcs) fleet sizing for an N-task instance.

    Container terminals run a small, fixed set of quay cranes but an AGV fleet that grows
    with throughput. The fleet is accordingly a pure function of N:

    - num_agvs = max(2, round(N / 12)) -- roughly one AGV per twelve container moves
      (floor of 2 so even tiny instances dual-cycle).
    - num_qcs  = clamp(round(N / 40), 3, 6) -- grows slowly from 3 to the 6 cranes
      present in the packaged distance matrix (AVAILABLE_QCS).

    Being a pure function of N is essential: the search instance and its BRKGA
    reference front are built in separate processes (see benchmark.runner), and
    they must be byte-for-byte the same instance. For N <= ~20 this reproduces the
    historical (2 AGVs, 3 QCs) configuration, keeping small-N results comparable.
    """
    if num_tasks < 1:
        raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
    num_agvs = max(2, round(num_tasks / 12))
    num_qcs = min(len(AVAILABLE_QCS), max(3, round(num_tasks / 40)))
    return num_agvs, num_qcs
