"""Schedule evaluator: makespan C_max and total AGV energy E.

This is the physical ground truth every other component is measured against, so
the timing model is stated explicitly. It is a faithful forward evaluation of the
single-indexed MILP of Fontes & Homayouni (2023) -- the timing constraints Eqs (2)-(4)
and (10)-(18) with the schedule's binaries fixed -- cross-validated to 0 discrepancy
against two independent implementations: a literal, line-by-line transcription of the
MILP timing constraints solved by fixed-point longest-path iteration
(scripts/verify_timing.py), and exhaustive 3^(2N) enumeration of the speed
assignments against the exact oracle (tests/unit/test_oracle.py).

Per task j (assigned to one AGV; its QC processes containers one at a time).
qc_ready is the completion c of the previous task on j's crane (Eq 10):

- The AGV first drives empty from its current position to pickup(j), then
  loaded from pickup(j) to dropoff(j).
- LOAD (export, yard -> ship): loaded leg LU -> QC. The AGV delivers the
  container to the quay (r_j) and is held until the QC lifts it off (Eq 11/12)::

      arr_dropoff = agv_ready + empty_time + loaded_time          # r_j: delivery to QC
      c_j         = max(arr_dropoff, qc_ready) + tau              # Eq 11/10: QC completion
      completion  = c_j                                           # Eq 2: Cmax >= c_j
      agv_free    = c_j - tau = max(arr_dropoff, qc_ready)        # Eq 12: released at pickup

- UNLOAD (import, ship -> yard): loaded leg QC -> LU. The QC handling overlaps
  the AGV's travel (Eqs 14/15/17 carry no +tau on arrival; the +tau accrues
  only along the QC chain via Eq 10, leaving the first task on a crane uncharged)::

      arr_pickup  = agv_ready + empty_time                        # AGV reaches QC
      c_j         = max(arr_pickup, qc_ready + tau)               # Eq 14/15/17 + 10
      arr_dropoff = c_j + loaded_time                             # r_j: delivery to yard (Eq 16)
      completion  = arr_dropoff                                   # Eq 3: Cmax >= r_j
      agv_free    = arr_dropoff                                   # released at the yard

A LOAD frees its AGV at the QC pickup (c_j - tau); an UNLOAD frees it at the yard
(arr_dropoff).

C_max = max_j completion[j]. Energy E = sum_j (empty_energy[j] + loaded_energy[j]).

Because makespan is a maximum over sums along AGV/QC precedence chains, it is exactly
a max-plus longest path over the schedule's precedence DAG -- the physics the
E-HGATv2 surrogate is built to embed. Evaluation processes tasks in a topological
order obtained from Kahn's algorithm over the precedence edges
{agv_prev -> j, qc_prev -> j}; if no topological order exists the schedule
contains an AGV/QC deadlock and ScheduleCycleError is raised.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import pairwise

from ehgat.environment.decoder import Schedule
from ehgat.environment.instance import Instance, TaskKind
from ehgat.environment.physics import SPEED_TABLE, leg_energy, travel_time

_EPS = 1e-9

__all__ = ["Evaluation", "ScheduleCycleError", "build_precedence", "evaluate"]


class ScheduleCycleError(ValueError):
    """Raised when AGV and QC precedence chains form a deadlock (no topo order)."""


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Resolved timings, leg costs and objectives for one schedule."""

    makespan: float
    energy: float
    completion: tuple[float, ...]
    qc_finish: tuple[float, ...]
    arr_pickup: tuple[float, ...]
    arr_dropoff: tuple[float, ...]
    agv_free_after: tuple[float, ...]
    empty_time: tuple[float, ...]
    loaded_time: tuple[float, ...]
    empty_energy: tuple[float, ...]
    loaded_energy: tuple[float, ...]
    topo_order: tuple[int, ...]
    # Resolved travel-leg start times (filled by the power-coupled simulator; under the
    # uncoupled model each leg starts as early as precedence allows). power_arcs are
    # the extra precedences the power budget induced, each as
    # (blocker_task, blocker_loaded, blocked_task, blocked_loaded) with the leg flag
    # 0=empty, 1=loaded -- the structure that has no closed form and that TAPE attributes
    # over (a power-delayed leg could only start once the blocker leg freed the budget).
    empty_start: tuple[float, ...] = ()
    loaded_start: tuple[float, ...] = ()
    power_arcs: tuple[tuple[int, int, int, int], ...] = ()
    # Per-leg power wait = realized leg start minus its precedence-ready time (>=0). Adding
    # it to the leg's travel time as an effective duration lets the precedence-only
    # activity DAG reproduce the coupled makespan exactly -- a continuous, learnable target
    # for the surrogate (no need to predict discrete resolution arcs).
    wait_empty: tuple[float, ...] = ()
    wait_loaded: tuple[float, ...] = ()

    @property
    def objectives(self) -> tuple[float, float]:
        """Bi-objective vector (C_max, E) (both minimised)."""
        return (self.makespan, self.energy)


def _predecessors(sequences: tuple[tuple[int, ...], ...], n: int) -> list[int]:
    """Map task_id -> immediate predecessor in its chain (or -1 if first)."""
    prev = [-1] * n
    for seq in sequences:
        for earlier, later in pairwise(seq):
            prev[later] = earlier
    return prev


def _topological_order(agv_prev: list[int], qc_prev: list[int], n: int) -> tuple[int, ...]:
    """Kahn's algorithm over {agv_prev -> j, qc_prev -> j} edges.

    Ties are broken by ascending task_id (min-heap) for deterministic ordering.
    Raises ScheduleCycleError if a topological order does not exist.
    """
    indegree = [0] * n
    successors: list[list[int]] = [[] for _ in range(n)]
    for j in range(n):
        for p in (agv_prev[j], qc_prev[j]):
            if p >= 0:
                successors[p].append(j)
                indegree[j] += 1

    ready = [j for j in range(n) if indegree[j] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        j = heapq.heappop(ready)
        order.append(j)
        for s in successors[j]:
            indegree[s] -= 1
            if indegree[s] == 0:
                heapq.heappush(ready, s)

    if len(order) != n:
        raise ScheduleCycleError(
            f"schedule has an AGV/QC precedence deadlock: only {len(order)}/{n} "
            "tasks could be ordered (a cycle exists in the precedence graph)."
        )
    return tuple(order)


def build_precedence(
    agv_sequences: tuple[tuple[int, ...], ...],
    qc_sequences: tuple[tuple[int, ...], ...],
    n: int,
) -> tuple[list[int], list[int], tuple[int, ...]]:
    """Resolve (agv_prev, qc_prev, topo_order) for the given chains.

    agv_prev[j]/qc_prev[j] are immediate predecessors (-1 if first).
    topo_order is a deterministic Kahn ordering; raises
    ScheduleCycleError on a deadlock. Shared by the evaluator and the oracle.
    """
    agv_prev = _predecessors(agv_sequences, n)
    qc_prev = _predecessors(qc_sequences, n)
    order = _topological_order(agv_prev, qc_prev, n)
    return agv_prev, qc_prev, order


def evaluate(schedule: Schedule, instance: Instance) -> Evaluation:
    """Compute (C_max, E) and full timings for schedule on instance.

    Dispatches on instance.peak_power: None runs the exact max-plus
    longest-path recurrence; a finite budget runs the power-coupled event simulator
    (_evaluate_power_coupled), whose makespan has no closed form.
    """
    if instance.peak_power is None:
        return _evaluate_uncoupled(schedule, instance)
    return _evaluate_power_coupled(schedule, instance)


def _evaluate_uncoupled(schedule: Schedule, instance: Instance) -> Evaluation:
    """Exact max-plus longest-path evaluation (no peak-power coupling)."""
    n = instance.num_tasks
    agv_prev, qc_prev, order = build_precedence(
        schedule.agv_sequences, schedule.qc_sequences, n
    )

    completion = [0.0] * n
    qc_finish = [0.0] * n
    arr_pickup = [0.0] * n
    arr_dropoff = [0.0] * n
    agv_free_after = [0.0] * n
    empty_time = [0.0] * n
    loaded_time = [0.0] * n
    empty_energy = [0.0] * n
    loaded_energy = [0.0] * n

    for j in order:
        task = instance.tasks[j]

        ap = agv_prev[j]
        if ap < 0:
            agv_ready = 0.0
            origin = instance.agv_start
        else:
            agv_ready = agv_free_after[ap]
            origin = instance.tasks[ap].dropoff

        empty_dist = instance.distance.distance(origin, task.pickup)
        loaded_dist = instance.loaded_distance(task)
        empty_time[j] = travel_time(empty_dist, schedule.empty_speed[j], loaded=False)
        loaded_time[j] = travel_time(loaded_dist, schedule.loaded_speed[j], loaded=True)
        empty_energy[j] = leg_energy(empty_dist, schedule.empty_speed[j], loaded=False)
        loaded_energy[j] = leg_energy(loaded_dist, schedule.loaded_speed[j], loaded=True)

        qp = qc_prev[j]
        qc_ready = qc_finish[qp] if qp >= 0 else 0.0

        arr_pickup[j] = agv_ready + empty_time[j]
        if task.kind is TaskKind.LOAD:
            # AGV delivers to the quay (r_j, Eq 12/13/18); the QC then lifts onto the ship
            # (Eq 11: c_j >= r_j + tau; Eq 10: c_j >= qc_ready + tau). The AGV is held at the
            # quay until the QC takes the container and is released at c_j - tau =
            # max(arrival, qc_ready), the term Eqs 12/17 apply to the next task (not
            # arrival).
            arr_dropoff[j] = arr_pickup[j] + loaded_time[j]  # r_j
            qc_finish[j] = max(arr_dropoff[j], qc_ready) + task.handling_time  # c_j (Eq 10/11)
            completion[j] = qc_finish[j]  # Eq 2: C_max >= c_j
            agv_free_after[j] = qc_finish[j] - task.handling_time  # Eq 12: c_i - tau_i
        else:
            # QC lifts off the ship onto the waiting AGV, then the AGV carries it to the yard.
            # Eq 14/15/17 bound c_j by the AGV arrival with NO +tau (the QC handling overlaps
            # the AGV's travel); the handling is charged only along the QC chain (Eq 10:
            # c_j >= qc_ready + tau). So c_j = max(arrival, qc_ready + tau), with no QC-chain
            # floor for the first task on a crane (qc_prev < 0).
            qc_floor = qc_ready + task.handling_time if qp >= 0 else 0.0
            qc_finish[j] = max(arr_pickup[j], qc_floor)  # c_j (Eq 10/14/15/17)
            arr_dropoff[j] = qc_finish[j] + loaded_time[j]  # r_j (Eq 16)
            completion[j] = arr_dropoff[j]  # Eq 3: C_max >= r_j
            agv_free_after[j] = arr_dropoff[j]  # r_j (Eq 14/18)

    makespan = max(completion)
    energy = sum(empty_energy) + sum(loaded_energy)

    return Evaluation(
        makespan=makespan,
        energy=energy,
        completion=tuple(completion),
        qc_finish=tuple(qc_finish),
        arr_pickup=tuple(arr_pickup),
        arr_dropoff=tuple(arr_dropoff),
        agv_free_after=tuple(agv_free_after),
        empty_time=tuple(empty_time),
        loaded_time=tuple(loaded_time),
        empty_energy=tuple(empty_energy),
        loaded_energy=tuple(loaded_energy),
        topo_order=order,
    )


def _evaluate_power_coupled(schedule: Schedule, instance: Instance) -> Evaluation:
    """Event-driven evaluation under a fleet-wide peak-power budget.

    Each AGV travel leg (empty, then loaded) draws its kinematic power for its whole
    duration; QC handling draws none. A leg may only start once its precedence is met
    and starting it keeps the sum of concurrently-running leg powers at or below
    instance.peak_power. Otherwise it waits until a running leg finishes and frees
    enough budget. The resolution is a deterministic parallel schedule-generation
    scheme (priority: earliest ready, then task_id, then empty-before-loaded), so
    (C_max, E) is a well-defined, exact function of the schedule -- but it is not
    a max-plus longest path: the power budget injects resource delays no closed form
    captures. Energy is unchanged (it does not depend on the timing).

    With peak_power large enough that every leg fits immediately this reduces to
    _evaluate_uncoupled (no leg ever waits), which the tests assert.
    """
    n = instance.num_tasks
    p_max = float(instance.peak_power)  # type: ignore[arg-type]
    agv_prev, qc_prev, order = build_precedence(
        schedule.agv_sequences, schedule.qc_sequences, n
    )

    empty_time = [0.0] * n
    loaded_time = [0.0] * n
    empty_energy = [0.0] * n
    loaded_energy = [0.0] * n
    empty_power = [0.0] * n
    loaded_power = [0.0] * n
    for j in range(n):
        task = instance.tasks[j]
        ap = agv_prev[j]
        origin = instance.agv_start if ap < 0 else instance.tasks[ap].dropoff
        empty_dist = instance.distance.distance(origin, task.pickup)
        loaded_dist = instance.loaded_distance(task)
        empty_time[j] = travel_time(empty_dist, schedule.empty_speed[j], loaded=False)
        loaded_time[j] = travel_time(loaded_dist, schedule.loaded_speed[j], loaded=True)
        empty_energy[j] = leg_energy(empty_dist, schedule.empty_speed[j], loaded=False)
        loaded_energy[j] = leg_energy(loaded_dist, schedule.loaded_speed[j], loaded=True)
        empty_power[j] = SPEED_TABLE[schedule.empty_speed[j]].empty_power
        loaded_power[j] = SPEED_TABLE[schedule.loaded_speed[j]].loaded_power

    # Activity model. Kinds: power-consuming travel legs "E" (empty) and "L" (loaded),
    # plus zero-power timing nodes "R" (LOAD AGV release at QC pickup, dur 0), "H" (QC
    # handling, dur tau) and "C" (UNLOAD completion = container-on-AGV, dur 0). The
    # precedence reproduces the corrected uncoupled recurrence (FSMJ Eqs 10-18) exactly;
    # only E and L draw power. Per task the "QC completion" node is H (load) / C (unload)
    # and the "AGV release" node is R (load) / L (unload).
    Act = tuple[int, str]

    def qc_done(j: int) -> Act:
        return (j, "H") if instance.tasks[j].kind is TaskKind.LOAD else (j, "C")

    def agv_rel(j: int) -> Act:
        return (j, "R") if instance.tasks[j].kind is TaskKind.LOAD else (j, "L")

    preds: dict[Act, list[Act]] = {}
    for j in range(n):
        e_preds: list[Act] = [] if agv_prev[j] < 0 else [agv_rel(agv_prev[j])]
        qc_chain: list[Act] = [] if qc_prev[j] < 0 else [qc_done(qc_prev[j])]
        preds[(j, "E")] = e_preds
        if instance.tasks[j].kind is TaskKind.LOAD:
            preds[(j, "L")] = [(j, "E")]
            preds[(j, "R")] = [(j, "L"), *qc_chain]  # released at max(arrival, qc_ready)
            preds[(j, "H")] = [(j, "R")]  # c_j = release + tau
        else:
            # QC handling overlaps AGV travel: H is gated by the QC chain only (no E), and
            # is omitted for the first task on a crane so it is not charged tau (Eq 14/15).
            if qc_prev[j] >= 0:
                preds[(j, "H")] = [*qc_chain]
                preds[(j, "C")] = [(j, "E"), (j, "H")]  # c_j = max(arrival, qc_ready + tau)
            else:
                preds[(j, "C")] = [(j, "E")]  # c_j = arrival
            preds[(j, "L")] = [(j, "C")]

    succ: dict[Act, list[Act]] = {a: [] for a in preds}
    pred_count: dict[Act, int] = {a: len(p) for a, p in preds.items()}
    for a, plist in preds.items():
        for p in plist:
            succ[p].append(a)

    avail: dict[Act, float] = dict.fromkeys(preds, 0.0)
    start: dict[Act, float] = {}
    finish: dict[Act, float] = {}

    def _is_power(a: Act) -> bool:
        return a[1] in ("E", "L")

    def _power(a: Act) -> float:
        return empty_power[a[0]] if a[1] == "E" else loaded_power[a[0]]

    def _dur(a: Act) -> float:
        return empty_time[a[0]] if a[1] == "E" else loaded_time[a[0]]

    def _zero_dur(a: Act) -> float:
        return instance.tasks[a[0]].handling_time if a[1] == "H" else 0.0

    finish_heap: list[tuple[float, int, Act]] = []
    ready_travel: list[Act] = []
    power_used = 0.0
    power_arcs: list[tuple[int, int, int, int]] = []
    seq = 0  # tie-break/order counter for the finish heap

    def _leg_flag(a: Act) -> int:
        return 0 if a[1] == "E" else 1

    def _schedule_zero_power(a: Act, t: float) -> None:
        nonlocal seq
        start[a] = t
        finish[a] = t + _zero_dur(a)
        heapq.heappush(finish_heap, (finish[a], seq, a))
        seq += 1

    def _try_start(t: float, trigger: Act | None) -> None:
        nonlocal power_used, seq
        ready_travel.sort(key=lambda a: (avail[a], a[0], 0 if a[1] == "E" else 1))
        remaining: list[Act] = []
        for a in ready_travel:
            p = _power(a)
            if power_used + p <= p_max + _EPS:
                start[a] = t
                finish[a] = t + _dur(a)
                heapq.heappush(finish_heap, (finish[a], seq, a))
                seq += 1
                power_used += p
                if trigger is not None and t > avail[a] + _EPS:
                    power_arcs.append((trigger[0], _leg_flag(trigger), a[0], _leg_flag(a)))
            else:
                remaining.append(a)
        ready_travel[:] = remaining

    # Seed: activities with no predecessors (first empty leg of each AGV).
    for a, c in pred_count.items():
        if c == 0:
            if _is_power(a):
                ready_travel.append(a)
            else:
                _schedule_zero_power(a, 0.0)
    _try_start(0.0, None)

    while finish_heap:
        t = finish_heap[0][0]
        trigger: Act | None = None
        while finish_heap and finish_heap[0][0] <= t + _EPS:
            _, _, a = heapq.heappop(finish_heap)
            if _is_power(a):
                power_used -= _power(a)
                trigger = a
            for s in succ[a]:
                pred_count[s] -= 1
                if finish[a] > avail[s]:
                    avail[s] = finish[a]
                if pred_count[s] == 0:
                    if _is_power(s):
                        ready_travel.append(s)
                    else:
                        _schedule_zero_power(s, avail[s])
        _try_start(t, trigger)

    arr_pickup = [finish[(j, "E")] for j in range(n)]
    qc_finish = [0.0] * n
    arr_dropoff = [0.0] * n
    completion = [0.0] * n
    agv_free_after = [0.0] * n
    wait_empty = [0.0] * n
    wait_loaded = [0.0] * n
    for j in range(n):
        prec_e = 0.0 if agv_prev[j] < 0 else finish[agv_rel(agv_prev[j])]
        wait_empty[j] = max(0.0, start[(j, "E")] - prec_e)
        if instance.tasks[j].kind is TaskKind.LOAD:
            arr_dropoff[j] = finish[(j, "L")]  # r_j
            qc_finish[j] = finish[(j, "H")]  # c_j
            completion[j] = qc_finish[j]
            agv_free_after[j] = finish[(j, "R")]  # released at QC pickup
            wait_loaded[j] = max(0.0, start[(j, "L")] - finish[(j, "E")])
        else:
            qc_finish[j] = finish[(j, "C")]  # c_j
            arr_dropoff[j] = finish[(j, "L")]  # r_j
            completion[j] = arr_dropoff[j]
            agv_free_after[j] = finish[(j, "L")]  # released at the yard
            wait_loaded[j] = max(0.0, start[(j, "L")] - finish[(j, "C")])

    return Evaluation(
        makespan=max(completion),
        energy=sum(empty_energy) + sum(loaded_energy),
        completion=tuple(completion),
        qc_finish=tuple(qc_finish),
        arr_pickup=tuple(arr_pickup),
        arr_dropoff=tuple(arr_dropoff),
        agv_free_after=tuple(agv_free_after),
        empty_time=tuple(empty_time),
        loaded_time=tuple(loaded_time),
        empty_energy=tuple(empty_energy),
        loaded_energy=tuple(loaded_energy),
        topo_order=order,
        empty_start=tuple(start[(j, "E")] for j in range(n)),
        loaded_start=tuple(start[(j, "L")] for j in range(n)),
        power_arcs=tuple(power_arcs),
        wait_empty=tuple(wait_empty),
        wait_loaded=tuple(wait_loaded),
    )
