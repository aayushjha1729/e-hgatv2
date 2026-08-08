"""Exact Pareto-front oracle for the dual-cycling instance.

Produces the true Pareto front PF* of (C_max, E) by exhaustive search,
and provides the exact ground truth against which convergence and attention
faithfulness are measured.

The naive search space is (#structures) x 3^(2N), where the 3^(2N) factor counts three
speed levels on each of the empty and loaded legs of N tasks. A per-structure speed
Pareto DP removes that factor:

For a fixed structure (AGV assignment + per-AGV order + per-QC order) the leg
distances are constant; only each leg's time/energy depends on its speed level.
The timing recurrence (see evaluator.py) is monotone in every AGV-free-time and
QC-finish-time, so tasks are swept in topological order carrying a Pareto set of
states (makespan, energy, [agv_free_a], [qc_finish_q]) with dominated states pruned.
Monotonicity implies a dominated state cannot lead to a non-dominated completion, so
the pruning is exact. Once a resource has no remaining
tasks its dimension is zeroed (it can no longer influence any objective), which keeps
the Pareto set tiny. This computes the structure's exact speed front in
O(N x |frontier| x 9) instead of 3^(2N).

The structure space itself still grows ~ (N+1)!, so a tractability guard caps
exhaustive exactness. The canonical exact instance is therefore small
(EXACT_TOY_TASKS); the N=10 instance is reserved for scaling studies.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import factorial
from pathlib import Path

import numpy as np

from ehgat.environment.evaluator import build_precedence
from ehgat.environment.instance import Instance, TaskKind
from ehgat.environment.physics import SPEED_TABLE, SpeedLevel

__all__ = [
    "DEFAULT_MAX_STRUCTURE_ITERATIONS",
    "OracleFront",
    "OracleTooLargeError",
    "Structure",
    "enumerate_structures",
    "evaluate_speeds",
    "exact_pareto_front",
    "pareto_front_2d",
    "save_front",
    "structure_speed_front",
]

# Upper bound on the (assignment x permutation) enumeration before deduplication.
# Protects against accidentally launching an intractable oracle run (e.g. N=10).
DEFAULT_MAX_STRUCTURE_ITERATIONS = 5_000_000

# The oracle computes objectives exactly (no float noise) and produces floats only
# for output, so Pareto filtering is exact (no tolerance) and reproducible.
_ROUND_DP = 6  # decimals for clean, deterministic float output of kept points

# Integer scales that render every leg time/energy exactly integral for the toy's
# physics (speeds are k/5 m/s, powers k/10 kW, distances integers). The DP runs in
# these scaled INTEGER units for speed, converting to exact Fractions only at the
# boundary. A non-integral leg trips an assertion (signal to widen the scales).
_TIME_SCALE = 1800  # 1 unit = 1/1800 s
_ENERGY_SCALE = 720  # 1 unit = 1/720 kJ


class OracleTooLargeError(ValueError):
    """Raised when the structure enumeration would exceed the tractability guard."""


@dataclass(frozen=True, slots=True)
class Structure:
    """A fixed routing: AGV assignment plus per-AGV and per-QC processing orders."""

    assignment: tuple[int, ...]
    agv_sequences: tuple[tuple[int, ...], ...]
    qc_sequences: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class OracleFront:
    """The exact Pareto front with provenance metadata."""

    front: tuple[tuple[float, float], ...]
    num_structures: int
    num_tasks: int
    num_agvs: int
    num_qcs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "objectives": ["makespan", "energy"],
            "num_tasks": self.num_tasks,
            "num_agvs": self.num_agvs,
            "num_qcs": self.num_qcs,
            "num_structures": self.num_structures,
            "front_size": len(self.front),
            "front": [list(point) for point in self.front],
        }


def _frac(value: float) -> Fraction:
    """Exact rational for a clean-decimal float (e.g. 4.8 -> 24/5)."""
    return Fraction(str(value))


def _leg_cost_scaled(distance: int, level: SpeedLevel, *, loaded: bool) -> tuple[int, int]:
    """Exact (time, energy) for one leg in scaled INTEGER units.

    time is in 1/_TIME_SCALE s and energy in 1/_ENERGY_SCALE kJ;
    both are exact integers for the toy's physics (a non-integral result asserts).
    """
    spec = SPEED_TABLE[level]
    speed = _frac(spec.loaded_speed if loaded else spec.empty_speed)
    power = _frac(spec.loaded_power if loaded else spec.empty_power)
    time = _TIME_SCALE * distance / speed
    energy = _ENERGY_SCALE * power * distance / speed
    if time.denominator != 1 or energy.denominator != 1:  # pragma: no cover - guard
        raise AssertionError(
            f"leg cost not integral under scales ({_TIME_SCALE}, {_ENERGY_SCALE}); "
            "widen the scales for this physics table."
        )
    return int(time), int(energy)


def _leg_options_scaled(distance: int, *, loaded: bool) -> tuple[tuple[int, int], ...]:
    """Scaled-integer (time, energy) for each of the 3 speed levels on one leg."""
    return tuple(_leg_cost_scaled(distance, lvl, loaded=loaded) for lvl in SpeedLevel)


def evaluate_speeds(
    structure: Structure,
    empty_levels: Sequence[SpeedLevel],
    loaded_levels: Sequence[SpeedLevel],
    instance: Instance,
) -> tuple[Fraction, Fraction]:
    """Exact (makespan, energy) for one structure under a full speed assignment.

    Mirrors the evaluator's timing recurrence in exact (scaled-integer) arithmetic.
    Used to generate surrogate-training targets and to cross-validate the speed DP.
    """
    n = instance.num_tasks
    qc_index = {qc: i for i, qc in enumerate(instance.qcs)}
    agv_prev, qc_prev, order = build_precedence(
        structure.agv_sequences, structure.qc_sequences, n
    )
    agv_of = structure.assignment

    agv_free = [0] * instance.num_agvs
    qc_finish = [0] * len(instance.qcs)
    makespan = 0
    energy = 0
    for j in order:
        task = instance.tasks[j]
        a = agv_of[j]
        q = qc_index[task.qc]
        ap = agv_prev[j]
        origin = instance.agv_start if ap < 0 else instance.tasks[ap].dropoff
        empty_time, empty_energy = _leg_cost_scaled(
            round(instance.distance.distance(origin, task.pickup)),
            empty_levels[j],
            loaded=False,
        )
        loaded_time, loaded_energy = _leg_cost_scaled(
            round(instance.loaded_distance(task)), loaded_levels[j], loaded=True
        )
        tau = _TIME_SCALE * round(task.handling_time)
        arr_pickup = agv_free[a] + empty_time
        # Matches evaluator._evaluate_uncoupled (FSMJ Eqs 10-18): a LOAD AGV is released at
        # the QC pickup (c_j - tau), and an UNLOAD's handling overlaps AGV travel (the +tau
        # floor applies only along the QC chain, not on the first task of a crane).
        if task.kind is TaskKind.LOAD:
            arr_dropoff = arr_pickup + loaded_time  # r_j
            qc_f = max(arr_dropoff, qc_finish[q]) + tau  # c_j (Eq 10/11)
            completion = qc_f  # Eq 2
            agv_free[a] = qc_f - tau  # Eq 12: c_i - tau_i
        else:
            qc_floor = qc_finish[q] + tau if qc_prev[j] >= 0 else 0  # Eq 10 only
            qc_f = max(arr_pickup, qc_floor)  # c_j (Eq 14/15/17)
            arr_dropoff = qc_f + loaded_time  # r_j (Eq 16)
            completion = arr_dropoff  # Eq 3
            agv_free[a] = arr_dropoff  # r_j (Eq 14/18)
        qc_finish[q] = qc_f
        if completion > makespan:
            makespan = completion
        energy += empty_energy + loaded_energy
    return Fraction(makespan, _TIME_SCALE), Fraction(energy, _ENERGY_SCALE)


def pareto_front_2d(
    points: list[tuple[Fraction, Fraction]],
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Exact non-dominated minimisation front over (makespan, energy) points.

    Ascending by (makespan, energy); a point is kept iff its energy is strictly
    below the best energy seen so far. Exact rational inputs make this exact.
    """
    front: list[tuple[Fraction, Fraction]] = []
    best_energy: Fraction | None = None
    for makespan, energy in sorted(points):  # ascending makespan, then energy
        if best_energy is None or energy < best_energy:
            front.append((makespan, energy))
            best_energy = energy
    return tuple(front)


def _pareto_front_2d_int(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Non-dominated minimisation front over scaled-integer (makespan, energy)."""
    front: list[tuple[int, int]] = []
    best_energy: int | None = None
    for makespan, energy in sorted(points):
        if best_energy is None or energy < best_energy:
            front.append((makespan, energy))
            best_energy = energy
    return front


def _pareto_min(states: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Keep non-dominated states under exact component-wise minimisation (all dims).

    Vectorised with numpy: dominated[i] is true iff some distinct state j is <= i in
    every dimension and < in at least one. All values are exact integers, which makes the
    comparison exact.
    """
    unique = list(set(states))
    if len(unique) <= 1:
        return unique
    arr = np.asarray(unique, dtype=np.int64)  # (M, D)
    leq = np.asarray((arr[None, :, :] <= arr[:, None, :]).all(axis=2))  # leq[i,j]: j <= i (all)
    lt = np.asarray((arr[None, :, :] < arr[:, None, :]).any(axis=2))  # lt[i,j]: j < i (some)
    # j == i has lt False, leaving the diagonal non-dominating without an explicit fill.
    dominated = np.asarray(leq & lt).any(axis=1)
    return [tuple(int(v) for v in row) for row in arr[~dominated]]


def _structure_speed_front_int(
    structure: Structure, instance: Instance
) -> list[tuple[int, int]]:
    """Scaled-integer (makespan, energy) Pareto front over speeds for one structure."""
    n = instance.num_tasks
    n_agv = instance.num_agvs
    qc_index = {qc: i for i, qc in enumerate(instance.qcs)}
    n_qc = len(instance.qcs)

    agv_prev, qc_prev, order = build_precedence(
        structure.agv_sequences, structure.qc_sequences, n
    )
    agv_of = structure.assignment

    empty_opts: list[tuple[tuple[int, int], ...]] = [()] * n
    loaded_opts: list[tuple[tuple[int, int], ...]] = [()] * n
    tau_scaled = [0] * n
    for j in range(n):
        task = instance.tasks[j]
        ap = agv_prev[j]
        origin = instance.agv_start if ap < 0 else instance.tasks[ap].dropoff
        empty_opts[j] = _leg_options_scaled(
            round(instance.distance.distance(origin, task.pickup)), loaded=False
        )
        loaded_opts[j] = _leg_options_scaled(round(instance.loaded_distance(task)), loaded=True)
        tau_scaled[j] = _TIME_SCALE * round(task.handling_time)

    # Position of each resource's last task in the topo order -> when it is "done".
    last_agv_pos = [-1] * n_agv
    last_qc_pos = [-1] * n_qc
    for s, j in enumerate(order):
        last_agv_pos[agv_of[j]] = s
        last_qc_pos[qc_index[instance.tasks[j].qc]] = s

    # State layout: (makespan, energy, agv_free[0..n_agv), qc_finish[0..n_qc)), scaled ints.
    frontier: list[tuple[int, ...]] = [(0, 0) + (0,) * n_agv + (0,) * n_qc]

    for s, j in enumerate(order):
        task = instance.tasks[j]
        a = agv_of[j]
        q = qc_index[task.qc]
        tau = tau_scaled[j]
        is_load = task.kind is TaskKind.LOAD
        agv_idx = 2 + a
        qc_idx = 2 + n_agv + q
        done_agv = [aa for aa in range(n_agv) if last_agv_pos[aa] <= s]
        done_qc = [qq for qq in range(n_qc) if last_qc_pos[qq] <= s]

        nxt: list[tuple[int, ...]] = []
        for st in frontier:
            agv_free = st[agv_idx]
            qc_finish_prev = st[qc_idx]
            for empty_time, empty_energy in empty_opts[j]:
                arr_pickup = agv_free + empty_time
                for loaded_time, loaded_energy in loaded_opts[j]:
                    # See evaluate_speeds / evaluator._evaluate_uncoupled (FSMJ Eqs 10-18).
                    if is_load:
                        arr_dropoff = arr_pickup + loaded_time  # r_j
                        qc_finish = max(arr_dropoff, qc_finish_prev) + tau  # c_j
                        completion = qc_finish
                        agv_free_new = qc_finish - tau  # Eq 12: c_i - tau_i
                    else:
                        qc_floor = qc_finish_prev + tau if qc_prev[j] >= 0 else 0  # Eq 10 only
                        qc_finish = max(arr_pickup, qc_floor)  # c_j (Eq 14/15/17)
                        arr_dropoff = qc_finish + loaded_time  # r_j (Eq 16)
                        completion = arr_dropoff
                        agv_free_new = arr_dropoff  # Eq 14/18

                    new = list(st)
                    new[0] = st[0] if st[0] >= completion else completion
                    new[1] = st[1] + empty_energy + loaded_energy
                    new[agv_idx] = agv_free_new
                    new[qc_idx] = qc_finish
                    for aa in done_agv:
                        new[2 + aa] = 0
                    for qq in done_qc:
                        new[2 + n_agv + qq] = 0
                    nxt.append(tuple(new))
        frontier = _pareto_min(nxt)

    return _pareto_front_2d_int([(st[0], st[1]) for st in frontier])


def structure_speed_front(
    structure: Structure, instance: Instance
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Exact (makespan, energy) Pareto front over all speed choices for one structure."""
    return tuple(
        (Fraction(m, _TIME_SCALE), Fraction(e, _ENERGY_SCALE))
        for m, e in _structure_speed_front_int(structure, instance)
    )


def enumerate_structures(
    instance: Instance, *, max_iterations: int = DEFAULT_MAX_STRUCTURE_ITERATIONS
) -> Iterator[Structure]:
    """Yield each unique deadlock-free structure exactly once.

    Enumerates (assignment, permutation) pairs and deduplicates by the induced
    (agv_sequences, qc_sequences). Raises OracleTooLargeError if the
    enumeration size exceeds max_iterations.
    """
    n = instance.num_tasks
    n_agv = instance.num_agvs
    estimate = (n_agv**n) * factorial(n)
    if estimate > max_iterations:
        raise OracleTooLargeError(
            f"structure enumeration ~{estimate:,} exceeds the guard {max_iterations:,} "
            f"for N={n}, A={n_agv}. Use a smaller instance (see EXACT_TOY_TASKS) or "
            "raise max_iterations deliberately."
        )

    seen: set[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = set()
    for assignment in product(range(n_agv), repeat=n):
        for perm in permutations(range(n)):
            agv_sequences = tuple(
                tuple(t for t in perm if assignment[t] == a) for a in range(n_agv)
            )
            qc_sequences = tuple(
                tuple(t for t in perm if instance.tasks[t].qc == qc) for qc in instance.qcs
            )
            key = (agv_sequences, qc_sequences)
            if key in seen:
                continue
            seen.add(key)
            yield Structure(
                assignment=tuple(assignment),
                agv_sequences=agv_sequences,
                qc_sequences=qc_sequences,
            )


def exact_pareto_front(
    instance: Instance, *, max_iterations: int = DEFAULT_MAX_STRUCTURE_ITERATIONS
) -> OracleFront:
    """Compute the exact Pareto front PF* for instance."""
    points: list[tuple[int, int]] = []
    num_structures = 0
    for structure in enumerate_structures(instance, max_iterations=max_iterations):
        num_structures += 1
        points.extend(_structure_speed_front_int(structure, instance))

    exact = _pareto_front_2d_int(points)
    front = tuple(
        (round(m / _TIME_SCALE, _ROUND_DP), round(e / _ENERGY_SCALE, _ROUND_DP))
        for m, e in exact
    )
    return OracleFront(
        front=front,
        num_structures=num_structures,
        num_tasks=instance.num_tasks,
        num_agvs=instance.num_agvs,
        num_qcs=len(instance.qcs),
    )


def save_front(front: OracleFront, path: str | Path) -> None:
    """Write the exact front to JSON (golden artefact for regression tests)."""
    Path(path).write_text(json.dumps(front.to_dict(), indent=2) + "\n", encoding="utf-8")
