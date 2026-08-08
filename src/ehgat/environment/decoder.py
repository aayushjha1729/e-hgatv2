"""Canonical schedule decoder: 4N random-key chromosome -> Schedule.

Implements the encoding/decoding of Fontes & Homayouni (2022), Section 4.2 / Fig. 3.
The chromosome is a length-4N vector of random keys in [0, 1] partitioned into
four contiguous blocks of length N (N = number of tasks):

============  ===================  ============================================
Block index   Semantics            Decode rule
============  ===================  ============================================
0             task-sequence keys   smallest-position-value (ascending argsort)
1             AGV-assignment keys  interval bucketing into |A| buckets
2             empty-leg speed keys interval bucketing into |V| speed levels
3             loaded-leg speed     interval bucketing into |V| speed levels
============  ===================  ============================================

The smallest-position-value (SPV) rule turns the first block into a total order
over tasks. The per-AGV and per-QC sequences are projections of this single total
order onto each AGV's assigned tasks and each QC's tasks respectively. Because both
projections inherit the same global order, the schedule produced by decode is
always acyclic (no AGV/QC precedence deadlock can arise from a freshly decoded
chromosome; deadlocks arise only after the targeted mutations of the guided search, and
the evaluator therefore re-validates acyclicity with Kahn's algorithm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ehgat.environment.instance import Instance
from ehgat.environment.physics import SpeedLevel

__all__ = ["NUM_BLOCKS", "Schedule", "bucket", "decode", "encode_canonical"]

NUM_BLOCKS = 4  # sequence, AGV assignment, empty-leg speed, loaded-leg speed
_SPEED_LEVELS: tuple[SpeedLevel, ...] = tuple(SpeedLevel)


@dataclass(frozen=True, slots=True)
class Schedule:
    """A fully decoded, structurally valid schedule.

    All per-task tuples are indexed by task_id. agv_sequences and
    qc_sequences list task_id\\ s in processing order.
    """

    global_order: tuple[int, ...]
    assignment: tuple[int, ...]
    agv_sequences: tuple[tuple[int, ...], ...]
    qc_sequences: tuple[tuple[int, ...], ...]
    empty_speed: tuple[SpeedLevel, ...]
    loaded_speed: tuple[SpeedLevel, ...]


def bucket(key: float, n_buckets: int) -> int:
    """Map a random key in [0, 1] to one of n_buckets equal intervals.

    Interval [k/n, (k+1)/n) -> bucket k; key == 1.0 maps to the last
    bucket. Matches the interval rule of Fontes & Homayouni (2022), Sec. 4.2.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    if not (0.0 <= key <= 1.0):
        raise ValueError(f"random key must be in [0, 1], got {key}")
    return min(int(key * n_buckets), n_buckets - 1)


def decode(keys: np.ndarray, instance: Instance) -> Schedule:
    """Decode a 4N random-key chromosome into a Schedule."""
    n = instance.num_tasks
    keys = np.asarray(keys, dtype=float)
    if keys.shape != (NUM_BLOCKS * n,):
        raise ValueError(
            f"chromosome must have shape ({NUM_BLOCKS * n},) for N={n}, got {keys.shape}"
        )
    if not np.all(np.isfinite(keys)):
        raise ValueError("chromosome contains non-finite keys")
    if np.any(keys < 0.0) or np.any(keys > 1.0):
        raise ValueError("all random keys must lie in [0, 1]")

    seq_keys = keys[0:n]
    agv_keys = keys[n : 2 * n]
    empty_keys = keys[2 * n : 3 * n]
    loaded_keys = keys[3 * n : 4 * n]

    # SPV: ascending argsort gives the global processing order of task_ids. Stable
    # sort makes equal keys deterministic (ordered by task_id).
    global_order = tuple(int(t) for t in np.argsort(seq_keys, kind="stable"))

    assignment = tuple(bucket(float(k), instance.num_agvs) for k in agv_keys)
    empty_speed = tuple(_SPEED_LEVELS[bucket(float(k), len(_SPEED_LEVELS))] for k in empty_keys)
    loaded_speed = tuple(_SPEED_LEVELS[bucket(float(k), len(_SPEED_LEVELS))] for k in loaded_keys)

    agv_sequences = tuple(
        tuple(t for t in global_order if assignment[t] == a) for a in range(instance.num_agvs)
    )
    qc_sequences = tuple(
        tuple(t for t in global_order if instance.tasks[t].qc == qc) for qc in instance.qcs
    )

    return Schedule(
        global_order=global_order,
        assignment=assignment,
        agv_sequences=agv_sequences,
        qc_sequences=qc_sequences,
        empty_speed=empty_speed,
        loaded_speed=loaded_speed,
    )


def encode_canonical(schedule: Schedule, instance: Instance) -> np.ndarray:
    """Produce a representative chromosome that decodes back to schedule.

    Keys are placed at the centre of their decoding interval so that decode is a
    left-inverse: decode(encode_canonical(s, inst), inst) == s. Useful for
    round-trip tests and for seeding BRKGA from a known schedule.
    """
    n = instance.num_tasks
    keys = np.empty(NUM_BLOCKS * n, dtype=float)

    # Sequence block: rank position in the global order -> ascending keys.
    for rank, task_id in enumerate(schedule.global_order):
        keys[task_id] = (rank + 0.5) / n

    n_agv = instance.num_agvs
    n_spd = len(_SPEED_LEVELS)
    for task_id in range(n):
        keys[n + task_id] = (schedule.assignment[task_id] + 0.5) / n_agv
        keys[2 * n + task_id] = (int(schedule.empty_speed[task_id]) + 0.5) / n_spd
        keys[3 * n + task_id] = (int(schedule.loaded_speed[task_id]) + 0.5) / n_spd

    return keys
