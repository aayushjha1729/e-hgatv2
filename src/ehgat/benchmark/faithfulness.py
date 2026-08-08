"""Attention faithfulness against the exact critical path.

The makespan of a resolved schedule is a max-plus longest path over the precedence DAG.
A disjunctive AGV arc is a true bottleneck iff it lies on that critical path: speeding
up the leg it carries is the only way to shorten the makespan. This module computes the
critical path exactly from the evaluator's resolved timings and checks whether the
surrogate's maximum-attention AGV arc lies on it.

Two complementary measures, averaged over a set of schedules:

- precision@1 -- fraction of schedules whose top-attention AGV arc is on the exact
  critical path (a hard, decision-relevant hit rate).
- Spearman rho -- rank correlation between per-task attention and the per-task
  marginal makespan reduction obtained by setting that task's legs to the fastest
  speed (a graded "does attention rank the levers correctly" signal).

A faithful explainer scores high on both; the random and post-hoc baselines provide the reference level.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from ehgat.environment.decoder import Schedule
from ehgat.environment.evaluator import build_precedence, evaluate
from ehgat.environment.instance import Instance, TaskKind
from ehgat.environment.physics import SpeedLevel
from ehgat.surrogate.ehgatv2 import EHGATv2
from ehgat.surrogate.graph import AGV_EDGE, build_hetero_graph

__all__ = [
    "FaithfulnessResult",
    "attention_per_task",
    "critical_agv_arcs",
    "critical_path_binding",
    "evaluate_faithfulness",
    "marginal_makespan_speedup",
]

_FASTEST: SpeedLevel = tuple(SpeedLevel)[-1]


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    """Aggregate attention-faithfulness over a set of schedules."""

    precision_at_1: float  # share of schedules whose top-alpha AGV arc is critical
    spearman_rho: float  # mean rank corr. between alpha and marginal makespan speedup
    num_schedules: int


def critical_path_binding(schedule: Schedule, instance: Instance) -> tuple[set[int], set[int]]:
    """Partition critical-path tasks by binding resource: (agv_bound, qc_bound).

    Backtracks from the makespan-defining task through its binding predecessor at each
    step (the AGV chain when the AGV-ready term meets/exceeds the QC-ready term, else the
    QC chain). agv_bound collects tasks gated by their incoming AGV arc; qc_bound
    those gated by the QC serialisation chain. The two sets are disjoint (each step picks
    exactly one binding resource). This is the exact Max-Plus bottleneck-type oracle
    the Channel-B operator-selection controller is graded against.
    """
    n = instance.num_tasks
    agv_prev, qc_prev, _ = build_precedence(schedule.agv_sequences, schedule.qc_sequences, n)
    ev = evaluate(schedule, instance)

    agv_bound: set[int] = set()
    qc_bound: set[int] = set()
    j: int | None = int(np.argmax(ev.completion))
    while j is not None:
        task = instance.tasks[j]
        ap, qp = agv_prev[j], qc_prev[j]
        agv_ready = ev.agv_free_after[ap] if ap >= 0 else 0.0
        qc_ready = ev.qc_finish[qp] if qp >= 0 else 0.0
        if task.kind is TaskKind.LOAD:
            agv_term = agv_ready + ev.empty_time[j] + ev.loaded_time[j]  # == arr_dropoff
        else:
            agv_term = agv_ready + ev.empty_time[j]  # == arr_pickup
        if agv_term >= qc_ready:  # AGV chain binds -> j's incoming AGV arc is critical
            agv_bound.add(j)
            j = ap if ap >= 0 else None
        else:  # QC chain binds
            qc_bound.add(j)
            j = qp if qp >= 0 else None
    return agv_bound, qc_bound


def critical_agv_arcs(schedule: Schedule, instance: Instance) -> set[int]:
    """Destination tasks of the AGV arcs lying on the exact makespan critical path.

    Thin wrapper over critical_path_binding (its AGV-bound partition); the graph
    stores exactly one AGV arc per task, keyed by destination.
    """
    return critical_path_binding(schedule, instance)[0]


def marginal_makespan_speedup(schedule: Schedule, instance: Instance) -> np.ndarray:
    """Per-task makespan reduction from setting that task's legs to the fastest speed.

    out[j] = C_max(base) - C_max(j at fastest) (>= 0); zero for off-critical tasks.
    """
    n = instance.num_tasks
    base = evaluate(schedule, instance).makespan
    out = np.zeros(n, dtype=float)
    empty = list(schedule.empty_speed)
    loaded = list(schedule.loaded_speed)
    for j in range(n):
        e0, l0 = empty[j], loaded[j]
        empty[j], loaded[j] = _FASTEST, _FASTEST
        faster = replace(schedule, empty_speed=tuple(empty), loaded_speed=tuple(loaded))
        out[j] = base - evaluate(faster, instance).makespan
        empty[j], loaded[j] = e0, l0
    return out


def attention_per_task(schedule: Schedule, instance: Instance, model: EHGATv2) -> np.ndarray:
    """Per-task attention weight on the incoming AGV arc ([N])."""
    n = instance.num_tasks
    data = build_hetero_graph(schedule, instance)
    edge_index, alpha = model.attention(data)[AGV_EDGE[1]]
    dst = edge_index[1].cpu().numpy()
    a = alpha.detach().cpu().numpy()
    if a.ndim > 1:
        a = a.mean(axis=tuple(range(1, a.ndim)))
    per_task = np.zeros(n, dtype=float)
    per_task[dst] = a
    return per_task


def evaluate_faithfulness(
    schedules: list[Schedule], instance: Instance, model: EHGATv2
) -> FaithfulnessResult:
    """Aggregate precision@1 and Spearman rho over schedules."""
    if not schedules:
        raise ValueError("need at least one schedule to evaluate faithfulness")
    hits = 0
    rhos: list[float] = []
    for sched in schedules:
        alpha = attention_per_task(sched, instance, model)
        critical = critical_agv_arcs(sched, instance)
        top = int(np.argmax(alpha))
        hits += int(top in critical)

        speedup = marginal_makespan_speedup(sched, instance)
        if np.std(alpha) > 0 and np.std(speedup) > 0:
            rho = spearmanr(alpha, speedup).correlation
            if not np.isnan(rho):
                rhos.append(float(rho))
    return FaithfulnessResult(
        precision_at_1=hits / len(schedules),
        spearman_rho=float(np.mean(rhos)) if rhos else 0.0,
        num_schedules=len(schedules),
    )
