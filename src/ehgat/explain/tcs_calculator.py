"""Trade-off Criticality Score (TCS) for TAPE explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ehgat.explain.tape_explainer import TapeExplanation

__all__ = ["ParetoPoint", "local_lambda", "tradeoff_criticality_scores"]


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """A Pareto schedule's objective vector and TAPE explanation."""

    solution_id: str
    makespan: float
    energy: float
    explanation: TapeExplanation


def local_lambda(points: list[ParetoPoint], idx: int) -> float:
    """Trade-off weight λ from the local front tangent at points[idx].

    For a bi-objective minimisation front sorted by makespan, the local tangent
    (ΔC, ΔE) satisfies λ ΔC + (1-λ) ΔE = 0 for a weighted-sum normal.
    """
    if len(points) < 2:
        return 0.5
    ordered = sorted(points, key=lambda p: (p.makespan, p.energy))
    p = ordered[idx]
    if idx == 0:
        q0, q1 = p, ordered[1]
    elif idx == len(ordered) - 1:
        q0, q1 = ordered[-2], p
    else:
        q0, q1 = ordered[idx - 1], ordered[idx + 1]
    dc = q1.makespan - q0.makespan
    de = q1.energy - q0.energy
    denom = dc - de
    if abs(denom) < 1e-12:
        return 0.5
    return max(0.0, min(1.0, -de / denom))


def _combine(lam: float, c_grad: float, e_grad: float) -> float:
    return abs(lam * c_grad + (1.0 - lam) * e_grad)


def tradeoff_criticality_scores(points: list[ParetoPoint]) -> list[dict[str, Any]]:
    """Compute edge/task TCS values for a Pareto set."""
    ordered = sorted(points, key=lambda p: (p.makespan, p.energy))
    rows: list[dict[str, Any]] = []
    for i, point in enumerate(ordered):
        lam = local_lambda(ordered, i)
        ex = point.explanation
        n = len(ex.empty_time_grad)
        tasks = []
        for j in range(n):
            empty_tcs = _combine(lam, ex.empty_time_grad[j], ex.empty_energy_grad[j])
            loaded_tcs = _combine(lam, ex.loaded_time_grad[j], ex.loaded_energy_grad[j])
            tasks.append(
                {
                    "task": j,
                    "empty_time_grad": ex.empty_time_grad[j],
                    "loaded_time_grad": ex.loaded_time_grad[j],
                    "empty_energy_grad": ex.empty_energy_grad[j],
                    "loaded_energy_grad": ex.loaded_energy_grad[j],
                    "tcs_empty": empty_tcs,
                    "tcs_loaded": loaded_tcs,
                    "tcs_total": empty_tcs + loaded_tcs,
                }
            )
        arcs = [
            {**meta, "tropical_grad": grad, "pts": abs(lam * grad)}
            for meta, grad in zip(ex.event_edges, ex.event_edge_grad, strict=True)
        ]
        rows.append(
            {
                "solution_id": point.solution_id,
                "objectives": {"makespan": point.makespan, "energy": point.energy},
                "lambda": lam,
                "tasks": tasks,
                "event_arcs": arcs,
            }
        )
    return rows
