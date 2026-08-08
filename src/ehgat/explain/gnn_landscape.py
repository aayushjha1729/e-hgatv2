"""GNN/TAPE-derived feature-importance landscape -- model-native and scalable.

This replaces the textbook variance-based (Sobol) landscape with one computed entirely
from the model's own explanations. Each schedule's exact critical-path subgradients
(dC_max/d(leg time), dC_max/d(tau), and the resource-arc indicators) are extracted
by the fused E-HGATv2 + TAPE head and aggregated across a sampled set into per-decision
-family importance and a Pareto-vs-dominated contrast.

The cost is flat in N per schedule: every score is one model forward and backward pass
per schedule, O(1) in the number of decision families and batchable on GPU. Grouped Sobol
costs base_samples * (2 + families) exact-evaluator calls and the critical-path cascade
costs O(N * samples) exact re-evaluations, both intractable at the DL instance sizes
(40-160 containers).

The decision families follow the formulation's decision variables and the random-key
blocks: sequence (task order and QC precedence), assignment (which AGV), and empty_speed
and loaded_speed (per-leg travel speed).

The same aggregator runs on the exact TAPE oracle
(ehgat.explain.tape_explainer.explain_schedule) at small N, producing a ground-truth
landscape against which the GNN landscape is checked (landscape_rank_agreement). At large
N only the GNN path is feasible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from ehgat.environment.decoder import Schedule
from ehgat.environment.instance import Instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.explain.fused_explainer import explain_fused_schedules
from ehgat.explain.tape_explainer import (
    TapeExplanation,
    explain_schedule,
    explain_schedule_coupled,
)
from ehgat.search.nsga2 import fast_non_dominated_sort

__all__ = [
    "FEATURE_FAMILIES",
    "GnnLandscapeResult",
    "aggregate_landscape",
    "exact_landscape",
    "gnn_landscape",
    "landscape_rank_agreement",
]

# The four decision-variable families (random-key chromosome blocks).
FEATURE_FAMILIES: tuple[str, ...] = ("sequence", "assignment", "empty_speed", "loaded_speed")

# Event-arc kind substrings (see event_dag.py) used to split critical-arc mass into
# the AGV resource (assignment/routing) vs the QC resource (sequencing/handling).
_AGV_KIND = "agv"
_QC_KIND_HINTS = ("qc", "handling", "arrival", "ready")
_EPS = 1e-12


def _family_criticality(ex: TapeExplanation) -> dict[str, float]:
    """Per-family makespan-criticality mass from one TAPE explanation.

    Uses the source-agnostic per-task subgradients (identical semantics for the fused model
    and the exact oracle): leg-time grads give the speed levers; the QC handling grad plus
    QC-resource critical arcs give sequence; AGV-resource critical arcs give assignment.
    """
    empty_speed = float(sum(abs(g) for g in ex.empty_time_grad))
    loaded_speed = float(sum(abs(g) for g in ex.loaded_time_grad))
    agv_arc = 0.0
    qc_arc = 0.0
    for meta, grad in zip(ex.event_edges, ex.event_edge_grad, strict=True):
        kind = str(meta.get("kind", ""))
        g = abs(grad)
        if _AGV_KIND in kind:
            agv_arc += g
        elif any(h in kind for h in _QC_KIND_HINTS):
            qc_arc += g
    sequence = float(sum(abs(g) for g in ex.node_grad)) + qc_arc
    return {
        "sequence": sequence,
        "assignment": agv_arc,
        "empty_speed": empty_speed,
        "loaded_speed": loaded_speed,
    }


def _normalise(values: dict[str, float]) -> dict[str, float]:
    """Normalise a per-family importance profile to sum to 1 (a comparable profile)."""
    total = float(sum(values.values()))
    if total <= _EPS:
        return {k: 0.0 for k in values}
    return {k: v / total for k, v in values.items()}


def _mean_profile(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {fam: 0.0 for fam in FEATURE_FAMILIES}
    return {fam: float(np.mean([r[fam] for r in rows])) for fam in FEATURE_FAMILIES}


@dataclass(frozen=True, slots=True)
class GnnLandscapeResult:
    """A model-native (or exact-oracle) landscape over the decision families."""

    source: str  # "gnn" or "exact"
    num_tasks: int
    num_samples: int
    makespan_importance: dict[str, float]   # normalised family criticality for C_max
    agv_share: float                        # AGV-routing mass / (AGV + QC) mass
    pareto_importance: dict[str, float]     # family profile on the non-dominated set
    dominated_importance: dict[str, float]  # family profile on the dominated set
    family_contrast: dict[str, float]       # pareto - dominated (why solutions are optimal)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "num_tasks": self.num_tasks,
            "num_samples": self.num_samples,
            "makespan_importance": self.makespan_importance,
            "agv_share": self.agv_share,
            "pareto_importance": self.pareto_importance,
            "dominated_importance": self.dominated_importance,
            "family_contrast": self.family_contrast,
        }


def aggregate_landscape(
    explanations: Sequence[TapeExplanation],
    objectives: Sequence[tuple[float, float]],
    *,
    source: str,
    num_tasks: int,
) -> GnnLandscapeResult:
    """Aggregate a list of TAPE explanations into a decision-family landscape.

    explanations may come from the fused model (gnn_landscape) or the exact
    oracle (exact_landscape); the aggregation is identical, which is what makes the
    GNN landscape directly comparable to the exact ground truth.
    """
    if len(explanations) != len(objectives):
        raise ValueError("explanations and objectives disagree on length")
    if not explanations:
        raise ValueError("need at least one explanation to aggregate")

    per = [_family_criticality(ex) for ex in explanations]
    mean = _mean_profile(per)
    makespan_importance = _normalise(mean)
    agv_mass = mean["assignment"] + mean["empty_speed"] + mean["loaded_speed"]
    qc_mass = mean["sequence"]
    denom = agv_mass + qc_mass
    agv_share = float(agv_mass / denom) if denom > _EPS else 0.5

    fronts = fast_non_dominated_sort([tuple(o) for o in objectives])
    front = set(fronts[0]) if fronts else set()
    front_rows = [_normalise(per[i]) for i in range(len(per)) if i in front]
    dom_rows = [_normalise(per[i]) for i in range(len(per)) if i not in front]
    pareto_importance = _mean_profile(front_rows)
    dominated_importance = _mean_profile(dom_rows)
    family_contrast = {
        fam: pareto_importance[fam] - dominated_importance[fam] for fam in FEATURE_FAMILIES
    }

    return GnnLandscapeResult(
        source=source,
        num_tasks=num_tasks,
        num_samples=len(explanations),
        makespan_importance=makespan_importance,
        agv_share=agv_share,
        pareto_importance=pareto_importance,
        dominated_importance=dominated_importance,
        family_contrast=family_contrast,
    )


def gnn_landscape(
    model: FusedEHGATv2, instance: Instance, schedules: list[Schedule]
) -> GnnLandscapeResult:
    """Model-native landscape: aggregate the fused model's TAPE attributions (scalable)."""
    exps = explain_fused_schedules(model, schedules, instance)
    objectives = [(ex.makespan, ex.energy) for ex in exps]
    return aggregate_landscape(
        exps, objectives, source="gnn", num_tasks=instance.num_tasks
    )


def exact_landscape(instance: Instance, schedules: list[Schedule]) -> GnnLandscapeResult:
    """Exact-oracle landscape (small-N ground truth) using the same aggregator.

    Uses the coupled oracle when the instance carries a peak-power budget, else the
    uncoupled one. Feasible only at small N (each call is an exact simulate + DP); this is
    the validation reference for gnn_landscape, not a scalable path.
    """
    coupled = instance.peak_power is not None
    exps = [
        explain_schedule_coupled(s, instance) if coupled else explain_schedule(s, instance)
        for s in schedules
    ]
    objectives = [(ex.makespan, ex.energy) for ex in exps]
    return aggregate_landscape(
        exps, objectives, source="exact", num_tasks=instance.num_tasks
    )


def landscape_rank_agreement(a: GnnLandscapeResult, b: GnnLandscapeResult) -> float:
    """Spearman rank correlation of the two makespan-importance profiles over the families.

    Used to validate the GNN landscape against the exact-oracle landscape at small N: a high
    value means the model recovers the same variable-importance ranking the exact oracle
    gives, which licenses trusting the GNN landscape at large N where the oracle is
    intractable.
    """
    fams = list(FEATURE_FAMILIES)
    va = [a.makespan_importance[f] for f in fams]
    vb = [b.makespan_importance[f] for f in fams]
    if np.allclose(va, va[0]) or np.allclose(vb, vb[0]):
        return 0.0
    rho = float(spearmanr(va, vb).correlation)
    return 0.0 if np.isnan(rho) else rho
