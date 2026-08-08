"""Graph-native explainability utilities for E-HGATv2."""

from ehgat.explain.event_dag import EventDag, assemble_event_dag, extract_precedence
from ehgat.explain.fused_ehgat import FusedEHGATv2, FusedPrediction
from ehgat.explain.fused_explainer import (
    FaithfulnessReport,
    explain_fused,
    explain_fused_schedules,
    faithfulness_report,
    fused_tradeoff_criticality_scores,
)
from ehgat.explain.gnn_landscape import (
    FEATURE_FAMILIES,
    GnnLandscapeResult,
    aggregate_landscape,
    exact_landscape,
    gnn_landscape,
    landscape_rank_agreement,
)
from ehgat.explain.tcs_calculator import ParetoPoint, tradeoff_criticality_scores
from ehgat.explain.tape_explainer import TapeExplanation, explain_schedule, explain_schedules
from ehgat.explain.tropical_dp import TropicalMaxPlus, tropical_longest_path, tropical_makespan

__all__ = [
    "FEATURE_FAMILIES",
    "EventDag",
    "FaithfulnessReport",
    "FusedEHGATv2",
    "FusedPrediction",
    "GnnLandscapeResult",
    "ParetoPoint",
    "TapeExplanation",
    "TropicalMaxPlus",
    "aggregate_landscape",
    "assemble_event_dag",
    "exact_landscape",
    "explain_fused",
    "explain_fused_schedules",
    "explain_schedule",
    "explain_schedules",
    "extract_precedence",
    "faithfulness_report",
    "fused_tradeoff_criticality_scores",
    "gnn_landscape",
    "landscape_rank_agreement",
    "tradeoff_criticality_scores",
    "tropical_longest_path",
    "tropical_makespan",
]
