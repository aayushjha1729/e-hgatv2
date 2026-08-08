"""Surrogate dataset: random schedules -> labelled HeteroData + normalisation stats.

Training samples are drawn by decoding uniformly random 4N chromosomes (covering the
full sequence / AGV-assignment / speed decision space the guided search explores),
evaluating them with the exact physics evaluator, and encoding each as a typed graph
whose y is the true (C_max, E). Standardisation statistics are fit on the
training split only to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.evaluator import evaluate
from ehgat.environment.instance import Instance
from ehgat.surrogate.ehgatv2 import EHGATv2
from ehgat.surrogate.graph import AGV_EDGE, NODE_TYPE, build_hetero_graph
from ehgat.utils.seeding import make_rng

__all__ = [
    "NormalizationStats",
    "fit_normalization",
    "generate_graphs",
    "split_graphs",
]


def generate_graphs(instance: Instance, num_samples: int, *, seed: int = 0) -> list[HeteroData]:
    """Decode/evaluate num_samples random chromosomes into labelled graphs."""
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    rng = make_rng(seed)
    n = instance.num_tasks
    graphs: list[HeteroData] = []
    for _ in range(num_samples):
        keys = rng.random(NUM_BLOCKS * n)
        schedule = decode(keys, instance)
        evaluation = evaluate(schedule, instance)
        graphs.append(build_hetero_graph(schedule, instance, targets=evaluation.objectives))
    return graphs


def split_graphs(
    graphs: list[HeteroData],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[HeteroData], list[HeteroData], list[HeteroData]]:
    """Deterministically shuffle and split graphs into (train, val, test)."""
    if not 0.0 <= val_frac + test_frac < 1.0:
        raise ValueError("val_frac + test_frac must be in [0, 1)")
    rng = make_rng(seed)
    order = rng.permutation(len(graphs)).tolist()
    shuffled = [graphs[i] for i in order]
    n = len(shuffled)
    n_test = round(test_frac * n)
    n_val = round(val_frac * n)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature/target standardisation statistics fit on the training split."""

    node_mean: Tensor
    node_std: Tensor
    agv_mean: Tensor
    agv_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def apply_to(self, model: EHGATv2) -> None:
        """Copy these statistics into model's normalisation buffers."""
        model.set_normalization(
            node_mean=self.node_mean,
            node_std=self.node_std,
            agv_mean=self.agv_mean,
            agv_std=self.agv_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
        )


def fit_normalization(train_graphs: list[HeteroData]) -> NormalizationStats:
    """Compute standardisation statistics over the training graphs.

    Node and AGV-edge features use their pooled mean/std; the QC arcs are structural
    zeros and are excluded (the model leaves them untouched). Targets use the per-graph
    (C_max, E) mean/std so the training MSE is balanced across the two objectives.
    """
    if not train_graphs:
        raise ValueError("cannot fit normalization on an empty training set")
    node_x = torch.cat([g[NODE_TYPE].x for g in train_graphs], dim=0)
    agv_attr = torch.cat([g[AGV_EDGE].edge_attr for g in train_graphs], dim=0)
    targets = torch.cat([g.y for g in train_graphs], dim=0)
    return NormalizationStats(
        node_mean=node_x.mean(dim=0),
        node_std=node_x.std(dim=0),
        agv_mean=agv_attr.mean(dim=0),
        agv_std=agv_attr.std(dim=0),
        target_mean=targets.mean(dim=0),
        target_std=targets.std(dim=0),
    )
