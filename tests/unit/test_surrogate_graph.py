"""Tests for the typed HeteroData graph builder (surrogate/graph.py).

Covers the physical encoding invariants (AGV-arc energy == total energy, QC arcs are
structural zeros) and the anti-Semantic-Tensor-Error must-crash contract.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ehgat.environment.decoder import NUM_BLOCKS, decode  # noqa: E402
from ehgat.environment.evaluator import evaluate  # noqa: E402
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance  # noqa: E402
from ehgat.surrogate.graph import (  # noqa: E402
    AGV_EDGE,
    NODE_TYPE,
    QC_EDGE,
    assert_graph_semantics,
    build_hetero_graph,
)
from ehgat.utils.assertions import SemanticTensorError  # noqa: E402

pytestmark = pytest.mark.learn


def _instance():
    return build_toy_instance(num_tasks=EXACT_TOY_TASKS)


def _schedule_and_eval(instance, seed: int = 0):
    keys = np.random.default_rng(seed).random(NUM_BLOCKS * instance.num_tasks)
    schedule = decode(keys, instance)
    return schedule, evaluate(schedule, instance)


def test_graph_shapes_and_signatures() -> None:
    inst = _instance()
    schedule, ev = _schedule_and_eval(inst)
    data = build_hetero_graph(schedule, inst, targets=ev.objectives)

    assert tuple(data[NODE_TYPE].x.shape) == (inst.num_tasks, 4)
    # Exactly one incoming AGV arc per task (predecessor or self-arc for AGV firsts).
    assert tuple(data[AGV_EDGE].edge_index.shape) == (2, inst.num_tasks)
    assert tuple(data[AGV_EDGE].edge_attr.shape) == (inst.num_tasks, 3)
    assert data[QC_EDGE].edge_index.shape[0] == 2
    assert data[QC_EDGE].edge_attr.shape[1] == 3
    assert tuple(data.y.shape) == (1, 2)
    assert tuple(data.node_signature) == ("Handling_Time", "Is_Load", "Is_Unload", "QC_ID")
    assert tuple(data.edge_signature) == ("Travel_Time", "Empty_Energy", "Loaded_Energy")


def test_agv_arc_energy_equals_total_energy() -> None:
    # The single most important invariant: summed Empty+Loaded arc energy == evaluator E.
    inst = _instance()
    for seed in range(5):
        schedule, ev = _schedule_and_eval(inst, seed)
        data = build_hetero_graph(schedule, inst)
        attr = data[AGV_EDGE].edge_attr
        graph_energy = float(attr[:, 1].sum() + attr[:, 2].sum())
        assert graph_energy == pytest.approx(ev.energy, rel=1e-6)


def test_qc_arcs_are_structural_zeros() -> None:
    inst = _instance()
    schedule, _ = _schedule_and_eval(inst)
    data = build_hetero_graph(schedule, inst)
    assert torch.all(data[QC_EDGE].edge_attr == 0.0)


def test_qc_arc_count_matches_precedence() -> None:
    inst = _instance()
    schedule, _ = _schedule_and_eval(inst)
    data = build_hetero_graph(schedule, inst)
    expected = sum(max(len(seq) - 1, 0) for seq in schedule.qc_sequences)
    assert data[QC_EDGE].edge_index.shape[1] == expected


def test_wrong_node_dim_crashes() -> None:
    inst = _instance()
    schedule, _ = _schedule_and_eval(inst)
    data = build_hetero_graph(schedule, inst)
    data[NODE_TYPE].x = torch.zeros((inst.num_tasks, 3))  # edge-shaped where node expected
    with pytest.raises(SemanticTensorError, match="feature-dimension mismatch"):
        assert_graph_semantics(data)


def test_tampered_node_signature_crashes() -> None:
    inst = _instance()
    schedule, _ = _schedule_and_eval(inst)
    data = build_hetero_graph(schedule, inst)
    data.node_signature = ["Travel_Time", "Empty_Energy", "Loaded_Energy", "Bogus"]
    with pytest.raises(SemanticTensorError, match="semantic-signature mismatch"):
        assert_graph_semantics(data)


def test_edge_index_attr_mismatch_crashes() -> None:
    inst = _instance()
    schedule, _ = _schedule_and_eval(inst)
    data = build_hetero_graph(schedule, inst)
    data[AGV_EDGE].edge_attr = data[AGV_EDGE].edge_attr[:-1]  # drop a row
    with pytest.raises(SemanticTensorError, match="structural/feature mismatch"):
        assert_graph_semantics(data)
