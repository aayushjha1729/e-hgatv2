"""Tests for the E-HGATv2 model (surrogate/ehgatv2.py).

Covers forward/predict shapes, batching, attention shape + determinism +
non-degeneracy, config validation, and the model-level semantic-assert crash.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch  # noqa: E402

from ehgat.environment.decoder import NUM_BLOCKS, decode  # noqa: E402
from ehgat.environment.evaluator import evaluate  # noqa: E402
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance  # noqa: E402
from ehgat.surrogate.ehgatv2 import AGV_EDGE, EHGATv2, EHGATv2Config  # noqa: E402
from ehgat.surrogate.graph import NODE_TYPE, build_hetero_graph  # noqa: E402
from ehgat.utils.assertions import SemanticTensorError  # noqa: E402
from ehgat.utils.seeding import seed_everything  # noqa: E402

pytestmark = pytest.mark.learn


def _graphs(n: int = 6):
    inst = build_toy_instance(num_tasks=EXACT_TOY_TASKS)
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n):
        schedule = decode(rng.random(NUM_BLOCKS * inst.num_tasks), inst)
        ev = evaluate(schedule, inst)
        graphs.append(build_hetero_graph(schedule, inst, targets=ev.objectives))
    return graphs


def _model(seed: int = 0) -> EHGATv2:
    seed_everything(seed)
    return EHGATv2(EHGATv2Config(hidden=32, layers=2, heads=4))


def test_forward_and_predict_shape() -> None:
    graphs = _graphs()
    model = _model()
    out, _ = model(graphs[0])
    assert tuple(out.shape) == (1, 2)
    pred = model.predict(graphs[0])
    assert tuple(pred.shape) == (1, 2)
    assert torch.isfinite(pred).all()


def test_batched_forward_shape() -> None:
    graphs = _graphs()
    model = _model()
    batch = Batch.from_data_list(graphs)
    out, _ = model(batch)
    assert tuple(out.shape) == (len(graphs), 2)


def test_attention_shapes_and_detached() -> None:
    graphs = _graphs()
    model = _model()
    g = graphs[0]
    attn = model.attention(g)
    agv_ei, agv_alpha = attn["agv"]
    assert tuple(agv_ei.shape) == (2, g[AGV_EDGE].edge_index.shape[1])
    assert agv_alpha.shape[0] == g[AGV_EDGE].edge_index.shape[1]
    assert not agv_alpha.requires_grad
    # attention weights are valid probabilities of a softmax over <= 2 relations
    assert float(agv_alpha.min()) >= 0.0
    assert float(agv_alpha.max()) <= 1.0 + 1e-6


def test_attention_is_non_degenerate() -> None:
    # Cross-resource attention is required to vary across arcs; a per-relation GATv2
    # softmax over a single predecessor would be the constant 1.
    graphs = _graphs()
    model = _model()
    spreads = []
    for g in graphs:
        alpha = model.attention(g)["agv"][1]
        spreads.append(float(alpha.max() - alpha.min()))
    assert max(spreads) > 1e-3


def test_attention_deterministic() -> None:
    graphs = _graphs()
    model = _model()
    g = graphs[1]
    a1 = model.attention(g)["agv"][1]
    a2 = model.attention(g)["agv"][1]
    assert torch.equal(a1, a2)


def test_init_and_forward_deterministic_across_seeded_models() -> None:
    graphs = _graphs()
    out1, _ = _model(0)(graphs[0])
    out2, _ = _model(0)(graphs[0])
    assert torch.equal(out1, out2)


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EHGATv2Config(hidden=30, heads=4)


def test_model_crashes_on_malformed_graph() -> None:
    graphs = _graphs()
    model = _model()
    bad = graphs[0]
    bad[NODE_TYPE].x = torch.zeros((bad[NODE_TYPE].num_nodes, 3))
    with pytest.raises(SemanticTensorError):
        model(bad)
