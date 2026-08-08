"""Schedule -> typed heterogeneous graph (HeteroData) for the E-HGATv2 surrogate.

The disjunctive schedule graph is encoded with one node type and two edge types,
mirroring the {agv_prev -> j, qc_prev -> j} precedence arcs that the evaluator's
max-plus longest path runs over (evaluator.py):

Node type "task" -- features in the canonical NODE_FEATURES order
    (Handling_Time, Is_Load, Is_Unload, QC_ID).

Edge type ("task", "agv", "task") -- disjunctive resource arc i -> j when
    task j immediately follows i on the same AGV. It carries j's physical
    movement: the empty leg dropoff(i) -> pickup(j) plus the loaded leg
    pickup(j) -> dropoff(j). The first task on each AGV gets a self-arc
    j -> j carrying the empty leg agv_start -> pickup(j) plus its loaded leg, so
    exactly one AGV arc per task holds its full travel/energy and the summed
    Empty_Energy + Loaded_Energy over AGV arcs equals the schedule's total energy
    E. These arcs are what the attention-guided mutation targets.

Edge type ("task", "qc", "task") -- serialisation arc i -> j when j
    immediately follows i on the same QC. No AGV motion is associated with a QC
    precedence arc, so its EDGE_FEATURES are all zero: the coupling is structural and
    the handling times live on the nodes.

Edge features are in the canonical EDGE_FEATURES order
(Travel_Time, Empty_Energy, Loaded_Energy); Travel_Time is the arc's total AGV
travel time empty_time + loaded_time.
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from ehgat.environment.decoder import Schedule
from ehgat.environment.evaluator import build_precedence
from ehgat.environment.instance import Instance, TaskKind
from ehgat.environment.physics import leg_energy, travel_time
from ehgat.utils.assertions import (
    EDGE_FEATURES,
    NODE_FEATURES,
    SemanticTensorError,
    assert_edge_features,
    assert_node_features,
    assert_semantic_signature,
)

__all__ = [
    "AGV_EDGE",
    "NODE_TYPE",
    "QC_EDGE",
    "assert_graph_semantics",
    "build_hetero_graph",
]

NODE_TYPE = "task"
AGV_EDGE = (NODE_TYPE, "agv", NODE_TYPE)
QC_EDGE = (NODE_TYPE, "qc", NODE_TYPE)


def _node_features(instance: Instance) -> torch.Tensor:
    """[N, 4] tensor in NODE_FEATURES order: (Handling_Time, Is_Load, Is_Unload, QC_ID)."""
    qc_index = {qc: i for i, qc in enumerate(instance.qcs)}
    rows = [
        [
            float(task.handling_time),
            1.0 if task.kind is TaskKind.LOAD else 0.0,
            1.0 if task.kind is TaskKind.UNLOAD else 0.0,
            float(qc_index[task.qc]),
        ]
        for task in instance.tasks
    ]
    return torch.tensor(rows, dtype=torch.float32)


def _agv_arc_features(
    origin: str, task_id: int, schedule: Schedule, instance: Instance
) -> list[float]:
    """[Travel_Time, Empty_Energy, Loaded_Energy] for the AGV arc delivering task_id."""
    task = instance.tasks[task_id]
    empty_dist = instance.distance.distance(origin, task.pickup)
    loaded_dist = instance.loaded_distance(task)
    empty_lvl = schedule.empty_speed[task_id]
    loaded_lvl = schedule.loaded_speed[task_id]
    empty_t = travel_time(empty_dist, empty_lvl, loaded=False)
    loaded_t = travel_time(loaded_dist, loaded_lvl, loaded=True)
    empty_e = leg_energy(empty_dist, empty_lvl, loaded=False)
    loaded_e = leg_energy(loaded_dist, loaded_lvl, loaded=True)
    return [empty_t + loaded_t, empty_e, loaded_e]


def build_hetero_graph(
    schedule: Schedule,
    instance: Instance,
    *,
    targets: tuple[float, float] | None = None,
) -> HeteroData:
    """Build a typed HeteroData for schedule (optionally with y targets).

    targets is the (makespan, energy) label attached as data.y of shape
    [1, 2] for surrogate training. Raises
    ehgat.environment.evaluator.ScheduleCycleError if the schedule deadlocks.
    """
    n = instance.num_tasks
    agv_prev, qc_prev, _ = build_precedence(
        schedule.agv_sequences, schedule.qc_sequences, n
    )

    agv_src: list[int] = []
    agv_dst: list[int] = []
    agv_attr: list[list[float]] = []
    for j in range(n):
        i = agv_prev[j]
        origin = instance.agv_start if i < 0 else instance.tasks[i].dropoff
        src = j if i < 0 else i  # self-arc carries the empty leg from agv_start
        agv_src.append(src)
        agv_dst.append(j)
        agv_attr.append(_agv_arc_features(origin, j, schedule, instance))

    qc_src: list[int] = []
    qc_dst: list[int] = []
    for j in range(n):
        i = qc_prev[j]
        if i >= 0:
            qc_src.append(i)
            qc_dst.append(j)

    data = HeteroData()
    data[NODE_TYPE].x = _node_features(instance)
    data[NODE_TYPE].num_nodes = n

    data[AGV_EDGE].edge_index = torch.tensor([agv_src, agv_dst], dtype=torch.long)
    data[AGV_EDGE].edge_attr = torch.tensor(agv_attr, dtype=torch.float32).reshape(-1, 3)

    data[QC_EDGE].edge_index = torch.tensor([qc_src, qc_dst], dtype=torch.long).reshape(2, -1)
    data[QC_EDGE].edge_attr = torch.zeros((len(qc_src), 3), dtype=torch.float32)

    # Stamp the ordered physical signatures so the model / tests can detect any
    # cross-contamination (an edge tensor flowing where node features are expected, etc.).
    data.node_signature = list(NODE_FEATURES)
    data.edge_signature = list(EDGE_FEATURES)

    if targets is not None:
        data.y = torch.tensor([list(targets)], dtype=torch.float32)

    assert_graph_semantics(data)
    return data


def _coerce_signature(raw: object, default: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    """Return a single signature tuple, tolerating PyG's batched list-of-signatures.

    A single graph stores node_signature = ["Handling_Time", ...]; a PyG mini-batch
    collates this into [[...], [...], ...] (one per graph). Both must reduce to one
    signature, and a batch must not mix signatures.
    """
    if raw is None:
        return tuple(default)
    seq = list(raw)  # type: ignore[call-overload]
    if seq and isinstance(seq[0], (list, tuple)):  # batched per-graph signatures
        first = tuple(seq[0])
        for other in seq[1:]:
            if tuple(other) != first:
                raise SemanticTensorError(
                    f"[{name}] a mini-batch mixes different feature signatures: "
                    f"{first} vs {tuple(other)}."
                )
        return first
    return tuple(str(s) for s in seq)


def assert_graph_semantics(data: HeteroData) -> None:
    """Validate node/edge feature dimensions and ordered semantic signatures.

    This is the violent failsafe of the anti-Semantic-Tensor-Error contract: it is
    invoked by the model's forward and is independently unit-tested by tampering with
    a graph's tensors/signatures to confirm it crashes loudly.
    """
    node_sig = _coerce_signature(
        getattr(data, "node_signature", None), NODE_FEATURES, name="task.node_signature"
    )
    edge_sig = _coerce_signature(
        getattr(data, "edge_signature", None), EDGE_FEATURES, name="edge_signature"
    )

    assert_node_features(data[NODE_TYPE].x, actual=node_sig, name="task.x")
    assert_semantic_signature(node_sig, NODE_FEATURES, name="task.node_signature")

    for edge_type in (AGV_EDGE, QC_EDGE):
        store = data[edge_type]
        edge_attr = store.edge_attr
        assert_edge_features(edge_attr, actual=edge_sig, name=f"{edge_type[1]}.edge_attr")
        if store.edge_index.shape[0] != 2:
            raise SemanticTensorError(
                f"[{edge_type[1]}.edge_index] must have shape [2, E], got "
                f"{tuple(store.edge_index.shape)}."
            )
        if store.edge_index.shape[1] != edge_attr.shape[0]:
            raise SemanticTensorError(
                f"[{edge_type[1]}] edge_index has {store.edge_index.shape[1]} arcs but "
                f"edge_attr has {edge_attr.shape[0]} rows; structural/feature mismatch."
            )
    assert_semantic_signature(edge_sig, EDGE_FEATURES, name="edge_signature")
