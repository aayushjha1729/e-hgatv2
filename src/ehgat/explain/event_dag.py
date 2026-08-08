"""Differentiable event-DAG assembler shared by the exact and learned paths.

The exact makespan of a resolved schedule is the max-plus longest path over an
expanded event DAG. Each task j is split into three events so that the AGV-free
time (which gates the next task on the same AGV) is distinguishable from the task's own
completion (which, for a LOAD, includes the quay-crane handling tau):

    source --(travel)--> m(j)  [AGV+QC meet at the gate]
    m(j)   --(0)-------> q(j)  [QC finishes handling:  +tau on the node]
    ... --> a(j)               [AGV becomes free again]

Both the exact oracle (true physical leg times) and the fused E-HGATv2 head (predicted
leg times empty_t/loaded_t and node delay d) assemble their DAG here. The two paths differ
only in the source of the leg and delay tensors, not in the precedence structure.
node_weights is built with index_add so it carries gradients back to a predicted node
delay; edge_weights is a stack so it carries gradients back to predicted leg times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "EventDag",
    "assemble_coupled_event_dag",
    "assemble_event_dag",
    "extract_precedence",
]


@dataclass(slots=True)
class EventDag:
    """An assembled max-plus event DAG ready for tropical_longest_path."""

    node_weights: Tensor          # [3N + 1]
    edge_index: Tensor            # [2, E] long
    edge_weights: Tensor          # [E]
    meta: list[dict[str, Any]]    # per-arc {src, dst, task, kind}
    completion_nodes: Tensor      # [N] long -- the node whose value is task j's completion


def extract_precedence(
    agv_index: Tensor, qc_index: Tensor, num_tasks: int
) -> tuple[list[int], list[int]]:
    """Recover agv_prev / qc_prev (-1 = first on resource) from edge indices.

    Mirrors ehgat.surrogate.graph.build_hetero_graph: exactly one AGV arc per
    task with dst = j; a self-arc (src == j) marks the first task on its AGV.
    """
    agv_prev = [-1] * num_tasks
    qc_prev = [-1] * num_tasks
    src = agv_index[0].detach().cpu().tolist()
    dst = agv_index[1].detach().cpu().tolist()
    for s, d in zip(src, dst, strict=True):
        if s != d:
            agv_prev[d] = s
    qsrc = qc_index[0].detach().cpu().tolist()
    qdst = qc_index[1].detach().cpu().tolist()
    for s, d in zip(qsrc, qdst, strict=True):
        qc_prev[d] = s
    return agv_prev, qc_prev


def assemble_event_dag(
    is_load: Tensor,
    agv_prev: list[int],
    qc_prev: list[int],
    empty_t: Tensor,
    loaded_t: Tensor,
    node_delay: Tensor,
) -> EventDag:
    """Build the exact max-plus event DAG from per-task leg times and node delays.

    Args:
        is_load: [N] boolean/0-1 tensor; True for LOAD tasks (completion = QC
            finish, which is gated by +tau), False for UNLOAD (completion = AGV
            delivery after the loaded leg).
        agv_prev / qc_prev: predecessor task per resource, -1 if first.
        empty_t / loaded_t: [N] predicted (or exact) empty/loaded leg times.
        node_delay: [N] quay-crane handling delay tau per task, applied at q(j).

    Returns:
        An EventDag. Gradients flow to empty_t/loaded_t through
        edge_weights and to node_delay through node_weights.
    """
    n = len(agv_prev)
    if not (empty_t.shape[0] == loaded_t.shape[0] == node_delay.shape[0] == n):
        raise ValueError("leg/delay tensors and precedence lists disagree on N")
    dtype = empty_t.dtype
    device = empty_t.device
    node_delay = node_delay.to(dtype)
    source = 0

    def m(j: int) -> int:
        return 1 + 3 * j

    def q(j: int) -> int:
        return 1 + 3 * j + 1

    def a(j: int) -> int:
        return 1 + 3 * j + 2

    def prev_a(j: int) -> int:
        return source if agv_prev[j] < 0 else a(agv_prev[j])

    def prev_q(j: int) -> int:
        return source if qc_prev[j] < 0 else q(qc_prev[j])

    zero = empty_t.new_zeros(())
    edge_src: list[int] = []
    edge_dst: list[int] = []
    weights: list[Tensor] = []
    meta: list[dict[str, Any]] = []
    completion: list[int] = []
    load_q: list[int] = []  # q-nodes that carry +tau on the node (LOAD only)
    for j in range(n):
        if bool(is_load[j]):
            # m = max(arr_dropoff, qc_ready); q = m + tau (node weight); the AGV is freed
            # at the QC pickup a = m = c_j - tau (not at delivery -- FSMJ Eq 12).
            agv_arrival = empty_t[j] + loaded_t[j]
            arcs = [
                (prev_a(j), m(j), agv_arrival, "agv_to_gate"),
                (prev_q(j), m(j), zero, "qc_to_gate"),
                (m(j), q(j), zero, "handling"),
                (m(j), a(j), zero, "agv_free_load"),
            ]
            completion.append(q(j))
            load_q.append(q(j))
        else:
            # Handling overlaps travel: m = arr_pickup; q = max(m, qc_ready + tau) with the
            # +tau on the QC-chain edge (and omitted on the first crane task), not a node
            # weight; the AGV is freed at the yard a = q + loaded = r_j.
            arcs = [
                (prev_a(j), m(j), empty_t[j], "agv_to_gate"),
                (m(j), q(j), zero, "arrival"),
                (q(j), a(j), loaded_t[j], "agv_free_unload"),
            ]
            if qc_prev[j] >= 0:
                arcs.append((prev_q(j), q(j), node_delay[j], "qc_chain"))
            completion.append(a(j))
        for u, v, w, kind in arcs:
            edge_src.append(u)
            edge_dst.append(v)
            weights.append(w)
            meta.append({"src": u, "dst": v, "task": j, "kind": kind})

    num_nodes = 1 + 3 * n
    node_weights = torch.zeros(num_nodes, dtype=dtype, device=device)
    if load_q:
        q_idx = torch.tensor(load_q, dtype=torch.long, device=device)
        node_weights = node_weights.index_add(0, q_idx, node_delay[torch.tensor(
            [j for j in range(n) if bool(is_load[j])], dtype=torch.long, device=device)])
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_weights = torch.stack(weights)
    completion_nodes = torch.tensor(completion, dtype=torch.long, device=device)
    return EventDag(node_weights, edge_index, edge_weights, meta, completion_nodes)


def assemble_coupled_event_dag(
    is_load: Tensor,
    agv_prev: list[int],
    qc_prev: list[int],
    empty_t: Tensor,
    loaded_t: Tensor,
    node_delay: Tensor,
    power_arcs: list[tuple[int, int, int, int]],
) -> EventDag:
    """Max-plus event DAG for the peak-power-coupled schedule.

    Unlike assemble_event_dag (which bundles the empty+loaded legs onto a single
    arc), this builder gives every activity end its own node -- empty leg E(j),
    QC handling H(j) and loaded leg L(j) -- so the resource delays the power
    budget injects can be attached at leg granularity. Each activity's duration is its
    node weight (empty/loaded leg time, or tau for handling) and every precedence
    edge has weight 0, so x[v] = duration[v] + max_pred x[pred] is exactly the
    realized completion time. power_arcs add zero-weight edges
    blocker_leg_end -> blocked_leg_end; with the simulator's binding arcs the tropical
    longest path reproduces the coupled makespan exactly and its subgradient is the
    true coupled critical path. Gradients flow to the leg times / tau through the node
    weights.
    """
    n = len(agv_prev)
    if not (empty_t.shape[0] == loaded_t.shape[0] == node_delay.shape[0] == n):
        raise ValueError("leg/delay tensors and precedence lists disagree on N")
    dtype = empty_t.dtype
    device = empty_t.device
    node_delay = node_delay.to(dtype)
    source = 0

    # Four node slots per task: E (empty leg), L (loaded leg), H (QC handling, dur tau),
    # and slot R/C: R = LOAD AGV release at QC pickup (dur 0), C = UNLOAD completion
    # (container on AGV, dur 0). This mirrors evaluator._evaluate_power_coupled exactly.
    def e(j: int) -> int:
        return 1 + 4 * j

    def ll(j: int) -> int:
        return 1 + 4 * j + 1

    def h(j: int) -> int:
        return 1 + 4 * j + 2

    def rc(j: int) -> int:  # R for a LOAD, C for an UNLOAD
        return 1 + 4 * j + 3

    def leg_node(task: int, flag: int) -> int:
        return e(task) if flag == 0 else ll(task)

    def qc_done(j: int) -> int:
        return h(j) if bool(is_load[j]) else rc(j)

    def agv_release(j: int) -> int:
        return rc(j) if bool(is_load[j]) else ll(j)

    def prev_agv_free(j: int) -> int:
        return source if agv_prev[j] < 0 else agv_release(agv_prev[j])

    edge_src: list[int] = []
    edge_dst: list[int] = []
    meta: list[dict[str, Any]] = []
    completion: list[int] = []

    def _edge(u: int, v: int, j: int, kind: str) -> None:
        edge_src.append(u); edge_dst.append(v)
        meta.append({"src": u, "dst": v, "task": j, "kind": kind})

    for j in range(n):
        _edge(prev_agv_free(j), e(j), j, "agv_chain")
        if bool(is_load[j]):
            # empty -> loaded; AGV released at R = max(arrival, qc_ready); H = R + tau.
            _edge(e(j), ll(j), j, "load_travel")
            _edge(ll(j), rc(j), j, "deliver")
            if qc_prev[j] >= 0:
                _edge(qc_done(qc_prev[j]), rc(j), j, "qc_chain")
            _edge(rc(j), h(j), j, "handling")
            completion.append(h(j))
        else:
            # QC handling overlaps travel: H gated by the QC chain only (omitted on the
            # first crane task); C = max(arrival, qc_ready + tau); loaded leg follows C.
            if qc_prev[j] >= 0:
                _edge(qc_done(qc_prev[j]), h(j), j, "qc_chain")
                _edge(h(j), rc(j), j, "qc_ready")
            _edge(e(j), rc(j), j, "pick_at_qc")
            _edge(rc(j), ll(j), j, "unload_travel")
            completion.append(ll(j))

    for b_task, b_flag, v_task, v_flag in power_arcs:
        u, v = leg_node(b_task, b_flag), leg_node(v_task, v_flag)
        _edge(u, v, v_task, "power")

    num_nodes = 1 + 4 * n
    idx = torch.tensor(
        [e(j) for j in range(n)] + [ll(j) for j in range(n)] + [h(j) for j in range(n)],
        dtype=torch.long, device=device,
    )
    vals = torch.cat([empty_t, loaded_t, node_delay])
    node_weights = torch.zeros(num_nodes, dtype=dtype, device=device).index_add(0, idx, vals)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_weights = torch.zeros(len(edge_src), dtype=dtype, device=device)
    completion_nodes = torch.tensor(completion, dtype=torch.long, device=device)
    return EventDag(node_weights, edge_index, edge_weights, meta, completion_nodes)
