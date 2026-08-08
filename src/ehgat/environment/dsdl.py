"""Loader for the FSMJ-2023 (Fontes & Homayouni) DS/DL container-terminal benchmark.

The published instances are given as dense tables (Table 5 task sets and Table 4
QC<->LU distance matrix). This module turns a transcribed JSON of those task sets into
validated ehgat.environment.instance.Instance objects so the surrogate / search /
explanation experiments run on the published instances rather than the generated family.

Two regimes are produced from the same record:
- uncoupled (peak_power=None) — the original bi-objective DS/DL problem (Papers A/B);
- coupled (peak_power=<kW>) — the author's peak-power extension applied on top of
  the published instance (published geometry and task lists, plus the coupling).

Geometry note: the packaged Table 4 matrix only spans QC1..QC6 + LU1..LU6 (the small
DS set). Large DL instances (8-16 QCs) exceed it; pass an extended DistanceMatrix or set
allow_synthetic_geometry=True to synthesize a deterministic grid for the missing nodes
(flagged in the returned metadata so it is never silently treated as "real").

JSON schema (one dataset file = mapping of instance-id -> record)::

    {
      "DS01": {
        "num_agvs": 2,
        "qcs": ["QC1", "QC2"],                 # optional; inferred + sorted from tasks if absent
        "lu_stations": ["LU1","LU2","LU3"],    # optional; default = LUs present in the distance matrix
        "kind_assignment": "alternate",        # "alternate" (dual-cycle) | "load" | "unload"; default alternate
        "tasks": [
          {"qc": "QC1", "handling_time": 45},                   # minimal: qc + handling_time
          {"qc": "QC1", "handling_time": 60, "kind": "UNLOAD"}, # optional explicit kind
          {"qc": "QC2", "handling_time": 30, "lu": "LU2"}       # optional explicit LU
        ]
      },
      "DS02": { ... }
    }

Only qc and handling_time are required per task. kind defaults to a dual-cycling
LOAD/UNLOAD alternation (index parity); lu defaults to a round-robin over lu_stations.
Task order is preserved (it defines each QC's ordered task list J_k).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ehgat.environment.distance import DistanceMatrix, load_default_distance_matrix
from ehgat.environment.instance import Instance, Task, TaskKind

__all__ = [
    "DsDlRecord",
    "load_dsdl_dataset",
    "load_dsdl_instance",
    "load_tables_4_5",
]


@dataclass(frozen=True)
class DsDlRecord:
    """A loaded DS/DL instance plus provenance metadata."""

    instance_id: str
    instance: Instance
    synthetic_geometry: bool  # True if any node fell back to synthesized distances


def _kind_for(index: int, explicit: str | None, strategy: str) -> TaskKind:
    if explicit is not None:
        return TaskKind[explicit.strip().upper()]
    strat = strategy.strip().lower()
    if strat == "load":
        return TaskKind.LOAD
    if strat == "unload":
        return TaskKind.UNLOAD
    if strat == "alternate":
        return TaskKind.LOAD if index % 2 == 0 else TaskKind.UNLOAD
    raise ValueError(f"unknown kind_assignment {strategy!r} (use alternate|load|unload)")


def _synthesize_distance(nodes: list[str]) -> DistanceMatrix:
    """Deterministic asymmetric grid geometry for nodes outside the published Table 4 matrix.

    QCs and LUs are laid on two parallel rows; distance = Manhattan-style with a small
    directional asymmetry so empty/loaded legs differ (mirroring the published matrix).
    Used only when the published matrix lacks a node (large DL instances) and explicitly opted in.
    """
    qcs = sorted(n for n in nodes if n.upper().startswith("QC"))
    lus = sorted(n for n in nodes if n.upper().startswith("LU"))
    pos: dict[str, tuple[float, float]] = {}
    for i, q in enumerate(qcs):
        pos[q] = (float(i) * 40.0, 0.0)
    for i, lu in enumerate(lus):
        pos[lu] = (float(i) * 40.0, 60.0)
    ordered = qcs + lus
    n = len(ordered)
    mat = np.zeros((n, n), dtype=float)
    for a, na in enumerate(ordered):
        for b, nb in enumerate(ordered):
            if a == b:
                continue
            (xa, ya), (xb, yb) = pos[na], pos[nb]
            base = abs(xa - xb) + abs(ya - yb)
            # +10% going "down" (increasing y) to break symmetry deterministically.
            mat[a, b] = base * (1.10 if yb > ya else 1.0) + 10.0
    return DistanceMatrix(ordered, mat)


def load_dsdl_instance(
    instance_id: str,
    record: dict,
    *,
    distance: DistanceMatrix | None = None,
    peak_power: float | None = None,
    allow_synthetic_geometry: bool = False,
) -> DsDlRecord:
    """Build one Instance from a transcribed DS/DL record (see module docstring)."""
    raw_tasks = record.get("tasks")
    if not raw_tasks:
        raise ValueError(f"{instance_id}: record has no 'tasks'")
    num_agvs = int(record.get("num_agvs", 2))
    kind_strategy = str(record.get("kind_assignment", "alternate"))

    qcs = tuple(record["qcs"]) if record.get("qcs") else tuple(
        sorted({str(t["qc"]) for t in raw_tasks}, key=lambda s: (len(s), s))
    )

    base_distance = distance or load_default_distance_matrix()
    known = set(base_distance.nodes)
    lu_stations = tuple(record.get("lu_stations") or [n for n in base_distance.nodes if n.upper().startswith("LU")])
    if not lu_stations:
        raise ValueError(f"{instance_id}: no LU stations available")

    # Determine the full node set the instance needs.
    needed = set(qcs) | set(lu_stations)
    synthetic = False
    if not needed.issubset(known):
        missing = needed - known
        if not allow_synthetic_geometry:
            raise ValueError(
                f"{instance_id}: distance matrix is missing nodes {sorted(missing)} "
                f"(the published Table 4 covers only QC1..QC6/LU1..LU6). Pass an extended "
                f"DistanceMatrix or allow_synthetic_geometry=True for large DL instances."
            )
        base_distance = _synthesize_distance(sorted(needed))
        synthetic = True

    tasks: list[Task] = []
    for idx, t in enumerate(raw_tasks):
        qc = str(t["qc"])
        handling = float(t["handling_time"])
        kind = _kind_for(idx, t.get("kind"), kind_strategy)
        lu = str(t["lu"]) if t.get("lu") else lu_stations[idx % len(lu_stations)]
        tasks.append(Task(task_id=idx, qc=qc, lu=lu, kind=kind, handling_time=handling))

    instance = Instance(
        tasks=tuple(tasks),
        qcs=qcs,
        num_agvs=num_agvs,
        agv_start=lu_stations[0],
        distance=base_distance,
        peak_power=peak_power,
    )
    return DsDlRecord(instance_id=instance_id, instance=instance, synthetic_geometry=synthetic)


def _distance_from_table4(table4: dict) -> DistanceMatrix:
    """Build a DistanceMatrix from the table4_distance_matrix_m block."""
    nodes = list(table4["nodes"])
    matrix = np.asarray(table4["matrix"], dtype=float)
    return DistanceMatrix(nodes, matrix)


def load_tables_4_5(
    path: str | Path,
    *,
    peak_power: float | None = None,
    num_agvs: int = 2,
    agv_start: str = "LU1",
    only: list[str] | None = None,
) -> list[DsDlRecord]:
    """Load data/tables_4_5.json (Homayouni & Fontes 2022 book chapter).

    This is the loading / single-cycle small-instance set (L01..L35): Table 4 is
    the QC<->LU distance matrix and Table 5 gives, per QC row, an ordered list of
    (processing_time, lu_index) tuples. Every task is a LOAD move (loaded leg
    LU -> QC). Task ids are assigned in QC-row order so each QC's ordered task list
    J_k is preserved (matching the paper's "tasks 1..n1 belong to QC1, ..." convention).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    distance = _distance_from_table4(payload["table4_distance_matrix_m"])
    instances = payload["table5_loading_instances"]["instances"]

    out: list[DsDlRecord] = []
    for inst_id, record in instances.items():
        if only is not None and inst_id not in only:
            continue
        qcs = tuple(str(row["qc"]) for row in record["rows"])
        tasks: list[Task] = []
        for row in record["rows"]:
            qc = str(row["qc"])
            for pt, lu_idx in row["tasks"]:
                tasks.append(
                    Task(
                        task_id=len(tasks),
                        qc=qc,
                        lu=f"LU{int(lu_idx)}",
                        kind=TaskKind.LOAD,
                        handling_time=float(pt),
                    )
                )
        instance = Instance(
            tasks=tuple(tasks),
            qcs=qcs,
            num_agvs=num_agvs,
            agv_start=agv_start,
            distance=distance,
            peak_power=peak_power,
        )
        out.append(DsDlRecord(instance_id=inst_id, instance=instance, synthetic_geometry=False))
    return out


def load_dsdl_dataset(
    path: str | Path,
    *,
    distance: DistanceMatrix | None = None,
    peak_power: float | None = None,
    allow_synthetic_geometry: bool = False,
    only: list[str] | None = None,
) -> list[DsDlRecord]:
    """Load every instance in a DS/DL dataset JSON file (optionally filtered by only)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):  # allow a list of records each carrying an "id"
        payload = {str(r["id"]): r for r in payload}
    out: list[DsDlRecord] = []
    for inst_id, record in payload.items():
        if only is not None and inst_id not in only:
            continue
        out.append(
            load_dsdl_instance(
                inst_id, record,
                distance=distance, peak_power=peak_power,
                allow_synthetic_geometry=allow_synthetic_geometry,
            )
        )
    return out
