"""Tests for the DS/DL benchmark loader (real published-instance ingestion)."""

from __future__ import annotations

import json

import pytest

from ehgat.environment.dsdl import load_dsdl_dataset, load_dsdl_instance
from ehgat.environment.instance import TaskKind


def _ds_record() -> dict:
    return {
        "num_agvs": 2,
        "qcs": ["QC1", "QC2"],
        "tasks": [
            {"qc": "QC1", "handling_time": 45},
            {"qc": "QC1", "handling_time": 52, "kind": "UNLOAD"},
            {"qc": "QC2", "handling_time": 30, "lu": "LU3"},
        ],
    }


def test_loads_real_geometry_ds_instance() -> None:
    rec = load_dsdl_instance("DS01", _ds_record())
    inst = rec.instance
    assert rec.synthetic_geometry is False
    assert inst.num_tasks == 3
    assert inst.qcs == ("QC1", "QC2")
    assert inst.num_agvs == 2
    # task_ids are a contiguous 0..N-1 range and order is preserved
    assert [t.task_id for t in inst.tasks] == [0, 1, 2]
    # explicit kind / lu respected; defaults applied otherwise
    assert inst.tasks[1].kind is TaskKind.UNLOAD
    assert inst.tasks[2].lu == "LU3"
    # default alternation: index 0 -> LOAD
    assert inst.tasks[0].kind is TaskKind.LOAD


def test_kind_alternation_and_lu_round_robin() -> None:
    rec = load_dsdl_instance(
        "DS02",
        {"num_agvs": 2, "qcs": ["QC1"], "lu_stations": ["LU1", "LU2"],
         "tasks": [{"qc": "QC1", "handling_time": 40} for _ in range(4)]},
    )
    kinds = [t.kind for t in rec.instance.tasks]
    assert kinds == [TaskKind.LOAD, TaskKind.UNLOAD, TaskKind.LOAD, TaskKind.UNLOAD]
    lus = [t.lu for t in rec.instance.tasks]
    assert lus == ["LU1", "LU2", "LU1", "LU2"]


def test_large_dl_requires_opt_in_for_synthetic_geometry() -> None:
    rec_dict = {"num_agvs": 4, "qcs": [f"QC{i}" for i in range(1, 9)],
                "tasks": [{"qc": "QC8", "handling_time": 50}]}
    with pytest.raises(ValueError, match="missing nodes"):
        load_dsdl_instance("DL01", rec_dict)  # QC7/QC8 not in real matrix
    rec = load_dsdl_instance("DL01", rec_dict, allow_synthetic_geometry=True)
    assert rec.synthetic_geometry is True
    assert rec.instance.num_tasks == 1


def test_coupled_variant_sets_peak_power() -> None:
    rec = load_dsdl_instance("DS01", _ds_record(), peak_power=30.0)
    assert rec.instance.peak_power == 30.0


def test_load_dataset_from_file(tmp_path) -> None:
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"DS01": _ds_record()}))
    recs = load_dsdl_dataset(path)
    assert len(recs) == 1 and recs[0].instance_id == "DS01"
