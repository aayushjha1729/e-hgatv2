"""Unit tests for the Table 4 distance matrix loader."""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.environment.distance import DistanceMatrix, load_default_distance_matrix


def test_default_matrix_loads_with_12_nodes() -> None:
    dm = load_default_distance_matrix()
    assert len(dm.nodes) == 12
    assert dm.nodes[0] == "QC1"
    assert dm.nodes[-1] == "LU6"


def test_known_distances() -> None:
    dm = load_default_distance_matrix()
    assert dm.distance("QC1", "QC1") == 0.0
    assert dm.distance("QC1", "LU6") == 400.0
    assert dm.distance("LU1", "QC6") == 400.0
    assert dm.distance("QC6", "QC1") == 450.0


def test_matrix_is_asymmetric() -> None:
    dm = load_default_distance_matrix()
    # Unidirectional paths => QC1->LU1 differs from LU1->QC1.
    assert dm.distance("QC1", "LU1") == 150.0
    assert dm.distance("LU1", "QC1") == 150.0
    assert dm.distance("QC1", "QC2") != dm.distance("QC2", "QC1")


def test_rejects_nonzero_diagonal() -> None:
    bad = np.array([[1.0, 2.0], [3.0, 0.0]])
    with pytest.raises(ValueError, match="diagonal"):
        DistanceMatrix(["A", "B"], bad)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        DistanceMatrix(["A", "B", "C"], np.zeros((2, 2)))
