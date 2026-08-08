"""Distance matrix loader and lookup (Homayouni & Fontes 2022, Table 4).

The matrix is asymmetric (unidirectional terminal paths). Rows are origins, columns
are destinations. Node labels are QC1..QC6 and LU1..LU6.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

__all__ = ["DistanceMatrix", "load_default_distance_matrix"]

_DATA_PACKAGE = "ehgat.environment.data"
_DEFAULT_FILENAME = "distance_matrix.json"


class DistanceMatrix:
    """Immutable origin->destination distance lookup keyed by node label."""

    def __init__(self, nodes: list[str], matrix: np.ndarray) -> None:
        n = len(nodes)
        if matrix.shape != (n, n):
            raise ValueError(f"matrix shape {matrix.shape} does not match {n} nodes")
        if len(set(nodes)) != n:
            raise ValueError("node labels must be unique")
        if np.any(matrix < 0):
            raise ValueError("distances must be non-negative")
        if np.any(np.diag(matrix) != 0):
            raise ValueError("self-distances (diagonal) must be zero")
        self._nodes = list(nodes)
        self._index = {label: i for i, label in enumerate(nodes)}
        self._matrix = matrix.astype(float)

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    def distance(self, origin: str, destination: str) -> float:
        """Distance (m) travelling from origin to destination."""
        try:
            i = self._index[origin]
            j = self._index[destination]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown node label: {exc.args[0]!r}") from exc
        return float(self._matrix[i, j])

    @classmethod
    def from_json(cls, path: str | Path) -> DistanceMatrix:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["nodes"], np.asarray(payload["matrix"], dtype=float))


def load_default_distance_matrix() -> DistanceMatrix:
    """Load the packaged Table 4 distance matrix.

    Falls back to the repository data/ directory when running from a source
    checkout in which the file has not been copied into the package.
    """
    try:
        with (
            resources.files(_DATA_PACKAGE)
            .joinpath(_DEFAULT_FILENAME)
            .open("r", encoding="utf-8") as fh
        ):
            payload = json.load(fh)
        return DistanceMatrix(payload["nodes"], np.asarray(payload["matrix"], dtype=float))
    except (ModuleNotFoundError, FileNotFoundError):
        repo_path = Path(__file__).resolve().parents[3] / "data" / _DEFAULT_FILENAME
        return DistanceMatrix.from_json(repo_path)
