"""Semantic-tensor assertions.

In a physically grounded GNN, tensor dimensions can align numerically while the
underlying physical quantities are cross-contaminated, for instance by multiplying
Empty_Energy data with weights reserved for Travel_Time. Such a mismatch does not raise
in PyTorch; the structural signal degrades into noise instead.

Every tensor entering a concatenation or linear projection is therefore validated for
(1) exact trailing-dimension size and (2) the exact ordered semantic feature signature
stamped on it at construction time. A mismatch raises SemanticTensorError with a
descriptive message.

Checks use an explicit raise rather than a bare assert and are not stripped under
python -O. The helpers require only a .shape attribute and are testable against NumPy
arrays without importing Torch.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "SemanticTensorError",
    "assert_edge_features",
    "assert_feature_dim",
    "assert_node_features",
    "assert_semantic_signature",
]


@runtime_checkable
class HasShape(Protocol):
    """Anything exposing a shape tuple (NumPy array, Torch tensor, ...)."""

    @property
    def shape(self) -> tuple[int, ...]: ...


# Canonical, ORDERED physical feature signatures. These are the single source of
# truth for the toy SA-AGV routing problem and must match the graph builder exactly.
NODE_FEATURES: tuple[str, ...] = ("Handling_Time", "Is_Load", "Is_Unload", "QC_ID")
EDGE_FEATURES: tuple[str, ...] = ("Travel_Time", "Empty_Energy", "Loaded_Energy")


class SemanticTensorError(AssertionError):
    """Raised when a tensor's shape or semantic signature violates the contract."""


def assert_feature_dim(tensor: HasShape, expected: tuple[str, ...], *, name: str) -> None:
    """Verify the trailing feature dimension equals len(expected).

    Raises
    SemanticTensorError
        If tensor is scalar/0-d or its last dimension size differs from the schema.
    """
    shape = tuple(tensor.shape)
    if len(shape) == 0:
        raise SemanticTensorError(
            f"[{name}] expected a feature tensor with a trailing dimension of size "
            f"{len(expected)} {expected}, but received a 0-d scalar."
        )
    last = shape[-1]
    if last != len(expected):
        raise SemanticTensorError(
            f"[{name}] feature-dimension mismatch: tensor has trailing dim {last} "
            f"(shape {shape}), but the physical schema requires {len(expected)} "
            f"features {expected}. Refusing to project mismatched physical quantities."
        )


def assert_semantic_signature(
    actual: tuple[str, ...], expected: tuple[str, ...], *, name: str
) -> None:
    """Verify the ORDERED semantic signature stamped on a tensor matches the schema.

    This is what prevents an edge_attr (Travel/Empty/Loaded energy) tensor from
    being silently consumed where node features (Handling_Time, ...) are expected,
    even if their sizes happened to coincide.
    """
    if tuple(actual) != tuple(expected):
        raise SemanticTensorError(
            f"[{name}] semantic-signature mismatch: tensor carries {tuple(actual)} "
            f"but {tuple(expected)} was required at this site. "
            f"A physical cross-contamination was prevented."
        )


def assert_node_features(
    tensor: HasShape, *, actual: tuple[str, ...] | None = None, name: str = "node_features"
) -> None:
    """Assert tensor is a valid node-feature tensor [*, 4] with the node schema."""
    assert_feature_dim(tensor, NODE_FEATURES, name=name)
    if actual is not None:
        assert_semantic_signature(actual, NODE_FEATURES, name=name)


def assert_edge_features(
    tensor: HasShape, *, actual: tuple[str, ...] | None = None, name: str = "edge_attr"
) -> None:
    """Assert tensor is a valid edge-feature tensor [*, 3] with the edge schema."""
    assert_feature_dim(tensor, EDGE_FEATURES, name=name)
    if actual is not None:
        assert_semantic_signature(actual, EDGE_FEATURES, name=name)
