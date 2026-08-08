"""Tests for the semantic-tensor assertion contract (Directive Section I).

These are 'violent failsafe' tests: they assert the code crashes loudly on physical
cross-contamination. NumPy arrays stand in for tensors (torch-agnostic by design).
"""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.utils.assertions import (
    EDGE_FEATURES,
    NODE_FEATURES,
    SemanticTensorError,
    assert_edge_features,
    assert_node_features,
    assert_semantic_signature,
)


def test_valid_node_features_pass() -> None:
    x = np.zeros((10, len(NODE_FEATURES)))
    assert_node_features(x, actual=NODE_FEATURES)


def test_valid_edge_features_pass() -> None:
    e = np.zeros((24, len(EDGE_FEATURES)))
    assert_edge_features(e, actual=EDGE_FEATURES)


def test_edge_tensor_rejected_where_node_expected() -> None:
    # 3-d edge tensor must not satisfy the 4-d node schema.
    e = np.zeros((10, len(EDGE_FEATURES)))
    with pytest.raises(SemanticTensorError, match="feature-dimension mismatch"):
        assert_node_features(e)


def test_signature_mismatch_crashes_even_when_size_matches() -> None:
    # Same trailing size as node schema (4), but the wrong ordered signature.
    wrong = ("Travel_Time", "Empty_Energy", "Loaded_Energy", "Bogus")
    x = np.zeros((5, 4))
    with pytest.raises(SemanticTensorError, match="semantic-signature mismatch"):
        assert_node_features(x, actual=wrong)


def test_scalar_rejected() -> None:
    with pytest.raises(SemanticTensorError, match="0-d scalar"):
        assert_edge_features(np.asarray(3.0))


def test_assert_semantic_signature_order_sensitive() -> None:
    permuted = ("Empty_Energy", "Travel_Time", "Loaded_Energy")
    with pytest.raises(SemanticTensorError, match="cross-contamination"):
        assert_semantic_signature(permuted, EDGE_FEATURES, name="edge_attr")
