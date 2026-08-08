"""Unit tests for the canonical 4N random-key schedule decoder."""

from __future__ import annotations

import numpy as np
import pytest

from ehgat.environment.decoder import NUM_BLOCKS, bucket, decode, encode_canonical
from ehgat.environment.instance import build_toy_instance
from ehgat.environment.physics import SpeedLevel
from ehgat.utils.seeding import make_rng


def _random_keys(seed: int, n: int) -> np.ndarray:
    return make_rng(seed).random(NUM_BLOCKS * n)


def test_bucket_interval_rule() -> None:
    assert bucket(0.0, 3) == 0
    assert bucket(0.33, 3) == 0
    assert bucket(0.34, 3) == 1
    assert bucket(0.99, 3) == 2
    assert bucket(1.0, 3) == 2  # right endpoint maps to last bucket


def test_bucket_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="random key"):
        bucket(1.5, 3)


def test_decode_rejects_wrong_length() -> None:
    inst = build_toy_instance()
    with pytest.raises(ValueError, match="shape"):
        decode(np.zeros(3 * inst.num_tasks), inst)


def test_decode_rejects_out_of_range_keys() -> None:
    inst = build_toy_instance()
    keys = _random_keys(0, inst.num_tasks)
    keys[0] = 2.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decode(keys, inst)


def test_global_order_is_permutation() -> None:
    inst = build_toy_instance()
    sched = decode(_random_keys(7, inst.num_tasks), inst)
    assert sorted(sched.global_order) == list(range(inst.num_tasks))


def test_agv_sequences_partition_all_tasks() -> None:
    inst = build_toy_instance()
    sched = decode(_random_keys(7, inst.num_tasks), inst)
    flat = [t for seq in sched.agv_sequences for t in seq]
    assert sorted(flat) == list(range(inst.num_tasks))
    assert len(sched.agv_sequences) == inst.num_agvs


def test_qc_sequences_partition_all_tasks() -> None:
    inst = build_toy_instance()
    sched = decode(_random_keys(7, inst.num_tasks), inst)
    flat = [t for seq in sched.qc_sequences for t in seq]
    assert sorted(flat) == list(range(inst.num_tasks))


def test_projections_respect_global_order() -> None:
    inst = build_toy_instance()
    sched = decode(_random_keys(11, inst.num_tasks), inst)
    rank = {t: i for i, t in enumerate(sched.global_order)}
    for seq in (*sched.agv_sequences, *sched.qc_sequences):
        ranks = [rank[t] for t in seq]
        assert ranks == sorted(ranks)  # each projection preserves the global order


def test_decode_is_deterministic() -> None:
    inst = build_toy_instance()
    keys = _random_keys(3, inst.num_tasks)
    assert decode(keys, inst) == decode(keys, inst)


def test_speed_levels_are_valid() -> None:
    inst = build_toy_instance()
    sched = decode(_random_keys(5, inst.num_tasks), inst)
    valid = set(SpeedLevel)
    assert set(sched.empty_speed) <= valid
    assert set(sched.loaded_speed) <= valid


def test_encode_decode_round_trip() -> None:
    inst = build_toy_instance()
    for seed in range(5):
        sched = decode(_random_keys(seed, inst.num_tasks), inst)
        regenerated = decode(encode_canonical(sched, inst), inst)
        assert regenerated == sched
