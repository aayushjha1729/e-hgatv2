"""Tests for the attention-guided NSGA-II (search/attention_nsga2.py).

The guarantees under test: (1) the no-deadlock invariant -- the direct AGV-sequence
swap operator can create an AGV/QC cycle but is always Kahn-re-validated, so every
schedule admitted to the population is acyclic; (2) Oracle-bound soundness -- because
objectives come from the exact evaluator, every front point is weakly dominated by the
golden PF*; plus determinism, evaluation accounting and a non-dominated front.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ehgat.benchmark.faithfulness import critical_agv_arcs, critical_path_binding  # noqa: E402
from ehgat.environment.decoder import NUM_BLOCKS, decode  # noqa: E402
from ehgat.environment.evaluator import build_precedence, evaluate  # noqa: E402
from ehgat.environment.instance import EXACT_TOY_TASKS, build_toy_instance  # noqa: E402
from ehgat.search.attention_nsga2 import (  # noqa: E402
    _MUTATION_OPS,
    AdaptivePursuit,
    AttentionNSGA2Config,
    attention_bottleneck_task,
    attention_bottleneck_type,
    default_config,
    mutate_reassign_agv,
    mutate_speed,
    mutate_swap_on_agv,
    mutate_swap_on_qc,
    operator_probabilities,
    operator_reward,
    run_attention_nsga2,
)
from ehgat.surrogate.train import TrainConfig, train_surrogate  # noqa: E402
from ehgat.utils.seeding import make_rng  # noqa: E402

pytestmark = pytest.mark.learn

GOLDEN_FRONT_N5 = Path(__file__).resolve().parents[1] / "data" / "golden" / "exact_front_n5.json"


def _golden_front() -> list[tuple[float, float]]:
    return [tuple(p) for p in json.loads(GOLDEN_FRONT_N5.read_text())["front"]]


def _instance():
    return build_toy_instance(num_tasks=EXACT_TOY_TASKS)


@pytest.fixture(scope="module")
def trained_model():
    inst = _instance()
    return train_surrogate(inst, TrainConfig(num_samples=400, epochs=10, seed=0)).model


def _deadlock_prone_schedule(inst):
    """A schedule where tasks 0 and 3 (both QC1) sit adjacently on one AGV.

    Swapping them on the AGV contradicts the QC1 order -> an AGV/QC deadlock.
    """
    n = inst.num_tasks
    keys = np.empty(NUM_BLOCKS * n)
    keys[0:n] = [0.1, 0.3, 0.5, 0.7, 0.9]  # global order = identity
    keys[n : 2 * n] = [0.1, 0.9, 0.9, 0.1, 0.9]  # AGV0 = {0, 3}; AGV1 = {1, 2, 4}
    keys[2 * n :] = 0.5  # arbitrary valid speeds
    return decode(keys, inst)


# --------------------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------------------
def test_mutate_speed_changes_only_target_speed() -> None:
    inst = _instance()
    sched = decode(make_rng(0).random(NUM_BLOCKS * inst.num_tasks), inst)
    rng = make_rng(1)
    mutated = mutate_speed(sched, task=2, rng=rng)
    changed = sum(
        (mutated.empty_speed[t] != sched.empty_speed[t])
        or (mutated.loaded_speed[t] != sched.loaded_speed[t])
        for t in range(inst.num_tasks)
    )
    assert changed == 1  # exactly the target task's speed changed
    evaluate(mutated, inst)  # still feasible


def test_mutate_reassign_changes_agv_and_stays_acyclic() -> None:
    inst = _instance()
    sched = decode(make_rng(2).random(NUM_BLOCKS * inst.num_tasks), inst)
    mutated = mutate_reassign_agv(sched, inst, task=1, rng=make_rng(3))
    assert mutated.assignment[1] != sched.assignment[1]
    evaluate(mutated, inst)  # re-projection keeps it acyclic


def test_swap_rejects_deadlock_returns_none() -> None:
    inst = _instance()
    sched = _deadlock_prone_schedule(inst)
    # Precondition: 0 and 3 are adjacent on the same AGV and share QC1.
    assert sched.assignment[0] == sched.assignment[3]
    # Swapping task 3 with its AGV predecessor (task 0) must deadlock -> None.
    assert mutate_swap_on_agv(sched, inst, task=3) is None
    # Task at sequence head has no predecessor -> None.
    assert mutate_swap_on_agv(sched, inst, task=0) is None


def test_swap_no_deadlock_invariant_property() -> None:
    """Over many schedules x tasks, every non-None swap result is acyclic."""
    inst = _instance()
    n = inst.num_tasks
    for seed in range(60):
        sched = decode(make_rng(seed).random(NUM_BLOCKS * n), inst)
        for task in range(n):
            mutated = mutate_swap_on_agv(sched, inst, task)
            if mutated is None:
                continue
            # Admitted schedules must have a valid topological order (no deadlock).
            build_precedence(mutated.agv_sequences, mutated.qc_sequences, n)
            evaluate(mutated, inst)


def test_swap_on_qc_no_deadlock_invariant_property() -> None:
    """Over many schedules x tasks, every non-None QC-swap result is acyclic + feasible."""
    inst = _instance()
    n = inst.num_tasks
    for seed in range(60):
        sched = decode(make_rng(seed).random(NUM_BLOCKS * n), inst)
        for task in range(n):
            mutated = mutate_swap_on_qc(sched, inst, task)
            if mutated is None:
                continue
            assert mutated.qc_sequences != sched.qc_sequences  # the QC order changed
            build_precedence(mutated.agv_sequences, mutated.qc_sequences, n)
            evaluate(mutated, inst)


def test_swap_on_qc_rejects_genuine_deadlock() -> None:
    """A non-head QC swap that contradicts the AGV order is Kahn-rejected (returns None)."""
    inst = _instance()
    sched = _deadlock_prone_schedule(inst)
    # Fixture precondition: tasks 0 and 3 share a QC (0 immediately before 3) AND the same
    # AGV (0 -> 3); reversing their QC order creates an AGV/QC cycle.
    qc_idx = next(i for i, s in enumerate(sched.qc_sequences) if 0 in s and 3 in s)
    seq = sched.qc_sequences[qc_idx]
    assert seq.index(0) + 1 == seq.index(3)
    assert sched.assignment[0] == sched.assignment[3]
    assert mutate_swap_on_qc(sched, inst, 3) is None  # genuine deadlock
    assert mutate_swap_on_qc(sched, inst, seq[0]) is None  # head of QC chain -> no predecessor


# --------------------------------------------------------------------------------------
# Attention bottleneck + full search
# --------------------------------------------------------------------------------------
def test_bottleneck_task_is_valid(trained_model) -> None:
    inst = _instance()
    sched = decode(make_rng(5).random(NUM_BLOCKS * inst.num_tasks), inst)
    task = attention_bottleneck_task(sched, inst, trained_model)
    assert 0 <= task < inst.num_tasks


def test_attention_bottleneck_type_is_per_task_simplex(trained_model) -> None:
    """w_agv and w_qc are a per-task softmax over the two relations -> sum to 1."""
    inst = _instance()
    n = inst.num_tasks
    sched = decode(make_rng(5).random(NUM_BLOCKS * n), inst)
    w_agv, w_qc = attention_bottleneck_type(sched, inst, trained_model)
    assert w_agv.shape == (n,) and w_qc.shape == (n,)
    assert np.all(w_agv >= -1e-6) and np.all(w_agv <= 1 + 1e-6)
    assert np.all(w_qc >= -1e-6) and np.all(w_qc <= 1 + 1e-6)
    np.testing.assert_allclose(w_agv + w_qc, np.ones(n), atol=1e-5)


def test_critical_path_binding_partitions_disjointly() -> None:
    """The exact bottleneck-type oracle: agv/qc-bound sets are disjoint, AGV side matches
    critical_agv_arcs, and the makespan-defining task is bound by exactly one resource.
    """
    inst = _instance()
    n = inst.num_tasks
    for seed in range(20):
        sched = decode(make_rng(seed).random(NUM_BLOCKS * n), inst)
        agv_bound, qc_bound = critical_path_binding(sched, inst)
        assert agv_bound.isdisjoint(qc_bound)
        assert agv_bound == critical_agv_arcs(sched, inst)
        last = int(np.argmax(evaluate(sched, inst).completion))
        assert last in (agv_bound | qc_bound)
        assert all(0 <= t < n for t in agv_bound | qc_bound)


def test_run_is_deterministic(trained_model) -> None:
    inst = _instance()
    cfg = AttentionNSGA2Config(pop_size=20, generations=8, seed=7)
    a = run_attention_nsga2(inst, trained_model, cfg)
    b = run_attention_nsga2(inst, trained_model, cfg)
    assert a.front == b.front
    assert a.evaluations == b.evaluations
    assert a.deadlocks_rejected == b.deadlocks_rejected


def test_evaluation_accounting(trained_model) -> None:
    inst = _instance()
    cfg = AttentionNSGA2Config(pop_size=20, generations=8, seed=0)
    res = run_attention_nsga2(inst, trained_model, cfg)
    assert res.evaluations == cfg.pop_size * (cfg.generations + 1)
    assert len(res.front_history) == cfg.generations + 1
    assert res.front_history[-1] == res.front


def test_front_is_mutually_non_dominated(trained_model) -> None:
    inst = _instance()
    res = run_attention_nsga2(inst, trained_model, AttentionNSGA2Config(20, generations=10, seed=1))
    front = res.front
    assert len(front) >= 1
    for i, (m_i, e_i) in enumerate(front):
        for j, (m_j, e_j) in enumerate(front):
            if i != j:
                assert not (m_j <= m_i and e_j <= e_i)
    assert len(res.schedules) == len(front)


def test_all_archive_schedules_are_feasible(trained_model) -> None:
    inst = _instance()
    res = run_attention_nsga2(inst, trained_model, AttentionNSGA2Config(20, generations=10, seed=2))
    for sched, obj in zip(res.schedules, res.front, strict=True):
        ev = evaluate(sched, inst)  # never raises ScheduleCycleError
        assert ev.objectives == pytest.approx(obj)


def test_pop_size_too_small_raises(trained_model) -> None:
    inst = _instance()
    with pytest.raises(ValueError, match="pop_size"):
        run_attention_nsga2(inst, trained_model, AttentionNSGA2Config(pop_size=1, generations=2))


@pytest.mark.slow
def test_front_bounded_by_oracle(trained_model) -> None:
    inst = _instance()
    res = run_attention_nsga2(inst, trained_model, default_config(inst, generations=60, seed=2))
    golden = _golden_front()
    tol = 1e-4
    for p in res.front:
        assert any(g[0] <= p[0] + tol and g[1] <= p[1] + tol for g in golden), p


@pytest.mark.slow
def test_swap_operator_is_exercised(trained_model) -> None:
    # The Kahn-guarded deadlock path is required to fire during a full run.
    inst = _instance()
    res = run_attention_nsga2(inst, trained_model, default_config(inst, generations=40, seed=0))
    assert res.deadlocks_rejected > 0


# --------------------------------------------------------------------------------------
# Channel-B adaptive operator selection (AOS)
# --------------------------------------------------------------------------------------
def test_operator_probabilities_simplex_and_bias() -> None:
    """Bottleneck bias steers operator mass; high temperature recovers a uniform policy."""
    k = len(_MUTATION_OPS)  # order: speed, reassign, swap_agv, swap_qc
    p_agv = operator_probabilities(1.0, temperature=0.1)
    assert p_agv.shape == (k,)
    np.testing.assert_allclose(p_agv.sum(), 1.0)
    assert np.all(p_agv >= 0.0)
    assert p_agv[1] > p_agv[3] and p_agv[2] > p_agv[3]  # AGV-bound -> AGV ops over swap_qc
    p_qc = operator_probabilities(0.0, temperature=0.1)
    assert p_qc[3] > p_qc[1] and p_qc[3] > p_qc[2]  # QC-bound -> swap_qc dominates
    p_hot = operator_probabilities(1.0, temperature=100.0)
    np.testing.assert_allclose(p_hot, np.full(k, 1.0 / k), atol=1e-2)  # tau -> infinity = uniform


def test_operator_probabilities_speed_not_crowded_out() -> None:
    """At high agv_bias the weighting keeps speed at or above the uniform share.

    A speed_weight of 0.5 pushes speed below the uniform 1/k when AGV-bound, starving the
    operator that drives the makespan-energy spread hypervolume rewards.
    """
    k = len(_MUTATION_OPS)
    p = operator_probabilities(0.84, temperature=0.5)  # the observed AGV-bound regime
    assert p[0] >= 1.0 / k  # speed not below uniform under the default speed_weight=1.0
    p_low = operator_probabilities(0.84, temperature=0.5, speed_weight=0.5)
    assert p_low[0] < p[0]  # a lower speed_weight gives speed strictly less mass
    assert p_low[0] < 1.0 / k  # ... and below the uniform share (the crowd-out)


def test_operator_reward_dominance_credit() -> None:
    """The MO credit: 1.0 if child dominates, 0.5 if incomparable, 0.0 if not better."""
    parent = (100.0, 50.0)
    assert operator_reward(parent, (90.0, 45.0)) == 1.0  # child dominates (both better)
    assert operator_reward(parent, (90.0, 50.0)) == 1.0  # better on makespan, ties energy
    assert operator_reward(parent, (90.0, 60.0)) == 0.5  # better makespan, worse energy
    assert operator_reward(parent, (110.0, 60.0)) == 0.0  # parent dominates child
    assert operator_reward(parent, (100.0, 50.0)) == 0.0  # equal -> no progress


def test_adaptive_pursuit_simplex_and_pursues_best() -> None:
    """Adaptive Pursuit keeps a valid simplex and chases the highest-reward operator."""
    k = len(_MUTATION_OPS)
    ap = AdaptivePursuit(k, alpha=0.3, beta=0.3, p_min=0.05)
    np.testing.assert_allclose(ap.probabilities(), np.full(k, 1.0 / k))
    # Operator index 1 consistently earns the maximal reward; the others earn nothing.
    for _ in range(50):
        ap.update([[0.0], [1.0], [0.0], [0.0]])
    p = ap.probabilities()
    np.testing.assert_allclose(p.sum(), 1.0)
    assert np.all(p >= 0.05 - 1e-9) and np.all(p <= 1.0 - (k - 1) * 0.05 + 1e-9)
    assert int(np.argmax(p)) == 1  # pursued the best operator
    assert p[1] > 1.0 / k  # ... above the uniform baseline
    # Operators that never fire must not move the policy.
    ap2 = AdaptivePursuit(k, alpha=0.3, beta=0.3, p_min=0.05)
    ap2.update([[], [], [], []])
    np.testing.assert_allclose(ap2.probabilities(), np.full(k, 1.0 / k))


@pytest.mark.parametrize("source", ["attention", "oracle", "reward"])
@pytest.mark.parametrize("window", ["full", "front", "best"])
def test_aos_modes_run_feasible_and_deterministic(trained_model, source, window) -> None:
    inst = _instance()
    cfg = AttentionNSGA2Config(
        pop_size=12,
        generations=4,
        seed=3,
        operator_selection=source,
        aggregation_window=window,
        operator_temperature=0.5,
    )
    a = run_attention_nsga2(inst, trained_model, cfg)
    b = run_attention_nsga2(inst, trained_model, cfg)
    assert a.front == b.front  # deterministic under a fixed seed
    for sched, obj in zip(a.schedules, a.front, strict=True):
        assert evaluate(sched, inst).objectives == pytest.approx(obj)  # exact-feasible


@pytest.mark.parametrize("source", ["attention", "oracle"])
def test_aos_per_task_routing_feasible_and_deterministic(trained_model, source) -> None:
    """Per-task Channel-B routing stays exact-feasible and deterministic under a fixed seed."""
    inst = _instance()
    cfg = AttentionNSGA2Config(
        pop_size=12,
        generations=4,
        seed=3,
        operator_selection=source,
        operator_granularity="per_task",
        operator_temperature=0.5,
    )
    a = run_attention_nsga2(inst, trained_model, cfg)
    b = run_attention_nsga2(inst, trained_model, cfg)
    assert a.front == b.front  # deterministic
    for sched, obj in zip(a.schedules, a.front, strict=True):
        assert evaluate(sched, inst).objectives == pytest.approx(obj)  # exact-feasible


def test_invalid_aos_config_raises(trained_model) -> None:
    inst = _instance()
    with pytest.raises(ValueError, match="operator_granularity"):
        run_attention_nsga2(
            inst,
            trained_model,
            AttentionNSGA2Config(pop_size=8, generations=2, operator_granularity="bogus"),
        )
    with pytest.raises(ValueError, match="operator_selection"):
        run_attention_nsga2(
            inst,
            trained_model,
            AttentionNSGA2Config(pop_size=8, generations=2, operator_selection="bogus"),
        )
    with pytest.raises(ValueError, match="aggregation_window"):
        run_attention_nsga2(
            inst,
            trained_model,
            AttentionNSGA2Config(pop_size=8, generations=2, aggregation_window="bogus"),
        )
