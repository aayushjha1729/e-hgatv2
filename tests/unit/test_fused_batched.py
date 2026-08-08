"""Batched fused forward must equal the per-graph fused forward for the same weights."""

from __future__ import annotations

import torch

from ehgat.environment.instance import build_toy_instance
from ehgat.explain.fused_ehgat import FusedEHGATv2
from ehgat.explain.train_fused import build_core, build_samples
from ehgat.explain.train_fused_batched import _forward_batch, build_batch, train_fused_batched
from ehgat.explain.train_fused import FusedTrainConfig


def test_batched_forward_matches_per_graph() -> None:
    inst = build_toy_instance(num_tasks=6, peak_power=30.0)
    core = build_core(inst, seed=0, num_samples=250, epochs=5)
    model = FusedEHGATv2(core, coupled=True)
    model.freeze_core()
    samples = build_samples(inst, 12, seed=1)

    b = build_batch(model, samples, inst)
    _, _, _, mk, en, _ = _forward_batch(model, b, coupled=True)

    for i, s in enumerate(samples):
        out = model(s.data)
        assert abs(float(out.makespan) - float(mk[i])) < 1e-4, f"makespan mismatch @ {i}"
        # Energy ~ O(1e4); float32 summation order differs negligibly -> relative tolerance.
        assert abs(float(out.energy) - float(en[i])) <= 1e-4 * abs(float(en[i])), f"energy @ {i}"


def test_batched_contention_matches_per_graph_helper() -> None:
    # The vectorised (per-sample-blocked) contention features must reproduce the per-graph
    # ones exactly on shared inputs. Tested directly (not via the makespan) because the hard
    # temporal-overlap threshold makes a cross-path makespan check sensitive to the two
    # longest-path implementations' float noise -- irrelevant to training (one path is used).
    import torch

    from ehgat.explain.train_fused_batched import _batched_contention

    inst = build_toy_instance(num_tasks=5, peak_power=30.0)
    core = build_core(inst, seed=0, num_samples=120, epochs=3)
    model = FusedEHGATv2(core, coupled=True, unroll_steps=2, peak_power=30.0)
    s, n = 7, 5
    g = torch.Generator().manual_seed(11)
    empty_eff = torch.rand(s * n, generator=g) + 0.2
    loaded_eff = torch.rand(s * n, generator=g) + 0.2
    e_end = torch.rand(s * n, generator=g) * 3 + empty_eff
    l_end = e_end + torch.rand(s * n, generator=g) * 3 + loaded_eff
    p_empty = torch.rand(s * n, generator=g) * 10 + 5
    p_loaded = torch.rand(s * n, generator=g) * 10 + 8

    batched = _batched_contention(e_end, l_end, empty_eff, loaded_eff, p_empty, p_loaded, s, n, 30.0)
    rows = []
    for i in range(s):
        sl = slice(i * n, (i + 1) * n)
        rows.append(model._contention_features(
            e_end[sl], l_end[sl], empty_eff[sl], loaded_eff[sl], p_empty[sl], p_loaded[sl]
        ))
    per_graph = torch.cat(rows, dim=0)
    assert torch.allclose(batched, per_graph, atol=1e-5), (batched - per_graph).abs().max()


def test_unroll_reduces_to_uncoupled_when_waits_are_zero() -> None:
    # With the wait head zeroed, the unrolled coupled forward must equal the uncoupled
    # makespan computed from the same model's predicted legs (the coupled DAG reduces to the
    # uncoupled one when no leg waits) -- isolating coupling, not head initialisation.
    import torch

    from ehgat.explain.event_dag import assemble_event_dag, extract_precedence
    from ehgat.explain.tropical_dp import tropical_longest_path
    from ehgat.surrogate.graph import AGV_EDGE, NODE_TYPE, QC_EDGE

    inst = build_toy_instance(num_tasks=6, peak_power=10_000.0)
    core = build_core(inst, seed=0, num_samples=200, epochs=5)
    cm = FusedEHGATv2(core, coupled=True, unroll_steps=2, peak_power=10_000.0)
    with torch.no_grad():
        cm.wait_head[-1].weight.zero_(); cm.wait_head[-1].bias.zero_()
    s = build_samples(inst, 1, seed=3)[0]
    out = cm(s.data)
    n = int(s.data[NODE_TYPE].x.shape[0])
    is_load = s.data[NODE_TYPE].x[:, 1] > 0.5
    agv_prev, qc_prev = extract_precedence(
        s.data[AGV_EDGE].edge_index, s.data[QC_EDGE].edge_index, n
    )
    dag = assemble_event_dag(is_load, agv_prev, qc_prev, out.empty_t, out.loaded_t, out.node_delay)
    mk_unc = tropical_longest_path(dag.node_weights, dag.edge_index, dag.edge_weights)[
        dag.completion_nodes
    ].max()
    assert abs(float(out.makespan) - float(mk_unc)) < 1e-4


def test_batched_training_runs_and_fits_energy() -> None:
    inst = build_toy_instance(num_tasks=6, peak_power=30.0)
    core = build_core(inst, seed=0, num_samples=400, epochs=20)
    res = train_fused_batched(
        inst, core, FusedTrainConfig(num_samples=300, epochs=40, seed=0), device="cpu"
    )
    # Energy is exact-additive; makespan should be a sensible positive fit.
    assert res.metrics["r2_energy"] >= 0.98
    assert res.metrics["r2_makespan"] >= 0.5
