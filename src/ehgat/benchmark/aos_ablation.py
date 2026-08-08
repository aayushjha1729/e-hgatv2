"""Adaptive-operator-selection ablation: isolate the Channel-B operator-selection signal.

Isolates the operator-selection channel. Four arms share an
identical NSGA-II skeleton -- the same Channel-A attention task selection, surrogate
screening, temperatures, population and generation budget -- and differ in exactly one
variable: the SOURCE that decides which mutation operator is applied:

- random    -- uniform operator selection (the null / lower bound);
- attention -- the surrogate's semantic AGV-vs-QC readout (the method under test);
- oracle    -- the exact Max-Plus critical-path binding (the structural upper bound,
  which assumes the bottleneck-type -> operator map);
- reward    -- field-standard online AOS (Adaptive Pursuit + measured fitness-improvement
  credit; Thierens 2005): the genuine operator-utility baseline/ceiling that learns
  which operator contributes, and whether the null result is intrinsic to the problem.

Attention exceeding random and approaching the oracle or reward ceiling with statistical
significance (stats module) establishes that the learned explanation is causally useful for
search rather than merely faithful. A tie between reward and random instead establishes the
null over the whole family: no AOS, learned or structural, helps under elitist selection on
this problem. Only operator_selection differs between arms, which isolates that
component.

Per seed it records final HV, HV-AUC (anytime convergence), IGD+, GD+, spread,
evaluations-to-threshold, wall-clock, exact evaluations and deadlock rejections, and
serialises everything to a structured JSON artifact the stats module consumes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

# Reuse the effectiveness runner's reference-front + CI machinery (same sub-package).
from ehgat.benchmark.runner import (
    Stat,
    _approximate_reference_front,
    _bootstrap_ci,
)
from ehgat.environment.instance import EXACT_TOY_TASKS, Instance, build_toy_instance
from ehgat.environment.oracle import exact_pareto_front
from ehgat.metrics import gd_plus, hypervolume, igd_plus, nadir_reference, spread
from ehgat.search.attention_nsga2 import AttentionNSGA2Config, run_attention_nsga2
from ehgat.surrogate.ehgatv2 import EHGATv2
from ehgat.surrogate.train import TrainConfig, train_surrogate
from ehgat.utils.seeding import make_rng

__all__ = [
    "AOS_ARMS",
    "AOSAblationConfig",
    "AOSAblationResult",
    "ArmResult",
    "SeedRecord",
    "run_aos_ablation",
    "to_json_dict",
]

Front = Sequence[tuple[float, float]]
_RANDOM = "random"
_ATTENTION = "attention"
_ORACLE = "oracle"
# reward is the standard online AOS (Adaptive Pursuit with measured fitness-improvement
# credit; Thierens 2005): an operator-utility baseline that learns which operator helps,
# against the oracle arm, which assumes the bottleneck-type -> operator map. A tie between
# reward and random establishes the null over the AOS family under elitist selection;
# attention matching reward indicates that the explanation recovers the learned routing
# without an online credit-assignment mechanism.
_REWARD = "reward"
AOS_ARMS: tuple[str, ...] = (_RANDOM, _ATTENTION, _ORACLE, _REWARD)


@dataclass(frozen=True, slots=True)
class AOSAblationConfig:
    """Configuration for the Channel-B operator-selection ablation.

    Everything except operator_selection is shared across the three arms so the
    comparison isolates the bottleneck-type signal. Defaults give the reported
    N=10 / 30-seed run.
    """

    num_tasks: int = 10
    generations: int = 60
    pop_size: int | None = None  # default 20N (matches BRKGA + attention search)
    num_seeds: int = 30
    base_seed: int = 0

    # ---- shared NSGA-II knobs (held constant across arms) --------------------------
    random_task: bool = False  # Channel A: False => attention-guided task selection
    mutation_temperature: float = 0.25  # Channel-A softmax temperature
    operator_temperature: float = 0.5  # Channel-B operator softmax tau
    operator_speed_weight: float = 1.0  # speed op score (>=structural; anti-crowd-out)
    operator_granularity: str = "population"  # Channel-B bias scope: population | per_task
    aggregation_window: str = "front"  # bottleneck readout window: full | front | best
    screening_factor: int = 1  # surrogate offspring screening (held equal across arms)

    # ---- reference front (exact for N<=5, else multi-start BRKGA approximation) -----
    oracle: str = "auto"  # "exact" | "approx" | "auto"
    oracle_seeds: int = 200
    oracle_generations: int = 200
    oracle_pop_size: int | None = None
    oracle_workers: int = 1

    # ---- surrogate (trained once, reused across all arms x seeds) -------------------
    surrogate_samples: int = 1000
    surrogate_epochs: int = 50
    surrogate_seed: int = 0

    # ---- metrics + statistics -------------------------------------------------------
    hv_margin: float = 0.1
    threshold_fraction: float = 0.95  # evals-to-threshold target = frac * golden_hv
    bootstrap_resamples: int = 2000
    ci: float = 0.95

    # ---- execution ------------------------------------------------------------------
    search_workers: int = 1  # parallel processes over (arm, seed) runs
    torch_threads: int = 1  # intra-op threads (tiny graphs: 1 is fastest)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.base_seed, self.base_seed + self.num_seeds))


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """All per-seed metrics for one arm (the paired samples for the stats module)."""

    seed: int
    final_hv: float
    hv_auc: float  # normalised area under the HV-vs-generation curve in [0, ~1]
    igd_plus: float
    gd_plus: float
    spread: float
    evals_to_threshold: int | None  # exact evals to reach frac*HV*, None if never
    wall_clock_s: float
    evaluations: int
    deadlocks_rejected: int
    hv_curve: tuple[float, ...]  # per-generation HV (for the convergence figure)
    final_front: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Per-arm aggregate across seeds (bootstrap CIs) plus the raw per-seed records."""

    name: str
    records: tuple[SeedRecord, ...]
    final_hv: Stat
    hv_auc: Stat
    igd_plus: Stat
    gd_plus: Stat
    spread: Stat
    hv_curve_mean: np.ndarray  # [generations + 1]


@dataclass(frozen=True, slots=True)
class AOSAblationResult:
    """Full ablation outcome; the stats module + paper tables derive from this."""

    config: AOSAblationConfig
    num_tasks: int
    num_agvs: int
    num_qcs: int
    pop_size: int
    generations: int
    golden_front: tuple[tuple[float, float], ...]
    golden_hv: float
    reference_point: tuple[float, float]
    threshold_hv: float
    arms: dict[str, ArmResult]


# --------------------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------------------
def _hv_auc(hv_curve: Sequence[float], golden_hv: float) -> float:
    """Normalised trapezoidal area under the HV-vs-generation curve.

    Divided by (num_intervals * HV*) so a method that instantly reaches PF* and
    holds it scores ~1.0. Captures anytime convergence, not just the final front.
    """
    hv = np.asarray(hv_curve, dtype=float)
    if golden_hv <= 0.0 or hv.size == 0:
        return 0.0
    if hv.size == 1:
        return float(hv[0] / golden_hv)
    area = float(np.sum((hv[:-1] + hv[1:]) / 2.0))  # unit spacing trapezoid
    return area / ((hv.size - 1) * golden_hv)


def _seed_record(
    seed: int,
    history: Sequence[Front],
    final: Front,
    evaluations: int,
    deadlocks_rejected: int,
    wall_clock_s: float,
    *,
    golden: Front,
    reference: tuple[float, float],
    golden_hv: float,
    pop_size: int,
    threshold_hv: float,
) -> SeedRecord:
    """Build a SeedRecord from one finished run's history + final front."""
    hv_curve = [hypervolume(front, reference) for front in history]
    evals_to_threshold: int | None = None
    for gen, hv in enumerate(hv_curve):
        if hv >= threshold_hv:
            evals_to_threshold = pop_size * (gen + 1)  # cumulative exact evals through gen
            break
    return SeedRecord(
        seed=seed,
        final_hv=hypervolume(final, reference),
        hv_auc=_hv_auc(hv_curve, golden_hv),
        igd_plus=igd_plus(final, golden),
        gd_plus=gd_plus(final, golden),
        spread=spread(final, golden),
        evals_to_threshold=evals_to_threshold,
        wall_clock_s=wall_clock_s,
        evaluations=evaluations,
        deadlocks_rejected=deadlocks_rejected,
        hv_curve=tuple(float(v) for v in hv_curve),
        final_front=tuple((float(m), float(e)) for m, e in final),
    )


def _aggregate_arm(
    name: str,
    records: Sequence[SeedRecord],
    *,
    config: AOSAblationConfig,
    stat_rng: np.random.Generator,
) -> ArmResult:
    rsp, ci = config.bootstrap_resamples, config.ci

    def st(values: Sequence[float]) -> Stat:
        return _bootstrap_ci(np.asarray(values, dtype=float), resamples=rsp, ci=ci, rng=stat_rng)

    curves = np.asarray([r.hv_curve for r in records], dtype=float)
    return ArmResult(
        name=name,
        records=tuple(records),
        final_hv=st([r.final_hv for r in records]),
        hv_auc=st([r.hv_auc for r in records]),
        igd_plus=st([r.igd_plus for r in records]),
        gd_plus=st([r.gd_plus for r in records]),
        spread=st([r.spread for r in records]),
        hv_curve_mean=curves.mean(axis=0),
    )


# --------------------------------------------------------------------------------------
# Per-arm search configuration (the only field that varies is operator_selection)
# --------------------------------------------------------------------------------------
def _arm_config(
    arm: str, *, pop_size: int, generations: int, seed: int, config: AOSAblationConfig
) -> AttentionNSGA2Config:
    return AttentionNSGA2Config(
        pop_size,
        generations,
        seed=seed,
        random_mutation=config.random_task,
        mutation_temperature=config.mutation_temperature,
        screening_factor=config.screening_factor,
        operator_selection=arm,
        operator_temperature=config.operator_temperature,
        operator_speed_weight=config.operator_speed_weight,
        operator_granularity=config.operator_granularity,
        aggregation_window=config.aggregation_window,
    )


def _run_one(
    instance: Instance, model: EHGATv2, cfg: AttentionNSGA2Config
) -> tuple[tuple[Front, ...], Front, int, int, float]:
    """Run a single (arm, seed) search; return (history, final, evals, deadlocks, wall)."""
    start = time.perf_counter()
    res = run_attention_nsga2(instance, model, cfg)
    wall = time.perf_counter() - start
    history = tuple(tuple((float(m), float(e)) for m, e in front) for front in res.front_history)
    final = tuple((float(m), float(e)) for m, e in res.front)
    return history, final, res.evaluations, res.deadlocks_rejected, wall


# --------------------------------------------------------------------------------------
# Parallel execution across (arm, seed) runs (mirrors runner.py's worker pattern)
# --------------------------------------------------------------------------------------
_AOS_INSTANCE: Instance | None = None
_AOS_MODEL: EHGATv2 | None = None


def _aos_worker_init(instance: Instance, model: EHGATv2) -> None:
    """Per-process setup: pin Torch to one thread and cache the shared instance + model."""
    import torch

    torch.set_num_threads(1)  # N processes x N threads each would thrash the box
    global _AOS_INSTANCE, _AOS_MODEL
    _AOS_INSTANCE = instance
    model.eval()
    _AOS_MODEL = model


def _run_arm_seed_task(
    spec: tuple[str, int, int, int, AOSAblationConfig],
) -> tuple[str, int, tuple[Front, ...], Front, int, int, float]:
    arm, seed, pop_size, generations, config = spec
    assert _AOS_INSTANCE is not None and _AOS_MODEL is not None
    cfg = _arm_config(arm, pop_size=pop_size, generations=generations, seed=seed, config=config)
    history, final, evals, deadlocks, wall = _run_one(_AOS_INSTANCE, _AOS_MODEL, cfg)
    return arm, seed, history, final, evals, deadlocks, wall


# --------------------------------------------------------------------------------------
# Reference front
# --------------------------------------------------------------------------------------
def _reference_front(instance: Instance, config: AOSAblationConfig) -> Front:
    mode = config.oracle
    if mode not in {"auto", "exact", "approx"}:
        raise ValueError("oracle must be one of: auto, exact, approx")
    if mode == "auto":
        mode = "exact" if instance.num_tasks <= EXACT_TOY_TASKS else "approx"
    if mode == "exact":
        return tuple((float(m), float(e)) for m, e in exact_pareto_front(instance).front)
    return _approximate_reference_front(
        instance,
        seeds=config.oracle_seeds,
        generations=config.oracle_generations,
        pop_size=config.oracle_pop_size,
        workers=config.oracle_workers,
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def run_aos_ablation(config: AOSAblationConfig | None = None) -> AOSAblationResult:
    """Run the three-arm Channel-B ablation and return the aggregated result."""
    config = config or AOSAblationConfig()

    import contextlib

    import torch

    torch.set_num_threads(max(1, config.torch_threads))
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)

    instance = build_toy_instance(num_tasks=config.num_tasks)
    pop_size = config.pop_size or 20 * instance.num_tasks
    generations = config.generations

    golden = _reference_front(instance, config)
    reference = nadir_reference(golden, margin=config.hv_margin)
    golden_hv = hypervolume(golden, reference)
    threshold_hv = config.threshold_fraction * golden_hv

    print(
        f"Training surrogate: {config.surrogate_samples} samples x "
        f"{config.surrogate_epochs} epochs",
        flush=True,
    )
    model = train_surrogate(
        instance,
        TrainConfig(
            num_samples=config.surrogate_samples,
            epochs=config.surrogate_epochs,
            seed=config.surrogate_seed,
        ),
    ).model
    model.eval()
    print("Training surrogate: complete", flush=True)

    stat_rng = make_rng(12345)  # dedicated bootstrap stream, independent of search seeds

    def make_record(
        seed: int,
        history: Sequence[Front],
        final: Front,
        evals: int,
        deadlocks: int,
        wall: float,
    ) -> SeedRecord:
        return _seed_record(
            seed,
            history,
            final,
            evals,
            deadlocks,
            wall,
            golden=golden,
            reference=reference,
            golden_hv=golden_hv,
            pop_size=pop_size,
            threshold_hv=threshold_hv,
        )

    collected: dict[str, dict[int, SeedRecord]] = {arm: {} for arm in AOS_ARMS}
    specs = [
        (arm, seed, pop_size, generations, config) for arm in AOS_ARMS for seed in config.seeds
    ]

    if config.search_workers and config.search_workers > 1:
        workers = max(1, min(config.search_workers, len(specs), os.cpu_count() or 1))
        print(f"Parallel AOS ablation: {len(specs)} runs on {workers} workers", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_aos_worker_init, initargs=(instance, model)
        ) as executor:
            futures = {executor.submit(_run_arm_seed_task, spec): spec for spec in specs}
            for done, future in enumerate(as_completed(futures), start=1):
                arm, seed, history, final, evals, deadlocks, wall = future.result()
                collected[arm][seed] = make_record(seed, history, final, evals, deadlocks, wall)
                print(f"AOS progress: {done}/{len(specs)}", flush=True)
    else:
        for done, (arm, seed, _ps, _g, _cfg) in enumerate(specs, start=1):
            cfg = _arm_config(
                arm, pop_size=pop_size, generations=generations, seed=seed, config=config
            )
            history, final, evals, deadlocks, wall = _run_one(instance, model, cfg)
            collected[arm][seed] = make_record(seed, history, final, evals, deadlocks, wall)
            print(f"AOS progress: {done}/{len(specs)}", flush=True)

    arms = {
        arm: _aggregate_arm(
            arm,
            [collected[arm][seed] for seed in config.seeds],
            config=config,
            stat_rng=stat_rng,
        )
        for arm in AOS_ARMS
    }

    return AOSAblationResult(
        config=config,
        num_tasks=instance.num_tasks,
        num_agvs=instance.num_agvs,
        num_qcs=len(instance.qcs),
        pop_size=pop_size,
        generations=generations,
        golden_front=tuple((float(m), float(e)) for m, e in golden),
        golden_hv=float(golden_hv),
        reference_point=(float(reference[0]), float(reference[1])),
        threshold_hv=float(threshold_hv),
        arms=arms,
    )


# --------------------------------------------------------------------------------------
# JSON serialisation (consumed by the stats module + paper tables)
# --------------------------------------------------------------------------------------
def _stat_dict(stat: Stat) -> dict[str, float]:
    return {"mean": float(stat.mean), "lo": float(stat.lo), "hi": float(stat.hi)}


def _record_dict(record: SeedRecord) -> dict[str, object]:
    return {
        "seed": record.seed,
        "final_hv": record.final_hv,
        "hv_auc": record.hv_auc,
        "igd_plus": record.igd_plus,
        "gd_plus": record.gd_plus,
        "spread": record.spread,
        "evals_to_threshold": record.evals_to_threshold,
        "wall_clock_s": record.wall_clock_s,
        "evaluations": record.evaluations,
        "deadlocks_rejected": record.deadlocks_rejected,
    }


def to_json_dict(result: AOSAblationResult) -> dict[str, object]:
    """JSON-serialisable view: per-arm CIs, mean HV curve, and per-seed scalar metrics."""
    cfg = result.config
    return {
        "experiment": "aos_ablation",
        "config": {
            "num_tasks": cfg.num_tasks,
            "generations": cfg.generations,
            "pop_size": result.pop_size,
            "num_seeds": cfg.num_seeds,
            "base_seed": cfg.base_seed,
            "random_task": cfg.random_task,
            "mutation_temperature": cfg.mutation_temperature,
            "operator_temperature": cfg.operator_temperature,
            "operator_speed_weight": cfg.operator_speed_weight,
            "operator_granularity": cfg.operator_granularity,
            "aggregation_window": cfg.aggregation_window,
            "screening_factor": cfg.screening_factor,
            "oracle": cfg.oracle,
            "surrogate_samples": cfg.surrogate_samples,
            "surrogate_epochs": cfg.surrogate_epochs,
            "threshold_fraction": cfg.threshold_fraction,
            "ci": cfg.ci,
        },
        "instance": {
            "num_tasks": result.num_tasks,
            "num_agvs": result.num_agvs,
            "num_qcs": result.num_qcs,
        },
        "golden_hv": result.golden_hv,
        "reference_point": list(result.reference_point),
        "threshold_hv": result.threshold_hv,
        "golden_front": [[m, e] for m, e in result.golden_front],
        "arms": {
            name: {
                "final_hv": _stat_dict(arm.final_hv),
                "hv_auc": _stat_dict(arm.hv_auc),
                "igd_plus": _stat_dict(arm.igd_plus),
                "gd_plus": _stat_dict(arm.gd_plus),
                "spread": _stat_dict(arm.spread),
                "hv_curve_mean": [float(v) for v in arm.hv_curve_mean],
                "seeds": [_record_dict(r) for r in arm.records],
            }
            for name, arm in result.arms.items()
        },
    }
