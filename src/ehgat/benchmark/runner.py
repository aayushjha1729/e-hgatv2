"""Multi-seed effectiveness benchmark -- BRKGA vs E-HGATv2-NSGA-II.

Runs three methods on the exact toy under a shared evaluation budget (identical
pop_size and generations => identical evaluations) and against the same golden
PF* and hypervolume reference point:

- BRKGA -- the stochastic baseline;
- E-HGATv2-NSGA-II -- attention-guided mutation;
- NSGA-II (random) -- the ablation control: the identical skeleton with attention replaced
  by uniform task selection, isolating the causal contribution of attention.

For each method it records the per-generation hypervolume curve (mean + normal CI band
across seeds) and final-front HV / IGD+ / GD+ / spread with
bootstrap CIs. It also reports attention faithfulness for the trained surrogate against
a random-selection baseline. The surrogate is trained once and reused across seeds.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from ehgat.baselines.brkga import BRKGAConfig, run_brkga
from ehgat.benchmark.faithfulness import (
    FaithfulnessResult,
    critical_agv_arcs,
    evaluate_faithfulness,
)
from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.instance import (
    AVAILABLE_QCS,
    EXACT_TOY_TASKS,
    Instance,
    build_toy_instance,
    scaled_fleet,
)
from ehgat.environment.oracle import exact_pareto_front
from ehgat.metrics import gd_plus, hypervolume, igd_plus, nadir_reference, spread
from ehgat.search.attention_nsga2 import AttentionNSGA2Config, run_attention_nsga2
from ehgat.surrogate.ehgatv2 import EHGATv2
from ehgat.surrogate.train import TrainConfig, train_surrogate
from ehgat.utils.seeding import make_rng

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "MethodResult",
    "Stat",
    "build_scaling_instance",
    "run_benchmark",
]

Front = Sequence[tuple[float, float]]
_BRKGA = "BRKGA"
_GUIDED = "E-HGATv2-NSGA-II"
_RANDOM = "NSGA-II (random)"
_Z_95 = 1.959963984540054  # normal two-sided 95% multiplier


@dataclass(frozen=True, slots=True)
class Stat:
    """A point estimate with a (bootstrap) confidence interval."""

    mean: float
    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for the effectiveness benchmark."""

    num_tasks: int = EXACT_TOY_TASKS
    num_agvs: int | None = None  # default: scaled_fleet(N) -- ~1 AGV per 12 tasks
    num_qcs: int | None = None  # default: scaled_fleet(N) -- 3..6 quay cranes
    generations: int = 60
    pop_size: int | None = None  # default 20N (matches BRKGA + attention search)
    num_seeds: int = 10
    base_seed: int = 0
    oracle: str = "auto"  # "exact", "approx", or "auto" (exact for N<=5, approx otherwise)
    oracle_seeds: int = 100
    oracle_generations: int = 100
    oracle_pop_size: int | None = None
    oracle_workers: int = 1
    search_workers: int = 1  # parallel processes for the 3 methods x N seeds search runs
    torch_threads: int = 1  # intra-op threads; 1 is fastest for these tiny graphs
    mutation_temperature: float = 0.25  # softmax temperature for soft bottleneck sampling
    screening_factor: int = 4  # guided method: surrogate-screen k*lambda offspring (1=off)
    surrogate_samples: int = 1000
    surrogate_epochs: int = 50
    hv_margin: float = 0.1
    faithfulness_samples: int = 60
    bootstrap_resamples: int = 2000
    ci: float = 0.95

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.base_seed, self.base_seed + self.num_seeds))


@dataclass(frozen=True, slots=True)
class MethodResult:
    """Per-method aggregate across seeds."""

    name: str
    hv_curve_mean: np.ndarray  # [generations + 1]
    hv_curve_lo: np.ndarray
    hv_curve_hi: np.ndarray
    hv_curves: np.ndarray  # [num_seeds, generations + 1] raw
    final_hv: Stat
    final_igd_plus: Stat
    final_gd_plus: Stat
    final_spread: Stat
    final_fronts: tuple[tuple[tuple[float, float], ...], ...]  # per seed


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Full benchmark outcome (charts + tables are derived from this)."""

    config: BenchmarkConfig
    num_tasks: int
    num_agvs: int
    num_qcs: int
    pop_size: int
    generations: int
    golden_front: tuple[tuple[float, float], ...]
    golden_hv: float
    reference_point: tuple[float, float]
    methods: dict[str, MethodResult] = field(default_factory=dict)
    faithfulness: dict[str, FaithfulnessResult] = field(default_factory=dict)
    random_precision_at_1: float = 0.0


def _bootstrap_ci(
    values: np.ndarray, *, resamples: int, ci: float, rng: np.random.Generator
) -> Stat:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < 2:
        return Stat(mean, mean, mean)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    boots = arr[idx].mean(axis=1)
    alpha = 1.0 - ci
    return Stat(
        mean,
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def _normal_band(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-generation mean and 95% normal CI band across seeds."""
    mean = curves.mean(axis=0)
    n = curves.shape[0]
    if n < 2:
        return mean, mean.copy(), mean.copy()
    sem = curves.std(axis=0, ddof=1) / math.sqrt(n)
    return mean, mean - _Z_95 * sem, mean + _Z_95 * sem


def _hv_curve(history: Sequence[Front], reference: tuple[float, float]) -> list[float]:
    return [hypervolume(front, reference) for front in history]


def _aggregate_method(
    name: str,
    per_seed: Sequence[tuple[Sequence[Front], Front]],
    *,
    config: BenchmarkConfig,
    golden: Front,
    reference: tuple[float, float],
    stat_rng: np.random.Generator,
) -> MethodResult:
    """Build a MethodResult from already-collected per-seed (history, final)."""
    curves: list[list[float]] = []
    final_fronts: list[tuple[tuple[float, float], ...]] = []
    final_hv: list[float] = []
    final_igd: list[float] = []
    final_gd: list[float] = []
    final_spread: list[float] = []
    for history, final in per_seed:
        curves.append(_hv_curve(history, reference))
        final_fronts.append(tuple((float(m), float(e)) for m, e in final))
        final_hv.append(hypervolume(final, reference))
        final_igd.append(igd_plus(final, golden))
        final_gd.append(gd_plus(final, golden))
        final_spread.append(spread(final, golden))

    curve_arr = np.asarray(curves, dtype=float)
    mean, lo, hi = _normal_band(curve_arr)
    _rsp = config.bootstrap_resamples
    _ci = config.ci
    return MethodResult(
        name=name,
        hv_curve_mean=mean,
        hv_curve_lo=lo,
        hv_curve_hi=hi,
        hv_curves=curve_arr,
        final_hv=_bootstrap_ci(np.asarray(final_hv), resamples=_rsp, ci=_ci, rng=stat_rng),
        final_igd_plus=_bootstrap_ci(np.asarray(final_igd), resamples=_rsp, ci=_ci, rng=stat_rng),
        final_gd_plus=_bootstrap_ci(np.asarray(final_gd), resamples=_rsp, ci=_ci, rng=stat_rng),
        final_spread=_bootstrap_ci(np.asarray(final_spread), resamples=_rsp, ci=_ci, rng=stat_rng),
        final_fronts=tuple(final_fronts),
    )


def _evaluate_method(
    name: str,
    run: Callable[[int], tuple[Sequence[Front], Front]],
    *,
    config: BenchmarkConfig,
    golden: Front,
    reference: tuple[float, float],
    stat_rng: np.random.Generator,
) -> MethodResult:
    """Serial path: run every seed in-process, then aggregate."""
    print(f"{name}: starting {len(config.seeds)} seeds", flush=True)
    per_seed: list[tuple[Sequence[Front], Front]] = []
    for index, seed in enumerate(config.seeds, start=1):
        print(f"{name}: seed {index}/{len(config.seeds)} (seed={seed})", flush=True)
        per_seed.append(run(seed))
    print(f"{name}: complete", flush=True)
    return _aggregate_method(
        name, per_seed, config=config, golden=golden, reference=reference, stat_rng=stat_rng
    )


# --------------------------------------------------------------------------------------
# Parallel search across CPU cores (the 3 methods x N seeds are independent runs)
# --------------------------------------------------------------------------------------
_SEARCH_INSTANCE: Instance | None = None
_SEARCH_MODEL: EHGATv2 | None = None


def _search_worker_init(instance: Instance, model: EHGATv2) -> None:
    """Per-process setup: pin Torch to 1 thread (avoid oversubscription) and cache state."""
    import torch

    torch.set_num_threads(1)  # CRITICAL: N processes x N threads each would thrash the box
    global _SEARCH_INSTANCE, _SEARCH_MODEL
    _SEARCH_INSTANCE = instance
    model.eval()
    _SEARCH_MODEL = model


def _run_search_task(
    spec: tuple[str, int, int, int, float, int],
) -> tuple[str, int, tuple[Front, ...], Front]:
    """Run a single (method, seed) search in a worker; returns (name, seed, history, final)."""
    name, seed, pop_size, generations, temperature, screening = spec
    assert _SEARCH_INSTANCE is not None and _SEARCH_MODEL is not None
    instance, model = _SEARCH_INSTANCE, _SEARCH_MODEL
    if name == _BRKGA:
        brkga = run_brkga(
            instance, BRKGAConfig(pop_size=pop_size, generations=generations, seed=seed)
        )
        raw_history, raw_final = brkga.front_history, brkga.front
    else:
        cfg = AttentionNSGA2Config(
            pop_size,
            generations,
            seed=seed,
            random_mutation=(name == _RANDOM),
            mutation_temperature=temperature,
            screening_factor=screening if name == _GUIDED else 1,
        )
        nsga = run_attention_nsga2(instance, model, cfg)
        raw_history, raw_final = nsga.front_history, nsga.front
    history = tuple(tuple((float(m), float(e)) for m, e in front) for front in raw_history)
    final = tuple((float(m), float(e)) for m, e in raw_final)
    return name, seed, history, final


def _run_methods_parallel(
    instance: Instance,
    model: EHGATv2,
    *,
    pop_size: int,
    generations: int,
    config: BenchmarkConfig,
    golden: Front,
    reference: tuple[float, float],
    stat_rng: np.random.Generator,
) -> dict[str, MethodResult]:
    """Distribute all (method, seed) runs across worker processes, then aggregate per method."""
    names = (_BRKGA, _GUIDED, _RANDOM)
    specs = [
        (name, seed, pop_size, generations, config.mutation_temperature, config.screening_factor)
        for name in names
        for seed in config.seeds
    ]
    workers = max(1, min(config.search_workers, len(specs), os.cpu_count() or 1))
    print(f"Parallel search: {len(specs)} runs on {workers} workers", flush=True)

    collected: dict[str, dict[int, tuple[Sequence[Front], Front]]] = {n: {} for n in names}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_search_worker_init, initargs=(instance, model)
    ) as executor:
        futures = {executor.submit(_run_search_task, spec): spec for spec in specs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, seed, history, final = future.result()
            collected[name][seed] = (history, final)
            print(f"Search progress: {done}/{len(specs)}", flush=True)

    return {
        name: _aggregate_method(
            name,
            [collected[name][seed] for seed in config.seeds],
            config=config,
            golden=golden,
            reference=reference,
            stat_rng=stat_rng,
        )
        for name in names
    }


def _random_precision_at_1(
    instance: Instance, model: EHGATv2, n_samples: int, seed: int
) -> float:
    """Expected precision@1 of selecting a uniformly random AGV arc (the random baseline)."""
    rng = make_rng(seed)
    n = instance.num_tasks
    chrom_len = NUM_BLOCKS * n
    total = 0.0
    for _ in range(n_samples):
        sched = decode(rng.random(chrom_len), instance)
        total += len(critical_agv_arcs(sched, instance)) / n  # P(random arc is critical)
    return total / n_samples


def _pareto_front_float(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Non-dominated minimisation front over float (makespan, energy) points."""
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for makespan, energy in points:
        key = (round(float(makespan), 6), round(float(energy), 6))
        if key not in seen:
            seen.add(key)
            unique.append((float(makespan), float(energy)))

    front: list[tuple[float, float]] = []
    best_energy: float | None = None
    for makespan, energy in sorted(unique):
        if best_energy is None or energy < best_energy:
            front.append((makespan, energy))
            best_energy = energy
    return tuple(front)


def _resolve_fleet(
    num_tasks: int, num_agvs: int | None, num_qcs: int | None
) -> tuple[int, int]:
    """Fill unset fleet sizes from the deterministic scaled_fleet policy."""
    default_agvs, default_qcs = scaled_fleet(num_tasks)
    return (num_agvs or default_agvs), (num_qcs or default_qcs)


def build_scaling_instance(
    num_tasks: int,
    *,
    num_agvs: int | None = None,
    num_qcs: int | None = None,
    peak_power: float | None = None,
) -> Instance:
    """Build a benchmark instance with an N-scaled (or explicit) AGV/QC fleet.

    Unset fleet sizes default to scaled_fleet, making the result a pure function of the
    arguments; the search and its separately built BRKGA reference front therefore act on
    identical instances.
    """
    agvs, qcs = _resolve_fleet(num_tasks, num_agvs, num_qcs)
    if qcs > len(AVAILABLE_QCS):
        raise ValueError(
            f"num_qcs={qcs} exceeds the {len(AVAILABLE_QCS)} cranes in the distance matrix"
        )
    return build_toy_instance(
        num_tasks=num_tasks, num_agvs=agvs, qcs=AVAILABLE_QCS[:qcs], peak_power=peak_power
    )


def _brkga_reference_points_for_seed(
    num_tasks: int, num_agvs: int, num_qcs: int, pop_size: int, generations: int, seed: int
) -> tuple[tuple[float, float], ...]:
    """Build one BRKGA reference front in a subprocess-friendly worker."""
    instance = build_scaling_instance(num_tasks, num_agvs=num_agvs, num_qcs=num_qcs)
    res = run_brkga(instance, BRKGAConfig(pop_size=pop_size, generations=generations, seed=seed))
    return tuple((float(m), float(e)) for m, e in res.front)


def _approximate_reference_front(
    instance: Instance,
    *,
    seeds: int,
    generations: int,
    pop_size: int | None,
    workers: int = 1,
) -> tuple[tuple[float, float], ...]:
    """Approximate PF* with a multi-start BRKGA reference run for scaling instances."""
    pop = pop_size or 20 * instance.num_tasks
    workers = max(1, min(workers, seeds, os.cpu_count() or 1))
    points: list[tuple[float, float]] = []

    n_agvs, n_qcs = instance.num_agvs, len(instance.qcs)
    if workers == 1:
        for seed in range(seeds):
            points.extend(
                _brkga_reference_points_for_seed(
                    instance.num_tasks, n_agvs, n_qcs, pop, generations, seed
                )
            )
        return _pareto_front_float(points)

    print(
        f"Approximate oracle: {seeds} BRKGA seeds x {generations} gens x pop={pop} "
        f"on {workers} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _brkga_reference_points_for_seed,
                instance.num_tasks, n_agvs, n_qcs, pop, generations, seed,
            ): seed
            for seed in range(seeds)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            points.extend(future.result())
            if completed == seeds or completed % max(1, seeds // 10) == 0:
                print(f"Approximate oracle progress: {completed}/{seeds}", flush=True)
    return _pareto_front_float(points)


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkResult:
    """Run the full multi-seed effectiveness benchmark and return its aggregate result."""
    config = config or BenchmarkConfig()

    # CRITICAL on many-core boxes: pin Torch intra-op threads. The graphs are tiny
    # (<=N nodes), so 32 threads per op thrash instead of help -- this is what made
    # surrogate training hang for hours at ~2700% CPU on a 32-vCPU VM.
    import contextlib

    import torch

    torch.set_num_threads(max(1, config.torch_threads))
    with contextlib.suppress(RuntimeError):
        # raises if Torch's interop pool already started; intra-op pinning is what matters
        torch.set_num_interop_threads(1)

    instance = build_scaling_instance(
        config.num_tasks, num_agvs=config.num_agvs, num_qcs=config.num_qcs
    )
    pop_size = config.pop_size or 20 * instance.num_tasks
    generations = config.generations

    oracle_mode = config.oracle
    if oracle_mode not in {"auto", "exact", "approx"}:
        raise ValueError("oracle must be one of: auto, exact, approx")
    if oracle_mode == "auto":
        oracle_mode = "exact" if instance.num_tasks <= EXACT_TOY_TASKS else "approx"

    if oracle_mode == "exact":
        oracle = exact_pareto_front(instance)
        golden = tuple((float(m), float(e)) for m, e in oracle.front)
    else:
        golden = _approximate_reference_front(
            instance,
            seeds=config.oracle_seeds,
            generations=config.oracle_generations,
            pop_size=config.oracle_pop_size,
            workers=config.oracle_workers,
        )
    reference = nadir_reference(golden, margin=config.hv_margin)
    golden_hv = hypervolume(golden, reference)

    print(
        f"Training surrogate: {config.surrogate_samples} samples x "
        f"{config.surrogate_epochs} epochs",
        flush=True,
    )
    model = train_surrogate(
        instance,
        TrainConfig(num_samples=config.surrogate_samples, epochs=config.surrogate_epochs, seed=0),
    ).model
    print("Training surrogate: complete", flush=True)

    stat_rng = make_rng(12345)  # dedicated stream for bootstrap, independent of search seeds

    def brkga_run(seed: int) -> tuple[Sequence[Front], Front]:
        res = run_brkga(
            instance, BRKGAConfig(pop_size=pop_size, generations=generations, seed=seed)
        )
        return res.front_history, res.front

    def guided_run(seed: int) -> tuple[Sequence[Front], Front]:
        res = run_attention_nsga2(
            instance,
            model,
            AttentionNSGA2Config(
                pop_size,
                generations,
                seed=seed,
                mutation_temperature=config.mutation_temperature,
                screening_factor=config.screening_factor,
            ),
        )
        return res.front_history, res.front

    def random_run(seed: int) -> tuple[Sequence[Front], Front]:
        res = run_attention_nsga2(
            instance,
            model,
            AttentionNSGA2Config(pop_size, generations, seed=seed, random_mutation=True),
        )
        return res.front_history, res.front

    if config.search_workers and config.search_workers > 1:
        methods = _run_methods_parallel(
            instance,
            model,
            pop_size=pop_size,
            generations=generations,
            config=config,
            golden=golden,
            reference=reference,
            stat_rng=stat_rng,
        )
    else:
        methods = {
            name: _evaluate_method(
                name, run, config=config, golden=golden, reference=reference, stat_rng=stat_rng
            )
            for name, run in ((_BRKGA, brkga_run), (_GUIDED, guided_run), (_RANDOM, random_run))
        }

    faith_rng = make_rng(999)
    chrom_len = NUM_BLOCKS * instance.num_tasks
    faith_schedules = [
        decode(faith_rng.random(chrom_len), instance) for _ in range(config.faithfulness_samples)
    ]
    faithfulness = {_GUIDED: evaluate_faithfulness(faith_schedules, instance, model)}
    random_p1 = _random_precision_at_1(instance, model, config.faithfulness_samples, seed=999)

    return BenchmarkResult(
        config=config,
        num_tasks=instance.num_tasks,
        num_agvs=instance.num_agvs,
        num_qcs=len(instance.qcs),
        pop_size=pop_size,
        generations=generations,
        golden_front=golden,
        golden_hv=golden_hv,
        reference_point=reference,
        methods=methods,
        faithfulness=faithfulness,
        random_precision_at_1=random_p1,
    )
