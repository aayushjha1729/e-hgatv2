"""Landscape analysis: exact, cascade-aware feature-importance.

An algorithm
that dissects how the decision variables shape the objectives and *why solutions are
Pareto-(near)optimal* -- and does so without the surrogate-fidelity problem that makes
tabular TreeSHAP unsafe here.

At the instance sizes considered here the exact max-plus evaluator is itself the
structural causal model from decision variables to (C_max, E). Importance is therefore
computed on the true objective rather than approximated on a fitted model:

- grouped_sobol -- variance-based global sensitivity (Saltelli 2010 first-order +
  Jansen 1999 total-order) over the random-key chromosome, grouped into the four decision
  families (sequence / AGV-assignment / empty-speed / loaded-speed). Total-order indices
  capture interactions, i.e. the cascading couplings a flat additive attributor misses.
- critical_path_landscape -- exact critical-path attribution: per-task AGV/QC
  binding frequency, the marginal (C_max, E) effect of accelerating each task, and the
  downstream cascade size (how many successors shift) -- the temporal-cascade signal.
- pareto_contrast -- why Pareto-optimal: which decision descriptors separate the
  non-dominated set from the dominated set (Cliff's delta effect size).
- tabular_failure_boundary -- where TreeSHAP (a fitted,
  additive, structure-blind attributor) agrees with the exact Sobol landscape (the
  kinematic / speed region) and where it structurally fails (the topological / sequencing
  region).

Everything in the core is pure NumPy/SciPy on the exact evaluator (no Torch / surrogate),
so it is fast, deterministic, and unit-testable without the heavy stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import qmc  # type: ignore[import-untyped]

from ehgat.environment.decoder import NUM_BLOCKS, Schedule, decode
from ehgat.environment.evaluator import evaluate
from ehgat.environment.instance import Instance
from ehgat.environment.physics import SpeedLevel
from ehgat.search.nsga2 import fast_non_dominated_sort

_FASTEST_SPEED: SpeedLevel = tuple(SpeedLevel)[-1]

__all__ = [
    "FEATURE_FAMILIES",
    "OBJECTIVES",
    "CascadeResult",
    "LandscapeResult",
    "ParetoContrast",
    "SobolIndex",
    "SobolResult",
    "critical_path_landscape",
    "grouped_sobol",
    "pareto_contrast",
    "run_landscape",
    "tabular_failure_boundary",
    "to_json_dict",
]

# The four decision-variable families, aligned with the random-key chromosome blocks
# (decoder.py): sequence keys (SPV order), AGV-assignment keys, empty-leg speed keys,
# loaded-leg speed keys. These are the decision variables of the formulation (task order,
# AGV assignment, travelling speed) in the representation the optimiser searches over.
FEATURE_FAMILIES: tuple[str, ...] = ("sequence", "assignment", "empty_speed", "loaded_speed")
OBJECTIVES: tuple[str, ...] = ("makespan", "energy")
# Partition used for the failure-boundary read-out: the kinematic (continuous, per-leg
# speed) region against the topological (sequencing and AGV-assignment) region. The exact
# Sobol landscape shows both objectives to be topology-dominated, which is the region in
# which a flat additive tabular attributor breaks down.
_SPEED_FAMILIES: tuple[str, ...] = ("empty_speed", "loaded_speed")
_STRUCTURAL_FAMILIES: tuple[str, ...] = ("sequence", "assignment")


# --------------------------------------------------------------------------------------
# Exact-evaluator objective batch
# --------------------------------------------------------------------------------------
def _evaluate_keys(keys_batch: np.ndarray, instance: Instance) -> np.ndarray:
    """Decode + exactly evaluate a [M, 4N] random-key batch -> [M, 2] objectives."""
    out = np.empty((keys_batch.shape[0], len(OBJECTIVES)), dtype=float)
    for i in range(keys_batch.shape[0]):
        ev = evaluate(decode(keys_batch[i], instance), instance)
        out[i, 0] = ev.makespan
        out[i, 1] = ev.energy
    return out


def _family_slices(n: int) -> dict[str, slice]:
    """Random-key column slice for each decision family (blocks of width N)."""
    return {
        "sequence": slice(0, n),
        "assignment": slice(n, 2 * n),
        "empty_speed": slice(2 * n, 3 * n),
        "loaded_speed": slice(3 * n, 4 * n),
    }


# --------------------------------------------------------------------------------------
# Grouped Sobol sensitivity (exact evaluator)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SobolIndex:
    """First-order and total-order Sobol indices of one family on one objective."""

    family: str
    objective: str
    first_order: float  # S_i: this family's own (main) variance share
    total_order: float  # ST_i: this family's share incl. all interactions (the cascade)


@dataclass(frozen=True, slots=True)
class SobolResult:
    """Grouped Sobol decomposition of objective variance over the decision families."""

    base_samples: int
    total_evaluations: int
    variance: dict[str, float]  # objective -> Var(objective) over the sample
    indices: dict[str, dict[str, SobolIndex]]  # objective -> family -> index


def grouped_sobol(
    instance: Instance, *, base_samples: int = 1024, seed: int = 0
) -> SobolResult:
    """Grouped Sobol indices of the four decision families on (C_max, E).

    Uses a scrambled Sobol' sequence (Saltelli's A/B design) over the 4N random-key
    chromosome, decoded and evaluated exactly. For each family the first-order index
    (Saltelli 2010) measures its standalone contribution and the total-order index (Jansen
    1999) measures its contribution including every interaction it participates in -- the
    cascading couplings (e.g. sequence x assignment determining the critical path) that an
    additive tabular attributor cannot represent.

    Cost is base_samples * (2 + |families|) exact evaluations; base_samples is
    rounded up to the next power of two for the Sobol' generator.
    """
    n = instance.num_tasks
    d = NUM_BLOCKS * n
    m = max(1, int(np.ceil(np.log2(max(base_samples, 2)))))
    # One Sobol' draw of dimension 2d, split into the independent A / B matrices.
    sample = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random_base2(m)
    a, b = sample[:, :d], sample[:, d:]
    m_samples = a.shape[0]

    fa = _evaluate_keys(a, instance)
    fb = _evaluate_keys(b, instance)
    slices = _family_slices(n)
    total_evals = 2 * m_samples

    # Total variance per objective over the pooled A/B sample.
    pooled = np.vstack([fa, fb])
    variance = {OBJECTIVES[c]: float(np.var(pooled[:, c], ddof=1)) for c in range(len(OBJECTIVES))}

    indices: dict[str, dict[str, SobolIndex]] = {obj: {} for obj in OBJECTIVES}
    for family in FEATURE_FAMILIES:
        ab = a.copy()
        ab[:, slices[family]] = b[:, slices[family]]
        fab = _evaluate_keys(ab, instance)
        total_evals += m_samples
        for c, obj in enumerate(OBJECTIVES):
            var_c = variance[obj]
            if var_c <= 0.0:
                first = total = 0.0
            else:
                # Saltelli (2010) first-order; Jansen (1999) total-order.
                first = float(np.mean(fb[:, c] * (fab[:, c] - fa[:, c])) / var_c)
                total = float(np.mean((fa[:, c] - fab[:, c]) ** 2) / (2.0 * var_c))
            indices[obj][family] = SobolIndex(
                family=family,
                objective=obj,
                first_order=float(np.clip(first, 0.0, 1.0)),
                total_order=float(max(total, 0.0)),
            )

    return SobolResult(
        base_samples=m_samples,
        total_evaluations=total_evals,
        variance=variance,
        indices=indices,
    )


# --------------------------------------------------------------------------------------
# Exact critical-path cascade attribution
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Per-task exact critical-path attribution averaged over a sampled landscape."""

    num_samples: int
    agv_bound_freq: tuple[float, ...]  # P(task on critical path, AGV-bound)
    qc_bound_freq: tuple[float, ...]  # P(task on critical path, QC-bound)
    agv_share: float  # overall fraction of critical-path mass that is AGV-bound
    marginal_makespan: tuple[float, ...]  # mean C_max reduction from accelerating task j
    marginal_energy: tuple[float, ...]  # mean Delta E from accelerating task j
    cascade_size: tuple[float, ...]  # mean # downstream tasks whose completion shifts


def _accelerate(schedule: Schedule, task: int) -> Schedule:
    """Return schedule with task's empty + loaded legs set to the fastest level."""
    empty = list(schedule.empty_speed)
    loaded = list(schedule.loaded_speed)
    empty[task] = _FASTEST_SPEED
    loaded[task] = _FASTEST_SPEED
    return replace(schedule, empty_speed=tuple(empty), loaded_speed=tuple(loaded))


def critical_path_landscape(
    instance: Instance, *, num_samples: int = 256, seed: int = 0
) -> CascadeResult:
    """Exact per-task critical-path attribution + downstream cascade over random schedules.

    For each sampled schedule the routine (1) records, via the exact max-plus binding
    (critical_path_binding), whether each task gates the makespan through its AGV
    arc or its QC chain; and (2) accelerate each task to the fastest speed and re-evaluate,
    measuring the marginal (C_max, E) change and the downstream cascade size -- the
    number of tasks whose completion time shifts. The cascade size is the precise temporal
    propagation a flat tabular attribution cannot express.
    """
    # Imported lazily: faithfulness.py transitively imports Torch (the surrogate), on which
    # the Sobol/Pareto core does not depend.
    from ehgat.benchmark.faithfulness import critical_path_binding
    from ehgat.utils.seeding import make_rng

    rng = make_rng(seed)
    n = instance.num_tasks
    chrom_len = NUM_BLOCKS * n

    agv_freq = np.zeros(n)
    qc_freq = np.zeros(n)
    marg_mk = np.zeros(n)
    marg_en = np.zeros(n)
    cascade = np.zeros(n)

    for _ in range(num_samples):
        schedule = decode(rng.random(chrom_len), instance)
        base = evaluate(schedule, instance)
        agv_bound, qc_bound = critical_path_binding(schedule, instance)
        for j in agv_bound:
            agv_freq[j] += 1.0
        for j in qc_bound:
            qc_freq[j] += 1.0
        base_completion = np.asarray(base.completion)
        for j in range(n):
            ev = evaluate(_accelerate(schedule, j), instance)
            marg_mk[j] += base.makespan - ev.makespan
            marg_en[j] += ev.energy - base.energy
            cascade[j] += float(np.sum(~np.isclose(np.asarray(ev.completion), base_completion)))

    inv = 1.0 / max(num_samples, 1)
    total_mass = float(agv_freq.sum() + qc_freq.sum())
    return CascadeResult(
        num_samples=num_samples,
        agv_bound_freq=tuple((agv_freq * inv).tolist()),
        qc_bound_freq=tuple((qc_freq * inv).tolist()),
        agv_share=float(agv_freq.sum() / total_mass) if total_mass > 0.0 else 0.5,
        marginal_makespan=tuple((marg_mk * inv).tolist()),
        marginal_energy=tuple((marg_en * inv).tolist()),
        cascade_size=tuple((cascade * inv).tolist()),
    )


# --------------------------------------------------------------------------------------
# Pareto-vs-dominated contrast: which descriptors separate the non-dominated set
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParetoContrast:
    """Decision descriptors that separate the non-dominated set from the dominated set."""

    num_samples: int
    num_pareto: int
    descriptors: tuple[str, ...]
    pareto_mean: dict[str, float]
    dominated_mean: dict[str, float]
    cliffs_delta: dict[str, float]  # in [-1, 1]; >0 => descriptor larger on the front


def _descriptors(schedule: Schedule, instance: Instance) -> dict[str, float]:
    """Interpretable decision descriptors used for the Pareto contrast."""
    empty = np.asarray([int(s) for s in schedule.empty_speed], dtype=float)
    loaded = np.asarray([int(s) for s in schedule.loaded_speed], dtype=float)
    counts = np.bincount(np.asarray(schedule.assignment), minlength=instance.num_agvs)
    return {
        "mean_empty_speed": float(empty.mean()),
        "mean_loaded_speed": float(loaded.mean()),
        "agv_load_imbalance": float(counts.max() - counts.min()),
    }


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size: (#(a>b) - #(a<b)) / (|a||b|) in [-1, 1]."""
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (a.size * b.size))


def pareto_contrast(
    instance: Instance, *, num_samples: int = 512, seed: int = 0
) -> ParetoContrast:
    """Contrast decision descriptors of non-dominated vs dominated sampled solutions.

    Samples random schedules, partitions them by exact Pareto dominance, and reports the
    Cliff's delta of each interpretable descriptor between the front and the dominated set
    -- a direct, model-free answer to why solutions are Pareto-(near)optimal (e.g. "front
    membership favours low loaded speed and balanced AGV load").
    """
    from ehgat.utils.seeding import make_rng

    rng = make_rng(seed)
    chrom_len = NUM_BLOCKS * instance.num_tasks

    schedules = [decode(rng.random(chrom_len), instance) for _ in range(num_samples)]
    objectives = [evaluate(s, instance).objectives for s in schedules]
    front = set(fast_non_dominated_sort(objectives)[0])

    descriptor_names = tuple(_descriptors(schedules[0], instance).keys())
    rows = np.asarray([list(_descriptors(s, instance).values()) for s in schedules])
    is_front = np.asarray([i in front for i in range(num_samples)])

    pareto_mean: dict[str, float] = {}
    dominated_mean: dict[str, float] = {}
    cliffs: dict[str, float] = {}
    for c, name in enumerate(descriptor_names):
        front_vals = rows[is_front, c]
        dom_vals = rows[~is_front, c]
        pareto_mean[name] = float(front_vals.mean()) if front_vals.size else 0.0
        dominated_mean[name] = float(dom_vals.mean()) if dom_vals.size else 0.0
        cliffs[name] = _cliffs_delta(front_vals, dom_vals)

    return ParetoContrast(
        num_samples=num_samples,
        num_pareto=int(is_front.sum()),
        descriptors=descriptor_names,
        pareto_mean=pareto_mean,
        dominated_mean=dominated_mean,
        cliffs_delta=cliffs,
    )


# --------------------------------------------------------------------------------------
# The topological failure boundary of tabular TreeSHAP
# --------------------------------------------------------------------------------------
def _family_normalised(values: dict[str, float]) -> dict[str, float]:
    """Normalise a per-family importance dict to sum to 1 (a comparable profile)."""
    total = float(sum(values.values()))
    if total <= 0.0:
        return {k: 0.0 for k in values}
    return {k: v / total for k, v in values.items()}


def tabular_failure_boundary(
    instance: Instance,
    sobol: SobolResult,
    *,
    num_samples: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Compare TreeSHAP's family-importance profile to the exact Sobol profile.

    Trains the tabular XGBoost surrogate, aggregates its per-feature TreeSHAP importance
    into the four decision families, and contrasts that with the exact Sobol total-order
    profile. The expected finding: TreeSHAP tracks Sobol on the speed (kinematic)
    families but diverges on the structural (sequence/assignment) families, because the
    additive tabular model cannot represent the cascading topological coupling -- the
    boundary where graph-native / exact methods must take over.

    Imports XGBoost + SHAP lazily so the core landscape module stays dependency-light.
    """
    from ehgat.surrogate.explainer_xgb import (
        XGBConfig,
        feature_names,
        shap_importance,
        train_xgb,
    )

    result = train_xgb(instance, XGBConfig(num_samples=num_samples, seed=seed))
    shap = shap_importance(result, instance, num_samples=min(256, num_samples), seed=seed)
    names = feature_names(instance)  # 4 features per task: seq_pos, agv, empty_spd, loaded_spd

    # Map the flat 4N feature importances onto the four decision families.
    family_of = ("sequence", "assignment", "empty_speed", "loaded_speed")
    out: dict[str, object] = {"r2": {k: result.metrics.get(f"r2_{k}", float("nan")) for k in OBJECTIVES}}
    boundary: dict[str, object] = {}
    shap_key = {"makespan": "makespan", "energy": "energy"}
    for obj in OBJECTIVES:
        shap_vals = np.abs(np.asarray(shap[shap_key[obj]]))
        shap_family = {fam: 0.0 for fam in FEATURE_FAMILIES}
        for idx in range(len(names)):
            shap_family[family_of[idx % NUM_BLOCKS]] += float(shap_vals[idx])
        shap_profile = _family_normalised(shap_family)
        sobol_profile = _family_normalised(
            {fam: sobol.indices[obj][fam].total_order for fam in FEATURE_FAMILIES}
        )
        # The exact (Sobol) vs fitted (TreeSHAP) importance mass on the topological region.
        sobol_struct = sum(sobol_profile[f] for f in _STRUCTURAL_FAMILIES)
        shap_struct = sum(shap_profile[f] for f in _STRUCTURAL_FAMILIES)
        boundary[obj] = {
            "shap_profile": shap_profile,
            "sobol_profile": sobol_profile,
            "sobol_structural_mass": float(sobol_struct),
            "shap_structural_mass": float(shap_struct),
            # >0 => TreeSHAP misses topological importance, spreading it onto the speed knobs.
            "structural_underweight": float(sobol_struct - shap_struct),
            "total_variation": float(
                0.5 * sum(abs(shap_profile[f] - sobol_profile[f]) for f in FEATURE_FAMILIES)
            ),
        }
    out["boundary"] = boundary
    return out


# --------------------------------------------------------------------------------------
# Bundle + serialisation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LandscapeResult:
    """Full landscape analysis for one instance (the landscape deliverable)."""

    num_tasks: int
    num_agvs: int
    num_qcs: int
    sobol: SobolResult
    cascade: CascadeResult
    contrast: ParetoContrast


def run_landscape(
    instance: Instance,
    *,
    sobol_base_samples: int = 1024,
    cascade_samples: int = 256,
    contrast_samples: int = 512,
    seed: int = 0,
) -> LandscapeResult:
    """Run the full exact landscape analysis (Sobol + cascade + Pareto contrast)."""
    return LandscapeResult(
        num_tasks=instance.num_tasks,
        num_agvs=instance.num_agvs,
        num_qcs=len(instance.qcs),
        sobol=grouped_sobol(instance, base_samples=sobol_base_samples, seed=seed),
        cascade=critical_path_landscape(instance, num_samples=cascade_samples, seed=seed),
        contrast=pareto_contrast(instance, num_samples=contrast_samples, seed=seed),
    )


def to_json_dict(result: LandscapeResult) -> dict[str, object]:
    """JSON-serialisable view of a LandscapeResult."""
    return {
        "experiment": "landscape",
        "instance": {
            "num_tasks": result.num_tasks,
            "num_agvs": result.num_agvs,
            "num_qcs": result.num_qcs,
        },
        "sobol": {
            "base_samples": result.sobol.base_samples,
            "total_evaluations": result.sobol.total_evaluations,
            "variance": result.sobol.variance,
            "indices": {
                obj: {
                    fam: {
                        "first_order": result.sobol.indices[obj][fam].first_order,
                        "total_order": result.sobol.indices[obj][fam].total_order,
                    }
                    for fam in FEATURE_FAMILIES
                }
                for obj in OBJECTIVES
            },
        },
        "cascade": {
            "num_samples": result.cascade.num_samples,
            "agv_share": result.cascade.agv_share,
            "agv_bound_freq": list(result.cascade.agv_bound_freq),
            "qc_bound_freq": list(result.cascade.qc_bound_freq),
            "marginal_makespan": list(result.cascade.marginal_makespan),
            "marginal_energy": list(result.cascade.marginal_energy),
            "cascade_size": list(result.cascade.cascade_size),
        },
        "pareto_contrast": {
            "num_samples": result.contrast.num_samples,
            "num_pareto": result.contrast.num_pareto,
            "descriptors": list(result.contrast.descriptors),
            "pareto_mean": result.contrast.pareto_mean,
            "dominated_mean": result.contrast.dominated_mean,
            "cliffs_delta": result.contrast.cliffs_delta,
        },
    }


def family_importance_table(sobol: SobolResult) -> Sequence[tuple[str, str, float, float]]:
    """Flatten Sobol indices into (objective, family, first_order, total_order) rows."""
    rows: list[tuple[str, str, float, float]] = []
    for obj in OBJECTIVES:
        for fam in FEATURE_FAMILIES:
            idx = sobol.indices[obj][fam]
            rows.append((obj, fam, idx.first_order, idx.total_order))
    return rows
