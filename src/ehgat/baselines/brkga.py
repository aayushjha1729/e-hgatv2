"""Multi-objective Biased Random-Key Genetic Algorithm baseline.

Follows Fontes & Homayouni (2022): a population of length-4N random-key
chromosomes is decoded by the shared canonical decoder, evaluated for
(makespan, energy), and ranked with NSGA-II non-dominated sorting + crowding
distance. Each generation the new population is::

    Ne elites  (carried over by rank/crowding)
  + No offspring  (biased uniform crossover: one elite + one non-elite parent)
  + Nm mutants  (fresh random-key chromosomes)

with an elite gene inherited with probability inherit_prob (p_e in the paper).

This single-population variant is the stochastic baseline the attention-guided
guided NSGA-II is compared against. An external, unbounded Pareto archive tracks
the best non-dominated solutions across all generations, and a per-generation front
history supports the convergence analysis. The exact Oracle (oracle.py)
upper-bounds achievable quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ehgat.environment.decoder import NUM_BLOCKS, decode
from ehgat.environment.evaluator import evaluate
from ehgat.environment.instance import Instance
from ehgat.search.nsga2 import fast_non_dominated_sort, order_by_rank_crowding
from ehgat.utils.seeding import make_rng

__all__ = ["BRKGAConfig", "BRKGAResult", "default_config", "run_brkga"]

Objectives = tuple[float, float]
_ARCHIVE_ROUND = 6  # dedup objective key precision


@dataclass(frozen=True, slots=True)
class BRKGAConfig:
    """BRKGA hyper-parameters (defaults follow Fontes & Homayouni 2022)."""

    pop_size: int
    generations: int = 300
    elite_frac: float = 0.2
    mutant_frac: float = 0.1
    inherit_prob: float = 0.7
    seed: int = 0


@dataclass(frozen=True, slots=True)
class BRKGAResult:
    """Outcome of a BRKGA run."""

    front: tuple[Objectives, ...]  # non-dominated (makespan, energy), sorted ascending
    chromosomes: tuple[np.ndarray, ...]  # chromosome for each front point (aligned)
    front_history: tuple[tuple[Objectives, ...], ...]  # archive front per generation
    evaluations: int


def default_config(instance: Instance, *, generations: int = 300, seed: int = 0) -> BRKGAConfig:
    """Population P = 20N and paper-default fractions for instance."""
    return BRKGAConfig(pop_size=20 * instance.num_tasks, generations=generations, seed=seed)


def _evaluate_population(pop: np.ndarray, instance: Instance) -> list[Objectives]:
    return [evaluate(decode(chrom, instance), instance).objectives for chrom in pop]


def _update_archive(
    archive_obj: list[Objectives],
    archive_chrom: list[np.ndarray],
    new_obj: list[Objectives],
    new_chrom: list[np.ndarray],
) -> tuple[list[Objectives], list[np.ndarray]]:
    """Merge new into the archive, keeping a deduplicated non-dominated set."""
    seen: set[tuple[float, float]] = set()
    merged_obj: list[Objectives] = []
    merged_chrom: list[np.ndarray] = []
    for obj, chrom in zip(archive_obj + new_obj, archive_chrom + new_chrom, strict=True):
        key = (round(obj[0], _ARCHIVE_ROUND), round(obj[1], _ARCHIVE_ROUND))
        if key in seen:
            continue
        seen.add(key)
        merged_obj.append(obj)
        merged_chrom.append(chrom)
    keep = fast_non_dominated_sort(merged_obj)[0]
    return [merged_obj[i] for i in keep], [merged_chrom[i] for i in keep]


def run_brkga(instance: Instance, config: BRKGAConfig) -> BRKGAResult:
    """Evolve a multi-objective BRKGA and return its non-dominated front."""
    n_elite = max(1, round(config.elite_frac * config.pop_size))
    n_mutant = max(1, round(config.mutant_frac * config.pop_size))
    n_offspring = config.pop_size - n_elite - n_mutant
    if n_offspring < 0:
        raise ValueError(
            f"pop_size {config.pop_size} too small for elite+mutant "
            f"({n_elite}+{n_mutant}); increase pop_size or reduce fractions."
        )

    rng = make_rng(config.seed)
    chrom_len = NUM_BLOCKS * instance.num_tasks
    pop = rng.random((config.pop_size, chrom_len))

    archive_obj: list[Objectives] = []
    archive_chrom: list[np.ndarray] = []
    history: list[tuple[Objectives, ...]] = []
    evaluations = 0

    for gen in range(config.generations + 1):
        objectives = _evaluate_population(pop, instance)
        evaluations += len(pop)

        fronts = fast_non_dominated_sort(objectives)
        front0 = fronts[0]
        archive_obj, archive_chrom = _update_archive(
            archive_obj, archive_chrom, [objectives[i] for i in front0], [pop[i] for i in front0]
        )
        history.append(tuple(sorted(archive_obj)))

        if gen == config.generations:
            break

        order = order_by_rank_crowding(objectives, fronts)
        elite_idx = order[:n_elite]
        non_elite_idx = order[n_elite:]
        elite_pop = pop[elite_idx]

        next_pop = np.empty_like(pop)
        next_pop[:n_elite] = elite_pop
        for k in range(n_offspring):
            elite_parent = elite_pop[rng.integers(n_elite)]
            non_elite_parent = pop[non_elite_idx[rng.integers(len(non_elite_idx))]]
            inherit_elite = rng.random(chrom_len) < config.inherit_prob
            next_pop[n_elite + k] = np.where(inherit_elite, elite_parent, non_elite_parent)
        next_pop[n_elite + n_offspring :] = rng.random((n_mutant, chrom_len))
        pop = next_pop

    order = sorted(range(len(archive_obj)), key=lambda i: archive_obj[i])
    return BRKGAResult(
        front=tuple(archive_obj[i] for i in order),
        chromosomes=tuple(archive_chrom[i].copy() for i in order),
        front_history=tuple(history),
        evaluations=evaluations,
    )
