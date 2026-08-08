# E-HGATv2

Reference implementation for bi-objective scheduling of quay cranes (QCs) and
speed-adjustable automated guided vehicles (SA-AGVs) in a dual-cycling container terminal.
The objectives are the makespan `C_max` and the total AGV travel energy `E`, both
minimised, and are treated in two regimes: the uncoupled regime, and a regime in which a
fleet-wide instantaneous power budget couples the vehicles.

The scheduling model, kinematics and distance data follow Homayouni and Fontes (2022) and
Fontes and Homayouni (2022, 2023). The surrogate model and its attribution readout are
introduced here.

The surrogate computes the makespan through a max-plus (tropical) longest-path layer, so
the prediction is produced by the same critical-path recurrence as the exact evaluator. The
gradient of that layer with respect to each transport leg is therefore an exact binary
indicator of membership in the critical path. The repository applies this readout in three
settings: post-hoc explanation of a single schedule, a steering signal within NSGA-II, and
a description of how the binding bottleneck changes along the Pareto front.

## Verification of the physical model

Every other component is measured against the exact evaluator, so its timing model is
checked independently:

| Check | Location |
|---|---|
| MILP Eqs. (2)–(4) and (10)–(18) as a forward recurrence with the binaries fixed | `src/ehgat/environment/evaluator.py` |
| Independent line-by-line MILP transcription, solved by fixed-point longest-path iteration | `scripts/verify_timing.py` |
| Exhaustive `3^(2N)` enumeration of speed assignments against the exact evaluator | `tests/unit/test_oracle.py` |
| Speed and power constants checked against the published kinematic relation `v = α·v₀` at import time | `src/ehgat/environment/physics.py` |
| Coupled evaluator reduces to the uncoupled one at a non-binding power budget | `tests/unit/test_power_evaluator.py` |

The assignment constraints Eqs. (5)–(9) and (19) are satisfied by construction by the
random-key decoder (`src/ehgat/environment/decoder.py`) and so do not appear as explicit
inequalities. This follows the encoding of Fontes and Homayouni (2022), §4.2.

The mp-BRKGA baseline in `src/ehgat/baselines/mp_brkga.py` is a re-implementation from the
published description. It has not been validated against published per-instance `C_max`
and `E` values, which are not available in the accessible material, so comparisons against
it are comparisons against a re-implementation.

## Layout

```
src/ehgat/
  environment/   physics, distances, instance generation, random-key decoder,
                 exact evaluator (uncoupled and power-coupled), exhaustive Pareto
                 enumeration
  baselines/     single-population BRKGA, multi-population BRKGA (mp-BRKGA)
  surrogate/     E-HGATv2 heterogeneous core, graph construction, training,
                 XGBoost/TreeSHAP flat-surrogate baseline
  explain/       tropical max-plus longest-path layer (custom autograd.Function),
                 batched variant, shared event-DAG assembler, fused head, TAPE
  search/        NSGA-II, surrogate-screened and attribution-guided variants
  metrics/       hypervolume, IGD+/GD+, spread
  utils/         central seeding, semantic tensor assertions
scripts/         experiment drivers, aggregation, figure and table generation
tests/unit/      29 test modules (173 tests without the optional learning extra)
experiments/     result artifacts backing every number reported in the paper
paper/           main.tex and figures
data/            published Table 4 distance matrix, Table 5 loading instances
```

## Installation

```bash
uv sync --python 3.12                              # environment, evaluator, BRKGA
uv sync --python 3.12 --extra learn --extra viz    # add the surrogate and plotting stack
```

## Tests

```bash
uv run pytest -m "not learn"     # 173 tests, no Torch required
uv run pytest                    # full suite including the surrogate
uv run python scripts/verify_timing.py   # independent MILP cross-check
```

## Reproducing the reported results

Each reported number derives from a seeded run whose artifact is stored under
`experiments/`. Tables and figures are regenerated from those artifacts alone:

```bash
uv run python scripts/compute_paper_stats.py   # -> experiments/fused_tape_guided/paper_stats.json
uv run python scripts/make_paper_figs.py       # -> paper/figs/*.pdf
uv run python scripts/emit_r3_numbers.py       # search-comparison table
```

The underlying experiments, should they be re-run rather than re-aggregated:

| Result | Driver |
|---|---|
| Headline 11-instance benchmark | `scripts/run_tape_guided_bench.py` |
| Scaling ladder, both regimes | `scripts/run_opt_scaling.sh` |
| Size generalisation (train small, evaluate large) | `scripts/run_scaling_generalization.py` |
| mp-BRKGA comparison | `scripts/run_mp_comparison.py` |
| Front composition, leave-one-instance-out | `scripts/run_front_learning.py`, `scripts/loro_composition_proof.py` |
| Screening ablation and fidelity probe | `scripts/screening_ablation.py`, `scripts/probe_screening_fidelity.py` |
| Critical-path demonstrations | `scripts/run_critical_path_demo.py` |
| Trade-off criticality along the front | `scripts/run_tcs_frontier.py` |

## Determinism

Randomness is seeded centrally through `ehgat.utils.seeding`. The surrogate runs on CPU for
all reported results, since scatter-max aggregation and the max-plus relaxation are
order-sensitive under non-deterministic GPU reductions; CPU execution makes each result
bit-reproducible from its recorded seed. Each experiment records its seed and configuration
alongside its output.

## Reading order

1. `src/ehgat/environment/evaluator.py`, the physical model, annotated line by line with
   the MILP equation each statement realises.
2. `src/ehgat/environment/decoder.py`, the random-key encoding and its decode rules.
3. `src/ehgat/explain/event_dag.py`, the event-DAG assembler. The exact evaluator and the
   learned head build the same precedence structure here and differ only in the source of
   the leg and delay values.
4. `src/ehgat/search/attention_nsga2.py`, the search loop, including offspring screening.
5. `paper/main.tex`. Appendices A and B cover the implementation and the hyper-parameters.
