#!/usr/bin/env bash
# R3 optimization-scaling sweep, seed-sharded for the high-core VM.
#
# Runs run_tape_guided_bench.py across an N-ladder at the SAME budget-matched config as the
# main table (p_mult=5 => 20N exact evals/gen for every method, screening=2, gens=40), so
# the HV/HV* trend vs N reconciles with tab:main. Each N's seeds are split into shards that
# run as independent single-thread processes; shards are merged per N, then aggregated into
# the scaling table. Shows guided search holding while mp-BRKGA / random NSGA-II stall.
#
#   bash scripts/run_opt_scaling.sh unc   experiments/fused_tape_guided/scaling_opt_unc
#   bash scripts/run_opt_scaling.sh pp30  experiments/fused_tape_guided/scaling_opt_pp30
set -u
cd /workspace/e-hgatv2

MODE="${1:-unc}"
OUTDIR="${2:-experiments/fused_tape_guided/scaling_opt_${MODE}}"
TOTAL_SEEDS="${TOTAL_SEEDS:-8}"
PER_SHARD="${PER_SHARD:-2}"
NS="${NS:-10 20 40 80 160}"

PP_ARG=""
[[ "$MODE" == pp* ]] && PP_ARG="--peak-power ${MODE#pp}"

# SEARCH_DEV=cuda routes the guided arms' GNN screening onto the GPU (see COMPUTE_SCALING.md);
# exact-eval arms (mp/random/sp) stay on CPU regardless. NGPU splits shards across the A40s.
SEARCH_DEV="${SEARCH_DEV:-cpu}"
NGPU="${NGPU:-2}"
BASE_POP="${BASE_POP:-}"   # if set, fixed per-population P across N (O(N) cost) => fixed-budget scaling
POP_ARG=""
[[ -n "$BASE_POP" ]] && POP_ARG="--base-pop $BASE_POP"

GENS="${GENS:-40}"
REFGENS="${REFGENS:-50}"
# Screening factor and surrogate-training size are now env-overridable (the batched tropical DP
# makes higher k budget-neutral-cheap; a bigger surrogate is the GNN-specific fidelity lever).
SCREENING="${SCREENING:-2}"
CORE_SAMPLES="${CORE_SAMPLES:-800}";  CORE_EPOCHS="${CORE_EPOCHS:-40}"
FUSED_SAMPLES="${FUSED_SAMPLES:-600}"; FUSED_EPOCHS="${FUSED_EPOCHS:-40}"
COMMON="--gens ${GENS} --screening ${SCREENING} --p-mult 5 $POP_ARG \
  --core-samples ${CORE_SAMPLES} --core-epochs ${CORE_EPOCHS} \
  --fused-samples ${FUSED_SAMPLES} --fused-epochs ${FUSED_EPOCHS} \
  --faith-samples 30 --ref-gens ${REFGENS} --device ${SEARCH_DEV} --search-device ${SEARCH_DEV}"

SH="$OUTDIR/shards"
mkdir -p "$SH/logs"
rm -f "$SH"/tape_bench_*_sh*.json

echo "[opt-scaling] mode=$MODE ns=[$NS] seeds=$TOTAL_SEEDS shard=$PER_SHARD search_dev=$SEARCH_DEV -> $OUTDIR"
declare -a pids
gi=0
for n in $NS; do
  s=0
  while [ "$s" -lt "$TOTAL_SEEDS" ]; do
    ns=$PER_SHARD
    if [ $((s + ns)) -gt "$TOTAL_SEEDS" ]; then ns=$((TOTAL_SEEDS - s)); fi
    tag="n${n}_sh${s}"
    GPU_ENV=""
    [[ "$SEARCH_DEV" == cuda ]] && GPU_ENV="CUDA_VISIBLE_DEVICES=$((gi % NGPU))"
    # $GPU_ENV must go through `env` -- a variable-expanded token is NOT treated as an
    # inline assignment by bash (it would be run as a command name otherwise).
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
      nohup env $GPU_ENV python scripts/run_tape_guided_bench.py --instance "toy:$n" $PP_ARG $COMMON \
        --seeds "$ns" --seed-start "$s" --out-tag "$tag" --out-dir "$SH" \
        > "$SH/logs/$tag.log" 2>&1 &
    pids+=($!)
    gi=$((gi + 1))
    s=$((s + ns))
  done
done
echo "[opt-scaling] launched ${#pids[@]} shard processes; waiting ..."

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "[opt-scaling] all shards done (fail=$fail); merging ..."

python scripts/merge_tape_shards.py --shards-dir "$SH" --out-dir "$OUTDIR"
python scripts/aggregate_opt_scaling.py --dir "$OUTDIR"
echo "[opt-scaling] DONE mode=$MODE"
