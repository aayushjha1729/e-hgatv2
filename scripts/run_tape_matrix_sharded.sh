#!/usr/bin/env bash
# Seed-sharded R3 matrix for the high-core VM. Each instance's seeds are split into
# shards that run as independent single-thread processes (each re-trains its own model
# but that overhead is dwarfed by running 20 seeds serially). All shards across all 11
# instances run concurrently; then shards are merged per instance and paper stats run.
#
# Usage: scripts/run_tape_matrix_sharded.sh [TOTAL_SEEDS] [SEEDS_PER_SHARD]
set -u
cd /workspace/e-hgatv2

TOTAL_SEEDS="${1:-20}"
PER_SHARD="${2:-4}"
OUT=experiments/fused_tape_guided
SH=$OUT/shards
mkdir -p "$SH/logs"
rm -f "$SH"/tape_bench_*_sh*.json

COMMON="--gens 40 --screening 2 --p-mult 5 \
  --core-samples 800 --core-epochs 40 --fused-samples 600 --fused-epochs 40 \
  --faith-samples 30 --ref-gens 50 --device cpu"

# instance base-tag | instance-specific args
INSTANCES=(
  "toy5|--instance toy:5"
  "toy8|--instance toy:8"
  "toy10|--instance toy:10"
  "toy15|--instance toy:15"
  "toy20|--instance toy:20"
  "toy10_pp30|--instance toy:10 --peak-power 30"
  "toy20_pp30|--instance toy:20 --peak-power 30"
  "L07|--instance L07"
  "L15|--instance L15"
  "L21|--instance L21"
  "L35|--instance L35"
)

echo "=== launching shards: ${#INSTANCES[@]} instances x $TOTAL_SEEDS seeds (shard=$PER_SHARD) ==="
n=0
for entry in "${INSTANCES[@]}"; do
  base="${entry%%|*}"; iargs="${entry#*|}"
  s=0
  while [ "$s" -lt "$TOTAL_SEEDS" ]; do
    ns=$PER_SHARD
    if [ $((s + ns)) -gt "$TOTAL_SEEDS" ]; then ns=$((TOTAL_SEEDS - s)); fi
    tag="${base}_sh${s}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
      nohup uv run python scripts/run_tape_guided_bench.py $iargs $COMMON \
        --seeds "$ns" --seed-start "$s" --out-tag "$tag" --out-dir "$SH" \
        > "$SH/logs/$tag.log" 2>&1 &
    n=$((n + 1)); s=$((s + ns))
  done
done
echo "launched $n shard processes; waiting ..."
wait
echo "=== all shards done; merging ==="
uv run python scripts/merge_tape_shards.py --shards-dir "$SH" --out-dir "$OUT"
echo "=== computing paper stats ==="
uv run python scripts/compute_paper_stats.py
echo "SHARDED_MATRIX_DONE"