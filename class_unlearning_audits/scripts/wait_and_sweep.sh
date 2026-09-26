#!/bin/bash
# Waits until the chosen GPUs are all free (low memory + low utilization), then launches a
# 60-run sweep. Polls every $POLL_INTERVAL seconds until every GPU is clear.
#
# Usage:
#   ./wait_and_sweep.sh                                  # DELETE, default run-dir
#   ./wait_and_sweep.sh badteacher                       # bad teacher
#   ./wait_and_sweep.sh scrub                            # SCRUB (scrub_r for SCRUB+R)
#   METHOD=delete RUN_DIR=runs/delete/cifar100_unlearn_lr_1e-4_bs_1 \
#     EXTRA="--unlearn-lr 1e-4" GPUS=6,7,8,9 ./wait_and_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

METHOD="${1:-${METHOD:-delete}}"
RUN_DIR="${RUN_DIR:-runs/${METHOD}/cifar100_bs_1}"
EXTRA="${EXTRA:-}"

GPUS="${GPUS:-6,7,8,9}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
NUM_RUNS="${NUM_RUNS:-60}"
MEM_THRESHOLD_MIB=100    # a GPU counts as "free" if used memory is below this
UTIL_THRESHOLD_PCT=5     # and utilization is below this
POLL_INTERVAL=30         # seconds between checks

PYTHON="${PYTHON:-python}"

gpus_are_free() {
  IFS=',' read -ra GPU_ARR <<< "$GPUS"
  for gpu in "${GPU_ARR[@]}"; do
    read -r used util <<< "$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$gpu")"
    used=$(echo "$used" | tr -d ' ,')
    util=$(echo "$util" | tr -d ' ,')
    if [ "$used" -ge "$MEM_THRESHOLD_MIB" ] || [ "$util" -ge "$UTIL_THRESHOLD_PCT" ]; then
      return 1
    fi
  done
  return 0
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] method=$METHOD run-dir=$RUN_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPUs $GPUS to be free (mem < ${MEM_THRESHOLD_MIB}MiB, util < ${UTIL_THRESHOLD_PCT}% each)..."
while ! gpus_are_free; do
  sleep "$POLL_INTERVAL"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] still waiting..."
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs $GPUS are free. Launching sweep."
# shellcheck disable=SC2086
"$PYTHON" -m pipeline.run_sweep --config "configs/${METHOD}.yaml" --run-dir "$RUN_DIR" \
  --num-runs "$NUM_RUNS" --gpus "$GPUS" --runs-per-gpu "$RUNS_PER_GPU" $EXTRA
