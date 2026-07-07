#!/bin/bash
set -euo pipefail

# Launcher for all combinations of 3 forget batches:
#  - Train then unlearn with each combination
#  - Each run uses a specific set of 3 forget batches (no random sampling)
#  - Outputs per-run folders under a single results root
#
# Usage:
#   ./run_combination_forget_unlearning.sh \
#       /lfs/mercury1/0/sahasras/r2d/data/cifar100/data_split/cifar100_bs_750 \
#       /lfs/mercury1/0/sahasras/r2d/test_cifar100_combination_forget_runs_bs_750 \
#       cifar100 \
#       tinynetcifar100 \
#       "1" \
#       2 \
#       "full" \
#       128 \
#       35 \
#       5 \
#       false \
#       0.01 \
#       "1"
#
# Args:
#   1) DATAROOT         Path to pre-split data root (train/val/test/forget/retain)  [required]
#   2) RESULTS_DIR_BASE Root to store per-run folders                               [required]
#   3) DATASET          Dataset name: cifar10 or cifar100                           [required]
#   4) MODEL            Model architecture: tinynet or tinynetcifar100              [required]
#   5) GPUS             Comma-separated GPU IDs (default: "1")
#   6) NUM_JOBS         Parallel jobs per GPU (default: 2)
#   7) BATCH_SIZE       Batch size for main.py (default: "full")
#   8) MICRO_BATCH_SIZE Micro batch size for gradient accumulation (default: 128)
#   9) TRAIN_EPOCHS     Number of epochs for initial training phase (default: 35)
#  10) UNLEARN_EPOCHS   Number of epochs for unlearning phase (default: 5)
#  11) SHUFFLE          true/false for DataLoader shuffling (default: false)
#  12) LEARNING_RATE    Learning rate for training (default: 0.01)
#  13) COMBO_INDEX      1-based index of a single combination to run (default: all)
# Note: COMBINATION_SIZE is computed automatically as NUM_FORGET_BATCHES / 2

DATAROOT=${1:-/lfs/mercury1/0/sahasras/r2d/data/cifar100/data_split/cifar100_uniform_bs_2250_seed1}
RESULTS_DIR_BASE=${2:-/lfs/mercury1/0/sahasras/r2d/cifar100_uniform_combination_forget_runs_bs_2250_lr_0_01}
DATASET=${3:-cifar100}
MODEL=${4:-tinynetcifar100}
GPUS_STR=${5:-"6"}
NUM_JOBS=${6:-2}
BATCH_SIZE=${7:-"full"}
MICRO_BATCH_SIZE=${8:-128}
TRAIN_EPOCHS=${9:-35}
UNLEARN_EPOCHS=${10:-5}
SHUFFLE=${11:-false}
LEARNING_RATE=${12:-0.01}
COMBO_INDEX=${13:-}
# Handle empty string case - if LEARNING_RATE is empty, use default
[ -z "$LEARNING_RATE" ] && LEARNING_RATE=0.01

# Count forget batches and compute COMBINATION_SIZE = NUM_FORGET_BATCHES / 2
FORGET_DIR="${DATAROOT}/forget"
NUM_FORGET_BATCHES=$(ls "${FORGET_DIR}"/batch_*.pkl 2>/dev/null | wc -l)

if [ "${NUM_FORGET_BATCHES}" -eq 0 ]; then
    echo "Error: No forget batches found in ${FORGET_DIR}"
    exit 1
fi

COMBINATION_SIZE=$(( NUM_FORGET_BATCHES / 2 ))
echo "NUM_FORGET_BATCHES=${NUM_FORGET_BATCHES}, using COMBINATION_SIZE=${COMBINATION_SIZE}"

# Generate RESULTS_DIR under RESULTS_DIR_BASE
# Format: {base}/bs_{batch_size}_train_{train_epochs}_unlearn_{unlearn_epochs}_lr_{learning_rate}_comb{combination_size}
if [ "${BATCH_SIZE}" = "full" ]; then
    BS_STR="full"
else
    BS_STR="${BATCH_SIZE}"
fi
LR_STR=$(echo "${LEARNING_RATE}" | sed 's/\./_/g')
RESULTS_DIR="${RESULTS_DIR_BASE}/bs_${BS_STR}_train_${TRAIN_EPOCHS}_unlearn_${UNLEARN_EPOCHS}_lr_${LR_STR}_comb${COMBINATION_SIZE}"

PY_MAIN=random_forget_runner_all_combinations.py

# Generate all combinations of COMBINATION_SIZE batches

# Generate all combinations using Python
COMBINATIONS=$(python3 <<PY
from itertools import combinations
batches = list(range(${NUM_FORGET_BATCHES}))
combs = list(combinations(batches, ${COMBINATION_SIZE}))
print(" ".join([",".join(map(str, c)) for c in combs]))
PY
)

# Convert to array
IFS=' ' read -ra COMB_ARRAY <<< "${COMBINATIONS}"
NUM_COMBINATIONS_TOTAL=${#COMB_ARRAY[@]}

declare -a LAUNCH_INDICES=()
if [ -n "${COMBO_INDEX}" ]; then
    if ! [[ "${COMBO_INDEX}" =~ ^[0-9]+$ ]]; then
        echo "Error: COMBO_INDEX must be a positive integer (1-based), got '${COMBO_INDEX}'"
        exit 1
    fi
    if [ "${COMBO_INDEX}" -lt 1 ] || [ "${COMBO_INDEX}" -gt "${NUM_COMBINATIONS_TOTAL}" ]; then
        echo "Error: COMBO_INDEX out of range. Valid range is 1..${NUM_COMBINATIONS_TOTAL}, got ${COMBO_INDEX}"
        exit 1
    fi

    TARGET_IDX=$((COMBO_INDEX - 1))
    LAUNCH_INDICES=("${TARGET_IDX}")
    echo "Filtering to COMBO_INDEX=${COMBO_INDEX}: [${COMB_ARRAY[$TARGET_IDX]}]"
else
    for ((idx=0; idx<NUM_COMBINATIONS_TOTAL; idx++)); do
        LAUNCH_INDICES+=("${idx}")
    done
fi
NUM_COMBINATIONS=${#LAUNCH_INDICES[@]}

IFS=',' read -ra GPU_ARRAY <<< "${GPUS_STR}"
NUM_GPUS=${#GPU_ARRAY[@]}
MAX_PARALLEL=$((NUM_GPUS * NUM_JOBS))

echo "Starting ${NUM_COMBINATIONS} runs across ${NUM_GPUS} GPUs (${NUM_JOBS} jobs/GPU, max ${MAX_PARALLEL} parallel)"
echo "Logs: ${RESULTS_DIR}/logs/"

mkdir -p "${RESULTS_DIR}"
export WANDB_DISABLED=true
# Improve PyTorch allocator behavior for large batches
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# Helpers
get_gpu_free_memory() {
    local GPU=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=${GPU} 2>/dev/null | head -1 | tr -d ' '
}

clear_pytorch_cache() {
    local GPU=$1
    python3 - <<PY 2>/dev/null || true
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(${GPU})
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
PY
}

# Track PIDs per GPU
IFS=',' read -ra GPU_ARRAY <<< "${GPUS_STR}"
NUM_GPUS=${#GPU_ARRAY[@]}

declare -A GPU_PIDS
for GPU in "${GPU_ARRAY[@]}"; do
    GPU_PIDS[$GPU]=""
done

# Map PID -> label for completion messages
declare -A RUN_LABELS

reap_finished() {
    local NEW_PIDS=()
    for PID in "${ALL_PIDS[@]:-}"; do
        if [ -z "${PID}" ]; then
            continue
        fi
        if kill -0 "${PID}" 2>/dev/null; then
            NEW_PIDS+=("${PID}")
        else
            if wait "${PID}" 2>/dev/null; then
                EXIT=$?
            else
                EXIT=$?
            fi
            if [ -n "${RUN_LABELS[$PID]+x}" ]; then
                # Extract run number and GPU from label for cleaner output
                if [[ "${RUN_LABELS[$PID]}" =~ run_([0-9]+).*gpu([0-9]+) ]]; then
                    if [ "${EXIT}" -eq 0 ]; then
                        echo "✓ Run ${BASH_REMATCH[1]} (GPU ${BASH_REMATCH[2]}) completed [exit ${EXIT}]"
                    else
                        echo "✗ Run ${BASH_REMATCH[1]} (GPU ${BASH_REMATCH[2]}) failed [exit ${EXIT}]"
                    fi
                else
                    if [ "${EXIT}" -eq 0 ]; then
                        echo "✓ ${RUN_LABELS[$PID]} completed [exit ${EXIT}]"
                    else
                        echo "✗ ${RUN_LABELS[$PID]} failed [exit ${EXIT}]"
                    fi
                fi
            fi
        fi
    done
    ALL_PIDS=("${NEW_PIDS[@]}")
}

wait_for_slot() {
    local GPU=$1
    # Adjust memory requirement based on micro batch size
    if [ "${MICRO_BATCH_SIZE}" -le 256 ]; then
        local MIN_FREE_MEM=8000   # ~8 GB for small micro batches
    elif [ "${MICRO_BATCH_SIZE}" -le 512 ]; then
        local MIN_FREE_MEM=10000  # ~10 GB for medium micro batches
    elif [ "${MICRO_BATCH_SIZE}" -le 750 ]; then
        local MIN_FREE_MEM=12000  # ~12 GB for 750 batch size
    else
        local MIN_FREE_MEM=15000  # ~15 GB for large batches
    fi
    while true; do
        local CURRENT=(${GPU_PIDS[$GPU]})
        local NEW_PIDS=()
        local RUNNING=0
        for PID in "${CURRENT[@]}"; do
            if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
                NEW_PIDS+=("${PID}")
                RUNNING=$((RUNNING + 1))
            else
                wait "${PID}" 2>/dev/null || true
            fi
        done
        GPU_PIDS[$GPU]="${NEW_PIDS[*]}"
        if [ "${RUNNING}" -lt "${NUM_JOBS}" ]; then
            local FREE_MEM
            FREE_MEM=$(get_gpu_free_memory "${GPU}")
            if [ -z "${FREE_MEM}" ]; then
                # nvidia-smi failed; proceed
                break
            fi
            if [ "${FREE_MEM}" -ge "${MIN_FREE_MEM}" ]; then
                break
            else
                # Suppress verbose memory wait messages
                clear_pytorch_cache "${GPU}"
                sleep 2
            fi
        else
            sleep 1
        fi
    done
}

RUN_COUNTER=0
declare -a ALL_PIDS=()
for i in "${LAUNCH_INDICES[@]}"; do
    reap_finished
    GPU_INDEX=$((RUN_COUNTER % NUM_GPUS))
    GPU=${GPU_ARRAY[$GPU_INDEX]}
    RUN_INDEX=$((i + 1))  # Start from 1

    wait_for_slot "${GPU}"
    clear_pytorch_cache "${GPU}"
    sleep 0.5

    # Final memory check before launch
    if [ "${MICRO_BATCH_SIZE}" -le 256 ]; then
        MIN_FREE_MEM_PRELAUNCH=8000
    elif [ "${MICRO_BATCH_SIZE}" -le 512 ]; then
        MIN_FREE_MEM_PRELAUNCH=10000
    elif [ "${MICRO_BATCH_SIZE}" -le 750 ]; then
        MIN_FREE_MEM_PRELAUNCH=12000
    else
        MIN_FREE_MEM_PRELAUNCH=15000
    fi
    FREE_MEM_PRELAUNCH=$(get_gpu_free_memory "${GPU}")
    while [ -n "${FREE_MEM_PRELAUNCH}" ] && [ "${FREE_MEM_PRELAUNCH}" -lt "${MIN_FREE_MEM_PRELAUNCH}" ]; do
        # Suppress verbose memory wait messages
        clear_pytorch_cache "${GPU}"
        sleep 2
        FREE_MEM_PRELAUNCH=$(get_gpu_free_memory "${GPU}")
    done

    # Reap any runs that may have finished while we waited for this slot
    reap_finished

    LOG_DIR="${RESULTS_DIR}/logs"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run_${RUN_INDEX}_gpu${GPU}.log"

    COMB_STR="${COMB_ARRAY[$i]}"
    RUN_LABEL="run_${RUN_INDEX}_comb[${COMB_STR}]_gpu${GPU}"
    echo "→ Run ${RUN_INDEX} started on GPU ${GPU} [${COMB_STR}]"
    
    SHUFFLE_ARG=()
    if [ "${SHUFFLE,,}" = "true" ]; then
        SHUFFLE_ARG+=(--shuffle)
    fi
    
    (
        {
            echo "CMD: CUDA_VISIBLE_DEVICES=${GPU} python3 ${PY_MAIN} --dataroot ${DATAROOT} --results_dir ${RESULTS_DIR} --run_index ${RUN_INDEX} --forget_batch_indices ${COMB_STR} --batch_size ${BATCH_SIZE} --micro_batch_size ${MICRO_BATCH_SIZE} --train_epochs ${TRAIN_EPOCHS} --unlearn_epochs ${UNLEARN_EPOCHS} --learning_rate ${LEARNING_RATE} --dataset ${DATASET} --model ${MODEL} --device 0 ${SHUFFLE_ARG[*]}"
            CUDA_VISIBLE_DEVICES=${GPU} \
            python3 "${PY_MAIN}" \
                --dataroot "${DATAROOT}" \
                --results_dir "${RESULTS_DIR}" \
                --run_index "${RUN_INDEX}" \
                --forget_batch_indices "${COMB_STR}" \
                --batch_size "${BATCH_SIZE}" \
                --micro_batch_size "${MICRO_BATCH_SIZE}" \
                --train_epochs "${TRAIN_EPOCHS}" \
                --unlearn_epochs "${UNLEARN_EPOCHS}" \
                --learning_rate "${LEARNING_RATE}" \
                --dataset "${DATASET}" \
                --model "${MODEL}" \
                --device 0 \
                "${SHUFFLE_ARG[@]}"
        } > "${LOG_FILE}" 2>&1
    ) &
    PID=$!
    RUN_LABELS[$PID]=$RUN_LABEL
    ALL_PIDS+=($PID)
    if [ -z "${GPU_PIDS[$GPU]}" ]; then
        GPU_PIDS[$GPU]="${PID}"
    else
        GPU_PIDS[$GPU]="${GPU_PIDS[$GPU]} ${PID}"
    fi
    RUN_COUNTER=$((RUN_COUNTER + 1))
done

while [ ${#ALL_PIDS[@]} -gt 0 ]; do
    reap_finished
    sleep 1
done

echo ""
echo "All ${NUM_COMBINATIONS} runs complete. Results: ${RESULTS_DIR}"

