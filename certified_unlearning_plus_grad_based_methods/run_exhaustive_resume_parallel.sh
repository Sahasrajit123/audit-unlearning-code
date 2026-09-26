#!/bin/bash
# Resume incomplete exhaustive-combo runs in parallel.
#
# This script looks for _run_*.dir files under a given results_dir. Each such file
# corresponds to a combo_idx whose run did not finish (the exhaustive launcher
# deletes _run_{idx}.dir on success). For each such combo, this script
# launches:
#   python experiment_unlearning_exhaustive_resume.py \
#       --results_dir RESULTS_DIR \
#       --combo_idx COMBO_IDX
# on an available GPU, with configurable parallelism.
#
# Example:
#   ./run_exhaustive_resume_parallel.sh \
#       --results_dir logs_multiple_runs_cifar100_bs_750 \
#       --gpus 4,5,7,8,9 \
#       --max_parallel 1
#
# Notes:
# - By default this script runs with WANDB_MODE=offline for each resume job.
#   You can override by exporting WANDB_MODE before invoking the script, e.g.:
#     WANDB_MODE=disabled ./run_exhaustive_resume_parallel.sh ...
# - CUDA_VISIBLE_DEVICES is used to pin each resume process to a single GPU.

usage() {
    echo "Usage: $0 --results_dir RESULTS_DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --results_dir RESULTS_DIR       Base directory containing _run_*.dir files"
    echo ""
    echo "Optional:"
    echo "  --gpus GPU_LIST                 Comma-separated GPU IDs (default: 0)"
    echo "  --combo_indices LIST            Optional comma-separated combo indices to resume."
    echo "                                  If omitted, all indices with _run_*.dir are resumed."
    echo "  --max_parallel N                Max jobs per GPU (default: 1)"
    echo "  --num_trial_workers N           Workers per combo splitting ckpt_trials (default: 1)"
    echo "  -h, --help                      Show this help"
    exit 1
}

RESULTS_DIR=""
GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
NUM_TRIAL_WORKERS="${NUM_TRIAL_WORKERS:-4}"
COMBO_INDICES_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir)  RESULTS_DIR="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --num_trial_workers) NUM_TRIAL_WORKERS="$2"; shift 2 ;;
        --combo_indices) COMBO_INDICES_OVERRIDE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$RESULTS_DIR" ]] && { echo "Error: --results_dir required"; usage; }

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: results_dir '$RESULTS_DIR' does not exist."
    exit 1
fi

# Discover candidate combos via _run_*.dir files
mapfile -t RUN_FILES < <(ls "$RESULTS_DIR"/_run_*.dir 2>/dev/null | sort)
if [[ ${#RUN_FILES[@]} -eq 0 ]]; then
    echo "No _run_*.dir files found under $RESULTS_DIR; nothing to resume."
    exit 0
fi

ALL_COMBO_INDICES=()
for f in "${RUN_FILES[@]}"; do
    # Extract the numeric part from _run_<idx>.dir
    base=$(basename "$f")
    idx=${base#_run_}
    idx=${idx%.dir}
    if [[ "$idx" =~ ^[0-9]+$ ]]; then
        ALL_COMBO_INDICES+=("$idx")
    fi
done

if [[ ${#ALL_COMBO_INDICES[@]} -eq 0 ]]; then
    echo "Found _run_*.dir files but could not parse any combo indices."
    exit 1
fi

# Optionally filter to a user-specified subset of combo indices
COMBO_INDICES=()
if [[ -n "$COMBO_INDICES_OVERRIDE" ]]; then
    IFS=',' read -ra REQ_COMBOS <<< "$COMBO_INDICES_OVERRIDE"
    declare -A allowed=()
    for c in "${ALL_COMBO_INDICES[@]}"; do
        allowed[$c]=1
    done
    for r in "${REQ_COMBOS[@]}"; do
        if [[ "$r" =~ ^[0-9]+$ ]]; then
            if [[ -n "${allowed[$r]}" ]]; then
                COMBO_INDICES+=("$r")
            else
                echo "Warning: requested combo_idx $r has no corresponding _run_$r.dir; skipping."
            fi
        else
            echo "Warning: ignoring invalid combo index '$r' in --combo_indices."
        fi
    done
else
    COMBO_INDICES=("${ALL_COMBO_INDICES[@]}")
fi

if [[ ${#COMBO_INDICES[@]} -eq 0 ]]; then
    echo "No valid combo indices to resume after applying filters."
    exit 0
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Resuming ${#COMBO_INDICES[@]} incomplete combos: ${COMBO_INDICES[*]}"
echo "Results dir: $RESULTS_DIR"
echo "GPUs: $GPUS, max_parallel: $MAX_PARALLEL, num_trial_workers: $NUM_TRIAL_WORKERS"
echo "WANDB_MODE: ${WANDB_MODE:-offline (default in this script>}"
echo "=========================================="

run_resume() {
    local combo_idx=$1
    local gpu_id=$2
    local worker_id=$3
    echo "Starting resume for combo $combo_idx worker $worker_id/$NUM_TRIAL_WORKERS on GPU $gpu_id"
    local mem_frac=$(python3 -c "print(0.9 / $MAX_PARALLEL)")
    XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_strict_conv_algorithm_picker=false" \
    JAX_DISABLE_MOST_FASTER_PATHS=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_frac \
    WANDB_MODE="${WANDB_MODE:-offline}" CUDA_VISIBLE_DEVICES=$gpu_id \
        python experiment_unlearning_exhaustive_resume.py \
        --results_dir "$RESULTS_DIR" \
        --combo_idx "$combo_idx" \
        --num_trial_workers "$NUM_TRIAL_WORKERS" \
        --trial_worker_id "$worker_id"
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Resume combo $combo_idx worker $worker_id completed"
    else
        echo "❌ Resume combo $combo_idx worker $worker_id failed (exit $exit_code)"
        run_dir=$(cat "$RESULTS_DIR/_run_$combo_idx.dir" 2>/dev/null)
        [[ -n "$run_dir" && -f "$run_dir/run_train.log" ]] && tail -20 "$run_dir/run_train.log"
    fi
}

declare -A gpu_jobs=()
declare -A gpu_pids=()
for g in "${GPU_ARRAY[@]}"; do gpu_jobs[$g]=0; gpu_pids[$g]=""; done

find_gpu() {
    local start=$1
    for o in $(seq 0 $((NUM_GPUS - 1))); do
        local i=$(((start + o) % NUM_GPUS))
        local g=${GPU_ARRAY[$i]}
        [[ ${gpu_jobs[$g]} -lt $MAX_PARALLEL ]] && { echo "$g"; return 0; }
    done
    echo ""; return 1
}

update_counts() {
    for g in "${GPU_ARRAY[@]}"; do
        local run=""
        local done=0
        for p in ${gpu_pids[$g]}; do
            kill -0 "$p" 2>/dev/null && run="$run $p" || ((done++))
        done
        gpu_pids[$g]=$(echo "$run" | xargs 2>/dev/null)
        ((done)) && gpu_jobs[$g]=$((${gpu_jobs[$g]} - done))
        [[ ${gpu_jobs[$g]} -lt 0 ]] && gpu_jobs[$g]=0
    done
}

last_gpu=0
for combo_idx in "${COMBO_INDICES[@]}"; do
    for worker_id in $(seq 0 $((NUM_TRIAL_WORKERS - 1))); do
        gpu_id=""
        while [[ -z "$gpu_id" ]]; do
            update_counts
            gpu_id=$(find_gpu $last_gpu)
            if [[ -z "$gpu_id" ]]; then
                wait -n
                update_counts
            else
                for idx in "${!GPU_ARRAY[@]}"; do
                    [[ "${GPU_ARRAY[$idx]}" == "$gpu_id" ]] && { last_gpu=$((idx + 1)); break; }
                done
                [[ ${gpu_jobs[$gpu_id]} -ge $MAX_PARALLEL ]] && gpu_id="" && continue
                gpu_jobs[$gpu_id]=$((${gpu_jobs[$gpu_id]} + 1))
            fi
        done
        run_resume "$combo_idx" "$gpu_id" "$worker_id" &
        pid=$!
        [[ -z "${gpu_pids[$gpu_id]}" ]] && gpu_pids[$gpu_id]=$pid || gpu_pids[$gpu_id]="${gpu_pids[$gpu_id]} $pid"
    done
done

wait
echo ""
echo "🎉 All resume jobs submitted by this script have completed."
echo "Results: $RESULTS_DIR"
echo "Timestamp: $(date)"
