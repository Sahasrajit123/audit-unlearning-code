#!/bin/bash
# Run exhaustive forget-combination experiments in parallel.
#
# For small n forget batches, iterates over all C(n,k) combinations. Each combo_idx
# is a combo index: 1 train + num_unlearn_per_combo trials (ckpt_trial_0, ...).
#
# total_runs defaults to num_combos (from --print_num_combos). Use --start_index and
# --total_runs to run a subset (e.g. for testing).
#
# Example:
#   ./run_exhaustive_combinations_parallel.sh --config configs/exp_cifar.yaml \
#       --results_dir logs_exhaustive_cifar_batch_6
#
#   # Get num_combos only:
#   python experiment_unlearning_exhaustive_combinations.py --print_num_combos \
#       --results_dir logs_exhaustive_cifar_batch_6 --forget_fraction 0.5 --file_batch_size 1

usage() {
    echo "Usage: $0 --config CONFIG_FILE --results_dir RESULTS_DIR --data_subfolder NAME [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --config CONFIG_FILE          Path to config YAML"
    echo "  --results_dir RESULTS_DIR     Base directory (must have data_split/ prepared)"
    echo "  --data_subfolder NAME         Data subfolder under data_split/ (required; e.g., cifar_750)"
    echo ""
    echo "Optional:"
    echo "  --start_index N               Start combo index (default: 0)"
    echo "  --total_runs N                Number of combos to run (default: num_combos from data)"
    echo "  --forget_fraction F           Forget fraction (default: 0.5)"
    echo "  --file_batch_size N           File batch size (default: 1)"
    echo "  --num_unlearn_per_combo N     Unlearn trials per combo (default: 5)"
    echo "  --no_shuffle_train_samples    Disable sample-level shuffle (enabled by default)"
    echo "  --shuffle_train_batches_each_epoch  Shuffle batch order each epoch (different order every epoch)"
    echo "  --unlearn_seed_offset N       Base for unlearn seeds (default: 10000)"
    echo "  --max_parallel N              Max jobs per GPU (default: 2)"
    echo "  --gpus GPU_LIST               Comma-separated GPU IDs (default: 8,9)"
    echo "  --deterministic               Use deterministic GPU ops"
    echo "  -h, --help                    Show this help"
    exit 1
}

CONFIG_FILE=""
RESULTS_DIR=""
START_INDEX="${START_INDEX:-0}"
TOTAL_RUNS=""   # empty => compute from --print_num_combos
FORGET_FRACTION="${FORGET_FRACTION:-0.5}"
DATA_SUBFOLDER=""
NUM_UNLEARN_PER_COMBO="${NUM_UNLEARN_PER_COMBO:-35}"
UNLEARN_SEED_OFFSET="${UNLEARN_SEED_OFFSET:-10000}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
GPUS="${GPUS:-2,7}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SHUFFLE_TRAIN_SAMPLES="${SHUFFLE_TRAIN_SAMPLES:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG_FILE="$2"; shift 2 ;;
        --results_dir)  RESULTS_DIR="$2"; shift 2 ;;
        --start_index)  START_INDEX="$2"; shift 2 ;;
        --total_runs)   TOTAL_RUNS="$2"; shift 2 ;;
        --forget_fraction) FORGET_FRACTION="$2"; shift 2 ;;
        --data_subfolder)  DATA_SUBFOLDER="$2"; shift 2 ;;
        --num_unlearn_per_combo) NUM_UNLEARN_PER_COMBO="$2"; shift 2 ;;
        --unlearn_seed_offset)   UNLEARN_SEED_OFFSET="$2"; shift 2 ;;
        --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --deterministic) DETERMINISTIC=1; shift ;;
        --no_shuffle_train_samples) SHUFFLE_TRAIN_SAMPLES=0; shift ;;
        --shuffle_train_batches_each_epoch) SHUFFLE_TRAIN_BATCHES_EACH_EPOCH=1; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$CONFIG_FILE" || -z "$RESULTS_DIR" || -z "$DATA_SUBFOLDER" ]] && {
    echo "Error: --config, --results_dir, and --data_subfolder are required"
    usage
}

# Resolve total_runs = num_combos if not provided
if [[ -z "$TOTAL_RUNS" ]]; then
    echo "[DEBUG] Python used for TOTAL_RUNS: $(which python)"
    python -c 'import sys; print("[DEBUG] sys.executable:", sys.executable)' 
    TOTAL_RUNS=$(python experiment_unlearning_exhaustive_combinations.py --print_num_combos \
        --results_dir "$RESULTS_DIR" \
        --forget_fraction "$FORGET_FRACTION" \
        --data_subfolder "$DATA_SUBFOLDER" \
        --config "$CONFIG_FILE" 2>/dev/null)

    if [[ -z "$TOTAL_RUNS" || ! "$TOTAL_RUNS" =~ ^[0-9]+$ ]]; then
        echo "Error: could not get num_combos. Ensure data_split/<data_subfolder>/forget/ exists."
        exit 1
    fi
fi

END_INDEX=$((START_INDEX + TOTAL_RUNS - 1))
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Exhaustive combinations: $TOTAL_RUNS combos (combo_idx $START_INDEX to $END_INDEX)"
echo "Config: $CONFIG_FILE"
echo "Results dir: $RESULTS_DIR"
echo "Forget fraction: $FORGET_FRACTION, file_batch_size: $FILE_BATCH_SIZE, data_subfolder: $DATA_SUBFOLDER"
echo "num_unlearn_per_combo: $NUM_UNLEARN_PER_COMBO"
echo "GPUs: $GPUS, max_parallel: $MAX_PARALLEL"
echo "=========================================="

run_experiment() {
    local combo_idx=$1
    local gpu_id=$2
    echo "Starting combo $combo_idx on GPU $gpu_id"
    SHUFFLE_ARGS=""
    [[ "$SHUFFLE_TRAIN_SAMPLES" != "0" ]] && SHUFFLE_ARGS="$SHUFFLE_ARGS --shuffle_train_samples"
    [[ -n "$SHUFFLE_TRAIN_BATCHES_EACH_EPOCH" && "$SHUFFLE_TRAIN_BATCHES_EACH_EPOCH" != "0" ]] && SHUFFLE_ARGS="$SHUFFLE_ARGS --shuffle_train_batches_each_epoch"
    local mem_frac=$(python3 -c "print(0.9 / $MAX_PARALLEL)")
    if [[ -n "$DETERMINISTIC" && "$DETERMINISTIC" != "0" ]]; then
        XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_strict_conv_algorithm_picker=false" \
        JAX_DISABLE_MOST_FASTER_PATHS=1 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_frac \
        WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python experiment_unlearning_exhaustive_combinations.py \
            --config "$CONFIG_FILE" \
            --results_dir "$RESULTS_DIR" \
            --combo_idx "$combo_idx" \
            --forget_fraction "$FORGET_FRACTION" \
            --data_subfolder "$DATA_SUBFOLDER" \
            --num_unlearn_per_combo "$NUM_UNLEARN_PER_COMBO" \
            --unlearn_seed_offset "$UNLEARN_SEED_OFFSET" \
            $SHUFFLE_ARGS
    else
        XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_frac \
        WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python experiment_unlearning_exhaustive_combinations.py \
            --config "$CONFIG_FILE" \
            --results_dir "$RESULTS_DIR" \
            --combo_idx "$combo_idx" \
            --forget_fraction "$FORGET_FRACTION" \
            --data_subfolder "$DATA_SUBFOLDER" \
            --num_unlearn_per_combo "$NUM_UNLEARN_PER_COMBO" \
            --unlearn_seed_offset "$UNLEARN_SEED_OFFSET" \
            $SHUFFLE_ARGS
    fi
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Combo $combo_idx completed"
    else
        echo "❌ Combo $combo_idx failed (exit $exit_code)"
        run_dir=$(cat "$RESULTS_DIR/_run_$combo_idx.dir" 2>/dev/null)
        [[ -n "$run_dir" && -f "$run_dir/run.log" ]] && tail -10 "$run_dir/run.log"
        [[ -f "$RESULTS_DIR/_run_${combo_idx}_pre.log" ]] && tail -10 "$RESULTS_DIR/_run_${combo_idx}_pre.log"
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
        gpu_pids[$g]=$(echo "$run" | xargs)
        ((done)) && gpu_jobs[$g]=$((${gpu_jobs[$g]} - done))
        [[ ${gpu_jobs[$g]} -lt 0 ]] && gpu_jobs[$g]=0
    done
}

last_gpu=0
for i in $(seq $START_INDEX $END_INDEX); do
    gpu_id=""
    while [[ -z "$gpu_id" ]]; do
        update_counts
        gpu_id=$(find_gpu $last_gpu)
        if [[ -z "$gpu_id" ]]; then wait -n; update_counts
        else
            for idx in "${!GPU_ARRAY[@]}"; do
                [[ "${GPU_ARRAY[$idx]}" == "$gpu_id" ]] && { last_gpu=$((idx + 1)); break; }
            done
            [[ ${gpu_jobs[$gpu_id]} -ge $MAX_PARALLEL ]] && gpu_id="" && continue
            gpu_jobs[$gpu_id]=$((${gpu_jobs[$gpu_id]} + 1))
        fi
    done
    run_experiment $i $gpu_id &
    pid=$!
    [[ -z "${gpu_pids[$gpu_id]}" ]] && gpu_pids[$gpu_id]=$pid || gpu_pids[$gpu_id]="${gpu_pids[$gpu_id]} $pid"
done

wait
echo ""
echo "🎉 All $TOTAL_RUNS exhaustive combo runs completed (indices $START_INDEX–$END_INDEX)"
echo "Results: $RESULTS_DIR"
echo "Timestamp: $(date)"
