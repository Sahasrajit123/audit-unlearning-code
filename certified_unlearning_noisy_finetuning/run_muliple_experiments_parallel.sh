#!/bin/bash
# Script to run experiments in parallel (if multiple GPUs available)
#
# Examples:
#
#   # Default: runs 1-80 (start_index=1, total_runs=60)
#   ./run_50_experiments_parallel.sh --config configs/exp_cifar.yaml --results_dir logs/
#
#   # Runs 21-70 (start_index=21, total_runs=50)
#   ./run_50_experiments_parallel.sh --config configs/exp_cifar.yaml --results_dir logs/ \
#       --start_index 21 --total_runs 50
#
#   # Via env: runs 81-130
#   START_INDEX=81 TOTAL_RUNS=50 ./run_50_experiments_parallel.sh --config configs/exp_cifar.yaml --results_dir logs/
#
#   # With custom GPUs and total_runs
#   ./run_50_experiments_parallel.sh --config configs/exp_cifar.yaml --results_dir logs/ \
#       --start_index 1 --total_runs 20 --gpus 2,3,4,5

# Usage function
usage() {
    echo "Usage: $0 --config CONFIG_FILE --results_dir RESULTS_DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --config CONFIG_FILE          Path to config YAML file"
    echo "  --results_dir RESULTS_DIR     Directory to save results"
    echo ""
    echo "Optional:"
    echo "  --start_index N               Start run index (default: 1). Runs from start_index to start_index+total_runs-1."
    echo "  --total_runs N               Number of experiments to run (default: 80)"
    echo "  --forget_fraction F           Forget fraction (default: 0.5)"
    echo "  --data_subfolder NAME         Data subfolder under data_split/ (default: auto from config)"
    echo "  --no_shuffle_train_samples    Disable sample-level train shuffle (default: enabled)"
    echo "  --shuffle_train_batches       Enable batch-order shuffle before rebatching (default: disabled)"
    echo "  --shuffle_train_batches_each_epoch"
    echo "                                Enable per-epoch train batch order shuffle (default: disabled)"
    echo "  --max_parallel N              Max operations per GPU (default: 2)"
    echo "  --gpus GPU_LIST               Comma-separated GPU IDs (default: 8,9)"
    echo "  --deterministic               Use deterministic GPU ops (XLA flags). Slower; for reproducibility across GPUs."
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Environment variables (can override defaults):"
    echo "  GPUS                          Comma-separated GPU IDs"
    echo "  START_INDEX                   Start run index"
    echo "  TOTAL_RUNS                    Number of experiments"
    echo "  FORGET_FRACTION               Forget fraction"
    echo "  DATA_SUBFOLDER                Data subfolder (default: auto from config)"
    echo "  SHUFFLE_TRAIN_SAMPLES         1 to enable sample-level train shuffle (default: 1)"
    echo "  SHUFFLE_TRAIN_BATCHES         1 to enable batch-order shuffle before rebatching (default: 0)"
    echo "  SHUFFLE_TRAIN_BATCHES_EACH_EPOCH"
    echo "                                1 to enable per-epoch batch shuffle (default: 0)"
    echo "  MAX_PARALLEL                  Max operations per GPU"
    echo "  DETERMINISTIC                 1 to enable deterministic GPU (same effect as --deterministic)"
    echo ""
    echo "Note: For a given run_index, training seeds and data are identical regardless of GPU."
    echo "      GPU execution can still differ across GPUs (float non-determinism). Use --deterministic to reduce this."
    exit 1
}

# Parse command-line arguments
CONFIG_FILE=""
RESULTS_DIR=""
START_INDEX="${START_INDEX:-1}"
TOTAL_RUNS="${TOTAL_RUNS:-60}"
FORGET_FRACTION="${FORGET_FRACTION:-0.5}"
# DATA_SUBFOLDER: optional; when unset, experiment script resolves default from config
DATA_SUBFOLDER="${DATA_SUBFOLDER:-}"
SHUFFLE_TRAIN_SAMPLES="${SHUFFLE_TRAIN_SAMPLES:-1}"
SHUFFLE_TRAIN_BATCHES="${SHUFFLE_TRAIN_BATCHES:-0}"
SHUFFLE_TRAIN_BATCHES_EACH_EPOCH="${SHUFFLE_TRAIN_BATCHES_EACH_EPOCH:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"  # Maximum number of operations per GPU
GPUS="${GPUS:-3,4}"  # Comma-separated list of GPU IDs (default: 8,9)
DETERMINISTIC="${DETERMINISTIC:-1}"  # If 1 or --deterministic: set XLA_FLAGS etc. for reproducible GPU

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --start_index)
            START_INDEX="$2"
            shift 2
            ;;
        --total_runs)
            TOTAL_RUNS="$2"
            shift 2
            ;;
        --forget_fraction)
            FORGET_FRACTION="$2"
            shift 2
            ;;
        --data_subfolder)
            DATA_SUBFOLDER="$2"
            shift 2
            ;;
        --no_shuffle_train_samples)
            SHUFFLE_TRAIN_SAMPLES=0
            shift
            ;;
        --shuffle_train_batches)
            SHUFFLE_TRAIN_BATCHES=1
            shift
            ;;
        --shuffle_train_batches_each_epoch)
            SHUFFLE_TRAIN_BATCHES_EACH_EPOCH=1
            shift
            ;;
        --max_parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --deterministic)
            DETERMINISTIC=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG_FILE" ] || [ -z "$RESULTS_DIR" ]; then
    echo "Error: --config and --results_dir are required"
    echo ""
    usage
fi

EFFECTIVE_DATA_SUBFOLDER="$DATA_SUBFOLDER"

# Parse GPU list into array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

END_INDEX=$((START_INDEX + TOTAL_RUNS - 1))
echo "Starting $TOTAL_RUNS experiments in parallel (run indices $START_INDEX to $END_INDEX)"
echo "Config: $CONFIG_FILE"
echo "Results dir: $RESULTS_DIR"
echo "Start index: $START_INDEX, total runs: $TOTAL_RUNS (indices $START_INDEX-$END_INDEX)"
echo "Forget fraction: $FORGET_FRACTION"
if [ -n "$DATA_SUBFOLDER" ]; then
    echo "Data subfolder: $DATA_SUBFOLDER (manual override)"
else
    echo "Data subfolder: (auto from config) -> $EFFECTIVE_DATA_SUBFOLDER"
fi
echo "Shuffle train samples: $SHUFFLE_TRAIN_SAMPLES"
echo "Shuffle train batches: $SHUFFLE_TRAIN_BATCHES"
echo "Shuffle train batches each epoch: $SHUFFLE_TRAIN_BATCHES_EACH_EPOCH"
echo "GPUs: $GPUS (${NUM_GPUS} GPU(s))"
echo "Max parallel per GPU: $MAX_PARALLEL"
echo "Total max parallel: $((NUM_GPUS * MAX_PARALLEL))"
echo "Deterministic GPU: $DETERMINISTIC"
echo "Logs will be saved in each run's directory: <results_dir>/<run_id>/run.log"
echo "=========================================="

# Function to run a single experiment
run_experiment() {
    local run_index=$1
    local gpu_id=$2
    local data_subfolder_args=()
    local shuffle_args=()

    if [ -n "$DATA_SUBFOLDER" ]; then
        data_subfolder_args=(--data_subfolder "$DATA_SUBFOLDER")
    fi
    if [ "$SHUFFLE_TRAIN_BATCHES" = "1" ]; then
        shuffle_args+=(--shuffle_train_batches)
    elif [ "$SHUFFLE_TRAIN_SAMPLES" = "1" ]; then
        shuffle_args+=(--shuffle_train_samples)
    fi
    if [ "$SHUFFLE_TRAIN_BATCHES_EACH_EPOCH" = "1" ]; then
        shuffle_args+=(--shuffle_train_batches_each_epoch)
    fi
    
    echo "Starting Experiment $run_index on GPU $gpu_id"
    
    # Use WANDB_MODE=offline to avoid authentication conflicts in parallel runs.
    # Optional: when DETERMINISTIC=1, set XLA/JAX env so GPU ops are deterministic across devices
    # (reduces float non-determinism; can be 10–30% slower).
    if [ -n "$DETERMINISTIC" ] && [ "$DETERMINISTIC" != "0" ]; then
        XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_strict_conv_algorithm_picker=false" \
        JAX_DISABLE_MOST_FASTER_PATHS=1 \
        WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python experiment_unlearning_random_forget_main.py \
            --config "$CONFIG_FILE" \
            --results_dir "$RESULTS_DIR" \
            --run_index "$run_index" \
            --forget_fraction "$FORGET_FRACTION" \
            "${data_subfolder_args[@]}" \
            "${shuffle_args[@]}"
    else
        WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu_id python experiment_unlearning_random_forget_main.py \
            --config "$CONFIG_FILE" \
            --results_dir "$RESULTS_DIR" \
            --run_index "$run_index" \
            --forget_fraction "$FORGET_FRACTION" \
            "${data_subfolder_args[@]}" \
            "${shuffle_args[@]}"
    fi

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Experiment $run_index completed successfully"
    else
        echo "❌ Experiment $run_index failed with exit code $exit_code"
        echo "Last 10 lines of run log:"
        run_dir=$(cat "$RESULTS_DIR/_run_$run_index.dir" 2>/dev/null)
        if [ -n "$run_dir" ] && [ -f "$run_dir/run.log" ]; then
            tail -10 "$run_dir/run.log"
        elif [ -f "$RESULTS_DIR/_run_${run_index}_pre.log" ]; then
            tail -10 "$RESULTS_DIR/_run_${run_index}_pre.log"
        else
            echo "(Run log not found; run may have failed before any log was created.)"
        fi
        echo "----------------------------------------"
    fi
}

# Run experiments in parallel with proper limiting per GPU
# Track running jobs per GPU using associative array
declare -A gpu_jobs=()
declare -A gpu_pids=()

# Initialize job counts for each GPU
for gpu in "${GPU_ARRAY[@]}"; do
    gpu_jobs[$gpu]=0
    gpu_pids[$gpu]=""
done

# Function to find a GPU with available capacity (round-robin for better distribution)
find_available_gpu() {
    local start_idx=$1
    local num_gpus=${#GPU_ARRAY[@]}
    
    # Try GPUs starting from start_idx (round-robin)
    for offset in $(seq 0 $((num_gpus - 1))); do
        local idx=$(((start_idx + offset) % num_gpus))
        local gpu=${GPU_ARRAY[$idx]}
        if [ ${gpu_jobs[$gpu]} -lt $MAX_PARALLEL ]; then
            echo "$gpu"
            return 0
        fi
    done
    echo ""
    return 1
}

# Function to update job counts by removing finished jobs
# This only removes finished PIDs and decrements counts, preserving manual increments
update_job_counts() {
    for gpu in "${GPU_ARRAY[@]}"; do
        local running_pids=""
        local finished_count=0
        
        for pid in ${gpu_pids[$gpu]}; do
            if kill -0 "$pid" 2>/dev/null; then
                # PID is still running
                running_pids="$running_pids $pid"
            else
                # PID has finished
                finished_count=$((finished_count + 1))
            fi
        done
        
        # Update PIDs list and decrement job count for finished jobs
        gpu_pids[$gpu]=$(echo "$running_pids" | xargs)
        if [ $finished_count -gt 0 ]; then
            gpu_jobs[$gpu]=$((gpu_jobs[$gpu] - finished_count))
            # Ensure count doesn't go negative
            if [ ${gpu_jobs[$gpu]} -lt 0 ]; then
                gpu_jobs[$gpu]=0
            fi
        fi
    done
}

# Track last GPU index used for round-robin
last_gpu_idx=0

for i in $(seq $START_INDEX $END_INDEX); do
    # Find an available GPU
    gpu_id=""
    while [ -z "$gpu_id" ]; do
        # Update job counts to reflect actual running jobs
        update_job_counts
        
        # Use round-robin to find available GPU
        gpu_id=$(find_available_gpu $last_gpu_idx)
        if [ -z "$gpu_id" ]; then
            # All GPUs are at capacity, wait for any job to complete
            wait -n
            # Update counts after a job completed
            update_job_counts
        else
            # Update last_gpu_idx to the GPU we found (for next round-robin)
            for idx in "${!GPU_ARRAY[@]}"; do
                if [ "${GPU_ARRAY[$idx]}" = "$gpu_id" ]; then
                    last_gpu_idx=$((idx + 1))
                    break
                fi
            done
            
            # Double-check this GPU still has capacity (race condition protection)
            if [ ${gpu_jobs[$gpu_id]} -ge $MAX_PARALLEL ]; then
                gpu_id=""  # Force retry
                continue
            fi
            
            # Increment job count BEFORE starting the job to reserve the slot
            gpu_jobs[$gpu_id]=$((gpu_jobs[$gpu_id] + 1))
        fi
    done
    
    # Run experiment in background
    run_experiment $i $gpu_id &
    pid=$!  # Remove 'local' - we're not in a function
    
    # Track this job's PID for the GPU
    if [ -z "${gpu_pids[$gpu_id]}" ]; then
        gpu_pids[$gpu_id]="$pid"
    else
        gpu_pids[$gpu_id]="${gpu_pids[$gpu_id]} $pid"
    fi
done

# Wait for all remaining processes to complete
wait

echo ""
echo "🎉 All $TOTAL_RUNS experiments completed (indices $START_INDEX-$END_INDEX)!"
echo "Results saved in: $RESULTS_DIR"
echo "Logs saved in each run folder: <run_id>/run.log"
echo "Final timestamp: $(date)"
