#!/usr/bin/env python3
"""
run_orchestrator.py - GPU-aware orchestrator for multi-run experiments

Dynamically assigns runs to GPUs based on availability.
Spawns worker processes to execute single runs in parallel.

Usage:
  python run_orchestrator.py --experiment_config config.json --gpus 0,1,2 --max_runs_per_gpu 3
  python run_orchestrator.py --experiment_config config.json --gpus "1 3 5"
  python run_orchestrator.py --experiment_config config.json  # defaults to GPU 0, max 2 runs/GPU
  python run_orchestrator.py --experiment_config config.json --num_runs 100 --start_index 50
"""

import argparse
import json
import subprocess
import time
import sys
from pathlib import Path
from collections import defaultdict
import threading
import queue
import shutil


def load_experiment_config(config_path):
    """Load experiment config from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


class GPUOrchestrator:
    """Manages GPU assignment and run execution."""

    def __init__(self, num_gpus, max_runs_per_gpu):
        """Initialize orchestrator."""
        self.num_gpus = num_gpus
        self.max_runs_per_gpu = max_runs_per_gpu
        self.gpu_jobs = defaultdict(int)
        self.lock = threading.Lock()

    def get_available_gpu(self):
        """Get GPU with least active jobs."""
        with self.lock:
            gpu_id = min(range(self.num_gpus), key=lambda g: self.gpu_jobs[g])
            current = self.gpu_jobs[gpu_id]

            if current < self.max_runs_per_gpu:
                self.gpu_jobs[gpu_id] += 1
                return gpu_id
            else:
                return None

    def release_gpu(self, gpu_id):
        """Release a GPU slot."""
        with self.lock:
            self.gpu_jobs[gpu_id] = max(0, self.gpu_jobs[gpu_id] - 1)

    def get_status(self):
        """Get orchestrator status."""
        with self.lock:
            return dict(self.gpu_jobs)


def wait_for_gpu_slot(orchestrator):
    """Wait for an available GPU slot (no timeout — waits indefinitely)."""
    while True:
        gpu_id = orchestrator.get_available_gpu()
        if gpu_id is not None:
            return gpu_id
        time.sleep(1)


def run_on_gpu(run_id, experiment_config, gpu_id, orchestrator, results_queue, run_folder, gpu_list):
    """Execute a single run on assigned GPU."""
    gpu_idx = None
    try:
        print("[RUN {}] LAUNCHED on GPU {}".format(run_id, gpu_id))

        # Map actual GPU ID back to index for orchestrator
        gpu_idx = gpu_list.index(gpu_id)

        # Setup log file for this run
        run_dir = Path(run_folder) / "run_{}".format(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "orchestrator.log"

        cmd = [
            "python", "single_run_unlearning.py",
            "--run_id", str(run_id),
            "--experiment_config", experiment_config,
            "--gpu", str(gpu_id),
        ]

        # Redirect output to run's log file
        with open(log_file, "w") as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=None)

        # Release GPU slot immediately after subprocess completes
        if gpu_idx is not None:
            orchestrator.release_gpu(gpu_idx)

        if result.returncode == 0:
            print("[RUN {}] FINISHED SUCCESS".format(run_id))
            results_queue.put((run_id, "success", None))
        else:
            print("[RUN {}] FINISHED FAILED (exit code: {})".format(run_id, result.returncode))
            results_queue.put((run_id, "failed", result.returncode))

    except subprocess.TimeoutExpired:
        print("[RUN {}] FINISHED TIMEOUT".format(run_id))
        if gpu_idx is not None:
            orchestrator.release_gpu(gpu_idx)
        results_queue.put((run_id, "timeout", None))
    except Exception as e:
        print("[RUN {}] FINISHED ERROR: {}".format(run_id, e))
        if gpu_idx is not None:
            orchestrator.release_gpu(gpu_idx)
        results_queue.put((run_id, "error", str(e)))


def main():
    parser = argparse.ArgumentParser(
        description="GPU-aware orchestrator for multi-run unlearning experiments"
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=True,
        help="Path to experiment config JSON file"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use (comma or space separated, e.g., '0,1,2' or '0 1 2')"
    )
    parser.add_argument(
        "--max_runs_per_gpu",
        type=int,
        default=2,
        help="Maximum concurrent runs per GPU"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=None,
        help="Override number of runs from config (if not specified, uses config value)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start run index (default: 0)"
    )

    args = parser.parse_args()

    # Parse GPU list
    gpu_str = args.gpus.replace(",", " ")
    gpu_list = [int(g.strip()) for g in gpu_str.split() if g.strip()]
    if not gpu_list:
        gpu_list = [0]
    num_gpus = len(gpu_list)

    # Load config
    print("\nLoading experiment config: {}".format(args.experiment_config))
    exp_config = load_experiment_config(args.experiment_config)

    exp_cfg = exp_config.get("experiment", {})
    num_runs = args.num_runs if args.num_runs is not None else exp_cfg.get("num_runs", 50)
    start_index = args.start_index
    run_folder = Path(exp_cfg.get("run_folder", "runs_unlearning"))
    strategy = exp_cfg.get("strategy", "finetune_retain")

    print("\n" + "="*80)
    print("GPU-AWARE ORCHESTRATOR")
    print("="*80 + "\n")
    print("Config: {}".format(args.experiment_config))
    print("Strategy: {}".format(strategy))
    print("Total runs: {}".format(num_runs))
    print("Start index: {}".format(start_index))
    print("Available GPUs: {}".format(gpu_list))
    print("Max runs per GPU: {}".format(args.max_runs_per_gpu))
    print("Max concurrent runs: {}".format(num_gpus * args.max_runs_per_gpu))
    print("Output folder: {}\n".format(run_folder))

    # Create orchestrator (use num_gpus, but map to actual GPU IDs later)
    orchestrator = GPUOrchestrator(num_gpus, args.max_runs_per_gpu)

    # Copy experiment config to run folder for debugging
    run_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.experiment_config, str(run_folder / "experiment_config.json"))

    # Queue for results
    results_queue = queue.Queue()

    # Track threads
    threads = {}
    completed = 0
    failed = 0
    start_time = time.time()

    print("="*80)
    print("Launching runs...")
    print("="*80 + "\n")

    # Launch runs
    for run_id in range(start_index, start_index + num_runs):
        # Wait for available GPU slot
        gpu_idx = wait_for_gpu_slot(orchestrator)

        # Map virtual index to actual GPU ID
        actual_gpu_id = gpu_list[gpu_idx]

        # Launch run in thread
        thread = threading.Thread(
            target=run_on_gpu,
            args=(run_id, args.experiment_config, actual_gpu_id, orchestrator, results_queue, run_folder, gpu_list),
        )
        thread.daemon = True
        thread.start()
        threads[run_id] = thread

        # Show status every 10 runs
        run_count = run_id - start_index + 1
        if run_count % 10 == 0:
            status = orchestrator.get_status()
            gpu_status_str = ", ".join(["GPU{}: {} jobs".format(gpu_list[idx], count) for idx, count in sorted(status.items())])
            print("[Status] Launched {}/{} runs | {}".format(run_count, num_runs, gpu_status_str))

    print("\nAll runs launched. Waiting for completion...\n")

    # Collect results
    results_collected = 0
    while results_collected < num_runs:
        try:
            run_id, status_code, error = results_queue.get(timeout=60)
            results_collected += 1

            if status_code == "success":
                completed += 1
            else:
                failed += 1

            # Show progress every 5 completions
            if results_collected % 5 == 0 or results_collected == num_runs:
                elapsed = time.time() - start_time
                print("[Progress] {}/{} completed | Succeeded: {}, Failed: {} | Elapsed: {:.1f}s".format(
                    results_collected, num_runs, completed, failed, elapsed))

        except queue.Empty:
            active_threads = sum(1 for t in threads.values() if t.is_alive())
            elapsed = time.time() - start_time
            print("[Waiting] {}/{} results | Active threads: {} | Elapsed: {:.1f}s".format(
                results_collected, num_runs, active_threads, elapsed))

    # Wait for all threads to complete
    for thread in threads.values():
        thread.join(timeout=10)

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "="*80)
    print("ORCHESTRATOR FINAL REPORT")
    print("="*80)
    print("Total runs requested: {}".format(num_runs))
    print("Completed successfully: {}".format(completed))
    print("Failed: {}".format(failed))
    print("Total time: {:.1f}s ({:.2f}h)".format(elapsed, elapsed/3600.0))
    print("\nResults folder: {}".format(run_folder))
    print("Run logs: runs_*/run_N/orchestrator.log")
    print("Metrics: runs_*/run_N/metrics.json")
    print("\nTo view results:")
    print("  python scripts/analyze_multi_run_results.py --run_folder {}".format(run_folder))
    print("="*80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
