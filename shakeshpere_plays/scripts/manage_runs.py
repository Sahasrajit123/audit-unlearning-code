#!/usr/bin/env python3
"""
manage_runs.py - Utility to manage and inspect multi-run experiments

Usage:
  python manage_runs.py --run_folder runs_finetune --action list
  python manage_runs.py --run_folder runs_finetune --action inspect --run_number 0
  python manage_runs.py --run_folder runs_finetune --action calculate_stats
  python manage_runs.py --run_folder runs_finetune --action compare_forget_sampling
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def list_runs(run_folder):
    """List all runs in the folder."""
    run_folder = Path(run_folder)
    runs = sorted(run_folder.glob("run_*"))

    print(f"\nFound {len(runs)} runs in {run_folder}\n")

    if not runs:
        print("No runs found.")
        return

    # Print experiment config if available
    exp_config_path = run_folder / "experiment_config.json"
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_config = json.load(f)
        print(f"Experiment config:")
        print(f"  Strategy: {exp_config.get('strategy')}")
        print(f"  Forget prob: {exp_config.get('forget_prob')}")
        print(f"  Num forget sets: {exp_config.get('num_forget_sets')}")
        print()

    # List runs with status
    print(f"{'Run':<8} {'Status':<15} {'Test Acc':<12} {'Test Loss':<12} {'Forget Sets':<20}")
    print("-" * 70)

    for run_dir in runs:
        run_num = int(run_dir.name.split("_")[1])
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            status = "✓ Complete"
            test_acc = f"{metrics.get('test_accuracy', 0):.4f}"
            test_loss = f"{metrics.get('test_loss', 0):.4f}"
            forget_sets = str(metrics.get('forget_indices', [])).replace(", ", ",")[:20]
        else:
            status = "○ Incomplete"
            test_acc = "-"
            test_loss = "-"
            forget_sets = "-"

        print(f"{run_num:<8} {status:<15} {test_acc:<12} {test_loss:<12} {forget_sets:<20}")


def inspect_run(run_folder, run_number):
    """Inspect a specific run."""
    run_folder = Path(run_folder)
    run_dir = run_folder / f"run_{run_number}"

    if not run_dir.exists():
        print(f"Run {run_number} not found in {run_folder}")
        return

    print(f"\n{'='*80}")
    print(f"RUN {run_number} DETAILS")
    print(f"{'='*80}\n")

    # Config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("Configuration:")
        print(f"  Strategy: {config.get('strategy')}")
        print(f"  Forget prob: {config.get('forget_prob')}")
        print(f"  Sampled forget indices: {config.get('sampled_forget_indices')}")
        print(f"  Training seed: {config.get('training', {}).get('seed')}")
        print(f"  Epochs: {config.get('training', {}).get('epochs')}")
        print(f"  Batch size: {config.get('training', {}).get('batch_size')}")
        print(f"  Learning rate: {config.get('training', {}).get('lr')}")
        print()

    # Metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("Results:")
        print(f"  Test loss: {metrics.get('test_loss', 'N/A'):.4f}")
        print(f"  Test accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Test perplexity: {metrics.get('test_perplexity', 'N/A'):.2f}")
        print()

        # Strategy-specific metrics
        if metrics.get('strategy') == 'ascent_descent':
            print("Ascent-Descent specific:")
            print(f"  Epochs: {metrics.get('epochs')}")
            print(f"  Q (retain steps): {metrics.get('q')}")
            print(f"  Lambda coef: {metrics.get('lambda_coef')}")
            print(f"  Forget epochs: {metrics.get('forget_epochs')}")
            print(f"  Elapsed: {metrics.get('elapsed_seconds', 'N/A'):.1f}s")
            print()

    # Files
    print(f"Files in run directory:")
    for fpath in sorted(run_dir.iterdir()):
        if fpath.is_file():
            size = fpath.stat().st_size
            if size > 1e6:
                size_str = f"{size/1e6:.1f}MB"
            elif size > 1e3:
                size_str = f"{size/1e3:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"  {fpath.name:<30} {size_str:>10}")
        elif fpath.is_dir():
            print(f"  {fpath.name}/ (directory)")

    print()


def calculate_stats(run_folder):
    """Calculate aggregate statistics."""
    run_folder = Path(run_folder)
    runs = sorted(run_folder.glob("run_*"))

    metrics_list = []
    for run_dir in runs:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics_list.append(metrics)

    if not metrics_list:
        print("No completed runs found.")
        return

    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}\n")

    test_accs = [m.get("test_accuracy", np.nan) for m in metrics_list]
    test_losses = [m.get("test_loss", np.nan) for m in metrics_list]
    test_ppls = [m.get("test_perplexity", np.nan) for m in metrics_list]

    print(f"Total completed runs: {len(metrics_list)}\n")

    print("Test Accuracy:")
    print(f"  Mean:     {np.nanmean(test_accs):.6f}")
    print(f"  Std:      {np.nanstd(test_accs):.6f}")
    print(f"  Min:      {np.nanmin(test_accs):.6f}")
    print(f"  Max:      {np.nanmax(test_accs):.6f}")
    print(f"  Median:   {np.nanmedian(test_accs):.6f}")
    print(f"  Q1:       {np.nanpercentile(test_accs, 25):.6f}")
    print(f"  Q3:       {np.nanpercentile(test_accs, 75):.6f}")

    print("\nTest Loss:")
    print(f"  Mean:     {np.nanmean(test_losses):.6f}")
    print(f"  Std:      {np.nanstd(test_losses):.6f}")
    print(f"  Min:      {np.nanmin(test_losses):.6f}")
    print(f"  Max:      {np.nanmax(test_losses):.6f}")

    print("\nTest Perplexity:")
    print(f"  Mean:     {np.nanmean(test_ppls):.2f}")
    print(f"  Std:      {np.nanstd(test_ppls):.2f}")
    print(f"  Min:      {np.nanmin(test_ppls):.2f}")
    print(f"  Max:      {np.nanmax(test_ppls):.2f}")

    print()


def compare_forget_sampling(run_folder):
    """Compare forget set sampling strategies."""
    run_folder = Path(run_folder)
    runs = sorted(run_folder.glob("run_*"))

    forget_usage_counts = defaultdict(list)
    num_sets_per_run = []

    for run_dir in runs:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            indices = config.get("sampled_forget_indices", [])
            num_sets_per_run.append(len(indices))
            for idx in indices:
                forget_usage_counts[idx].append(config.get("run_num", -1))

    print(f"\n{'='*80}")
    print("FORGET SET SAMPLING ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Total runs: {len(runs)}")
    print(f"Forget sets sampled per run: {sorted(set(num_sets_per_run))}")
    print(f"Distribution: min={min(num_sets_per_run)}, max={max(num_sets_per_run)}, "
          f"mean={np.mean(num_sets_per_run):.1f}\n")

    print("Forget set usage frequency:")
    for idx in sorted(forget_usage_counts.keys()):
        count = len(forget_usage_counts[idx])
        pct = 100.0 * count / len(runs)
        print(f"  forget_{idx}: used in {count:3d} runs ({pct:5.1f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Manage and inspect multi-run experiments")
    parser.add_argument(
        "--run_folder",
        type=str,
        default="runs_unlearning",
        help="Path to run folder"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["list", "inspect", "calculate_stats", "compare_forget_sampling"],
        default="list",
        help="Action to perform"
    )
    parser.add_argument(
        "--run_number",
        type=int,
        default=0,
        help="Run number for inspect action"
    )

    args = parser.parse_args()

    if args.action == "list":
        list_runs(args.run_folder)
    elif args.action == "inspect":
        inspect_run(args.run_folder, args.run_number)
    elif args.action == "calculate_stats":
        calculate_stats(args.run_folder)
    elif args.action == "compare_forget_sampling":
        compare_forget_sampling(args.run_folder)


if __name__ == "__main__":
    main()
