#!/usr/bin/env python3
"""
analyze_multi_run_results.py - Analyze results from multi-run unlearning experiments

Usage:
  python analyze_multi_run_results.py --run_folder runs_finetune
  python analyze_multi_run_results.py --run_folder runs_ascent_descent --detailed
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_all_metrics(run_folder: str):
    """Load metrics from all runs."""
    run_folder = Path(run_folder)
    metrics_list = []

    for run_dir in sorted(run_folder.glob("run_*")):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"

        if metrics_path.exists() and config_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            with open(config_path) as f:
                config = json.load(f)

            metrics["run_num"] = config.get("run_num", int(run_dir.name.split("_")[1]))
            metrics["forget_indices"] = config.get("sampled_forget_indices", [])
            metrics_list.append(metrics)

    return metrics_list


def print_summary_stats(metrics_list):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    if not metrics_list:
        print("No runs found!")
        return

    num_runs = len(metrics_list)
    strategy = metrics_list[0].get("strategy", "unknown")

    print(f"\nTotal runs: {num_runs}")
    print(f"Strategy: {strategy}")

    # Test metrics
    test_losses = [m.get("test_loss", np.nan) for m in metrics_list]
    test_accs = [m.get("test_accuracy", np.nan) for m in metrics_list]
    test_ppls = [m.get("test_perplexity", np.nan) for m in metrics_list]

    print(f"\n--- Test Loss ---")
    print(f"  Mean:   {np.nanmean(test_losses):.4f}")
    print(f"  Std:    {np.nanstd(test_losses):.4f}")
    print(f"  Min:    {np.nanmin(test_losses):.4f}")
    print(f"  Max:    {np.nanmax(test_losses):.4f}")
    print(f"  Median: {np.nanmedian(test_losses):.4f}")

    print(f"\n--- Test Accuracy ---")
    print(f"  Mean:   {np.nanmean(test_accs):.4f} ({100*np.nanmean(test_accs):.2f}%)")
    print(f"  Std:    {np.nanstd(test_accs):.4f}")
    print(f"  Min:    {np.nanmin(test_accs):.4f}")
    print(f"  Max:    {np.nanmax(test_accs):.4f}")
    print(f"  Median: {np.nanmedian(test_accs):.4f}")

    print(f"\n--- Test Perplexity ---")
    print(f"  Mean:   {np.nanmean(test_ppls):.2f}")
    print(f"  Std:    {np.nanstd(test_ppls):.2f}")
    print(f"  Min:    {np.nanmin(test_ppls):.2f}")
    print(f"  Max:    {np.nanmax(test_ppls):.2f}")
    print(f"  Median: {np.nanmedian(test_ppls):.2f}")

    # Forget indices usage
    print(f"\n--- Forget Set Usage ---")
    forget_usage = defaultdict(int)
    for m in metrics_list:
        for idx in m.get("forget_indices", []):
            forget_usage[idx] += 1

    if forget_usage:
        print(f"  Sets sampled per run: {sorted(forget_usage.values())}")
        print(f"  Forget set usage:")
        for idx in sorted(forget_usage.keys()):
            count = forget_usage[idx]
            pct = 100.0 * count / num_runs
            print(f"    forget_{idx}: {count:3d} times ({pct:.1f}%)")


def print_detailed_results(metrics_list):
    """Print detailed results for each run."""
    print("\n" + "="*80)
    print("DETAILED RESULTS (per run)")
    print("="*80)

    # Create dataframe
    rows = []
    for m in metrics_list:
        row = {
            "Run": m.get("run_num", -1),
            "Test Loss": m.get("test_loss", np.nan),
            "Test Acc": m.get("test_accuracy", np.nan),
            "Test PPL": m.get("test_perplexity", np.nan),
            "Forget Indices": str(m.get("forget_indices", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # Print winners
    best_acc_idx = df["Test Acc"].idxmax()
    worst_acc_idx = df["Test Acc"].idxmin()

    print(f"\nBest accuracy:  Run {df.loc[best_acc_idx, 'Run']:.0f} ({df.loc[best_acc_idx, 'Test Acc']:.4f})")
    print(f"Worst accuracy: Run {df.loc[worst_acc_idx, 'Run']:.0f} ({df.loc[worst_acc_idx, 'Test Acc']:.4f})")


def print_experiment_config(run_folder: str):
    """Print experiment-level configuration."""
    experiment_config_path = Path(run_folder) / "experiment_config.json"

    if experiment_config_path.exists():
        with open(experiment_config_path) as f:
            config = json.load(f)

        print("\n" + "="*80)
        print("EXPERIMENT CONFIGURATION")
        print("="*80)
        print(f"Number of runs: {config.get('num_runs', 'N/A')}")
        print(f"Strategy: {config.get('strategy', 'N/A')}")
        print(f"Forget prob: {config.get('forget_prob', 'N/A')}")
        print(f"Dataset: {config.get('dataset', 'N/A')}")
        print(f"Num forget sets: {config.get('num_forget_sets', 'N/A')}")
        print(f"Forget files: {config.get('forget_files', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-run unlearning results")
    parser.add_argument(
        "--run_folder",
        type=str,
        default="runs_unlearning",
        help="Path to run folder"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-run results"
    )

    args = parser.parse_args()

    # Load and display
    print_experiment_config(args.run_folder)

    metrics_list = load_all_metrics(args.run_folder)
    print_summary_stats(metrics_list)

    if args.detailed:
        print_detailed_results(metrics_list)

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
