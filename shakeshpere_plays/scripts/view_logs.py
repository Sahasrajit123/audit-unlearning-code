#!/usr/bin/env python3
"""
view_logs.py - Utility to view and analyze logs from multi-run experiments

Usage:
  python view_logs.py --run_folder runs_finetune --run_number 0
  python view_logs.py --run_folder runs_finetune --run_number 0 --last_n_lines 50
  python view_logs.py --run_folder runs_finetune --search "Test Results"
  python view_logs.py --run_folder runs_finetune --extract_metrics
"""

import argparse
from pathlib import Path
import re
import json


def view_full_log(run_folder, run_number):
    """View complete log for a run."""
    run_dir = Path(run_folder) / f"run_{run_number}"
    log_path = run_dir / "run.log"

    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return

    print(f"\n{'='*80}")
    print(f"FULL LOG - Run {run_number}")
    print(f"{'='*80}\n")

    with open(log_path) as f:
        print(f.read())


def view_last_n_lines(run_folder, run_number, n=50):
    """View last N lines of log."""
    run_dir = Path(run_folder) / f"run_{run_number}"
    log_path = run_dir / "run.log"

    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return

    print(f"\n{'='*80}")
    print(f"LAST {n} LINES - Run {run_number}")
    print(f"{'='*80}\n")

    with open(log_path) as f:
        lines = f.readlines()
        for line in lines[-n:]:
            print(line, end="")


def search_logs(run_folder, pattern, run_number=None):
    """Search logs for a pattern."""
    run_folder = Path(run_folder)

    if run_number is not None:
        runs = [f"run_{run_number}"]
    else:
        runs = [d.name for d in run_folder.glob("run_*")]

    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS: '{pattern}'")
    print(f"{'='*80}\n")

    total_matches = 0

    for run_name in sorted(runs):
        log_path = run_folder / run_name / "run.log"
        if not log_path.exists():
            continue

        with open(log_path) as f:
            lines = f.readlines()

        matches = [line for line in lines if pattern.lower() in line.lower()]

        if matches:
            print(f"\n{run_name}:")
            for match in matches:
                print(f"  {match.rstrip()}")
            total_matches += len(matches)

    print(f"\n{'='*80}")
    print(f"Total matches: {total_matches}")
    print(f"{'='*80}\n")


def extract_metrics(run_folder):
    """Extract and display metrics from all runs."""
    run_folder = Path(run_folder)

    print(f"\n{'='*80}")
    print(f"EXTRACTED METRICS - All Runs")
    print(f"{'='*80}\n")

    print(f"{'Run':<6} {'Strategy':<15} {'Test Acc':<12} {'Test Loss':<12} {'Test PPL':<10} {'Forget Sets':<20}")
    print("-" * 85)

    for run_dir in sorted(run_folder.glob("run_*")):
        run_num = int(run_dir.name.split("_")[1])
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            strategy = metrics.get("strategy", "?")[:13]
            acc = f"{metrics.get('test_accuracy', 0):.4f}"
            loss = f"{metrics.get('test_loss', 0):.4f}"
            ppl = f"{metrics.get('test_perplexity', 0):.2f}"
            forget_sets = str(metrics.get('forget_indices', [])).replace(", ", ",")[:20]

            print(f"{run_num:<6} {strategy:<15} {acc:<12} {loss:<12} {ppl:<10} {forget_sets:<20}")

    print()


def extract_epochs(run_folder, run_number):
    """Extract epoch-by-epoch metrics from log."""
    run_dir = Path(run_folder) / f"run_{run_number}"
    log_path = run_dir / "run.log"

    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return

    print(f"\n{'='*80}")
    print(f"EPOCH-BY-EPOCH METRICS - Run {run_number}")
    print(f"{'='*80}\n")

    with open(log_path) as f:
        content = f.read()

    # Extract epoch lines
    epoch_pattern = r"Epoch\s+(\d+).*?"
    lines = content.split("\n")

    epoch_lines = [l for l in lines if "Epoch" in l and ("Train" in l or "Loss:" in l)]

    print(f"{'Epoch':<8} {'Phase':<12} Line")
    print("-" * 80)

    for line in epoch_lines:
        # Try to extract epoch number
        match = re.search(r"Epoch\s+(\d+)", line)
        if match:
            epoch_num = match.group(1)
            if "Phase 2" in content[:content.find(line)] and content[:content.find(line)].count("Phase 2") > 0:
                phase = "Phase 2"
            else:
                phase = "Phase 1/AD"
            print(f"{epoch_num:<8} {phase:<12} {line[:60]}...")
        else:
            print(f"{'--':<8} {'--':<12} {line[:60]}...")

    print()


def compare_runs(run_folder, run_numbers):
    """Compare multiple runs."""
    run_folder = Path(run_folder)

    print(f"\n{'='*80}")
    print(f"COMPARING RUNS: {run_numbers}")
    print(f"{'='*80}\n")

    runs_data = []

    for run_num in run_numbers:
        run_dir = run_folder / f"run_{run_num}"
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            runs_data.append((run_num, metrics))

    if not runs_data:
        print("No metrics found for specified runs")
        return

    # Print comparison
    print(f"{'Metric':<25} ", end="")
    for run_num, _ in runs_data:
        print(f"Run {run_num:<6} ", end="")
    print()

    print("-" * (25 + len(runs_data) * 15))

    # Test accuracy
    print(f"{'Test Accuracy':<25} ", end="")
    for _, metrics in runs_data:
        acc = metrics.get("test_accuracy", 0)
        print(f"{acc:<13.4f} ", end="")
    print()

    # Test loss
    print(f"{'Test Loss':<25} ", end="")
    for _, metrics in runs_data:
        loss = metrics.get("test_loss", 0)
        print(f"{loss:<13.4f} ", end="")
    print()

    # Test perplexity
    print(f"{'Test Perplexity':<25} ", end="")
    for _, metrics in runs_data:
        ppl = metrics.get("test_perplexity", 0)
        print(f"{ppl:<13.2f} ", end="")
    print()

    # Strategy-specific
    if runs_data[0][1].get("strategy") == "finetune_retain":
        print(f"{'Phase 1 Duration (s)':<25} ", end="")
        for _, metrics in runs_data:
            dur = metrics.get("phase_1_duration", 0)
            print(f"{dur:<13.1f} ", end="")
        print()

        print(f"{'Phase 2 Duration (s)':<25} ", end="")
        for _, metrics in runs_data:
            dur = metrics.get("phase_2_duration", 0)
            print(f"{dur:<13.1f} ", end="")
        print()
    elif runs_data[0][1].get("strategy") == "ascent_descent":
        print(f"{'Q Value':<25} ", end="")
        for _, metrics in runs_data:
            q = metrics.get("q", 0)
            print(f"{q:<13} ", end="")
        print()

        print(f"{'Lambda':<25} ", end="")
        for _, metrics in runs_data:
            lam = metrics.get("lambda_coef", 0)
            print(f"{lam:<13.2f} ", end="")
        print()

    print()


def main():
    parser = argparse.ArgumentParser(description="View and analyze logs from runs")
    parser.add_argument("--run_folder", type=str, default="runs_unlearning")
    parser.add_argument("--run_number", type=int, default=None)
    parser.add_argument("--action", type=str, choices=["full", "tail", "search", "metrics", "epochs", "compare"],
                       default="tail")
    parser.add_argument("--last_n_lines", type=int, default=50)
    parser.add_argument("--search", type=str, default=None)
    parser.add_argument("--compare_runs", type=int, nargs="+", default=None)

    args = parser.parse_args()

    if args.action == "full":
        if args.run_number is None:
            print("--run_number required for full action")
            return
        view_full_log(args.run_folder, args.run_number)

    elif args.action == "tail":
        if args.run_number is None:
            print("--run_number required for tail action")
            return
        view_last_n_lines(args.run_folder, args.run_number, args.last_n_lines)

    elif args.action == "search":
        if args.search is None:
            print("--search required for search action")
            return
        search_logs(args.run_folder, args.search, args.run_number)

    elif args.action == "metrics":
        extract_metrics(args.run_folder)

    elif args.action == "epochs":
        if args.run_number is None:
            print("--run_number required for epochs action")
            return
        extract_epochs(args.run_folder, args.run_number)

    elif args.action == "compare":
        if args.compare_runs is None or len(args.compare_runs) == 0:
            print("--compare_runs required (e.g., --compare_runs 0 1 2)")
            return
        compare_runs(args.run_folder, args.compare_runs)


if __name__ == "__main__":
    main()
