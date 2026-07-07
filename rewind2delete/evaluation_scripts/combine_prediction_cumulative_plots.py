#!/usr/bin/env python3
"""
Combine all prediction_cumulative JSON files and create plots for median and mean lower bounds.
"""

import argparse
import json
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def safe_epsilon_lb(value):
    """Convert None or negative values to 0."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return 0.0


def extract_epsilon_from_filename(filename):
    """Extract epsilon value from filename."""
    if 'epsinf' in filename:
        return float('inf')
    match = re.search(r'eps(\d+)', filename)
    if match:
        return float(match.group(1))
    return None


def load_and_extract_data(json_file: Path):
    """Load JSON file and extract epsilon and lower bound values."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        epsilon = data.get('epsilon')
        if epsilon is None:
            epsilon = extract_epsilon_from_filename(json_file.name)

        avg_lb = safe_epsilon_lb(data.get('avg_v_test', {}).get('epsilon_lb'))
        median_lb = safe_epsilon_lb(data.get('median_v_test', {}).get('epsilon_lb'))

        return {
            'epsilon': epsilon,
            'mean_lb': avg_lb,
            'median_lb': median_lb,
            'filename': json_file.name,
        }
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Combine prediction_cumulative JSON files and plot epsilon lower bounds."
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing prediction_cumulative_*.json files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save plots and array-check JSON "
                             "(default: same as --input-dir)")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or args.input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("prediction_cumulative*.json"))
    if not json_files:
        print(f"No prediction_cumulative JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")

    all_data = [load_and_extract_data(f) for f in json_files]
    all_data = [d for d in all_data if d is not None]

    if not all_data:
        print("No valid data extracted from JSON files")
        return

    def sort_key(x):
        eps = x['epsilon']
        if eps is None:
            return (2, 0)
        if eps == float('inf'):
            return (1, 0)
        return (0, eps)

    all_data.sort(key=sort_key)

    epsilons = []
    mean_lbs = []
    median_lbs = []

    for d in all_data:
        eps = d['epsilon']
        if eps is None:
            continue
        epsilons.append(np.inf if eps == float('inf') else eps)
        mean_lbs.append(d['mean_lb'])
        median_lbs.append(d['median_lb'])

    array_check = {
        'num_files': len(all_data),
        'epsilons': epsilons,
        'mean_lower_bounds': mean_lbs,
        'median_lower_bounds': median_lbs,
        'files_processed': [d['filename'] for d in all_data],
    }

    output_json = output_dir / "prediction_cumulative_array_check.json"
    with open(output_json, 'w') as f:
        json.dump(array_check, f, indent=2)
    print(f"Saved array check to {output_json}")

    epsilons_plot = [e if e != np.inf else 1e7 for e in epsilons]
    epsilons_labels = [
        'inf' if e == np.inf else (str(int(e)) if e is not None else 'None')
        for e in epsilons
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epsilons_plot, median_lbs, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Epsilon', fontsize=12)
    ax1.set_ylabel('Median Lower Bound', fontsize=12)
    ax1.set_title('Median Epsilon Lower Bound', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xticks(epsilons_plot)
    ax1.set_xticklabels(epsilons_labels, rotation=45, ha='right')

    ax2.plot(epsilons_plot, mean_lbs, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Epsilon', fontsize=12)
    ax2.set_ylabel('Mean Lower Bound', fontsize=12)
    ax2.set_title('Mean Epsilon Lower Bound', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xticks(epsilons_plot)
    ax2.set_xticklabels(epsilons_labels, rotation=45, ha='right')

    plt.tight_layout()

    output_plot = output_dir / "prediction_cumulative_lower_bounds.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_plot}")

    print(f"\nSummary:")
    print(f"  Total files processed: {len(all_data)}")
    print(f"  Epsilon values:        {epsilons_labels}")
    print(f"  Mean lower bounds:     {mean_lbs}")
    print(f"  Median lower bounds:   {median_lbs}")


if __name__ == "__main__":
    main()
