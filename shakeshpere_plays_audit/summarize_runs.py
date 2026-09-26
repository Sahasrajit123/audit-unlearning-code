#!/usr/bin/env python3
"""
Summarize run metrics across all run_* folders in one or more experiment directories.

Usage:
    python summarize_runs.py <folder1> [<folder2> ...]

For each folder, reads run.log in every run_* subdirectory, extracts:
  - Before unlearning: forget / retain / combined train metrics
  - After forget phase: forget / retain / combined train / val-test metrics
  - After retain phase: forget / retain / combined train / val-test metrics
  - Final evaluation: test loss / acc / ppl

Averages all metrics across runs and writes accuracy_summary.json to each folder.
"""

import re
import sys
import json
import glob
import os
from collections import defaultdict


# ── regex helpers ─────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(
    r"(Forget|Retain|Combined Train|Val/Test):\s+"
    r"loss=([\d.]+),\s+acc=([\d.]+),\s+ppl=([\d.]+)"
)
_TEST_LOSS_RE  = re.compile(r"Test loss:\s+([\d.]+)")
_TEST_ACC_RE   = re.compile(r"Test accuracy:\s+([\d.]+)")
_TEST_PPL_RE   = re.compile(r"Test perplexity:\s+([\d.]+)")

_SECTION_HEADERS = {
    "before_unlearning":  "Phase 1 (Before Unlearning) - Val Metrics",
    "after_forget_phase": "Phase 1 (After Forget Phase) - Val Metrics",
    "after_retain_phase": "Phase 2 (After Retain Phase) - Val Metrics",
    "final_evaluation":   "FINAL EVALUATION",
}

_SPLIT_KEYS = {
    "Forget":        "forget",
    "Retain":        "retain",
    "Combined Train":"combined_train",
    "Val/Test":      "val_test",
}


def parse_log(path: str) -> dict:
    """Parse a single run.log and return a dict of metric sections."""
    with open(path) as f:
        lines = f.readlines()

    sections = {k: {} for k in _SECTION_HEADERS}
    current = None

    for line in lines:
        # detect which section we're entering
        for key, header in _SECTION_HEADERS.items():
            if header in line:
                current = key
                break

        if current == "final_evaluation":
            m = _TEST_LOSS_RE.search(line)
            if m:
                sections["final_evaluation"]["test_loss"] = float(m.group(1))
            m = _TEST_ACC_RE.search(line)
            if m:
                sections["final_evaluation"]["test_acc"] = float(m.group(1))
            m = _TEST_PPL_RE.search(line)
            if m:
                sections["final_evaluation"]["test_ppl"] = float(m.group(1))
        elif current is not None:
            m = _SPLIT_RE.search(line)
            if m:
                split_label, loss, acc, ppl = m.groups()
                key = _SPLIT_KEYS[split_label]
                sections[current][key] = {
                    "loss": float(loss),
                    "acc":  float(acc),
                    "ppl":  float(ppl),
                }

    return sections


def average_metrics(all_runs: list[dict]) -> dict:
    """Average metrics across a list of per-run dicts."""
    totals: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for run in all_runs:
        for section, splits in run.items():
            if section == "final_evaluation":
                for metric, value in splits.items():
                    totals[section][metric]["sum"] += value
                    counts[section][metric]["n"] += 1
            else:
                for split, metrics in splits.items():
                    for metric, value in metrics.items():
                        totals[section][split][metric] += value
                        counts[section][split][metric] += 1

    averaged: dict = {}
    for section in totals:
        if section == "final_evaluation":
            averaged[section] = {
                metric: round(totals[section][metric]["sum"] / counts[section][metric]["n"], 6)
                for metric in totals[section]
            }
        else:
            averaged[section] = {}
            for split in totals[section]:
                averaged[section][split] = {
                    metric: round(totals[section][split][metric] / counts[section][split][metric], 6)
                    for metric in totals[section][split]
                }

    return averaged


def summarize_folder(folder: str) -> None:
    run_dirs = sorted(glob.glob(os.path.join(folder, "run_*")))
    if not run_dirs:
        print(f"  [SKIP] No run_* subdirectories found in {folder}")
        return

    all_runs = []
    skipped = []
    for rd in run_dirs:
        log_path = os.path.join(rd, "run.log")
        if not os.path.isfile(log_path):
            skipped.append(rd)
            continue
        try:
            metrics = parse_log(log_path)
            # only include runs that have at least a final evaluation
            if metrics.get("final_evaluation"):
                all_runs.append(metrics)
            else:
                skipped.append(rd)
        except Exception as e:
            print(f"  [WARN] Could not parse {log_path}: {e}")
            skipped.append(rd)

    if not all_runs:
        print(f"  [SKIP] No parseable run logs in {folder}")
        return

    avg = average_metrics(all_runs)

    summary = {
        "folder": os.path.abspath(folder),
        "n_runs_included": len(all_runs),
        "n_runs_skipped": len(skipped),
        "skipped_dirs": [os.path.basename(d) for d in skipped],
        "averages": avg,
    }

    out_path = os.path.join(folder, "accuracy_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Wrote {out_path}  ({len(all_runs)} runs averaged, {len(skipped)} skipped)")

    # print a compact human-readable digest
    fe = avg.get("final_evaluation", {})
    if fe:
        print(f"  Final test  →  acc={fe.get('test_acc','?'):.4f}  "
              f"loss={fe.get('test_loss','?'):.4f}  ppl={fe.get('test_ppl','?'):.4f}")
    bu = avg.get("before_unlearning", {})
    if bu:
        for split in ("forget", "retain", "combined_train"):
            s = bu.get(split, {})
            if s:
                print(f"  Before unlearn [{split:14s}]  acc={s['acc']:.4f}  "
                      f"loss={s['loss']:.4f}  ppl={s['ppl']:.4f}")
    ar = avg.get("after_retain_phase", {})
    if ar:
        for split in ("forget", "retain", "combined_train", "val_test"):
            s = ar.get(split, {})
            if s:
                print(f"  After retain  [{split:14s}]  acc={s['acc']:.4f}  "
                      f"loss={s['loss']:.4f}  ppl={s['ppl']:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_runs.py <folder1> [<folder2> ...]")
        sys.exit(1)

    folders = sys.argv[1:]
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"[ERROR] Not a directory: {folder}")
            continue
        print(f"\nProcessing: {folder}")
        summarize_folder(folder)


if __name__ == "__main__":
    main()
