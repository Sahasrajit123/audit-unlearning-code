#!/usr/bin/env python3
"""
Collect the audit_summary block out of every
prediction_cumulative_model_results_*.json in a directory into one CSV,
one row per noise setting, sorted by epsilon (or rho).

Written for the eval_rho_audit/ folders produced by run_rho_audit_comb3.sh,
but works on any output directory of evaluate_prediction_cumulative_model.py.

Columns, and what they mean:
  epsilon / rho            the NOISE setting the models were perturbed with
  accountant               "eps_delta" or "zcdp" -- which one calibrated the noise
  epsilon_lb_*             audited certified-unlearning epsilon at delta = 0,
                           already halved from the LDP parameter
  epsilon_lb_*_ldp         the un-halved LDP parameter of the audit mechanism
  rho_lb_*                 audited certified-unlearning rho (zCDP)
  eps_from_rho_*           eps_estimate_from_rho(rho_lb, conv_delta): a FORWARD
                           conversion of the audited rho, reported at
                           delta = conv_delta. NOT a lower bound on eps, and not
                           on the same delta axis as epsilon_lb_* (delta = 0).

Older result files that predate the rho audit simply leave those columns blank.
"""

import argparse
import csv
import json
import math
from pathlib import Path

# (csv column, path into the JSON). audit_summary is the flat block added
# alongside the rho audit; the nested fallbacks keep this working on files
# written before that block existed.
FIELDS = [
    ("epsilon",                 ("epsilon",)),
    ("rho",                     ("rho",)),
    ("accountant",              ("accountant",)),
    ("add_noise",               ("add_noise",)),
    ("sigma_mean",              ("sigma_stats", "mean")),
    ("m",                       ("m",)),
    ("r",                       ("r",)),
    ("T",                       ("T",)),
    ("ci_delta",                ("ci_delta",)),
    ("v_list_mean",             ("v_list_mean",)),
    ("v_list_median",           ("v_list_median",)),
    ("epsilon_lb_avg_v",        ("audit_summary", "epsilon_lb_avg_v"),
                                ("avg_v_test", "epsilon_lb")),
    ("epsilon_lb_median_v",     ("audit_summary", "epsilon_lb_median_v"),
                                ("median_v_test", "epsilon_lb")),
    ("epsilon_lb_avg_v_ldp",    ("audit_summary", "epsilon_lb_avg_v_ldp"),
                                ("avg_v_test", "epsilon_lb_ldp")),
    ("epsilon_lb_median_v_ldp", ("audit_summary", "epsilon_lb_median_v_ldp"),
                                ("median_v_test", "epsilon_lb_ldp")),
    ("rho_lb_avg_v",            ("audit_summary", "rho_lb_avg_v"),
                                ("avg_v_test_rho", "rho_lb")),
    ("rho_lb_median_v",         ("audit_summary", "rho_lb_median_v"),
                                ("median_v_test_rho", "rho_lb")),
    ("eps_from_rho_avg_v",      ("audit_summary", "eps_from_rho_avg_v"),
                                ("avg_v_test_rho", "eps_estimate")),
    ("eps_from_rho_median_v",   ("audit_summary", "eps_from_rho_median_v"),
                                ("median_v_test_rho", "eps_estimate")),
    ("conv_delta",              ("audit_summary", "conv_delta"),
                                ("avg_v_test_rho", "conv_delta")),
]


def dig(data, path):
    """Follow a key path, returning None if any level is missing or not a dict."""
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(data, paths):
    """First non-None value among several candidate key paths."""
    for path in paths:
        value = dig(data, path)
        if value is not None:
            return value
    return None


def sort_key(row):
    """Order by the noise setting, pushing 'inf' (no noise) to the end."""
    for key in ("rho", "epsilon"):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return (0, math.inf) if math.isinf(value) else (0, value)
    return (1, math.inf)


def main():
    parser = argparse.ArgumentParser(
        description="Collect audit_summary blocks into one CSV."
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory holding prediction_cumulative_model_results_*.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default: <results-dir>/rho_audit_summary.csv)")
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("prediction_cumulative_model_results_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No prediction_cumulative_model_results_*.json found in {args.results_dir}"
        )

    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"[warn] skipping unreadable {path.name}: {e}")
            continue
        row = {"file": path.name}
        for field in FIELDS:
            column, paths = field[0], field[1:]
            row[column] = first_present(data, paths)
        rows.append(row)

    if not rows:
        raise ValueError(f"No readable result files in {args.results_dir}")

    rows.sort(key=sort_key)

    output = args.output or (args.results_dir / "rho_audit_summary.csv")
    columns = ["file"] + [f[0] for f in FIELDS]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}\n")

    # Console view of the headline numbers.
    show = ["epsilon", "rho", "epsilon_lb_avg_v", "epsilon_lb_median_v",
            "rho_lb_avg_v", "rho_lb_median_v",
            "eps_from_rho_avg_v", "eps_from_rho_median_v"]
    widths = {c: max(len(c), 12) for c in show}
    print("  ".join(c.rjust(widths[c]) for c in show))
    for row in rows:
        cells = []
        for c in show:
            v = row.get(c)
            if v is None:
                text = "-"
            elif isinstance(v, float):
                text = "inf" if math.isinf(v) else f"{v:.6g}"
            else:
                text = str(v)
            cells.append(text.rjust(widths[c]))
        print("  ".join(cells))


if __name__ == "__main__":
    main()
