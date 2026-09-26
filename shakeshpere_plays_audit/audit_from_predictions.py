#!/usr/bin/env python3
"""
audit_from_predictions.py

Re-run the epsilon / rho / mu audits at a different r, reusing the LLR scores
already saved by evaluate_llr_predictions.py.

llr_predictions_<model_type>[_T<T>].json stores, for every test run, the full
llr_scores dict (one score per forget file) and the true forget_indices. The
attack for a given r is just "take the top r/2 and bottom r/2 of those scores",
so a new r needs no models, no GPU and no losses file -- only the predictions
JSON. That makes an r sweep essentially free compared with re-running
evaluate_llr_predictions.py, which would re-evaluate every test model per r.

Outputs one summary per (runs_dir, r) as

    <runs_dir>/llr_epsilon_lb_<model_type>_T<T>_r<r>.json

(the r is in the name because the audit outputs of evaluate_llr_predictions.py
are keyed only by T, so two different r at the same T would collide), plus a
combined CSV across all runs_dirs and r values.

Example:
    python audit_from_predictions.py --r 300
    python audit_from_predictions.py --r 100 300 --T 10 --csv sweep.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb,
    compute_median_v_test_epsilon_lb,
    compute_avg_v_test_rho_lb,
    compute_median_v_test_rho_lb,
    compute_avg_v_test_mu_lb,
    compute_median_v_test_mu_lb,
)

DEFAULT_RUNS_DIRS = [
    "runs_ascent_descent_fs400_q_1",
    "runs_ascent_descent_fs400_q_1_var1",
    "runs_ascent_descent_fs400_q_1_var2",
    "runs_ascent_descent_fs400_q_1_var3",
    "runs_ascent_descent_fs400_mid_q",
    "runs_ascent_descent_fs400_q_9",
    "runs_ascent_descent_fs400_q_None",
    "runs_hessian_unlearning_fs400",
    "runs_finetune_fs400",
]


def find_predictions_file(runs_dir, model_type, T):
    """Locate the predictions JSON, preferring the _T<T> variant when T is given."""
    candidates = []
    if T is not None:
        candidates.append(runs_dir / "llr_predictions_{}_T{}.json".format(model_type, T))
    candidates.append(runs_dir / "llr_predictions_{}.json".format(model_type))
    for c in candidates:
        if c.exists():
            return c
    return None


def overlap_v_for_r(run_result, r):
    """
    Recompute the overlap statistic v at budget r for one test run.

    Mirrors evaluate_llr_predictions.py exactly: rank the forget files by LLR
    descending, guess the top r/2 as members and the bottom r/2 as non-members,
    and score the two sides on opposite events. Returns (v, top_correct,
    bottom_correct, m).

    The llr_scores dict is read in file order (which is forget_idx 0,1,2,...),
    never re-sorted by key, so that ties in the LLR are broken the same way
    Python's stable sort broke them in the original run.
    """
    k = r // 2
    scores = run_result["llr_scores"]              # {str(forget_idx): score}
    actual = set(run_result["actual_forget_indices"])
    m = len(scores)

    ranked = sorted(((int(idx), val) for idx, val in scores.items()),
                    key=lambda kv: kv[1], reverse=True)
    top = set(idx for idx, _ in ranked[:k])
    bottom = set(idx for idx, _ in ranked[-k:])

    top_correct = len(top & actual)
    bottom_correct = len(bottom - actual)
    return top_correct + bottom_correct, top_correct, bottom_correct, m


def audit_v_list(m, r, T, v_list, args):
    """Run all six audits on a v_list and return the summary dict."""
    avg_eps = compute_avg_v_test_epsilon_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=args.ci_delta,
        direction=args.avg_direction, theta_max=args.theta_max,
    )
    median_eps = compute_median_v_test_epsilon_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=args.ci_delta,
    )
    avg_rho = compute_avg_v_test_rho_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=args.ci_delta,
        gamma_max=args.gamma_max, theta_max=args.theta_max, conv_delta=args.conv_delta,
    )
    median_rho = compute_median_v_test_rho_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=args.ci_delta,
        gamma_max=args.gamma_max, conv_delta=args.conv_delta,
    )
    avg_mu = compute_avg_v_test_mu_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=args.ci_delta,
        theta_max=args.theta_max, conv_delta=args.conv_delta,
    )
    median_mu = compute_median_v_test_mu_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=args.ci_delta, conv_delta=args.conv_delta,
    )

    return {
        "m": m,
        "r": r,
        "k_per_side": r // 2,
        "T": T,
        "v_list": v_list,
        "ci_delta": args.ci_delta,
        "avg_direction": args.avg_direction,
        "gamma_max": args.gamma_max,
        "conv_delta": args.conv_delta,
        "avg_v_test": avg_eps,
        "median_v_test": median_eps,
        "avg_v_test_rho": avg_rho,
        "median_v_test_rho": median_rho,
        "avg_v_test_mu": avg_mu,
        "median_v_test_mu": median_mu,
        "eps_estimate_rho_avg": avg_rho.get("eps_estimate"),
        "eps_estimate_rho_median": median_rho.get("eps_estimate"),
        "eps_estimate_mu_avg": avg_mu.get("eps_estimate"),
        "eps_estimate_mu_median": median_mu.get("eps_estimate"),
        "source": "recomputed from saved llr_scores by audit_from_predictions.py",
        "note": "epsilon_lb entries are (eps, 0) certified-unlearning lower bounds "
                "(LDP bound halved). eps_estimate_* are forward conversions of rho_lb / "
                "mu_lb at delta = conv_delta and are NOT lower bounds on eps.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-audit saved LLR predictions at one or more values of r.")
    parser.add_argument("--runs_dirs", nargs="+", default=DEFAULT_RUNS_DIRS,
                        help="Run folders to process (default: the nine fs400 folders)")
    parser.add_argument("--model_type", choices=["trained", "unlearnt"], default="unlearnt")
    parser.add_argument("--r", type=int, nargs="+", default=[300],
                        help="Audit parameter r (total budget; top r/2 + bottom r/2). "
                             "Must be even and <= m. Accepts several values")
    parser.add_argument("--T", type=int, default=10,
                        help="Number of test runs to audit, taking the T smallest run ids. "
                             "Also selects the _T<T> predictions file when present")
    parser.add_argument("--ci_delta", type=float, default=0.05)
    parser.add_argument("--theta_max", type=float, default=50.0)
    parser.add_argument("--avg_direction", choices=["ge", "le"], default="ge")
    parser.add_argument("--gamma_max", type=float, default=1e4)
    parser.add_argument("--conv_delta", type=float, default=1e-3)
    parser.add_argument("--csv", type=str, default="audit_sweep.csv",
                        help="Path for the combined across-folder CSV")
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute and print, but write no files")
    args = parser.parse_args()

    for r in args.r:
        if r <= 0 or r % 2 != 0:
            print("[main] Error: r must be a positive even number, got {}".format(r))
            return 1

    rows = []
    for dname in args.runs_dirs:
        runs_dir = Path(dname)
        if not runs_dir.is_dir():
            print("[{}] SKIP: not a directory".format(dname))
            continue

        pred_path = find_predictions_file(runs_dir, args.model_type, args.T)
        if pred_path is None:
            print("[{}] SKIP: no llr_predictions_{}[_T{}].json".format(
                dname, args.model_type, args.T))
            continue

        with open(str(pred_path), "r") as f:
            preds = json.load(f)

        # Take the T smallest run ids, matching evaluate_llr_predictions.py.
        preds = sorted(preds, key=lambda x: x["run_id"])
        if args.T is not None:
            if len(preds) < args.T:
                print("[{}] SKIP: only {} runs in {}, need T={}".format(
                    dname, len(preds), pred_path.name, args.T))
                continue
            preds = preds[:args.T]
        T = len(preds)

        print("[{}] {} ({} runs, ids {}..{})".format(
            dname, pred_path.name, T, preds[0]["run_id"], preds[-1]["run_id"]))

        for r in args.r:
            v_list, m = [], None
            for run_result in preds:
                v, top_c, bot_c, m_run = overlap_v_for_r(run_result, r)
                v_list.append(int(v))
                m = m_run if m is None else m
                if m_run != m:
                    print("   ! inconsistent forget-file count across runs "
                          "({} vs {})".format(m_run, m))
                    return 1

            if r > m:
                print("   r={}: SKIP, exceeds m={}".format(r, m))
                continue

            summary = audit_v_list(m, r, T, v_list, args)

            out_path = runs_dir / "llr_epsilon_lb_{}_T{}_r{}.json".format(
                args.model_type, T, r)
            if not args.dry_run:
                with open(str(out_path), "w") as f:
                    json.dump(summary, f, indent=2)

            mean_v = sum(v_list) / float(len(v_list))
            print("   r={:<4} v mean={:.2f} median={:.1f} of {}   "
                  "eps_lb avg/med = {} / {}   mu_lb avg/med = {} / {}".format(
                      r, mean_v, sorted(v_list)[len(v_list) // 2], r,
                      _fmt(summary["avg_v_test"].get("epsilon_lb")),
                      _fmt(summary["median_v_test"].get("epsilon_lb")),
                      _fmt(summary["avg_v_test_mu"].get("mu_lb")),
                      _fmt(summary["median_v_test_mu"].get("mu_lb"))))
            if not args.dry_run:
                print("        -> {}".format(out_path))

            rows.append({
                "runs_dir": dname,
                "model_type": args.model_type,
                "m": m,
                "r": r,
                "k_per_side": r // 2,
                "T": T,
                "v_mean": mean_v,
                "v_median": sorted(v_list)[len(v_list) // 2],
                "eps_lb_avg": summary["avg_v_test"].get("epsilon_lb"),
                "eps_lb_median": summary["median_v_test"].get("epsilon_lb"),
                "eps_lb_ldp_avg": summary["avg_v_test"].get("epsilon_lb_ldp"),
                "eps_lb_ldp_median": summary["median_v_test"].get("epsilon_lb_ldp"),
                "rho_lb_avg": summary["avg_v_test_rho"].get("rho_lb"),
                "rho_lb_median": summary["median_v_test_rho"].get("rho_lb"),
                "mu_lb_avg": summary["avg_v_test_mu"].get("mu_lb"),
                "mu_lb_median": summary["median_v_test_mu"].get("mu_lb"),
                "eps_est_rho_avg": summary["eps_estimate_rho_avg"],
                "eps_est_rho_median": summary["eps_estimate_rho_median"],
                "eps_est_mu_avg": summary["eps_estimate_mu_avg"],
                "eps_est_mu_median": summary["eps_estimate_mu_median"],
                "v_list": " ".join(str(v) for v in summary["v_list"]),
            })

    if rows and not args.dry_run:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n[main] Wrote {} rows to {}".format(len(rows), args.csv))
    elif not rows:
        print("\n[main] Nothing computed.")
        return 1
    return 0


def _fmt(x):
    if x is None:
        return "None"
    return "{:.4f}".format(x) if isinstance(x, float) else str(x)


if __name__ == "__main__":
    sys.exit(main())
