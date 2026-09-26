#!/usr/bin/env python3
"""
evaluate_llr_predictions.py

For each test run:
1. Load model (trained or unlearnt)
2. Evaluate on all forget files using disjoint chunks
3. Compute cumulative LLR for each forget_idx:
   - LLR = sum over chunks of [log(likelihood | chosen) - log(likelihood | not_chosen)]
   - Uses Gaussian likelihood from mean/var in losses JSON
4. Predict top r and bottom r forget indices
5. Compare with actual forget_indices.json
"""

import json
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path
from scipy.stats import norm

from model import ShakespeareLSTM
from cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb,
    compute_median_v_test_epsilon_lb,
    compute_avg_v_test_rho_lb,
    compute_median_v_test_rho_lb,
    compute_avg_v_test_mu_lb,
    compute_median_v_test_mu_lb,
)


def ensure_losses_file_exists(runs_dir, data_dir, device, target_path):
    """Generate losses JSON via compute_forget_set_losses.py if it is missing."""
    target_path = Path(target_path)
    if target_path.exists():
        return

    script_path = Path(__file__).resolve().parent / "compute_forget_set_losses.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "Missing required generator script: {}".format(script_path)
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--runs_dir",
        str(runs_dir),
        "--data_dir",
        str(data_dir),
        "--device",
        str(device),
    ]

    print("[main] Missing losses file; generating with: {}".format(" ".join(cmd)))
    subprocess.run(cmd, check=True)

    if not target_path.exists():
        raise FileNotFoundError(
            "Losses file still missing after generation: {}".format(target_path)
        )


def load_losses_json(json_path):
    """Load pre-computed losses statistics."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_config_and_meta(data_dir):
    """Load metadata including vocab and seq_len."""
    meta_path = Path(data_dir) / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return {
        "vocab": meta["vocab"],
        "vocab_size": meta["vocab_size"],
        "seq_len": meta["seq_len"],
    }


def build_char_to_idx(vocab):
    """Build character to index mapping."""
    return {c: i for i, c in enumerate(vocab)}


def load_forget_files(data_dir):
    """Load all forget_*.txt files from forget/ subfolder."""
    data_path = Path(data_dir)
    forget_dir = data_path / "forget"

    # Load from forget/ subfolder
    forget_files = sorted([p for p in forget_dir.glob("forget_*.txt")])
    if not forget_files:
        forget_file = forget_dir / "forget.txt"
        if forget_file.exists():
            forget_files = [forget_file]

    # Fallback to root if forget/ folder doesn't exist
    if not forget_files:
        forget_files = sorted([p for p in data_path.glob("forget_*.txt")])
        if not forget_files:
            forget_file = data_path / "forget.txt"
            if forget_file.exists():
                forget_files = [forget_file]

    texts = []
    for f in forget_files:
        with open(f, "r", encoding="utf-8") as fp:
            texts.append(fp.read())

    return texts


def load_model(model_path, vocab_size, device):
    """Load a model from checkpoint."""
    model = ShakespeareLSTM(vocab_size, embed_dim=8, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


def gaussian_log_likelihood(x, mean, var):
    """Compute log likelihood of x under N(mean, var)."""
    if var < 1e-6:
        var = 1e-6
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var


def compute_cumulative_llr(test_model, forget_texts, seq_len, char2idx, device, vocab_size, losses_stats, num_forget_files):
    """
    Compute cumulative LLR for each forget_idx by evaluating test_model on forget chunks.

    test_model: loaded model to evaluate
    forget_texts: list of forget file texts
    losses_stats: {str(forget_idx): {str(position): {...}}}
    Returns: {forget_idx: llr_score}
    """
    from compute_forget_set_losses import compute_loss_on_text

    llr_scores = {}

    for forget_idx in range(num_forget_files):
        forget_idx_str = str(forget_idx)

        if forget_idx_str not in losses_stats:
            llr_scores[forget_idx] = 0.0
            continue

        # Evaluate test model on this forget file
        actual_losses = compute_loss_on_text(test_model, forget_texts[forget_idx], seq_len, char2idx, device, vocab_size)

        cumulative_llr = 0.0

        for position, actual_loss in enumerate(actual_losses):
            position_str = str(position)

            if position_str not in losses_stats[forget_idx_str]:
                continue

            stats = losses_stats[forget_idx_str][position_str]
            chosen_stats = stats["models_chosen"]
            not_chosen_stats = stats["models_not_chosen"]

            # Skip if no data
            if chosen_stats["count"] == 0 or not_chosen_stats["count"] == 0:
                continue

            chosen_mean = chosen_stats["mean"]
            chosen_var = chosen_stats["var"]
            not_chosen_mean = not_chosen_stats["mean"]
            not_chosen_var = not_chosen_stats["var"]

            # Log likelihood of actual_loss under chosen distribution
            ll_chosen = gaussian_log_likelihood(actual_loss, chosen_mean, chosen_var)

            # Log likelihood of actual_loss under not_chosen distribution
            ll_not_chosen = gaussian_log_likelihood(actual_loss, not_chosen_mean, not_chosen_var)

            # LLR for this position
            llr_position = ll_chosen - ll_not_chosen
            cumulative_llr += llr_position

        llr_scores[forget_idx] = cumulative_llr

    return llr_scores


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs_ascent_descent",
                        help="Path to main runs directory")
    parser.add_argument("--data_dir", type=str, default="data_splits_speakers300_fs400",
                        help="Path to data directory")
    parser.add_argument("--model_type", type=str, choices=["trained", "unlearnt"],
                        default="unlearnt",
                        help="Model type to evaluate")
    parser.add_argument("--r", type=int, default=100,
                        help="Audit parameter r: the TOTAL guess budget across both sides. "
                             "The attack guesses the top r/2 and the bottom r/2 by LLR, so the "
                             "overlap statistic v ranges over [0, r]. Must be even and <= the "
                             "number of forget files. This is the same r passed to the epsilon/"
                             "rho/mu tests -- no factor of 2 is applied anywhere downstream")
    parser.add_argument("--T", type=int, default=None,
                        help="Number of test runs (L) to audit, taking the T smallest run ids "
                             "from test_run/. Default: all test runs present")
    # No --delta: the epsilon audit is (eps, 0) only. Its reduction
    # "(eps,0)-certified unlearning => M is (2 eps,0)-LDP" does not survive
    # delta > 0 (see the note above epsilon_lb_from_logM in cum_runs_eps_lab).
    # The only delta here is --conv_delta, used to convert an audited rho/mu
    # into an (eps, delta) pair for reporting.
    parser.add_argument("--ci_delta", type=float, default=0.05,
                        help="confidence tail probability for epsilon lower-bound computation")
    parser.add_argument("--theta_max", type=float, default=50.0,
                        help="theta upper bound used by Chernoff optimization in avg-v epsilon test")
    parser.add_argument("--avg_direction", type=str, choices=["ge", "le"], default="ge",
                        help="direction for avg-v epsilon test: ge for avg(v)>=a, le for avg(v)<=a")
    parser.add_argument("--gamma_max", type=float, default=1e4,
                        help="upper bound for the Renyi-order search in the zCDP rho lower-bound conversion")
    parser.add_argument("--conv_delta", type=float, default=1e-3,
                        help="delta at which the audited rho/mu are reported as an epsilon; "
                             "one-directional conversion, not part of the rho/mu audits themselves")

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    test_runs_dir = runs_dir / "test_run"
    # r is the audit parameter itself (total budget); each side guesses r/2.
    r_value = args.r
    if r_value <= 0 or r_value % 2 != 0:
        print("[main] Error: --r must be a positive even number, got {}".format(r_value))
        return
    k_per_side = r_value // 2
    model_type = args.model_type

    # Require test_run/ subfolder to exist
    if not test_runs_dir.exists():
        print("[main] Error: {} not found".format(test_runs_dir))
        return

    print("[main] Model type: {}".format(model_type))
    print("[main] Audit parameter r: {} (top {} + bottom {} guesses, v in [0,{}])".format(
        r_value, k_per_side, k_per_side, r_value))

    # Load metadata
    data_dir = Path(args.data_dir)
    print("[main] Loading metadata from: {}".format(data_dir))
    meta = load_config_and_meta(data_dir)
    char2idx = build_char_to_idx(meta["vocab"])
    seq_len = meta["seq_len"]
    vocab_size = meta["vocab_size"]

    # Load forget files
    print("[main] Loading forget files...")
    forget_texts = load_forget_files(data_dir)
    print("[main] Loaded {} forget files".format(len(forget_texts)))

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print("[main] Using device: {}".format(device))

    # Load losses statistics from main runs
    if model_type == "trained":
        losses_json_path = runs_dir / "losses_trained_models.json"
    else:
        losses_json_path = runs_dir / "losses_unlearnt_models.json"

    ensure_losses_file_exists(
        runs_dir=runs_dir,
        data_dir=data_dir,
        device=device,
        target_path=losses_json_path,
    )

    print("[main] Loading losses from: {}".format(losses_json_path))
    losses_stats = load_losses_json(losses_json_path)

    num_forget_files = len(losses_stats)
    print("[main] Number of forget files: {}".format(num_forget_files))

    # The lab needs m >= r; the same condition keeps the top r/2 and bottom r/2
    # guesses disjoint, so a single index can never be counted on both sides.
    if r_value > num_forget_files:
        print("[main] Error: --r={} exceeds the {} forget files (need r <= m)".format(
            r_value, num_forget_files))
        return

    # Find all test runs in test_run/ subfolder. Sorted by run id numerically
    # (not lexicographically, which would order run_10 before run_2), so that
    # --T truncates to the T smallest run ids rather than an arbitrary subset.
    def _run_id_of(path):
        try:
            return int(path.name.split("_")[1])
        except (ValueError, IndexError):
            return None

    test_run_dirs = [d for d in test_runs_dir.iterdir()
                     if d.is_dir() and d.name.startswith("run_") and _run_id_of(d) is not None]
    test_run_dirs.sort(key=_run_id_of)
    num_available_test_runs = len(test_run_dirs)

    if args.T is not None:
        if args.T <= 0:
            print("[main] Error: --T must be positive, got {}".format(args.T))
            return
        if args.T > num_available_test_runs:
            print("[main] Error: --T={} exceeds the {} test runs available in {}".format(
                args.T, num_available_test_runs, test_runs_dir))
            return
        test_run_dirs = test_run_dirs[:args.T]

    print("[main] Found {} test runs; using {}{}".format(
        num_available_test_runs, len(test_run_dirs),
        " (--T={})".format(args.T) if args.T is not None else " (all)"))

    results = []

    for test_run_dir in test_run_dirs:
        try:
            run_id = int(test_run_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        # Load model for this test run
        if model_type == "trained":
            model_path = test_run_dir / "model_trained.pt"
        else:
            model_path = test_run_dir / "model_unlearnt.pt"

        if not model_path.exists():
            print("[main] Warning: {} not found, skipping run_{}".format(model_path, run_id))
            continue

        test_model = load_model(str(model_path), vocab_size, device)

        # Load forget_indices for this run
        forget_indices_path = test_run_dir / "forget_indices.json"
        if not forget_indices_path.exists():
            print("[main] Warning: {} not found, skipping".format(forget_indices_path))
            continue

        with open(str(forget_indices_path), "r") as f:
            forget_info = json.load(f)

        actual_forget_indices = set(forget_info["forget_indices"])

        # Compute cumulative LLR for this test model
        llr_scores = compute_cumulative_llr(test_model, forget_texts, seq_len, char2idx, device, vocab_size, losses_stats, num_forget_files)

        # Guess the top r/2 and the bottom r/2, so that the two sides together
        # spend the audit's budget r (a1, a2 in [0, r/2] with v = a1 + a2).
        sorted_by_llr = sorted(llr_scores.items(), key=lambda x: x[1], reverse=True)
        top_r_indices = set([idx for idx, _ in sorted_by_llr[:k_per_side]])
        bottom_r_indices = set([idx for idx, _ in sorted_by_llr[-k_per_side:]])

        # Compute metrics. The two sides count opposite events: the top guesses
        # are scored on membership, the bottom guesses on non-membership.
        top_r_correct = len(top_r_indices & actual_forget_indices)
        bottom_r_correct = len(bottom_r_indices - actual_forget_indices)

        result = {
            "run_id": run_id,
            "actual_forget_indices": sorted(list(actual_forget_indices)),
            "top_r_predicted": sorted(list(top_r_indices)),
            "bottom_r_predicted": sorted(list(bottom_r_indices)),
            "top_r_correct": top_r_correct,
            "top_r_total": k_per_side,
            "bottom_r_correct": bottom_r_correct,
            "bottom_r_total": k_per_side,
            "top_r_accuracy": top_r_correct / k_per_side,
            "bottom_r_accuracy": bottom_r_correct / k_per_side,
            "overlap_v": int(top_r_correct + bottom_r_correct),
            "llr_scores": {str(k): v for k, v in llr_scores.items()},
        }

        results.append(result)

        print("[main] run_{}: top_r={:.1%}, bottom_r={:.1%}".format(
            run_id, result["top_r_accuracy"], result["bottom_r_accuracy"]))

        del test_model
        torch.cuda.empty_cache()

    # Output names carry the T suffix only when --T was given, so the default
    # (all test runs) keeps the historical filenames and successive --T sweeps
    # do not overwrite one another.
    suffix = "" if args.T is None else "_T{}".format(args.T)

    # Save per-run prediction results (existing output format preserved)
    output_path = runs_dir / "llr_predictions_{}{}.json".format(model_type, suffix)
    with open(str(output_path), "w") as f:
        json.dump(results, f, indent=2)

    print("[main] Saved results to: {}".format(output_path))

    epsilon_summary = None
    epsilon_output_path = runs_dir / "llr_epsilon_lb_{}{}.json".format(model_type, suffix)

    # Compute aggregate statistics
    if results:
        avg_top_r = np.mean([r["top_r_accuracy"] for r in results])
        avg_bottom_r = np.mean([r["bottom_r_accuracy"] for r in results])
        total_top_r_correct = sum(r["top_r_correct"] for r in results)
        total_bottom_r_correct = sum(r["bottom_r_correct"] for r in results)
        v_list = [int(r["overlap_v"]) for r in results]

        # r goes to the audits unchanged: --r already IS the audit's r, and the
        # attack above spent it as top r/2 + bottom r/2.
        epsilon_r = int(r_value)
        epsilon_T = int(len(v_list))
        epsilon_m = int(num_forget_files)

        # (eps, 0) audit. epsilon_lb is the certified-unlearning epsilon, i.e. the
        # audited LDP parameter of M halved (the lemma gives "(eps,0)-certified
        # unlearning => M is (2 eps,0)-LDP"); the un-halved value is epsilon_lb_ldp.
        avg_eps = compute_avg_v_test_epsilon_lb(
            m=epsilon_m,
            r=epsilon_r,
            T=epsilon_T,
            v_list=v_list,
            ci_delta=float(args.ci_delta),
            direction=args.avg_direction,
            theta_max=float(args.theta_max),
        )
        median_eps = compute_median_v_test_epsilon_lb(
            m=epsilon_m,
            r=epsilon_r,
            T=epsilon_T,
            v_list=v_list,
            ci_delta=float(args.ci_delta),
        )

        # zCDP rho lower bounds (mean/"ge" test only).
        avg_rho = compute_avg_v_test_rho_lb(
            m=epsilon_m,
            r=epsilon_r,
            T=epsilon_T,
            v_list=v_list,
            ci_delta=float(args.ci_delta),
            gamma_max=float(args.gamma_max),
            theta_max=float(args.theta_max),
            conv_delta=float(args.conv_delta),
        )
        median_rho = compute_median_v_test_rho_lb(
            m=epsilon_m,
            r=epsilon_r,
            T=epsilon_T,
            v_list=v_list,
            ci_delta=float(args.ci_delta),
            gamma_max=float(args.gamma_max),
            conv_delta=float(args.conv_delta),
        )

        # mu-GDP lower bounds (mean/"ge" test only). mu_lb needs no halving: the
        # factor of 2 from the reference-law reduction is already inside mu_loc.
        avg_mu = compute_avg_v_test_mu_lb(
            m=epsilon_m,
            r=epsilon_r,
            T=epsilon_T,
            v_list=v_list,
            ci_delta=float(args.ci_delta),
            theta_max=float(args.theta_max),
            conv_delta=float(args.conv_delta),
        )
        median_mu = compute_median_v_test_mu_lb(
            m=epsilon_m,
            r=epsilon_r,
            T=epsilon_T,
            v_list=v_list,
            ci_delta=float(args.ci_delta),
            conv_delta=float(args.conv_delta),
        )

        # Epsilons implied by each audited rho/mu at conv_delta, computed inside
        # the lab module. Reporting only: these are forward conversions of a
        # lower bound, so they are estimates, not certified lower bounds on eps.
        eps_estimate_rho_avg = avg_rho.get("eps_estimate")
        eps_estimate_rho_median = median_rho.get("eps_estimate")
        eps_estimate_mu_avg = avg_mu.get("eps_estimate")
        eps_estimate_mu_median = median_mu.get("eps_estimate")

        epsilon_summary = {
            "m": epsilon_m,
            "r": epsilon_r,
            "k_per_side": int(k_per_side),
            "T": epsilon_T,
            "T_requested": args.T,
            "T_available": num_available_test_runs,
            "v_list": v_list,
            "ci_delta": float(args.ci_delta),
            "avg_direction": args.avg_direction,
            "avg_v_test": avg_eps,
            "median_v_test": median_eps,
            "gamma_max": float(args.gamma_max),
            "conv_delta": float(args.conv_delta),
            "avg_v_test_rho": avg_rho,
            "median_v_test_rho": median_rho,
            "avg_v_test_mu": avg_mu,
            "median_v_test_mu": median_mu,
            "eps_estimate_rho_avg": eps_estimate_rho_avg,
            "eps_estimate_rho_median": eps_estimate_rho_median,
            "eps_estimate_mu_avg": eps_estimate_mu_avg,
            "eps_estimate_mu_median": eps_estimate_mu_median,
            "note": "epsilon_lb entries are (eps, 0) certified-unlearning lower bounds "
                    "(LDP bound halved). eps_estimate_* are forward conversions of rho_lb / "
                    "mu_lb at delta = conv_delta and are NOT lower bounds on eps.",
        }

        with open(str(epsilon_output_path), "w") as f:
            json.dump(epsilon_summary, f, indent=2)

        print("\n[main] Aggregate Statistics:")
        print("[main]   Avg top_r accuracy: {:.1%}".format(avg_top_r))
        print("[main]   Avg bottom_r accuracy: {:.1%}".format(avg_bottom_r))
        print("[main]   Total top_r correct: {} / {}".format(
            total_top_r_correct, len(results) * k_per_side))
        print("[main]   Total bottom_r correct: {} / {}".format(
            total_bottom_r_correct, len(results) * k_per_side))
        print("[main]   Avg overlap v: {:.4f} / {}".format(float(np.mean(v_list)), r_value))
        print("[main]   Median overlap v: {:.4f} / {}".format(float(np.median(v_list)), r_value))

        if epsilon_summary is not None:
            print("\n[main] Epsilon Lower-Bound Summary:")
            print("[main]   r = {} (top {} + bottom {})".format(
                epsilon_summary["r"], k_per_side, k_per_side))
            print(
                "[main]   Avg-v epsilon_lb (direction={}): {}".format(
                    args.avg_direction,
                    epsilon_summary["avg_v_test"].get("epsilon_lb"),
                )
            )
            print("[main]   Median-v epsilon_lb: {}".format(epsilon_summary["median_v_test"].get("epsilon_lb")))
            print("[main]   Avg-v rho_lb: {}".format(epsilon_summary["avg_v_test_rho"].get("rho_lb")))
            print("[main]   Median-v rho_lb: {}".format(epsilon_summary["median_v_test_rho"].get("rho_lb")))
            print("[main]   Avg-v mu_lb: {}".format(epsilon_summary["avg_v_test_mu"].get("mu_lb")))
            print("[main]   Median-v mu_lb: {}".format(epsilon_summary["median_v_test_mu"].get("mu_lb")))
            print("[main]   (estimates at delta={}, not lower bounds)".format(args.conv_delta))
            print("[main]     eps from rho (avg):    {}".format(epsilon_summary["eps_estimate_rho_avg"]))
            print("[main]     eps from rho (median): {}".format(epsilon_summary["eps_estimate_rho_median"]))
            print("[main]     eps from mu  (avg):    {}".format(epsilon_summary["eps_estimate_mu_avg"]))
            print("[main]     eps from mu  (median): {}".format(epsilon_summary["eps_estimate_mu_median"]))
            print("[main]   Saved epsilon summary to: {}".format(epsilon_output_path))

    print("[main] Done!")


if __name__ == "__main__":
    main()
