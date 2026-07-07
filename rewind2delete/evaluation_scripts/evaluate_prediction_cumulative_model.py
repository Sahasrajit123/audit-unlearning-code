#!/usr/bin/env python3
"""
Predict the run using per-model distributions and cumulative log-likelihood,
then compute overlap arrays and epsilon lower bounds.
"""

import argparse
import json
import math
import random
import re
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))          # for evaluate_grouped_predictions
sys.path.insert(0, str(SCRIPT_DIR.parent))   # for cum_runs_eps_lab, models, forget_phi_noisy_loader

from evaluate_grouped_predictions import (
    load_grouped_stats,
    load_model_from_run,
    compute_phi_and_loss,
    load_forget_batch_indices,
    log_pdf_gaussian,
)

from cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb,
    compute_median_v_test_epsilon_lb,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict run via cumulative log-likelihood and compute epsilon bounds."
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Path to directory containing run_XXX folders (model outputs)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to data directory with forget/ sub-directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write prediction_cumulative_* JSON results "
                             "(default: --results-dir/eval_smaller_test_run)")
    parser.add_argument("--stats-dir", type=Path, default=None,
                        help="Directory containing pre-computed stats JSONs "
                             "(default: same as --output-dir)")
    parser.add_argument("--stats-file", type=Path, default=None,
                        help="Explicit path to a stats JSON (overrides --stats-dir lookup)")
    parser.add_argument("--epsilon", type=str, default="inf",
                        help="Epsilon value(s), comma- or space-separated "
                             "(e.g. 'inf' or 'inf,0.1,0.5')")
    parser.add_argument("--delta", type=float, default=1e-3,
                        help="Delta used for model loading")
    parser.add_argument("--metric", type=str, default="phi", choices=["phi", "loss"])
    parser.add_argument("--model-type", type=str, default="unlearnt")
    parser.add_argument("--sampling-seed", type=int, default=123)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:9")
    parser.add_argument("--num-shadow-reloads", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=256,
                        help="Batch size used by stats generation in evaluate_combination_models.py")
    parser.add_argument("--no-generate-if-missing", action="store_true")
    parser.add_argument("--m", type=int, default=None,
                        help="Total number of forget batches used in m parameter for epsilon "
                             "lower bound (default: auto-computed from --data-dir/forget/)")
    parser.add_argument("--r", type=int, default=None,
                        help="Total number of forget batches used in r parameter for epsilon "
                             "lower bound (default: same as --m)")
    parser.add_argument("--epsilon-delta", type=float, default=1e-8)
    parser.add_argument("--ci-delta", type=float, default=0.05)
    return parser.parse_args()


def count_forget_batches(data_dir: Path) -> int:
    """Count the number of batch_*.pkl files in data_dir/forget/."""
    forget_dir = data_dir / "forget"
    if not forget_dir.exists():
        raise FileNotFoundError(f"Forget directory not found: {forget_dir}")
    batch_files = list(forget_dir.glob("batch_*.pkl"))
    if not batch_files:
        raise ValueError(f"No batch_*.pkl files found in {forget_dir}")
    return len(batch_files)


def parse_epsilon(epsilon_str: str) -> float:
    if epsilon_str.lower() in {"inf", "infty", "infinite"}:
        return float("inf")
    return float(epsilon_str)


def parse_epsilon_list(epsilon_str: str) -> List[float]:
    if "," in epsilon_str:
        parts = [p.strip() for p in epsilon_str.split(",")]
    else:
        parts = epsilon_str.split()
    values = [parse_epsilon(p) for p in parts if p]
    return values if values else [float("inf")]


def format_epsilon_for_filename(epsilon_value: float, raw: str) -> str:
    if math.isinf(epsilon_value):
        return "epsinf"
    return "eps" + raw.strip().lower().replace(".", "_")


def format_delta_for_filename(delta_value: float) -> str:
    if delta_value == 0:
        return "delta0"
    exponent = int(math.floor(math.log10(delta_value)))
    mantissa = delta_value / (10 ** exponent)
    mantissa_str = f"{mantissa:g}".replace(".", "")
    exp_str = f"em{abs(exponent):02d}" if exponent < 0 else f"e{exponent:02d}"
    return f"delta{mantissa_str}{exp_str}"


def format_delta_alternates(delta_value: float) -> List[str]:
    delta_tags = {format_delta_for_filename(delta_value)}
    if delta_value != 0:
        simple = f"{delta_value:g}".replace(".", "p")
        delta_tags.add(f"delta{simple}")
    return sorted(delta_tags)


def resolve_stats_file(stats_dir: Path, epsilon_value: float, epsilon_raw: str,
                       delta_value: float, num_shadow_reloads: int,
                       explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        return explicit_path
    eps_tag = format_epsilon_for_filename(epsilon_value, epsilon_raw)
    candidates = [
        stats_dir / f"unlearnt_{eps_tag}_{delta_tag}_shadow-reloads{num_shadow_reloads}.json"
        for delta_tag in format_delta_alternates(delta_value)
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = sorted(stats_dir.glob(f"unlearnt_{eps_tag}_delta*_shadow-reloads*.json"))
    raise FileNotFoundError(
        "Stats file not found. Tried:\n"
        + "\n".join(str(c) for c in candidates)
        + "\nAvailable:\n"
        + "\n".join(str(p) for p in available)
    )


def build_output_file(output_dir: Path, epsilon_value: float, epsilon_raw: str,
                      delta_value: float, num_shadow_reloads: int,
                      metric_name: str) -> Path:
    eps_tag = format_epsilon_for_filename(epsilon_value, epsilon_raw)
    delta_tag = format_delta_for_filename(delta_value)
    return output_dir / (
        f"prediction_cumulative_model_results_{metric_name}_{eps_tag}_{delta_tag}_"
        f"shadow-reloads{num_shadow_reloads}.json"
    )


def generate_stats_file(results_dir: Path, data_dir: Path, stats_dir: Path,
                        model_type: str, epsilon_raw: str, delta_value: float,
                        num_shadow_reloads: int, device: Optional[str],
                        eval_batch_size: int) -> None:
    script_path = SCRIPT_DIR / "evaluate_combination_models.py"
    cmd = [
        sys.executable, str(script_path),
        "--results-dir", str(results_dir),
        "--data-dir", str(data_dir),
        "--model-type", model_type,
        "--epsilon", epsilon_raw,
        "--delta", str(delta_value),
        "--num-shadow-reloads", str(num_shadow_reloads),
        "--eval-batch-size", str(eval_batch_size),
        "--output-dir", str(stats_dir),
    ]
    if device:
        cmd.extend(["--device", device])
    subprocess.run(cmd, check=True)


def extract_images_labels(batch_data):
    if isinstance(batch_data, tuple):
        images, labels = batch_data[0], batch_data[1]
    elif isinstance(batch_data, dict):
        images = batch_data.get("data", batch_data.get("images", batch_data.get("x")))
        labels = batch_data.get("labels", batch_data.get("targets", batch_data.get("y")))
    else:
        raise ValueError(f"Unexpected batch data format: {type(batch_data)}")

    if isinstance(images, torch.Tensor):
        images = images.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if images.ndim != 4:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            labels = np.array([labels])
        elif labels.ndim > 1:
            labels = labels.flatten()
    else:
        labels = np.array([labels])

    return images, labels


def compute_obs_values_for_model(model, batch_files, device):
    obs_values: Dict[str, Tuple[float, float]] = {}
    for batch_file in batch_files:
        batch_match = re.match(r"batch_(\d+)", batch_file.stem)
        if not batch_match:
            continue
        batch_idx = int(batch_match.group(1))

        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f)
        images, labels = extract_images_labels(batch_data)

        for point_idx in range(images.shape[0]):
            point_key = f"batch_{batch_idx}_point_{point_idx}"
            img_tensor = torch.tensor(
                images[point_idx], dtype=torch.float32
            ).permute(2, 0, 1).to(device)
            phi_val, loss_val = compute_phi_and_loss(model, img_tensor, int(labels[point_idx]))
            obs_values[point_key] = (phi_val, loss_val)

    return obs_values


def metric_for_point(obs_values, point_key, metric_name: str):
    if point_key not in obs_values:
        return None
    phi_val, loss_val = obs_values[point_key]
    return phi_val if metric_name == "phi" else loss_val


def cumulative_loglik_per_model(stats, obs_values, metric_name: str):
    points = stats.get("points", {})
    model_scores: Dict[str, float] = {}
    model_counts: Dict[str, int] = {}

    for point_key in obs_values:
        point_stats = points.get(point_key)
        if not point_stats:
            continue
        x = metric_for_point(obs_values, point_key, metric_name)
        if x is None:
            continue

        for model_name, model_stats in point_stats.get("models", {}).items():
            if metric_name == "phi":
                mu, var = model_stats.get("mean_phi"), model_stats.get("var_phi")
            else:
                mu, var = model_stats.get("mean_loss"), model_stats.get("var_loss")

            if mu is None or var is None:
                continue

            model_scores[model_name] = model_scores.get(model_name, 0.0) + log_pdf_gaussian(x, mu, var)
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

    return model_scores, model_counts


def obs_values_for_run(run_dir, batch_files, device, model_type, epsilon, delta,
                       add_noise, dataroot=None,
                       obs_cache: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None):
    run_name = run_dir.name
    if obs_cache is not None and run_name in obs_cache:
        return obs_cache[run_name], None  # sigma unknown when served from cache

    model, sigma = load_model_from_run(
        run_dir, model_type=model_type, device=device,
        epsilon=epsilon, delta=delta, add_noise=add_noise, dataroot=dataroot,
    )
    obs_values = compute_obs_values_for_model(model, batch_files, device)
    if obs_cache is not None:
        obs_cache[run_name] = obs_values
    return obs_values, sigma


def predict_run_by_loglik(stats, obs_values, metric_name: str):
    model_scores, model_counts = cumulative_loglik_per_model(stats, obs_values, metric_name)
    if not model_scores:
        stats_points = stats.get("points", {})
        sample_stat_key = next(iter(stats_points), None)
        sample_obs_key = next(iter(obs_values), None)
        raise RuntimeError(
            f"No model scores accumulated — point keys in obs_values do not match stats['points'].\n"
            f"  obs_values count: {len(obs_values)}, sample key: {sample_obs_key!r}\n"
            f"  stats['points'] count: {len(stats_points)}, sample key: {sample_stat_key!r}"
        )
    best_model = max(model_scores.items(), key=lambda kv: kv[1])[0]
    return best_model, model_scores, model_counts


def load_forget_ids_set(run_dir):
    forget_ids = load_forget_batch_indices(run_dir)
    return set(forget_ids) if forget_ids is not None else set()


def overlap_stats(chosen_set, predicted_set):
    intersection = chosen_set & predicted_set
    union = chosen_set | predicted_set
    overlap = len(intersection)
    chosen_ratio = overlap / len(chosen_set) if chosen_set else 0.0
    jaccard = overlap / len(union) if union else 0.0
    return overlap, chosen_ratio, jaccard


def process_single_epsilon(epsilon_value: float, epsilon_raw: str, args,
                           m: int, r: int,
                           stats_dir: Path, results_dir: Path,
                           data_dir: Path, output_dir: Path):
    obs_cache: Dict[str, Dict[str, Tuple[float, float]]] = {}
    add_noise = not math.isinf(epsilon_value)

    try:
        stats_file = resolve_stats_file(
            stats_dir=stats_dir,
            epsilon_value=epsilon_value,
            epsilon_raw=epsilon_raw,
            delta_value=args.delta,
            num_shadow_reloads=args.num_shadow_reloads,
            explicit_path=args.stats_file,
        )
    except FileNotFoundError:
        if args.no_generate_if_missing:
            raise
        print(f"Stats file missing for epsilon={epsilon_raw}; generating via evaluate_combination_models.py...")
        generate_stats_file(
            results_dir=results_dir, data_dir=data_dir, stats_dir=stats_dir,
            model_type=args.model_type, epsilon_raw=epsilon_raw,
            delta_value=args.delta, num_shadow_reloads=args.num_shadow_reloads,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
        )
        stats_file = resolve_stats_file(
            stats_dir=stats_dir,
            epsilon_value=epsilon_value,
            epsilon_raw=epsilon_raw,
            delta_value=args.delta,
            num_shadow_reloads=args.num_shadow_reloads,
            explicit_path=args.stats_file,
        )

    output_file = build_output_file(
        output_dir=output_dir, epsilon_value=epsilon_value, epsilon_raw=epsilon_raw,
        delta_value=args.delta, num_shadow_reloads=args.num_shadow_reloads,
        metric_name=args.metric,
    )

    random.seed(args.sampling_seed)
    np.random.seed(args.sampling_seed)
    torch.manual_seed(args.sampling_seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading per-model statistics...")
    print(f"  stats_file:   {stats_file.resolve()}")
    print(f"  results_dir:  {results_dir.resolve()}")
    stats = load_grouped_stats(stats_file)
    print(f"Loaded statistics for {len(stats.get('points', {}))} points")
    print(f"Epsilon: {stats.get('epsilon', 'N/A')}  Delta: {stats.get('delta', 'N/A')}  "
          f"Runs: {stats.get('num_runs', 'N/A')}  Reloads: {stats.get('num_reloads', 'N/A')}")
    _sample_stat_key = next(iter(stats.get("points", {})), None)
    if _sample_stat_key:
        _sample_models = list(stats["points"][_sample_stat_key].get("models", {}).keys())[:3]
        print(f"  Sample stat key: {_sample_stat_key!r}, models: {_sample_models}")
    else:
        print("  WARNING: stats['points'] is empty — stats file may be stale or from a different dataset")
    print(f"m={m}  r={r}")

    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    run_dir_by_name = {d.name: d for d in run_dirs}
    print(f"Found {len(run_dirs)} run directories")

    batch_files = sorted((data_dir / "forget").glob("batch_*.pkl"))
    print(f"Found {len(batch_files)} forget batch files")

    overlap_sizes = []
    overlap_ratios = []
    jaccard_scores = []
    chosen_runs = []
    predicted_runs = []
    chosen_sets_track = []
    predicted_sets_track = []
    sigma_values = []

    for iteration in range(args.num_samples):
        chosen_run = random.choice(run_dirs)
        obs_values, sigma = obs_values_for_run(
            chosen_run, batch_files, device,
            model_type=args.model_type,
            epsilon=epsilon_value,
            delta=args.delta,
            add_noise=add_noise,
            dataroot=data_dir,
            obs_cache=obs_cache,
        )
        if sigma is not None:
            sigma_values.append(float(sigma))
        if iteration == 0:
            _sample_obs_key = next(iter(obs_values), None)
            print(f"  obs_values count: {len(obs_values)}, sample key: {_sample_obs_key!r}")
        predicted_model, _, _ = predict_run_by_loglik(stats, obs_values, args.metric)

        chosen_set = load_forget_ids_set(chosen_run)
        predicted_run = run_dir_by_name.get(predicted_model) if predicted_model else None
        predicted_set = load_forget_ids_set(predicted_run) if predicted_run else set()

        overlap, chosen_ratio, jaccard = overlap_stats(chosen_set, predicted_set)
        overlap_sizes.append(overlap)
        overlap_ratios.append(chosen_ratio)
        jaccard_scores.append(jaccard)
        chosen_runs.append(chosen_run.name)
        predicted_runs.append(predicted_model)
        chosen_sets_track.append(frozenset(chosen_set))
        predicted_sets_track.append(frozenset(predicted_set))

        if args.progress_every and iteration % args.progress_every == 0:
            print(f"Iteration {iteration} complete")

    if sigma_values:
        print(f"\nSigma statistics ({len(sigma_values)} unique model loads):")
        print(f"  mean={np.mean(sigma_values):.6e}  std={np.std(sigma_values):.6e}"
              f"  min={np.min(sigma_values):.6e}  max={np.max(sigma_values):.6e}")
    else:
        print("\nNo noise added (sigma=N/A)")

    exact_matches = sum(1 for c, p in zip(chosen_runs, predicted_runs) if c == p)
    print(f"\nExact run match: {exact_matches}/{len(chosen_runs)} = "
          f"{100 * exact_matches / len(chosen_runs):.1f}%")
    print("First 5 (chosen, predicted, match):")
    for i in range(min(5, len(chosen_runs))):
        m_str = "yes" if chosen_runs[i] == predicted_runs[i] else "no"
        print(f"  {chosen_runs[i]} -> {predicted_runs[i]}  match={m_str}")

    v_list = [2 * v for v in overlap_sizes]
    T = int(args.num_samples)

    avg_lb = compute_avg_v_test_epsilon_lb(
        m=m, r=r, T=T, v_list=v_list,
        delta=args.epsilon_delta, ci_delta=args.ci_delta, direction="ge",
    )
    median_lb = compute_median_v_test_epsilon_lb(
        m=m, r=r, T=T, v_list=v_list,
        delta=args.epsilon_delta, ci_delta=args.ci_delta,
    )

    print("\nEpsilon lower bounds:")
    avg_epsilon_lb = avg_lb['epsilon_lb']
    median_epsilon_lb = median_lb['epsilon_lb']
    print(f"Avg-v test epsilon_lb:    "
          f"{avg_epsilon_lb:.6f}" if avg_epsilon_lb is not None else "None (infeasible)")
    print(f"Median-v test epsilon_lb: "
          f"{median_epsilon_lb:.6f}" if median_epsilon_lb is not None else "None (infeasible)")

    # Direct Clopper-Pearson epsilon lower bound (only when M=2)
    m2_cp_result = None
    if m == 2:
        try:
            from scipy.stats import beta as _beta_dist
            all_seen_sets = set(chosen_sets_track) | set(predicted_sets_track)
            all_seen_sets.discard(frozenset())
            unique_forget_sets = sorted(all_seen_sets, key=lambda s: tuple(sorted(s)))
            if len(unique_forget_sets) == 2:
                set_neg, set_pos = unique_forget_sets[0], unique_forget_sets[1]

                tp = sum(1 for cs, ps in zip(chosen_sets_track, predicted_sets_track) if cs == set_pos and ps == set_pos)
                fp = sum(1 for cs, ps in zip(chosen_sets_track, predicted_sets_track) if cs == set_neg and ps == set_pos)
                fn = sum(1 for cs, ps in zip(chosen_sets_track, predicted_sets_track) if cs == set_pos and ps == set_neg)
                tn = sum(1 for cs, ps in zip(chosen_sets_track, predicted_sets_track) if cs == set_neg and ps == set_neg)
                n_pos = tp + fn
                n_neg = fp + tn

                alpha = args.ci_delta / 2
                fp_high = _beta_dist.ppf(1 - alpha, fp + 1, n_neg - fp) if n_neg > fp else 1.0
                fn_high = _beta_dist.ppf(1 - alpha, fn + 1, n_pos - fn) if n_pos > fn else 1.0
                if n_neg == 0:
                    fp_high = 1.0
                if n_pos == 0:
                    fn_high = 1.0

                def _safe_log_ratio(a, b):
                    if a <= 0 or b <= 0:
                        return None
                    return math.log(a / b)

                term1 = _safe_log_ratio(1 - args.epsilon_delta - fp_high, fn_high)
                term2 = _safe_log_ratio(1 - args.epsilon_delta - fn_high, fp_high)
                valid_terms = [t for t in (term1, term2) if t is not None]
                m2_eps_lb = max(valid_terms) if valid_terms else None

                tpr = tp / n_pos if n_pos > 0 else None
                fpr = fp / n_neg if n_neg > 0 else None
                m2_cp_result = {
                    "set_pos": sorted(set_pos),
                    "set_neg": sorted(set_neg),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "n_pos": n_pos, "n_neg": n_neg,
                    "tpr": tpr,
                    "fpr": fpr,
                    "fp_high": fp_high,
                    "fn_high": fn_high,
                    "ci_alpha_each": alpha,
                    "epsilon_lb": m2_eps_lb,
                }
                print(f"\n[m=2] Direct Clopper-Pearson epsilon lower bound:")
                print(f"  set_pos={sorted(set_pos)}, set_neg={sorted(set_neg)}")
                print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}  (n_pos={n_pos}, n_neg={n_neg})")
                if tpr is not None:
                    print(f"  TPR={tpr:.4f}")
                if fpr is not None:
                    print(f"  FPR={fpr:.4f}")
                print(f"  FP^high={fp_high:.6f}, FN^high={fn_high:.6f}  (each alpha={alpha})")
                if m2_eps_lb is not None:
                    print(f"  epsilon_lb (m=2 direct CP) = {m2_eps_lb:.6f}")
                else:
                    print(f"  epsilon_lb (m=2 direct CP) = None (infeasible)")
            else:
                print(f"[m=2] Skipping CP bound: found {len(all_seen_sets)} unique forget sets (expected 2)")
                m2_cp_result = {"skipped": f"Found {len(all_seen_sets)} unique forget sets (expected 2)"}
        except Exception as e:
            import traceback as _tb
            print(f"[warn] Error computing m=2 Clopper-Pearson bound: {e}")
            m2_cp_result = {"error": str(e), "error_traceback": _tb.format_exc()}

    results = {
        "stats_file": str(stats_file),
        "results_dir": str(results_dir),
        "data_dir": str(data_dir),
        "metric": args.metric,
        "model_type": args.model_type,
        "epsilon": epsilon_value,
        "delta": args.delta,
        "add_noise": add_noise,
        "sampling_seed": args.sampling_seed,
        "num_samples": args.num_samples,
        "m": m,
        "r": r,
        "T": T,
        "epsilon_delta": args.epsilon_delta,
        "ci_delta": args.ci_delta,
        "v_list_mean": float(np.mean(v_list)) if v_list else None,
        "v_list_median": float(np.median(v_list)) if v_list else None,
        "avg_v_test": avg_lb,
        "median_v_test": median_lb,
        "m2_cp_result": m2_cp_result,
        "sigma_stats": {
            "mean": float(np.mean(sigma_values)) if sigma_values else None,
            "std": float(np.std(sigma_values)) if sigma_values else None,
            "min": float(np.min(sigma_values)) if sigma_values else None,
            "max": float(np.max(sigma_values)) if sigma_values else None,
            "num_unique_loads": len(sigma_values),
        },
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_file}")


def main():
    args = parse_args()
    epsilon_list = parse_epsilon_list(args.epsilon)

    results_dir = args.results_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or args.results_dir / "eval_smaller_test_run").resolve()
    stats_dir = (args.stats_dir or output_dir).resolve()

    # Auto-compute m and r from the number of forget batches
    num_forget_batches = count_forget_batches(data_dir)
    m = args.m if args.m is not None else num_forget_batches
    r = args.r if args.r is not None else m
    print(f"Forget batches: {num_forget_batches}  ->  m={m}  r={r}")

    print(f"Processing {len(epsilon_list)} epsilon value(s): {epsilon_list}")

    epsilon_parts = (
        [p.strip() for p in args.epsilon.split(",")]
        if "," in args.epsilon
        else args.epsilon.split()
    )

    for idx, epsilon_value in enumerate(epsilon_list):
        epsilon_raw = epsilon_parts[idx] if idx < len(epsilon_parts) else str(epsilon_value)

        print(f"\n{'=' * 60}")
        print(f"Epsilon {idx + 1}/{len(epsilon_list)}: {epsilon_raw} (value: {epsilon_value})")
        print(f"{'=' * 60}")

        try:
            process_single_epsilon(
                epsilon_value=epsilon_value,
                epsilon_raw=epsilon_raw,
                args=args,
                m=m,
                r=r,
                stats_dir=stats_dir,
                results_dir=results_dir,
                data_dir=data_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"Error processing epsilon {epsilon_raw}: {e}")
            import traceback
            traceback.print_exc()
            if idx < len(epsilon_list) - 1:
                print("Continuing with next epsilon value...")
                continue
            else:
                raise

    print(f"\n{'=' * 60}")
    print(f"Completed all {len(epsilon_list)} epsilon value(s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
