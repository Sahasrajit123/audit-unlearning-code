#!/usr/bin/env python3
"""
Evaluate trained and unlearnt models from combination runs on forget data points.

For each model (trained/unlearnt) from each combination run, evaluate on all forget
data points, with noise loading, and compute mean/variance of loss and phi.

Output JSON files are named based on model type, epsilon, and other parameters.
"""

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import models
from forget_phi_noisy_loader import (
    compute_n_and_num_forget,
    extract_training_params_from_path,
    compute_T_and_K,
    compute_sigma_from_log_new_structure,
)
from r2d import add_gaussian_noise_to_weights


_MODEL_CONFIGS = {
    "tinynet": {"num_classes": 10},
    "tinynetcifar100": {"num_classes": 100},
}


def infer_model_from_checkpoint(ckpt_path: Path) -> Tuple[str, int, float]:
    """
    Infer model architecture, num_classes, and filters_percentage from checkpoint filename.

    Expected format: {dataset}_{model_arch}_{fp_int}_{fp_dec}_...
    e.g. cifar10_tinynet_1_0_... or cifar100_tinynetcifar100_1_0_...

    Returns:
        (model_arch, num_classes, filters_percentage)

    Raises:
        ValueError if model_arch is not supported or filename does not match expected format.
    """
    parts = ckpt_path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot infer model from checkpoint filename '{ckpt_path.name}'. "
            f"Expected format: {{dataset}}_{{model_arch}}_{{fp_int}}_{{fp_dec}}_..."
        )

    model_arch = parts[1]
    if model_arch not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model architecture '{model_arch}' in checkpoint '{ckpt_path.name}'. "
            f"Supported: {list(_MODEL_CONFIGS.keys())}"
        )

    num_classes = _MODEL_CONFIGS[model_arch]["num_classes"]

    try:
        filters_percentage = float(f"{parts[2]}.{parts[3]}")
    except (ValueError, IndexError):
        filters_percentage = 1.0

    return model_arch, num_classes, filters_percentage


def find_run_directories(results_dir: Path) -> List[Tuple[Path, int]]:
    """Find all run_XXX directories, sorted by index."""
    runs = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = re.match(r'run_(\d+)', run_dir.name)
        if match:
            runs.append((run_dir, int(match.group(1))))
    runs.sort(key=lambda x: x[1])
    return runs


def load_forget_batch_indices(run_path: Path) -> Optional[List[int]]:
    """Load forget batch indices from a run directory."""
    forget_ids_file = run_path / "forget_ids.json"
    if not forget_ids_file.exists():
        return None

    try:
        with open(forget_ids_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and "type" in data and "values" in data:
            if data["type"] == "batch_indices":
                return sorted(data["values"])
            elif data["type"] == "identity_ids":
                return None
        elif isinstance(data, list):
            return sorted(data)
    except Exception as e:
        print(f"Warning: Failed to load forget_ids from {forget_ids_file}: {e}")

    return None


def find_checkpoints(run_path: Path, model_type: str) -> Tuple[Path, Path]:
    """
    Find checkpoint and log file paths for a model in a run directory.

    Args:
        run_path: Path to the run directory
        model_type: 'trained' or 'unlearnt'

    Returns:
        (checkpoint_path, log_path)
    """
    run_path = run_path.resolve()

    if model_type == "trained":
        train_dir = run_path / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"train directory not found in {run_path}")

        ckpts = sorted((train_dir / "checkpoints").glob("*_final.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No trained checkpoint found in {train_dir / 'checkpoints'}")
        ckpt_path = ckpts[-1].resolve()

        logs = sorted((train_dir / "logs").glob("*.pkl"))
        if not logs:
            raise FileNotFoundError(f"No trained log found in {train_dir / 'logs'}")
        log_path = logs[0].resolve()

    else:  # unlearnt
        unlearn_dir = run_path / "unlearn"
        if not unlearn_dir.exists():
            raise FileNotFoundError(f"unlearn directory not found in {run_path}")

        all_ckpts = list((unlearn_dir / "checkpoints").glob("*_final.pt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No unlearnt checkpoint found in {unlearn_dir / 'checkpoints'}")

        def extract_unlearn_epoch(p: Path) -> int:
            match = re.search(r'_(\d+)_final$', p.stem)
            return int(match.group(1)) if match else -1

        sorted_ckpts = sorted(
            all_ckpts,
            key=lambda p: (extract_unlearn_epoch(p), p.stat().st_mtime),
            reverse=True
        )
        ckpt_path = sorted_ckpts[0].resolve()

        ckpt_name = ckpt_path.stem
        if '_loadedfromfinal_' in ckpt_name:
            log_base = ckpt_name.split('_loadedfromfinal_')[0] + '_loadedfromfinal'
        elif '_loadedfrom' in ckpt_name:
            log_base = re.sub(r'_loadedfrom.*?_\d+_final$', '_loadedfromfinal', ckpt_name)
        else:
            log_base = re.sub(r'_\d+_final$', '', ckpt_name)

        logs = list((unlearn_dir / "logs").glob(f"{log_base}.pkl"))
        if not logs:
            raise FileNotFoundError(
                f"No unlearnt log matching '{log_base}.pkl' in {unlearn_dir / 'logs'}"
            )
        log_path = logs[0].resolve()

    return ckpt_path, log_path


def load_all_forget_points(forget_dir: Path) -> List[Tuple[torch.Tensor, int, int, int]]:
    """Pre-load all forget data points from all batches on CPU."""
    all_points = []
    for batch_file in sorted(forget_dir.glob("batch_*.pkl")):
        batch_match = re.match(r'batch_(\d+)', batch_file.stem)
        if not batch_match:
            continue
        batch_idx = int(batch_match.group(1))

        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        if isinstance(batch_data, tuple):
            images, labels = batch_data[0], batch_data[1]
        elif isinstance(batch_data, dict):
            images = batch_data.get('data', batch_data.get('images', batch_data.get('x')))
            labels = batch_data.get('labels', batch_data.get('targets', batch_data.get('y')))
        else:
            raise ValueError(f"Unexpected batch data format: {type(batch_data)}")

        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        if images.ndim == 3:
            images = images[np.newaxis, ...]
        elif images.ndim != 4:
            raise ValueError(f"Unexpected image shape: {images.shape}")

        if isinstance(labels, np.ndarray):
            if labels.ndim == 0:
                labels = np.array([labels])
            elif labels.ndim > 1:
                labels = labels.flatten()
        else:
            labels = np.array([labels])

        for point_idx in range(images.shape[0]):
            img_tensor = torch.tensor(
                images[point_idx], dtype=torch.float32
            ).permute(2, 0, 1)
            all_points.append((img_tensor, int(labels[point_idx]), batch_idx, point_idx))

    return all_points


@torch.no_grad()
def compute_phi_and_loss_batch(
    model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phi (log-odds) and cross-entropy loss for a batch of samples."""
    logits = model(images)
    probs = torch.softmax(logits, dim=1)

    batch_indices = torch.arange(len(labels), device=labels.device)
    p = probs[batch_indices, labels].double().clamp(min=1e-9, max=1 - 1e-9)
    phi_values = torch.log(p / (1.0 - p)).cpu().numpy()
    loss_values = F.cross_entropy(logits, labels, reduction="none").cpu().numpy()

    return phi_values, loss_values


def evaluate_model_on_all_points(
    model_ckpt: Path,
    model_log: Path,
    all_points: List[Tuple[torch.Tensor, int, int, int]],
    device: torch.device,
    epsilon: float,
    delta: float,
    num_reloads: int,
    model_name: str,
    dataroot: Path,
    eval_batch_size: int,
) -> Dict:
    """
    Evaluate a model on all forget data points with noise reloading.
    Model architecture is inferred from the checkpoint filename.
    """
    results = {}
    add_noise = not math.isinf(epsilon)
    effective_reloads = 1 if math.isinf(epsilon) else num_reloads

    if math.isinf(epsilon):
        print(f"  Evaluating {model_name} on {len(all_points)} forget points "
              f"(epsilon=inf, no noise, single evaluation)...")
    else:
        print(f"  Evaluating {model_name} on {len(all_points)} forget points "
              f"(epsilon={epsilon}, {num_reloads} shadow-reloads)...")

    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")

    images_list = [p[0] for p in all_points]
    labels_list = [p[1] for p in all_points]
    batch_indices = [p[2] for p in all_points]
    point_indices = [p[3] for p in all_points]

    all_results = {
        f"batch_{b}_point_{p}": []
        for b, p in zip(batch_indices, point_indices)
    }

    if effective_reloads <= 10:
        progress_interval = 1
    elif effective_reloads <= 100:
        progress_interval = 10
    else:
        progress_interval = 50

    sigma = None
    if add_noise:
        try:
            sigma = compute_sigma_from_log_new_structure(
                model_log, model_ckpt.name, epsilon, delta, dataroot=dataroot, verbose=False
            )
            if sigma is None or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Computed sigma is invalid")
            print(f"  Computed noise sigma: {sigma:.6f}")
        except Exception as e:
            raise ValueError(f"Failed to compute sigma: {e}")

    model_arch, num_classes, filters_pct = infer_model_from_checkpoint(model_ckpt)

    for reload_idx in range(effective_reloads):
        try:
            model = models.get_model(
                model_arch, num_classes=num_classes, filters_percentage=filters_pct
            ).to(device)
            state = torch.load(model_ckpt, map_location=device)
            state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state_dict)
            model.eval()

            if add_noise and sigma is not None:
                add_gaussian_noise_to_weights(model, sigma, device)

            total_points = len(images_list)
            for start in range(0, total_points, eval_batch_size):
                end = min(start + eval_batch_size, total_points)
                images_batch = torch.stack(images_list[start:end]).to(device, non_blocking=True)
                labels_batch = torch.tensor(labels_list[start:end], device=device)

                phi_values, loss_values = compute_phi_and_loss_batch(model, images_batch, labels_batch)

                for i in range(end - start):
                    b = batch_indices[start + i]
                    p = point_indices[start + i]
                    all_results[f"batch_{b}_point_{p}"].append((phi_values[i], loss_values[i]))

            reload_num = reload_idx + 1
            if reload_num % progress_interval == 0 or reload_num == effective_reloads:
                print(f"    Progress: {reload_num}/{effective_reloads} shadow-reloads "
                      f"({100 * reload_num / effective_reloads:.1f}%)")

        except Exception as e:
            print(f"    Error: Failed to load model for shadow-reload {reload_idx}: {e}")
            raise

    for point_key, values_list in all_results.items():
        if values_list:
            phi_vals = [v[0] for v in values_list]
            loss_vals = [v[1] for v in values_list]
            results[point_key] = {
                "mean_phi": float(np.mean(phi_vals)),
                "var_phi": 0.0 if math.isinf(epsilon) else float(np.var(phi_vals)),
                "mean_loss": float(np.mean(loss_vals)),
                "var_loss": 0.0 if math.isinf(epsilon) else float(np.var(loss_vals)),
                "count": num_reloads if math.isinf(epsilon) else len(phi_vals),
            }

    return results


def compute_sigma_metadata(
    run_path: Path,
    model_log: Path,
    dataroot: Path,
    epsilon: float,
    delta: float,
) -> Dict:
    """Compute and return sigma-related metadata for logging."""
    batch_size_str, train_epochs, unlearn_epochs = extract_training_params_from_path(run_path)
    n, num_forget = compute_n_and_num_forget(model_log, dataroot=dataroot)
    T, K = compute_T_and_K(batch_size_str, train_epochs, unlearn_epochs, n, num_forget)
    sigma = compute_sigma_from_log_new_structure(
        model_log, model_log.stem, epsilon, delta, dataroot=dataroot, verbose=False
    )
    return {
        "batch_size": batch_size_str,
        "train_epochs": train_epochs,
        "unlearn_epochs": unlearn_epochs,
        "n": n,
        "num_forget": num_forget,
        "T": T,
        "K": K,
        "sigma": float(sigma) if sigma is not None else None,
        "epsilon": float(epsilon) if not math.isinf(epsilon) else "inf",
        "delta": float(delta),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate combination run models on forget data points"
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Path to results directory containing run_XXX folders")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to data directory with forget/ sub-directory")
    parser.add_argument("--model-type", type=str, default="unlearnt", choices=["trained", "unlearnt"],
                        help="Model type to evaluate (default: unlearnt)")
    parser.add_argument("--epsilon", type=str, default="inf",
                        help="Noise epsilon(s), comma-separated (e.g. 'inf' or '1.0,2.0,inf')")
    parser.add_argument("--delta", type=float, default=1e-3, help="Noise delta")
    parser.add_argument("--num-shadow-reloads", type=int, default=100,
                        help="Number of shadow-reloads per model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")
    parser.add_argument("--eval-batch-size", type=int, default=256,
                        help="Number of forget points per forward pass during evaluation")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for JSON files (default: current directory)")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    epsilon_strs = [eps.strip() for eps in args.epsilon.split(',')]
    epsilon_values = []
    for eps_str in epsilon_strs:
        if eps_str.lower() == "inf":
            epsilon_values.append(float("inf"))
        else:
            try:
                epsilon_values.append(float(eps_str))
            except ValueError:
                raise ValueError(f"Invalid epsilon value: {eps_str}. Must be a number or 'inf'")

    output_dir = args.output_dir if args.output_dir is not None else Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Finding run directories...")
    runs = find_run_directories(args.results_dir)
    print(f"Found {len(runs)} run directories")
    if not runs:
        raise ValueError(f"No run directories found in {args.results_dir}")

    forget_dir = args.data_dir / "forget"
    if not forget_dir.exists():
        raise FileNotFoundError(f"Forget directory not found: {forget_dir}")

    print(f"\nLoading forget data points from {forget_dir}...")
    all_points = load_all_forget_points(forget_dir)
    print(f"Loaded {len(all_points)} forget data points")

    point_labels = {
        f"batch_{b}_point_{p}": label
        for _, label, b, p in all_points
    }

    print("=" * 70)
    print("Model Evaluation on Forget Data Points")
    print("=" * 70)
    print(f"Model type:          {args.model_type.upper()}")
    print(f"Number of runs:      {len(runs)}")
    print(f"Forget points:       {len(all_points)}")
    print(f"Epsilon values:      {epsilon_strs}")
    print(f"Delta:               {args.delta}")
    print(f"Shadow-reloads:      {args.num_shadow_reloads}")
    print(f"Device:              {device}")
    print(f"Output directory:    {output_dir}")
    print()

    all_output_paths = []

    for eps_idx, epsilon in enumerate(epsilon_values, 1):
        epsilon_str = epsilon_strs[eps_idx - 1]
        print(f"\n{'#' * 70}")
        print(f"Processing Epsilon {eps_idx}/{len(epsilon_values)}: {epsilon_str}")
        print(f"{'#' * 70}\n")

        all_results = {}
        sigma_metadata = {}
        completed_count = 0

        if len(runs) <= 10:
            progress_interval = 1
        elif len(runs) <= 50:
            progress_interval = 5
        else:
            progress_interval = max(5, len(runs) // 10)

        delta_str = str(args.delta).replace('.', 'p').replace('-', 'm')
        eps_str_filename = epsilon_str.replace('.', 'p').replace('-', 'm')
        sigma_output_path = None
        if not math.isinf(epsilon):
            sigma_output_path = output_dir / (
                f"{args.model_type}_eps{eps_str_filename}_delta{delta_str}_"
                f"shadow-reloads{args.num_shadow_reloads}_sigma-info.log"
            )

        for run_idx, (run_path, run_index) in enumerate(runs, 1):
            run_name = f"run_{run_index:03d}"
            print(f"{'=' * 70}")
            print(f"Processing {run_name} ({run_idx}/{len(runs)})")
            print(f"{'=' * 70}")

            model_ckpt, model_log = find_checkpoints(run_path, args.model_type)
            print(f"Checkpoint: {model_ckpt.name}")

            point_results = evaluate_model_on_all_points(
                model_ckpt, model_log, all_points, device,
                epsilon, args.delta, args.num_shadow_reloads, run_name,
                dataroot=args.data_dir,
                eval_batch_size=args.eval_batch_size,
            )

            if not math.isinf(epsilon):
                sigma_metadata[run_name] = compute_sigma_metadata(
                    run_path=run_path, model_log=model_log,
                    dataroot=args.data_dir, epsilon=epsilon, delta=args.delta,
                )
                if sigma_output_path is not None:
                    with open(sigma_output_path, "w") as f:
                        json.dump(sigma_metadata, f, indent=2)

            forget_batch_indices = load_forget_batch_indices(run_path)

            for point_key, stats in point_results.items():
                if point_key not in all_results:
                    all_results[point_key] = {}
                match = re.match(r'batch_(\d+)_point_(\d+)', point_key)
                included = (
                    match is not None
                    and forget_batch_indices is not None
                    and int(match.group(1)) in forget_batch_indices
                )
                stats_with_inclusion = stats.copy()
                stats_with_inclusion["included"] = included
                all_results[point_key][run_name] = stats_with_inclusion

            completed_count += 1
            print(f"  ✓ Completed {run_name}")

            if run_idx % progress_interval == 0 or run_idx == len(runs):
                pct = 100 * run_idx / len(runs)
                print(f"\n  Progress: {run_idx}/{len(runs)} ({pct:.1f}%)  "
                      f"Completed: {completed_count}\n")

        print(f"\n{'=' * 70}")
        print(f"Epsilon={epsilon_str}: {completed_count}/{len(runs)} runs completed")
        print(f"{'=' * 70}\n")

        # Save per-point per-model results
        points_dict = {}
        for point_key, models_dict in all_results.items():
            match = re.match(r'batch_(\d+)_point_(\d+)', point_key)
            if not match:
                continue
            points_dict[point_key] = {
                "batch_idx": int(match.group(1)),
                "point_idx": int(match.group(2)),
                "label": int(point_labels.get(point_key, -1)),
                "models": models_dict,
            }

        output_data = {
            "epsilon": epsilon_str,
            "delta": args.delta,
            "num_reloads": args.num_shadow_reloads,
            "model_type": args.model_type,
            "num_runs": len(runs),
            "points": points_dict,
        }

        filename = (f"{args.model_type}_eps{eps_str_filename}_delta{delta_str}"
                    f"_shadow-reloads{args.num_shadow_reloads}.json")
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved {args.model_type} results: {output_path}")
        all_output_paths.append((args.model_type, output_path))

        if sigma_metadata:
            sigma_filename = (
                f"{args.model_type}_eps{eps_str_filename}_delta{delta_str}_"
                f"shadow-reloads{args.num_shadow_reloads}_sigma-info.json"
            )
            sigma_path = output_dir / sigma_filename
            with open(sigma_path, "w") as f:
                json.dump(sigma_metadata, f, indent=2)
            print(f"  Saved sigma metadata: {sigma_path}")
            all_output_paths.append((f"{args.model_type}_sigma_info", sigma_path))

        # Save grouped (included vs excluded) statistics
        print(f"\n{'=' * 70}")
        print(f"Generating grouped statistics for epsilon={epsilon_str}...")

        grouped_points_dict = {}
        for point_key, models_dict in all_results.items():
            match = re.match(r'batch_(\d+)_point_(\d+)', point_key)
            if not match:
                continue

            included_phi, included_loss = [], []
            excluded_phi, excluded_loss = [], []

            for run_name, model_stats in models_dict.items():
                if model_stats.get("included", False):
                    included_phi.append(model_stats["mean_phi"])
                    included_loss.append(model_stats["mean_loss"])
                else:
                    excluded_phi.append(model_stats["mean_phi"])
                    excluded_loss.append(model_stats["mean_loss"])

            def _group_stats(phi_vals, loss_vals):
                if phi_vals:
                    return {
                        "count": len(phi_vals),
                        "mean_phi": float(np.mean(phi_vals)),
                        "var_phi": float(np.var(phi_vals)),
                        "mean_loss": float(np.mean(loss_vals)),
                        "var_loss": float(np.var(loss_vals)),
                    }
                return {"count": 0, "mean_phi": None, "var_phi": None,
                        "mean_loss": None, "var_loss": None}

            grouped_points_dict[point_key] = {
                "batch_idx": int(match.group(1)),
                "point_idx": int(match.group(2)),
                "label": int(point_labels.get(point_key, -1)),
                "included": _group_stats(included_phi, included_loss),
                "excluded": _group_stats(excluded_phi, excluded_loss),
            }

        grouped_output_data = {
            "epsilon": epsilon_str,
            "delta": args.delta,
            "num_reloads": args.num_shadow_reloads,
            "model_type": args.model_type,
            "num_runs": len(runs),
            "points": grouped_points_dict,
        }

        grouped_filename = (f"{args.model_type}_eps{eps_str_filename}_delta{delta_str}"
                            f"_shadow-reloads{args.num_shadow_reloads}_grouped.json")
        grouped_path = output_dir / grouped_filename
        with open(grouped_path, 'w') as f:
            json.dump(grouped_output_data, f, indent=2)
        print(f"  Saved grouped statistics: {grouped_path}")
        all_output_paths.append((f"{args.model_type}_grouped", grouped_path))

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete. Processed {len(epsilon_values)} epsilon value(s).")
    print("Generated files:")
    for label, path in all_output_paths:
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
