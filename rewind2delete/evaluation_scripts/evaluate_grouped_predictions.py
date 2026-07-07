#!/usr/bin/env python3
"""
Helper functions for evaluating predictions using grouped statistics.

This module provides utilities for:
- Loading grouped statistics JSON
- Computing LLRs (log-likelihood ratios) for predictions
- Evaluating prediction accuracy
"""

import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import models
from forget_phi_noisy_loader import compute_sigma_from_log_new_structure
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

    Raises ValueError if model_arch is not supported.
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


def log_pdf_gaussian(x: float, mu: float, var: float, eps: float = 1e-10) -> float:
    """Compute log probability density of x under Gaussian(mu, var)."""
    var = max(var, eps)
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mu) ** 2) / var


def load_grouped_stats(stats_path: Path) -> Dict:
    """Load grouped statistics JSON file."""
    with open(stats_path, 'r') as f:
        data = json.load(f)
    return data


def load_forget_point(batch_path: Path, point_idx: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Load a specific point from a forget batch file."""
    with open(batch_path, 'rb') as f:
        batch_data = pickle.load(f)

    if isinstance(batch_data, tuple):
        images = batch_data[0]
        labels = batch_data[1]
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

    if point_idx >= len(images):
        raise ValueError(f"Point index {point_idx} out of range (batch has {len(images)} points)")

    img_np = images[point_idx]
    label = int(labels[point_idx])
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).to(device)
    return img_tensor, label


def load_model_from_run(run_dir: Path, model_type: str, device: torch.device,
                        epsilon: float = float("inf"), delta: float = 1e-5,
                        add_noise: bool = False, dataroot: Path = None) -> torch.nn.Module:
    """
    Load a model from a run directory.

    Model architecture is inferred from the checkpoint filename — avoids the
    hardcoded tinynet/num_classes=10 in forget_phi_noisy_loader.load_noisy_model.
    """
    run_dir = run_dir.resolve()

    if model_type in ("unlearnt", "unlearn"):
        checkpoints_dir = run_dir / "unlearn" / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Unlearn checkpoints directory not found: {checkpoints_dir}")

        all_ckpts = list(checkpoints_dir.glob("*_final.pt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No unlearnt checkpoint found in {checkpoints_dir}")

        def extract_unlearn_epoch(ckpt_path: Path) -> int:
            match = re.search(r'_(\d+)_final$', ckpt_path.stem)
            return int(match.group(1)) if match else -1

        sorted_ckpts = sorted(
            all_ckpts,
            key=lambda p: (extract_unlearn_epoch(p), p.stat().st_mtime),
            reverse=True
        )
        ckpt_path = sorted_ckpts[0]

        log_pattern = f"{ckpt_path.stem.replace('_final', '')}.pkl"
        logs = list((run_dir / "unlearn" / "logs").glob(log_pattern))
        if not logs:
            logs = sorted((run_dir / "unlearn" / "logs").glob("*.pkl"))
        if not logs:
            raise FileNotFoundError(f"No unlearnt log found in {run_dir / 'unlearn' / 'logs'}")
        log_path = logs[0]
    else:  # trained
        checkpoints_dir = run_dir / "train" / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Train checkpoints directory not found: {checkpoints_dir}")

        ckpts = sorted(checkpoints_dir.glob("*_final.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No trained checkpoint found in {checkpoints_dir}")
        ckpt_path = ckpts[-1]

        logs = sorted((run_dir / "train" / "logs").glob("*.pkl"))
        if not logs:
            raise FileNotFoundError(f"No trained log found in {run_dir / 'train' / 'logs'}")
        log_path = logs[0]

    # Infer architecture from checkpoint filename instead of relying on
    # load_noisy_model which hardcodes tinynet/num_classes=10.
    model_arch, num_classes, filters_pct = infer_model_from_checkpoint(ckpt_path)
    model = models.get_model(model_arch, num_classes=num_classes,
                             filters_percentage=filters_pct).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()

    sigma = None
    if add_noise:
        sigma = compute_sigma_from_log_new_structure(
            log_path, ckpt_path.name, epsilon, delta, dataroot=dataroot, verbose=False
        )
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid sigma={sigma} for {ckpt_path.name}")
        add_gaussian_noise_to_weights(model, sigma, device)

    return model, sigma


def compute_phi_and_loss(model: torch.nn.Module, x: torch.Tensor, label: int) -> Tuple[float, float]:
    """Compute phi (log-odds) and cross-entropy loss for a single sample."""
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)

        p = probs[0, label].double().clamp(min=1e-9, max=1 - 1e-9)
        phi_val = math.log(p.item() / (1.0 - p.item()))

        y_t = torch.tensor([label], device=logits.device)
        loss_val = F.cross_entropy(logits, y_t, reduction="mean").item()

        return phi_val, loss_val


def compute_llrs_for_points(grouped_stats: Dict, obs_values: Dict[str, Tuple[float, float]],
                            metric: str = "phi") -> List[Tuple[str, float]]:
    """
    Compute LLR = log p_included(x) - log p_excluded(x) for each point.

    Args:
        grouped_stats: The grouped statistics dictionary
        obs_values: Dictionary mapping point_key to (phi_value, loss_value)
        metric: "phi" or "loss"

    Returns:
        List of (point_key, llr) tuples
    """
    points = grouped_stats.get("points", {})
    llrs = []

    for point_key, point_data in points.items():
        if point_key not in obs_values:
            continue

        phi_val, loss_val = obs_values[point_key]
        x = phi_val if metric == "phi" else loss_val

        included = point_data.get("included", {})
        excluded = point_data.get("excluded", {})

        if metric == "phi":
            mu_in = included.get("mean_phi")
            var_in = included.get("var_phi")
            mu_out = excluded.get("mean_phi")
            var_out = excluded.get("var_phi")
        else:
            mu_in = included.get("mean_loss")
            var_in = included.get("var_loss")
            mu_out = excluded.get("mean_loss")
            var_out = excluded.get("var_loss")

        if mu_in is None or var_in is None or mu_out is None or var_out is None:
            continue

        lp_in = log_pdf_gaussian(x, mu_in, var_in)
        lp_out = log_pdf_gaussian(x, mu_out, var_out)
        llrs.append((point_key, lp_in - lp_out))

    return llrs


def load_forget_batch_indices(run_dir: Path) -> Optional[List[int]]:
    """Load forget batch indices from a run directory."""
    forget_ids_file = run_dir / "forget_ids.json"
    if not forget_ids_file.exists():
        return None

    try:
        with open(forget_ids_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and "type" in data and "values" in data:
            if data["type"] == "batch_indices":
                return sorted(data["values"])
        elif isinstance(data, list):
            return sorted(data)
    except Exception as e:
        print(f"Warning: Failed to load forget_ids from {forget_ids_file}: {e}")

    return None


def evaluate_predictions(llrs: List[Tuple[str, float]], k: int,
                         ground_truth_points: set) -> Dict:
    """
    Evaluate predictions using top-k and bottom-k points.

    Args:
        llrs: List of (point_key, llr) tuples
        k: Number of top/bottom points to select
        ground_truth_points: Set of point keys that were actually in the forget set

    Returns:
        Dictionary with evaluation metrics
    """
    if len(llrs) == 0:
        return {
            "accuracy": 0.0,
            "top_k_correct": 0,
            "bottom_k_correct": 0,
            "top_k_total": 0,
            "bottom_k_total": 0,
            "overlap": 0
        }

    llrs_sorted = sorted(llrs, key=lambda t: t[1], reverse=True)
    point_keys = [key for key, _ in llrs_sorted]
    scores = np.array([score for _, score in llrs_sorted], dtype=float)

    pred_forgotten = scores > 0
    gt = np.array([key in ground_truth_points for key in point_keys], dtype=bool)
    accuracy = (pred_forgotten == gt).mean()

    k = min(k, len(point_keys))
    top_k_keys = point_keys[:k]
    top_k_gt = np.array([key in ground_truth_points for key in top_k_keys], dtype=bool)
    top_k_correct = top_k_gt.sum()

    bottom_k_keys = point_keys[-k:]
    bottom_k_gt = np.array([key in ground_truth_points for key in bottom_k_keys], dtype=bool)
    bottom_k_correct = (~bottom_k_gt).sum()

    overlap = int(top_k_correct) + int(bottom_k_correct)

    return {
        "accuracy": float(accuracy),
        "top_k_correct": int(top_k_correct),
        "bottom_k_correct": int(bottom_k_correct),
        "top_k_total": k,
        "bottom_k_total": k,
        "overlap": overlap,
        "max_overlap": 2 * k,
        "overlap_ratio": float(overlap) / (2 * k) if k > 0 else 0.0
    }
