# audit_utils.py
"""
Utilities for auditing unlearning experiments:
 - Batch I/O and loaders
 - Rebatching
 - Eval step builders
 - Likelihood-based batch selection
 - Run setup and checkpoint restore
 - Audit epsilon bound computation
"""

import json, pickle, random, math
from pathlib import Path
from typing import List, Iterable, Union, Optional, Tuple, Dict, Set

import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jax import tree_map

import gc
import os
# ---------------------------------------------------------------------
# Batch I/O
# ---------------------------------------------------------------------
Batch = Union[dict, tuple, list]

def load_batches(run_dir: str, split: str) -> List[Batch]:
    """Load pickled batches for a given split."""
    root = Path(run_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(f"split '{split}' not found under {root.parent}")
    return [pickle.load(p.open("rb")) for p in sorted(root.glob("batch_*.pkl"))]


class BatchLoader(Iterable):
    """Dataloader-like wrapper over in-memory batches."""
    def __init__(self, batches: List[Batch], shuffle: bool = False, seed: Optional[int] = None):
        self._batches = list(batches)
        self._shuffle = shuffle
        self._rng = random.Random(seed)

    def __iter__(self):
        idxs = list(range(len(self._batches)))
        if self._shuffle:
            self._rng.shuffle(idxs)
        for i in idxs:
            yield self._batches[i]

    def __len__(self):
        return len(self._batches)


# ---------------------------------------------------------------------
# Rebatch
# ---------------------------------------------------------------------
def rebatch(batches: List[Tuple[jnp.ndarray, jnp.ndarray]], new_bs: int, drop_remainder: bool = True):
    """Concatenate all batches and re-slice to new batch size."""
    imgs = jnp.concatenate([b[0] for b in batches], axis=0)
    labs = jnp.concatenate([b[1] for b in batches], axis=0)
    n_total, n_full = imgs.shape[0], imgs.shape[0] // new_bs
    new_batches = [(imgs[i*new_bs:(i+1)*new_bs], labs[i*new_bs:(i+1)*new_bs]) for i in range(n_full)]
    if n_total % new_bs and not drop_remainder:
        new_batches.append((imgs[n_full*new_bs:], labs[n_full*new_bs:]))
    return new_batches


# ---------------------------------------------------------------------
# Eval Step
# ---------------------------------------------------------------------
@jax.jit
def _compute_loss(logits, labels, *, num_classes: int):
    one_hot = jax.nn.one_hot(labels, num_classes)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))

@jax.jit
def _compute_phi(logits, labels, *, num_classes: int):
    """Compute phi (log-odds): log(p/(1-p)) where p is probability for true class."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Get log probability for the true class
    log_p = jnp.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1)
    p = jnp.exp(log_p)
    # Clamp to avoid numerical issues
    eps = 1e-9
    p_clamped = jnp.clip(p, eps, 1.0 - eps)
    log_one_minus_p = jnp.log(1.0 - p_clamped)
    phi = log_p - log_one_minus_p
    return jnp.mean(phi)  # Return mean over batch


@jax.jit
def _compute_phi_per_sample(logits, labels, *, num_classes: int):
    """Per-sample phi (log-odds). Same as _compute_phi but no mean."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_p = jnp.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1)
    p = jnp.exp(log_p)
    eps = 1e-9
    p_clamped = jnp.clip(p, eps, 1.0 - eps)
    log_one_minus_p = jnp.log(1.0 - p_clamped)
    return log_p - log_one_minus_p


def _compute_loss_per_sample(logits, labels, num_classes: int):
    """Cross-entropy loss per sample (no reduction). num_classes must be concrete (use via closure)."""
    one_hot = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot)


def make_eval_step(model, use_phi: bool = True):
    """Return eval_step(params, batch). Defaults to phi (log-odds) instead of loss."""
    @jax.jit
    def _eval_step(params, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        images, labels = batch
        logits = model.apply({"params": params}, images, train=False)
        if use_phi:
            value = _compute_phi(logits, labels, num_classes=model.num_classes)
        else:
            one_hot = jax.nn.one_hot(labels, model.num_classes)
            value = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        return {"value": value, "accuracy": acc}  # "value" is phi or loss depending on use_phi
    return _eval_step


def make_eval_step_per_point(model):
    """Return eval_step(params, batch) that yields per-sample phi and loss. batch is (x, y)."""
    nc = model.num_classes

    @jax.jit
    def _eval_step(params, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        images, labels = batch[0], batch[1]
        logits = model.apply({"params": params}, images, train=False)
        phi = _compute_phi_per_sample(logits, labels, num_classes=nc)
        loss = _compute_loss_per_sample(logits, labels, num_classes=nc)
        return {"phi": phi, "loss": loss}
    return _eval_step


# ---------------------------------------------------------------------
# Likelihood-based Selection
# ---------------------------------------------------------------------
def select_batches_by_likelihood(value_arr: np.ndarray, stats_json: str, k: int, naming_style: str = "auto", eps = 1e-11):
    """
    Select top-k and bottom-k batches by likelihood ratio.
    
    Args:
        value_arr: Array of values (phi or loss) for each batch
        stats_json: Path to JSON file with statistics
        k: Number of top/bottom batches to select
        naming_style: "auto" to auto-detect (default), "numeric" for pure numbers (0, 1, 2, ...), 
                     "batch_x" for batch_X format (batch_0, batch_1, ...)
    """
    stats = json.loads(Path(stats_json).read_text())
    scores = []
    ##eps = 1e-8

    # Auto-detect naming style if needed
    if naming_style == "auto":
        sample_keys = list(stats.keys())[:10]  # Check first 10 keys
        has_batch_prefix = any(key.startswith("batch_") for key in sample_keys)
        has_numeric = any(key.isdigit() for key in sample_keys)
        
        if has_batch_prefix:
            naming_style = "batch_x"
        elif has_numeric:
            naming_style = "numeric"
        else:
            # Try to detect from first key
            first_key = list(stats.keys())[0] if stats else ""
            if first_key.startswith("batch_"):
                naming_style = "batch_x"
            elif first_key.isdigit():
                naming_style = "numeric"
            else:
                raise ValueError(f"Could not auto-detect naming style. First key: '{first_key}'. "
                               f"Expected either numeric keys (0, 1, 2, ...) or batch_X format (batch_0, batch_1, ...)")

    # Track statistics for debugging
    keys_processed = 0
    keys_skipped_format = 0
    keys_skipped_invalid = 0
    keys_skipped_zero_count = 0

    for key_str, entry in stats.items():
        keys_processed += 1
        
        # Parse key based on naming style
        if naming_style == "batch_x":
            # Handle batch_X format: extract X as global batch index
            if key_str.startswith("batch_"):
                try:
                    j = int(key_str.split("_")[1])  # Extract X from batch_X
                except (ValueError, IndexError):
                    keys_skipped_format += 1
                    continue
            else:
                keys_skipped_format += 1
                continue
        elif naming_style == "numeric":
            # Handle pure numeric keys (backward compatibility)
            try:
                j = int(key_str)
            except ValueError:
                keys_skipped_format += 1
                continue
        else:
            raise ValueError(f"Unknown naming_style: {naming_style}. Use 'auto', 'batch_x', or 'numeric'.")

        # Validate entry structure
        if not isinstance(entry, dict) or "selected" not in entry or "remaining" not in entry:
            keys_skipped_invalid += 1
            continue

        # unpack
        sel, rem = entry["selected"], entry["remaining"]
        n_s, μ_s, σ2_s = sel.get("count"), sel.get("mean"), sel.get("var")
        n_r, μ_r, σ2_r = rem.get("count"), rem.get("mean"), rem.get("var")

        # skip entries with missing or zero counts (as notebook did)
        if n_s is None or n_r is None or n_s == 0 or n_r == 0:
            keys_skipped_zero_count += 1
            continue

        # Validate batch index is within value_arr bounds
        if j >= len(value_arr):
            keys_skipped_invalid += 1
            continue

        x = value_arr[j]

        # clip variances to avoid division by zero
        
        σ2_s = max(σ2_s, eps)
        σ2_r = max(σ2_r, eps)

        lp_s = -0.5 * (((x - μ_s) ** 2) / σ2_s)
        lp_r = -0.5 * (((x - μ_r) ** 2) / σ2_r)
        llr  = lp_s - lp_r

        scores.append((j, llr))

    if not scores:
        # Provide detailed error message
        error_msg = (
            f"No valid batches found in stats_json: {stats_json}\n"
            f"  Total keys processed: {keys_processed}\n"
            f"  Keys skipped (format mismatch): {keys_skipped_format}\n"
            f"  Keys skipped (invalid structure): {keys_skipped_invalid}\n"
            f"  Keys skipped (zero/missing counts): {keys_skipped_zero_count}\n"
            f"  Detected naming_style: {naming_style}\n"
            f"  Sample keys: {list(stats.keys())[:5]}\n"
            f"  value_arr length: {len(value_arr)}"
        )
        raise ValueError(error_msg)

    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
    ##k = min(k, len(scores_sorted))  # handle case fewer than k valid entries

    print ("scores_sorted [:k]", scores_sorted[:k], "scores_sorted[-k:]", scores_sorted[-k:])

    selected_batches  = np.array([j for j, _ in scores_sorted[:k]], dtype=int)
    remaining_batches = np.array([j for j, _ in scores_sorted[-k:]], dtype=int)
    return selected_batches, remaining_batches


# ---------------------------------------------------------------------
# Batch-level predictions from pointwise stats (cumulative LLR)
# ---------------------------------------------------------------------
# For batches with size > 1: compute per-point LLR under Gaussian(mean, var)
# for selected vs remaining; sum over points -> cumulative batch LLR; predict
# selected (forgotten) iff cumulative_llr > 0. Top-k / bottom-k: sort batches
# by cumulative LLR descending; top-k = predicted selected, bottom-k = predicted remaining.

def _log_pdf_gaussian(x: float, mu: float, var: float, eps: float = 1e-10) -> float:
    """Log probability density of x under Gaussian(mu, var)."""
    var = max(var, eps)
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mu) ** 2) / var


def compute_point_llr_pointwise(
    point_stats: Dict,
    metric: str,
    obs_value: float,
    *,
    selected_key: str = "selected",
    remaining_key: str = "remaining",
    eps: float = 1e-10,
) -> float:
    """
    LLR = log p_selected(obs) - log p_remaining(obs) for one point under Gaussians.

    Args:
        point_stats: has selected_key and remaining_key, each with mean_phi, var_phi
            or mean_loss, var_loss depending on metric
        metric: "phi" or "loss"
        obs_value: observed phi or loss for this point
        selected_key, remaining_key: keys for selected/remaining stats (default "selected"/"remaining")
        eps: minimum variance to avoid div by zero

    Returns:
        LLR (log p_selected - log p_remaining)
    """
    sel = point_stats.get(selected_key, {})
    rem = point_stats.get(remaining_key, {})
    if metric == "phi":
        mu_s, var_s = sel.get("mean_phi"), sel.get("var_phi")
        mu_r, var_r = rem.get("mean_phi"), rem.get("var_phi")
    elif metric == "loss":
        mu_s, var_s = sel.get("mean_loss"), sel.get("var_loss")
        mu_r, var_r = rem.get("mean_loss"), rem.get("var_loss")
    else:
        raise ValueError("metric must be 'phi' or 'loss'")
    if mu_s is None or var_s is None or mu_r is None or var_r is None:
        raise ValueError(
            f"Point stats missing for {selected_key}/{remaining_key} and metric={metric}: "
            f"need mean and var; got selected=({mu_s}, {var_s}), remaining=({mu_r}, {var_r})"
        )
    lp_s = _log_pdf_gaussian(obs_value, mu_s, var_s, eps)
    lp_r = _log_pdf_gaussian(obs_value, mu_r, var_r, eps)
    return lp_s - lp_r


def batch_level_predictions_from_pointwise_stats(
    pointwise_stats: Union[Dict, str, Path],
    observed_map: Dict[str, float],
    metric: str = "phi",
    *,
    selected_key: str = "selected",
    remaining_key: str = "remaining",
    eps: float = 1e-10,
    threshold: float = 0.0,
) -> Dict:
    """
    Batch-level prediction using cumulative LLR per batch (sum of per-point LLRs).
    - For each point: LLR = log p_selected(x) - log p_remaining(x) under Gaussian.
    - Cumulative LLR for batch = sum of point LLRs.
    - Predicted selected (forgotten) iff cumulative_llr > threshold.
    - batch_ids_sorted_by_llr: descending by LLR (for top-k / bottom-k).

    Args:
        pointwise_stats: dict with "points" or the points dict; or path to JSON.
        observed_map: "batch_{b}_point_{p}" -> observed phi or loss (float).
        metric: "phi" or "loss"
        selected_key, remaining_key: keys in each point's stats
        eps: min variance for Gaussian log-pdf
        threshold: predict selected when cumulative_llr > threshold (default 0)

    Returns:
        {"batch_cumulative_llrs": {batch_idx: cum_llr}, "predictions": {batch_idx: bool},
         "batch_ids_sorted_by_llr": [batch_idx, ...] (descending by LLR)}
    """
    if isinstance(pointwise_stats, (str, Path)):
        pointwise_stats = json.loads(Path(pointwise_stats).read_text())
    pts = pointwise_stats.get("points", pointwise_stats)
    if not isinstance(pts, dict):
        pts = {}

    batch_sums: Dict[int, float] = {}
    for key, obs in observed_map.items():
        if not key.startswith("batch_") or "_point_" not in key:
            continue
        parts = key.split("_")
        if len(parts) < 4:
            continue
        try:
            batch_idx = int(parts[1])
            point_idx = int(parts[3])
        except (ValueError, IndexError):
            continue
        point_stat = pts.get(key)
        if point_stat is None:
            raise ValueError(
                f"Pointwise stats missing for point {key!r}; it appears in observed_map "
                "but not in pointwise_stats['points']"
            )
        llr = compute_point_llr_pointwise(
            point_stat, metric, float(obs),
            selected_key=selected_key, remaining_key=remaining_key, eps=eps,
        )
        batch_sums[batch_idx] = batch_sums.get(batch_idx, 0.0) + llr

    if not batch_sums:
        return {"batch_cumulative_llrs": {}, "predictions": {}, "batch_ids_sorted_by_llr": []}

    sorted_items = sorted(batch_sums.items(), key=lambda t: t[1], reverse=True)
    batch_ids_sorted = [b for b, _ in sorted_items]
    cumulative_llrs = {b: v for b, v in sorted_items}
    predictions = {b: (v > threshold) for b, v in cumulative_llrs.items()}

    return {
        "batch_cumulative_llrs": cumulative_llrs,
        "predictions": predictions,
        "batch_ids_sorted_by_llr": batch_ids_sorted,
    }


def evaluate_batch_llr_predictions(
    batch_cumulative_llrs: Dict[Union[int, str], float],
    ground_truth_included: Set[Union[int, str]],
    k: int,
    *,
    threshold: float = 0.0,
) -> Dict:
    """
    Evaluate batch-level cumulative-LLR predictions: top-k, bottom-k, combined
    accuracy, precision, recall.

    Args:
        batch_cumulative_llrs: batch_idx -> cumulative_llr
        ground_truth_included: set of batch indices in the forget set
        k: top-k and bottom-k size
        threshold: predict selected when llr > threshold (default 0)

    Returns:
        dict with top_k_accuracy, bottom_k_accuracy, combined_accuracy, precision,
        recall, true_positives, false_positives, true_negatives, false_negatives,
        num_batches, num_forget_batches.
    """
    gt = set(int(x) for x in ground_truth_included)
    llrs = {int(b): float(v) for b, v in batch_cumulative_llrs.items()}
    if not llrs:
        return {
            "top_k_accuracy": 0.0, "bottom_k_accuracy": 0.0, "combined_accuracy": 0.0,
            "precision": 0.0, "recall": 0.0,
            "true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0,
            "num_batches": 0, "num_forget_batches": len(gt),
        }
    sorted_items = sorted(llrs.items(), key=lambda t: t[1], reverse=True)
    batch_ids = [b for b, _ in sorted_items]
    cumulative_arr = np.array([v for _, v in sorted_items], dtype=float)
    k_actual = min(k, len(batch_ids))
    top_k_ids = batch_ids[:k_actual]
    bottom_k_ids = batch_ids[-k_actual:]

    top_k_correct = sum(1 for b in top_k_ids if b in gt)
    top_k_accuracy = top_k_correct / k_actual if k_actual else 0.0
    bottom_k_correct = sum(1 for b in bottom_k_ids if b not in gt)
    bottom_k_accuracy = bottom_k_correct / k_actual if k_actual else 0.0
    combined_accuracy = (top_k_correct + bottom_k_correct) / (2 * k_actual) if k_actual else 0.0

    gt_arr = np.array([b in gt for b in batch_ids], dtype=bool)
    pred_forgotten = cumulative_arr > threshold
    tp = int(((pred_forgotten == True) & (gt_arr == True)).sum())
    fp = int(((pred_forgotten == True) & (gt_arr == False)).sum())
    tn = int(((pred_forgotten == False) & (gt_arr == False)).sum())
    fn = int(((pred_forgotten == False) & (gt_arr == True)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "top_k_accuracy": float(top_k_accuracy),
        "bottom_k_accuracy": float(bottom_k_accuracy),
        "combined_accuracy": float(combined_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "num_batches": len(llrs),
        "num_forget_batches": len(gt),
    }


# ---------------------------------------------------------------------
# Run Setup & Checkpoint Restore
# ---------------------------------------------------------------------
def load_run_vars(run_dir: Path) -> dict:
    vars_file = Path(run_dir) / "run_vars.json"
    with vars_file.open("r") as f:
        return json.load(f)


def _dataset_name_from_config(config_path: Optional[str]) -> Optional[str]:
    if not config_path:
        return None
    p = Path(config_path)
    if not p.exists():
        return None
    try:
        with p.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        ds = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
        name = ds.get("name") if isinstance(ds, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip().lower()
    except Exception:
        return None
    return None


def resolve_config_path(
    main_folder: str,
    run_dir: Optional[Path] = None,
    default_config: str = "configs/exp_cifar.yaml",
) -> str:
    # Strict mode: require only the top-level logs config for reproducibility.
    top_level_config = Path(main_folder) / "config.yaml"
    if top_level_config.exists():
        return str(top_level_config)
    raise FileNotFoundError(
        "Missing required top-level config file: "
        f"{top_level_config}. Please ensure the logs folder contains config.yaml."
    )


def infer_dataset_type(
    run_vars: dict,
    main_folder: str,
    run_dir: Optional[Path] = None,
    config_path: Optional[str] = None,
) -> str:
    """Infer dataset type robustly across older/newer run_vars schemas.

    Preference order:
      1) run_vars['data_subfolder'] (if present/non-empty)
      2) run_vars['dataset_type'] (if present/non-empty and not legacy default)
      3) infer from config.yaml dataset.name (if available)
      4) fallback to 'cifar_1'
    """
    data_subfolder = run_vars.get("data_subfolder")
    if isinstance(data_subfolder, str) and data_subfolder.strip():
        return data_subfolder.strip()

    dataset_type = run_vars.get("dataset_type")
    if isinstance(dataset_type, str) and dataset_type.strip() and dataset_type.strip() != "cifar_1":
        return dataset_type.strip()

    # Use dataset.name from YAML config when available.
    cfg_candidates = [config_path]
    if run_dir is not None:
        cfg_candidates.append(str(Path(run_dir) / "config.yaml"))
    cfg_candidates.append(str(Path(main_folder) / "config.yaml"))
    for c in cfg_candidates:
        cfg_name = _dataset_name_from_config(c)
        if cfg_name:
            return cfg_name

    return "cifar_1"

def get_cache_root(
    main_folder: str,
    dataset_type: Optional[str] = None,
    dataset: str = "cifar",
) -> str:
    """Resolve cache root for CIFAR/CIFAR100 and synthetic-style layouts.

    Resolution order:
      1) <main_folder>/data_split/<dataset_type>
      2) <project_root>/data/<cifar|cifar100>/data_split/<dataset_type>
      3) <project_root>/data/<cifar|cifar100>
    """
    main_path = Path(main_folder)

    dataset_candidates = []
    if dataset_type is not None and str(dataset_type).strip():
        dataset_candidates.append(str(dataset_type).strip())

    # Try explicit logs-root data_split path first.
    for ds_name in dataset_candidates:
        p = main_path / "data_split" / ds_name
        if p.exists():
            return str(p)

    # CIFAR fallback: project-level data/cifar or data/cifar100.
    is_cifar = any(name.lower().startswith("cifar") for name in dataset_candidates)
    if is_cifar:
        is_cifar100 = (
            any(name.lower().startswith("cifar100") for name in dataset_candidates)
            or "cifar100" in str(main_folder).lower()
        )
        cifar_root = main_path.parent / "data" / ("cifar100" if is_cifar100 else "cifar")

        for ds_name in dataset_candidates:
            p = cifar_root / "data_split" / ds_name
            if p.exists():
                return str(p)

        if cifar_root.exists():
            return str(cifar_root)

    raise FileNotFoundError(
        "Could not resolve cache root. Tried dataset candidates "
        f"{dataset_candidates} under {main_path / 'data_split'} and project-level data folders."
    )

def latest_checkpoint(run_dir: Path, prefix: str = "unlearn_") -> str:
    ckpt_root = os.path.join(run_dir, "ckpt")
    latest = max(
        (os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root) if d.startswith(prefix)),
        key=os.path.getmtime
    )
    return os.path.abspath(latest)

# def load_checkpoint(main_folder: str, prefix: str = "unlearn_") -> str:
#     ckpt_root = os.path.join(main_folder, "ckpt")
#     latest = max(
#         (os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root) if d.startswith(prefix)),
#         key=os.path.getmtime
#     )
#     return os.path.abspath(latest)

def restore_orbax_state(latest_path: str, state_structure=None, device: str = "cpu"):
    """
    Restore orbax checkpoint with proper device mapping and sharding handling.
    
    This function handles the common issues with checkpoint restoration:
    1. GPU memory issues (uses available GPU instead of cuda:0)
    2. Sharding issues (provides fallback device mapping)
    3. Device topology mismatches

    latest_path: str required to be absolute path
    """
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Try to find an available GPU with sufficient memory
    available_devices = jax.devices()
    target_device = None
    
    if device != "cpu":
        # Look for GPUs with available memory
        for dev in available_devices:
            if hasattr(dev, 'id') and 'cuda' in str(dev):
                try:
                    # Test if we can create a small array on this device
                    test_array = jax.device_put(jnp.array([1.0]), dev)
                    target_device = dev
                    break
                except Exception:
                    continue
    
    # Fallback to CPU if no GPU available or if device="cpu"
    if target_device is None:
        target_device = jax.devices("cpu")[0]
    
    print(f"Using device: {target_device}")

    # Try multiple approaches to restore the checkpoint
    print("=== Approach 1: Try to restore with device mapping ===")
    try:
        # Set the default device and try to restore
        jax.config.update('jax_default_device', target_device)
        
        if state_structure is not None:
            dev = target_device
            template = tree_map(
                lambda x: jax.device_put(jnp.zeros_like(x) if hasattr(x, "shape") else x, dev),
                state_structure,
            )
            
            try:
                # For Orbax ≥ 0.5
                restore_args = ocp.args.PyTreeRestore(template)
            except AttributeError:
                # For older Orbax, just pass template directly
                restore_args = template
            
            state = checkpointer.restore(latest_path, args=restore_args)
        else:
            state = checkpointer.restore(latest_path)
        
        print("✅ Checkpoint restored successfully with device mapping!")
        return state
        
    except Exception as e:
        print(f"❌ Approach 1 failed: {e}")
        
        print("\n=== Approach 2: Try to restore with PyTreeRestore ===")
        try:
            # Try with PyTreeRestore
            restore_args = ocp.args.PyTreeRestore()
            state = checkpointer.restore(latest_path, args=restore_args)
            print("✅ Checkpoint restored successfully with PyTreeRestore!")
            return state
        except Exception as e2:
            print(f"❌ Approach 2 failed: {e2}")
            
            print("\n=== Approach 3: Try to restore with StandardRestore ===")
            try:
                # Try with StandardRestore
                restore_args = ocp.args.StandardRestore()
                state = checkpointer.restore(latest_path, args=restore_args)
                print("✅ Checkpoint restored successfully with StandardRestore!")
                return state
            except Exception as e3:
                print(f"❌ Approach 3 failed: {e3}")
                
                print("\n=== Approach 4: Try to restore without any args ===")
                try:
                    # Try without any args
                    state = checkpointer.restore(latest_path)
                    print("✅ Checkpoint restored successfully without args!")
                    return state
                except Exception as e4:
                    print(f"❌ All approaches failed. Final error: {e4}")
                    print("\nThis checkpoint appears to require the original device topology to restore properly.")
                    print("The checkpoint was likely saved with all parameters on a specific device (e.g., cuda:0),")
                    print("but that device is out of memory or unavailable.")
                    print("Consider:")
                    print("1. Freeing up memory on the original device")
                    print("2. Using a different checkpoint that was saved with different device topology")
                    print("3. Modifying the checkpoint files to change device references")
                    raise e4




# ---------------------------------------------------------------------
# Audit epsilon bound computation
# ---------------------------------------------------------------------
def build_v_s_vectors(selected_idx, non_selected_idx, chosen_idx, N):
    """Build v and s indicator vectors."""
    v = np.zeros(N, dtype=int)
    v[selected_idx] = +1
    v[non_selected_idx] = -1

    s = np.full(N, -1, dtype=int)
    s[chosen_idx] = +1
    return v, s

def compute_overlap_score(v, s):
    """Compute sum of max(0, v_i * s_i)."""
    return np.maximum(v * s, 0).sum()

def kl_bernoulli(p, q, eps=1e-12):
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def epsilon_lower_bound_from_vs(result, k, N, confidence_level=0.95, delta=0.0):
    """
    Compute epsilon lower bound given overlap result(s), top-k size, and forget set size.
    
    Args:
        result: overlap score (sum of max(0, v*s)) for single run, or list of overlap scores for multiple runs
        k: number of top/bottom batches
        N: total number of forget batches (m)
        confidence_level: float (maps to ci_delta = 1 - confidence_level)
        delta: audit noise delta parameter (default: 0.0)
    
    Returns:
        epsilon lower bound (float) or dict with detailed results if multiple runs
    """
    try:
        from cum_runs_eps_lab import compute_avg_v_test_epsilon_lb, compute_median_v_test_epsilon_lb
    except ImportError:
        # Fallback: try importing from current directory
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from cum_runs_eps_lab import compute_avg_v_test_epsilon_lb, compute_median_v_test_epsilon_lb
    
    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level
    
    # Handle both single result and list of results
    # Check for numpy scalar types as well
    if isinstance(result, (int, float, np.integer, np.floating)):
        # Single run: use median function
        v_list = [float(result)]
        T = 1
    elif isinstance(result, (list, np.ndarray, tuple)):
        # Multiple runs
        v_list = [float(v) for v in result]
        T = len(v_list)
    else:
        raise TypeError(f"result must be int, float, numpy scalar, list, tuple, or numpy array, got {type(result)}")
    
    # For single run, use median function (but handle T=1 as special case)
    if T == 1:
        v = v_list[0]
        if v/r < 0.5:
            print("Less than half of points rightly identified, returning 0")
            return {
                'mean': None,  # Mean method not applicable for T=1
                'median': 0.0,
                'mean_details': None,
                'median_details': None,
                'T': T,
                'v_list': v_list
            }
        
        # For T=1, the median test might not be appropriate, so try it but fall back to old formula if needed
        try:
            result_dict = compute_median_v_test_epsilon_lb(
                m=m,
                r=r,
                T=T,
                v_list=v_list,
                delta=delta,
                ci_delta=ci_delta
            )
            
            eps_lb = result_dict.get("epsilon_lb", 0.0)
            # Check if the result is valid and finite
            if eps_lb is not None and np.isfinite(eps_lb) and eps_lb != float('inf'):
                print(f"ε lower bound (single run, median method): {eps_lb:.6f}")
                return {
                    'mean': None,  # Mean method not applicable for T=1
                    'median': float(eps_lb),
                    'mean_details': None,
                    'median_details': result_dict,
                    'T': T,
                    'v_list': v_list
                }
            
            # If median function returned inf or None, fall back to old formula
            print("Median function returned invalid result for T=1, using fallback formula")
        except Exception as e:
            raise Exception(f"Median function failed for T=1: {e}, using fallback formula")
        
        # Should not reach here, but if it does, return None for both
        return {
            'mean': None,
            'median': None,
            'mean_details': None,
            'median_details': None,
            'T': T,
            'v_list': v_list
        }
    
    # For multiple runs, use both avg and median functions
    else:
        results = {}
        
        # Compute using average (mean) function
        try:
            result_dict_avg = compute_avg_v_test_epsilon_lb(
                m=m,
                r=r,
                T=T,
                v_list=v_list,
                delta=delta,
                ci_delta=ci_delta,
                direction="ge",
                theta_max=50.0
            )
            eps_lb_avg = result_dict_avg.get("epsilon_lb", 0.0)
            if eps_lb_avg is not None and np.isfinite(eps_lb_avg) and eps_lb_avg != float('inf'):
                results['mean'] = float(eps_lb_avg)
                results['mean_details'] = result_dict_avg
            else:
                results['mean'] = None
                results['mean_details'] = None
        except Exception as e:
            print(f"Average function failed: {e}")
            results['mean'] = None
            results['mean_details'] = None
        
        # Compute using median function
        try:
            result_dict_median = compute_median_v_test_epsilon_lb(
                m=m,
                r=r,
                T=T,
                v_list=v_list,
                delta=delta,
                ci_delta=ci_delta
            )
            eps_lb_median = result_dict_median.get("epsilon_lb", 0.0)
            if eps_lb_median is not None and np.isfinite(eps_lb_median) and eps_lb_median != float('inf'):
                results['median'] = float(eps_lb_median)
                results['median_details'] = result_dict_median
            else:
                results['median'] = None
                results['median_details'] = None
        except Exception as e:
            print(f"Median function failed: {e}")
            results['median'] = None
            results['median_details'] = None
        
        # Display both results
        print(f"ε lower bound (from {T} runs):")
        if results['mean'] is not None:
            print(f"  Mean method: {results['mean']:.6f}")
        else:
            print(f"  Mean method: Failed or invalid")
        
        if results['median'] is not None:
            print(f"  Median method: {results['median']:.6f}")
        else:
            print(f"  Median method: Failed or invalid")
        
        # Return dictionary with both results
        return {
            'mean': results['mean'],
            'median': results['median'],
            'mean_details': results.get('mean_details'),
            'median_details': results.get('median_details'),
            'T': T,
            'v_list': v_list
        }

# =============================================================
# === Additional helper utilities (added by refactor pass) ===
# =============================================================

def evaluate_batches(eval_step, state, batch_iter, limit=None):
    """Run eval_step over up to `limit` batches; return (loss_arr, stats_list)."""
    losses, stats = [], []
    for i, (xb, yb) in enumerate(batch_iter):
        if limit is not None and i >= limit:
            break
        out = eval_step(state, xb, yb)
        # "value" from audit_utils.make_eval_step (phi or loss); "loss" for other eval_steps
        loss = float(out.get("value", out.get("loss", 0.0)))
        losses.append(loss)
        stats.append({"i": i, "n": len(yb), "loss": loss})
    return np.asarray(losses, dtype=float), stats


def select_indices_from_losses(loss_arr, k):
    """Convenience wrapper that builds minimal stats_json for selection."""
    from audit_utils import select_batches_by_likelihood
    stats_json = {"per_batch": [{"i": i, "loss": float(v)} for i, v in enumerate(loss_arr)]}
    return select_batches_by_likelihood(loss_arr, stats_json, k)

from src.models.model import ModelFactory
import yaml
import subprocess
import sys


def generate_forget_stats_if_missing(
    main_folder: str,
    unlearn_style: str,
    unlearn_itr: int,
    config_path: str = "configs/exp_cifar.yaml",
    dataset_type: str = "cifar_1",
    single_forget_subset: bool = False
):
    """
        Generate forget stats files (phi and loss) if missing by calling analyze_forget_batch_stats.py.
    The script produces both forget_stats_phi_*.json and forget_stats_loss_*.json. We run if
    either is missing so that callers using use_phi=True or use_phi=False both get the right file.
    
    Args:
        main_folder: Main folder containing runs
        unlearn_style: Unlearning style (epoch or step)
        unlearn_itr: Unlearning iteration number
        config_path: Path to config file
        dataset_type: Dataset type (default: cifar_1)
        single_forget_subset: Whether to use single_forget_subset mode
    """
    stats_file_phi = Path(main_folder) / f"forget_stats_phi_{unlearn_style}_{unlearn_itr}.json"
    stats_file_loss = Path(main_folder) / f"forget_stats_loss_{unlearn_style}_{unlearn_itr}.json"
    
    if stats_file_phi.exists() and stats_file_loss.exists():
        print(f"Forget stats files already exist: {stats_file_phi.name}, {stats_file_loss.name}")
        return
    
    missing = [f.name for f in (stats_file_phi, stats_file_loss) if not f.exists()]
    print(f"Forget stats file(s) not found: {missing}")
    print(f"Generating forget stats by calling analyze_forget_batch_stats.py...")
    
    script_path = Path(__file__).parent / "analyze_forget_batch_stats.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find analyze_forget_batch_stats.py at {script_path}")
    
    cmd = [
        sys.executable,
        str(script_path),
        "--runs_root", main_folder,
        "--unlearn_style", unlearn_style,
        "--unlearn_itr", str(unlearn_itr),
        "--config_path", config_path,
        "--dataset_type", dataset_type,
        "--skip-missing"
    ]

    if single_forget_subset:
        cmd.append("--single_forget_subset")
    
    print(f"Running: {' '.join(cmd)}")
    # Use the same env as parent. Do NOT set JAX_PLATFORMS=cpu: the checkpoints
    # were saved on GPU with sharding device_str="cuda:0". Orbax needs that
    # device in jax.local_devices() to restore; with CPU-only, to_jax_sharding()
    # fails and deserialization gets "sharding... Got None".
    # The subprocess runs to completion before the parent loads its model, so
    # GPU is not shared simultaneously. To use a specific GPU: set
    # CUDA_VISIBLE_DEVICES in the parent or FORGET_STATS_CUDA_DEVICES here.
    env = os.environ.copy()
    if os.environ.get("FORGET_STATS_CUDA_DEVICES") is not None:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["FORGET_STATS_CUDA_DEVICES"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to generate forget stats. Error:\n{result.stderr}\n"
            f"Command: {' '.join(cmd)}"
        )
    
    if not stats_file_phi.exists():
        raise FileNotFoundError(
            f"Stats file was not created after running script: {stats_file_phi}"
        )
    if not stats_file_loss.exists():
        raise FileNotFoundError(
            f"Stats file was not created after running script: {stats_file_loss}"
        )
    
    print(f"Successfully generated forget stats: {stats_file_phi.name}, {stats_file_loss.name}")


def generate_forget_stats_pointwise_if_missing(
    main_folder: str,
    unlearn_style: str,
    unlearn_itr: int,
    config_path: str = "configs/exp_cifar.yaml",
    dataset_type: str = "cifar_1",
    single_forget_subset: bool = False,
    trained_stats_only = False,
):
    """
    Generate pointwise forget stats (forget_stats_pointwise_phi_*.json and
    forget_stats_pointwise_loss_*.json) if missing by calling
    analyze_forget_stats_pointwise_batch.py. Run if either is missing so that
    callers using use_phi=True or use_phi=False get the right file.

    Args:
        main_folder: Main folder containing runs
        unlearn_style: Unlearning style (epoch or step)
        unlearn_itr: Unlearning iteration number
        config_path: Path to config file
        dataset_type: Dataset type (default: cifar_1)
        single_forget_subset: Whether to use single_forget_subset mode
    """

    if trained_stats_only:
        stats_loss = Path(main_folder) / f"forget_stats_pointwise_loss_trained_{unlearn_itr}.json"
        stats_phi = Path(main_folder) / f"forget_stats_pointwise_phi_trained_{unlearn_itr}.json"
    else:
        stats_phi = Path(main_folder) / f"forget_stats_pointwise_phi_{unlearn_style}_{unlearn_itr}.json"
        stats_loss = Path(main_folder) / f"forget_stats_pointwise_loss_{unlearn_style}_{unlearn_itr}.json"
    if stats_phi.exists() and stats_loss.exists():
        print(f"Pointwise forget stats already exist: {stats_phi.name}, {stats_loss.name}")
        return
    missing = [f.name for f in (stats_phi, stats_loss) if not f.exists()]
    print(f"Pointwise forget stats file(s) not found: {missing}")
    print("Generating by calling analyze_forget_stats_pointwise_batch.py...")
    script_path = Path(__file__).parent / "analyze_forget_stats_pointwise_batch.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find analyze_forget_stats_pointwise_batch.py at {script_path}")
    if not trained_stats_only:
        cmd = [
            sys.executable,
            str(script_path),
            "--runs_root", main_folder,
            "--unlearn_style", unlearn_style,
            "--unlearn_itr", str(unlearn_itr),
            "--config_path", config_path,
            "--dataset_type", dataset_type,
        ]
    else:
        cmd = [
            sys.executable,
            str(script_path),
            "--runs_root", main_folder,
            "--unlearn_style", unlearn_style,
            "--unlearn_itr", str(unlearn_itr),
            "--config_path", config_path,
            "--dataset_type", dataset_type,
            "--trained_stats_only",
        ]
    if single_forget_subset:
        cmd.append("--single_forget_subset")
    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    if os.environ.get("FORGET_STATS_CUDA_DEVICES") is not None:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["FORGET_STATS_CUDA_DEVICES"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to generate pointwise forget stats. Error:\n{result.stderr}\n"
            f"Command: {' '.join(cmd)}"
        )
    if not stats_phi.exists():
        raise FileNotFoundError(f"Pointwise stats was not created: {stats_phi}")
    if not stats_loss.exists():
        raise FileNotFoundError(f"Pointwise stats was not created: {stats_loss}")
    print(f"Successfully generated pointwise forget stats: {stats_phi.name}, {stats_loss.name}")


def pack_vs_result(selected_idx, N):
    """Bundle v, s, and index sets consistently."""
    from audit_utils import build_v_s_vectors
    selected_idx = sorted(set(int(i) for i in selected_idx))
    non_selected_idx = sorted(set(range(N)) - set(selected_idx))
    v, s = build_v_s_vectors(selected_idx, non_selected_idx, selected_idx, N)
    return {"v": v, "s": s, "selected_idx": selected_idx, "non_selected_idx": non_selected_idx}

def compute_eps_lower_bound(main_folder, wandb_run_id, unlearn_style, unlearn_itr, k, verbose=False, use_phi=True):    
    cfg_file = resolve_config_path(main_folder)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    model = ModelFactory.create_model(
        model_name = cfg["model"]["name"],
        num_classes = cfg["model"]["n_classes"],
    )


    run_dir = Path(main_folder) / "test_run" / wandb_run_id
    
    # Check if test_run directory exists, if not try direct path
    if not (Path(main_folder) / "test_run").exists():
        # Try direct path structure
        run_dir = Path(main_folder) / wandb_run_id
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Could not find run directory. Tried:\n"
                f"  1. {Path(main_folder) / 'test_run' / wandb_run_id}\n"
                f"  2. {Path(main_folder) / wandb_run_id}\n"
                f"Neither exists."
            )
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    load_path = os.path.abspath(Path(run_dir) /"ckpt" / f"unlearn_{unlearn_style}_{unlearn_itr}")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    
    vars_file = Path(run_dir) / "run_vars.json"
    
    if not vars_file.exists():
        raise FileNotFoundError(f"run_vars.json not found: {vars_file}")

    with vars_file.open("r") as f:
        run_vars = json.load(f)

    # 3) extract your parameters
    forget_fraction   = run_vars["forget_fraction"]

    # Check if single_forget_subset mode
    single_forget_subset = run_vars.get("single_forget_subset", False)

    # Determine dataset type from run_vars or use default
    cfg_file = resolve_config_path(main_folder, run_dir=run_dir)
    dataset_type = infer_dataset_type(
        run_vars,
        main_folder,
        run_dir=run_dir,
        config_path=cfg_file,
    )

    print ("Restoring from checkpoint", load_path)
    state = restore_orbax_state(load_path) ##, state_structure=state_structure)
    params = state["params"] if isinstance(state, dict) else state.params
    del state  # keep only params to reduce memory

    if verbose:
        print (
            "cache_root",
            get_cache_root(
                main_folder,
                dataset_type=dataset_type,
            ),
        )

    # load forget batches
    forget_batches = load_batches(
        get_cache_root(
            main_folder,
            dataset_type=dataset_type,
        ),
        "forget",
    )

    # Generate forget stats if missing. Use phi stats when use_phi=True, loss stats when use_phi=False
    # so that value_arr and the stats file are the same metric (phi vs phi, loss vs loss).
    stats_file = Path(main_folder) / f"forget_stats_{'phi' if use_phi else 'loss'}_{unlearn_style}_{unlearn_itr}.json"
    if not stats_file.exists():
        print(f"Forget stats file not found ({stats_file.name}), generating it...")
        generate_forget_stats_if_missing(
            main_folder=main_folder,
            unlearn_style=unlearn_style,
            unlearn_itr=unlearn_itr,
            config_path=cfg_file,
            dataset_type=dataset_type,
            single_forget_subset=single_forget_subset
        )

    # evaluate values (phi by default) and select indices
    eval_step = make_eval_step(model, use_phi=use_phi)
    value_arr = np.array([float(eval_step(params, (x, y))["value"]) for (x, y) in forget_batches])

    if verbose:
        metric_name = "phi" if use_phi else "loss"
        print (f"{metric_name}_arr", value_arr)
    ##print ("loss_arr_ref", loss_arr_ref, "loss_arr", loss_arr)
    selected_k_idx, non_selected_k_idx = select_batches_by_likelihood(value_arr, str(stats_file), k=k)

    if verbose:
        print ("selected_k_idx", selected_k_idx)
        print ("non_selected_k_idx", non_selected_k_idx)
    
    # build v,s and compute epsilon bound
    chosen_idx = np.load(run_dir / "chosen_forget_batches.npy")
    v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=len(forget_batches))
    result = compute_overlap_score(v, s)
    if verbose:
        print ("shape", s.shape, v.shape, "result", result)
    eps_lb = epsilon_lower_bound_from_vs(result, k=k, N=len(forget_batches), confidence_level=0.95)
    print("ε lower bound:", eps_lb)

    # Free GPU memory before returning
    del params, eval_step, model
    gc.collect()
    jax.clear_caches()
    return eps_lb


def _resolve_chosen_idx(run_dir: Path, cache_root: str, verbose: bool = False):
    """Resolve chosen_idx from chosen_forget_batches.npy or .json (single_forget_subset)."""
    chosen_file = run_dir / "chosen_forget_batches.npy"
    if chosen_file.exists():
        return np.load(chosen_file)
    chosen_file_json = run_dir / "chosen_forget_batches.json"
    if not chosen_file_json.exists():
        raise FileNotFoundError(f"No chosen_forget_batches.npy or .json in {run_dir}")
    with chosen_file_json.open("r") as f:
        chosen_data = json.load(f)
    if "chosen_subset_idx" not in chosen_data:
        raise ValueError(f"chosen_forget_batches.json missing chosen_subset_idx: {list(chosen_data.keys())}")
    forget_root = Path(cache_root) / "forget"
    if not forget_root.exists() or not forget_root.is_dir():
        raise FileNotFoundError(f"Cannot determine forget subset structure: {forget_root}")
    forget_subset_mapping = []
    forget_dirs = sorted([d for d in forget_root.iterdir() if d.is_dir() and d.name.startswith("forget_")])
    for forget_dir in forget_dirs:
        subset_name = forget_dir.name
        batch_files = sorted(forget_dir.glob("batch_*.pkl"))
        for local_idx in range(len(batch_files)):
            forget_subset_mapping.append((subset_name, local_idx))
    chosen_subset_name = f"forget_{chosen_data['chosen_subset_idx']}"
    chosen_idx = []
    for batch_idx, (subset_name, _) in enumerate(forget_subset_mapping):
        if subset_name == chosen_subset_name:
            chosen_idx.append(batch_idx)
    return np.array(chosen_idx)


def compute_eps_lower_bound_batch_pointwise(
    main_folder, wandb_run_id, unlearn_style, unlearn_itr, k, verbose=False, use_phi=True
):
    """
    Compute epsilon lower bound for a single run using batch-level pointwise stats
    and cumulative LLR. For batches with size > 1: per-point LLR under Gaussian
    (selected vs remaining), sum -> cumulative batch LLR; predict selected iff
    cumulative_llr > 0; top-k / bottom-k from batches sorted by cumulative LLR.

    Uses forget_stats_pointwise_phi_*.json or forget_stats_pointwise_loss_*.json.
    If the needed file is missing, calls generate_forget_stats_pointwise_if_missing
    (which runs analyze_forget_stats_pointwise_batch.py).

    Model loading, run discovery, chosen_idx resolution, and get_cache_root/load_batches
    match compute_eps_lower_bound and compute_eps_lower_bound_multiple_runs.
    """
    cfg_file = resolve_config_path(main_folder)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    model = ModelFactory.create_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["n_classes"],
    )

    run_dir = Path(main_folder) / "test_run" / wandb_run_id
    if not (Path(main_folder) / "test_run").exists():
        run_dir = Path(main_folder) / wandb_run_id
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Could not find run directory. Tried "
                f"{Path(main_folder) / 'test_run' / wandb_run_id} and {run_dir}"
            )
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    load_path = os.path.abspath(run_dir / "ckpt" / f"unlearn_{unlearn_style}_{unlearn_itr}")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    with (run_dir / "run_vars.json").open("r") as f:
        run_vars = json.load(f)
    dataset_type = infer_dataset_type(
        run_vars,
        main_folder,
        run_dir=run_dir,
        config_path=cfg_file,
    )
    single_forget_subset = run_vars.get("single_forget_subset", False)

    cache_root = get_cache_root(
        main_folder,
        dataset_type=dataset_type,
    )
    forget_batches = load_batches(cache_root, "forget")
    chosen_idx = _resolve_chosen_idx(run_dir, cache_root, verbose=verbose)

    stats_file = Path(main_folder) / f"forget_stats_pointwise_{'phi' if use_phi else 'loss'}_{unlearn_style}_{unlearn_itr}.json"
    if not stats_file.exists():
        print(f"Pointwise forget stats not found ({stats_file.name}), generating...")
        generate_forget_stats_pointwise_if_missing(
            main_folder=main_folder,
            unlearn_style=unlearn_style,
            unlearn_itr=unlearn_itr,
            config_path=cfg_file,
            dataset_type=dataset_type,
            single_forget_subset=single_forget_subset,
        )

    print("Restoring from checkpoint", load_path)
    state = restore_orbax_state(load_path)
    params = state["params"] if isinstance(state, dict) else state.params
    del state

    eval_step = make_eval_step_per_point(model)
    observed_map: Dict[str, float] = {}
    metric = "phi" if use_phi else "loss"
    for batch_idx, batch in enumerate(forget_batches):
        x, y = batch[0], batch[1]
        out = eval_step(params, (x, y))
        phi_arr = np.array(out["phi"])
        loss_arr = np.array(out["loss"])
        for i in range(phi_arr.shape[0]):
            key = f"batch_{batch_idx}_point_{i}"
            observed_map[key] = float(phi_arr[i]) if use_phi else float(loss_arr[i])
        del out  # release JAX device arrays promptly

    res = batch_level_predictions_from_pointwise_stats(stats_file, observed_map, metric=metric)
    del observed_map  # no longer needed; can be large (one float per point)
    batch_ids = res["batch_ids_sorted_by_llr"]
    if len(batch_ids) < 2 * k:
        raise ValueError(
            f"Need at least 2*k={2*k} batches for top-k/bottom-k; got {len(batch_ids)}. "
            "Use smaller k or ensure pointwise stats and forget batches match."
        )
    selected_k_idx = np.array(batch_ids[:k], dtype=int)
    non_selected_k_idx = np.array(batch_ids[-k:], dtype=int)

    N = len(forget_batches)
    v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=N)
    del forget_batches  # release forget data; we only need N from here on
    result = compute_overlap_score(v, s)
    if verbose:
        evals = evaluate_batch_llr_predictions(res["batch_cumulative_llrs"], set(int(i) for i in np.atleast_1d(chosen_idx).tolist()), k)
        print(f"Top-k acc: {evals['top_k_accuracy']:.4f}, bottom-k: {evals['bottom_k_accuracy']:.4f}, combined: {evals['combined_accuracy']:.4f}")
        print("shape", s.shape, v.shape, "result", result)
    del res  # done with res after verbose; release before eps computation
    eps_lb = epsilon_lower_bound_from_vs(result, k=k, N=N, confidence_level=0.95)
    print("ε lower bound (batch-pointwise):", eps_lb)

    del params, eval_step, model
    gc.collect()
    jax.clear_caches()
    return eps_lb


def compute_eps_lower_bound_multiple_runs(
    main_folder, 
    unlearn_style, 
    unlearn_itr, 
    k, 
    verbose=False,
    confidence_level=0.95,
    delta=0.0,
    use_phi=False,
    skip_failed_stats=True
):
    """
    Compute epsilon lower bound across multiple runs in test_run folder.
    
    Args:
        main_folder: Main folder containing test_run directory
        unlearn_style: Unlearning style (e.g., "epoch", "step")
        unlearn_itr: Unlearning iteration number
        k: number of top/bottom batches
        verbose: Whether to print detailed information
        confidence_level: Confidence level (default: 0.95)
        delta: Audit noise delta parameter (default: 0.0)
        use_phi: Whether to use phi (log-odds) instead of loss (default: True)
    
    Returns:
        epsilon lower bound (float) computed from all runs
    """
    test_run_dir = Path(main_folder) / "test_run"
    
    if not test_run_dir.exists():
        raise FileNotFoundError(f"test_run directory not found: {test_run_dir}")
    
    # Find all run directories, skipping ignored_runs
    run_dirs = sorted([d for d in test_run_dir.iterdir() if d.is_dir() and d.name != "ignored_runs"])

    if len(run_dirs) == 0:
        raise ValueError(f"No run directories found in {test_run_dir}")
    
    if verbose:
        print(f"Found {len(run_dirs)} runs in {test_run_dir}")
    
    # Load config once (assuming all runs use same config)
    cfg_file = resolve_config_path(main_folder, run_dir=run_dirs[0] if run_dirs else None)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    model = ModelFactory.create_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["n_classes"],
    )
    
    # Check for stats file and generate if missing. Use phi stats when use_phi=True, loss when use_phi=False
    # so that value_arr and the stats file are the same metric (phi vs phi, loss vs loss).
    stats_file = Path(main_folder) / f"forget_stats_{'phi' if use_phi else 'loss'}_{unlearn_style}_{unlearn_itr}.json"
    if not stats_file.exists():
        # Try to get parameters from first run
        first_run_dir = run_dirs[0] if run_dirs else None
        if first_run_dir:
            vars_file = first_run_dir / "run_vars.json"
            if vars_file.exists():
                with vars_file.open("r") as f:
                    first_run_vars = json.load(f)
                dataset_type = infer_dataset_type(
                    first_run_vars,
                    main_folder,
                    run_dir=first_run_dir,
                    config_path=cfg_file,
                )
                single_forget_subset = first_run_vars.get("single_forget_subset", False)
            else:
                dataset_type = "cifar_1"
                single_forget_subset = False
        else:
            dataset_type = "cifar_1"
            single_forget_subset = False

        print(f"Forget stats file not found ({stats_file.name}), generating it...")
        try:
            generate_forget_stats_if_missing(
                main_folder=main_folder,
                unlearn_style=unlearn_style,
                unlearn_itr=unlearn_itr,
                config_path=cfg_file,
                dataset_type=dataset_type,
                single_forget_subset=single_forget_subset
            )
        except (RuntimeError, FileNotFoundError) as e:
            if skip_failed_stats:
                print(f"[WARNING] Failed to generate stats for {stats_file.name}: {e}. Skipping.")
                return None
            else:
                raise
    
    eval_step = make_eval_step(model, use_phi=use_phi)
    v_list = []
    successful_runs = []
    N_batches = None  # Will be set from first successful run
    
    # Process each run
    for run_dir in run_dirs:
        wandb_run_id = run_dir.name
        
        try:
            # Load run variables
            vars_file = run_dir / "run_vars.json"
            if not vars_file.exists():
                if verbose:
                    print(f"Skipping {wandb_run_id}: no run_vars.json")
                continue
            
            with vars_file.open("r") as f:
                run_vars = json.load(f)
            
            dataset_type = infer_dataset_type(
                run_vars,
                main_folder,
                run_dir=run_dir,
                config_path=cfg_file,
            )
            
            # Load checkpoint
            load_path = os.path.abspath(run_dir / "ckpt" / f"unlearn_{unlearn_style}_{unlearn_itr}")
            if not os.path.exists(load_path):
                if verbose:
                    print(f"Skipping {wandb_run_id}: checkpoint not found at {load_path}")
                continue
            
            if verbose:
                print(f"Processing {wandb_run_id}...")
            
            state = restore_orbax_state(load_path)
            params = state["params"] if isinstance(state, dict) else state.params
            del state  # free checkpoint struct before next run

            # Load forget batches
            cache_root = get_cache_root(
                main_folder,
                dataset_type=dataset_type,
            )
            forget_batches = load_batches(cache_root, "forget")
            
            # Evaluate values (phi by default) and select indices
            value_arr = np.array([float(eval_step(params, (x, y))["value"]) for (x, y) in forget_batches])
            
            selected_k_idx, non_selected_k_idx = select_batches_by_likelihood(
                value_arr, str(stats_file), k=k
            )
            
            # Build v,s and compute overlap score
            chosen_file = run_dir / "chosen_forget_batches.npy"
            if not chosen_file.exists():
                # Try JSON format for synthetic data
                chosen_file_json = run_dir / "chosen_forget_batches.json"
                if chosen_file_json.exists():
                    with chosen_file_json.open("r") as f:
                        chosen_data = json.load(f)
                    if "chosen_subset_idx" in chosen_data:
                        # For synthetic data with single forget subset
                        # Need to map subset to batch indices
                        # Load forget batches to determine structure
                        cache_root_path = Path(cache_root)
                        forget_root = cache_root_path / "forget"
                        
                        if forget_root.exists() and forget_root.is_dir():
                            # Build mapping from batch index to subset
                            forget_subset_mapping = []
                            forget_dirs = sorted([d for d in forget_root.iterdir() 
                                                if d.is_dir() and d.name.startswith("forget_")])
                            
                            for forget_dir in forget_dirs:
                                subset_name = forget_dir.name
                                batch_files = sorted(forget_dir.glob("batch_*.pkl"))
                                for local_idx in range(len(batch_files)):
                                    forget_subset_mapping.append((subset_name, local_idx))
                            
                            # Find batches from chosen subset
                            chosen_subset_name = f"forget_{chosen_data['chosen_subset_idx']}"
                            chosen_idx = []
                            for batch_idx, (subset_name, local_idx) in enumerate(forget_subset_mapping):
                                if subset_name == chosen_subset_name:
                                    chosen_idx.append(batch_idx)
                            chosen_idx = np.array(chosen_idx)
                        else:
                            if verbose:
                                print(f"Skipping {wandb_run_id}: cannot determine forget subset structure")
                            continue
                    else:
                        if verbose:
                            print(f"Skipping {wandb_run_id}: unexpected chosen_forget_batches.json format")
                        continue
                else:
                    if verbose:
                        print(f"Skipping {wandb_run_id}: no chosen_forget_batches file found")
                    continue
            else:
                chosen_idx = np.load(chosen_file)
            
            # Store number of batches from first successful run
            if N_batches is None:
                N_batches = len(forget_batches)
            elif N_batches != len(forget_batches):
                if verbose:
                    print(f"Warning: {wandb_run_id} has {len(forget_batches)} batches, expected {N_batches}")
            
            v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=len(forget_batches))
            result = compute_overlap_score(v, s)
            
            v_list.append(result)
            successful_runs.append(wandb_run_id)

            if verbose:
                print(f"  {wandb_run_id}: overlap score = {result}")
            del params  # free before loading next run

        except Exception as e:
            if verbose:
                print(f"Error processing {wandb_run_id}: {e}")
            continue
    
    if len(v_list) == 0:
        raise ValueError("No successful runs found")
    
    if N_batches is None:
        raise ValueError("Could not determine number of batches")
    
    if verbose:
        print(f"\nSuccessfully processed {len(v_list)} runs: {successful_runs}")
        print(f"Overlap scores: {v_list}")
        print(f"Number of batches per run: {N_batches}")
    
    # Compute epsilon bound from all runs
    N = N_batches
    eps_lb = epsilon_lower_bound_from_vs(
        v_list, 
        k=k, 
        N=N, 
        confidence_level=confidence_level,
        delta=delta
    )
    
    print(f"ε lower bound (from {len(v_list)} runs): {eps_lb}")

    # Free GPU memory before returning (params from last run; may not exist if all runs skipped before load)
    try:
        del params
    except NameError:
        pass
    del model, eval_step
    gc.collect()
    jax.clear_caches()
    return eps_lb


def compute_eps_bounds_for_all_runs_batch_pointwise(
    main_folder,
    unlearn_style,
    unlearn_itr,
    k,
    verbose=False,
    confidence_level=0.95,
    delta=0.0,
    use_phi=True,
    trained_stats_only = False
):
    """
    Compute epsilon lower bound across multiple runs using batch-level pointwise
    stats and cumulative LLR. For each run: per-point phi/loss -> observed_map;
    batch_level_predictions_from_pointwise_stats -> cumulative LLR per batch;
    top-k / bottom-k from batches sorted by LLR; overlap with chosen_idx -> v.
    Collect v_list over runs, then epsilon_lower_bound_from_vs(v_list, ...).

    Uses forget_stats_pointwise_phi_*.json or forget_stats_pointwise_loss_*.json.
    If the needed file is missing, calls generate_forget_stats_pointwise_if_missing.

    Model loading, run discovery, chosen_idx (including single_forget_subset),
    get_cache_root, load_batches match compute_eps_lower_bound_multiple_runs.
    """
    test_run_dir = Path(main_folder) / "test_run"
    if not test_run_dir.exists():
        raise FileNotFoundError(f"test_run directory not found: {test_run_dir}")
    run_dirs = sorted([d for d in test_run_dir.iterdir() if d.is_dir() and d.name != "ignored_runs"])
    if not run_dirs:
        raise ValueError(f"No run directories in {test_run_dir}")

    cfg_file = resolve_config_path(main_folder, run_dir=run_dirs[0] if run_dirs else None)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    model = ModelFactory.create_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["n_classes"],
    )

    metric = "phi" if use_phi else "loss"
    if not trained_stats_only:
        stats_file = Path(main_folder) / f"forget_stats_pointwise_{metric}_{unlearn_style}_{unlearn_itr}.json"
    else:
        stats_file = Path(main_folder) / f"forget_stats_pointwise_{metric}_trained_{unlearn_itr}.json"
    if not stats_file.exists():
        dr = run_dirs[0]
        rv = json.loads((dr / "run_vars.json").read_text()) if (dr / "run_vars.json").exists() else {}
        dataset_type = infer_dataset_type(
            rv,
            main_folder,
            run_dir=dr,
            config_path=cfg_file,
        )
        single_forget_subset = rv.get("single_forget_subset", False)
        print(f"Pointwise forget stats not found ({stats_file.name}), generating...")
        generate_forget_stats_pointwise_if_missing(
            main_folder=main_folder,
            unlearn_style=unlearn_style,
            unlearn_itr=unlearn_itr,
            config_path=cfg_file,
            dataset_type=dataset_type,
            single_forget_subset=single_forget_subset,
            trained_stats_only=trained_stats_only
        )

    eval_step = make_eval_step_per_point(model)
    v_list = []
    run_ids = []
    failed_runs: List[Tuple[str, str]] = []
    N_batches = None

    for run_dir in run_dirs:
        wid = run_dir.name
        try:
            with (run_dir / "run_vars.json").open("r") as f:
                run_vars = json.load(f)
            dataset_type = run_vars.get("data_subfolder") or run_vars.get("dataset_type")
            if not dataset_type:
                raise ValueError(f"No data_subfolder or dataset_type found in run_vars.json for run {wid}")
            if not trained_stats_only:
                load_path = os.path.abspath(run_dir / "ckpt" / f"unlearn_{unlearn_style}_{unlearn_itr}")
            else:
                load_path = os.path.abspath(run_dir / "ckpt" / f"checkpoint_{unlearn_itr}")
            if not os.path.exists(load_path):
                if verbose:
                    print(f"Skipping {wid}: checkpoint not found at {load_path}")
                continue
            if verbose:
                print(f"Processing {wid}...")
            state = restore_orbax_state(load_path)
            params = state["params"] if isinstance(state, dict) else state.params
            del state

            cache_root = get_cache_root(
                main_folder,
                dataset_type=dataset_type,
            )
            forget_batches = load_batches(cache_root, "forget")
            chosen_idx = _resolve_chosen_idx(run_dir, cache_root, verbose=verbose)

            observed_map = {}
            for batch_idx, batch in enumerate(forget_batches):
                x, y = batch[0], batch[1]
                out = eval_step(params, (x, y))
                phi_arr = np.array(out["phi"])
                loss_arr = np.array(out["loss"])
                for i in range(phi_arr.shape[0]):
                    key = f"batch_{batch_idx}_point_{i}"
                    observed_map[key] = float(phi_arr[i]) if use_phi else float(loss_arr[i])
                del out  # release JAX device arrays promptly each batch

            res = batch_level_predictions_from_pointwise_stats(stats_file, observed_map, metric=metric)
            batch_ids = res["batch_ids_sorted_by_llr"]
            if len(batch_ids) < 2 * k:
                raise ValueError(f"Need at least 2*k={2*k} batches; got {len(batch_ids)}")
            selected_k_idx = np.array(batch_ids[:k], dtype=int)
            non_selected_k_idx = np.array(batch_ids[-k:], dtype=int)

            v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=len(forget_batches))
            ov = compute_overlap_score(v, s)
            v_list.append(ov)
            run_ids.append(wid)
            if N_batches is None:
                N_batches = len(forget_batches)
            if verbose:
                print(f"  {wid}: overlap = {ov}")
            # Free run-specific data before next run to avoid OOM over many runs
            del params, forget_batches, observed_map, res
            gc.collect()
        except Exception as e:
            failed_runs.append((wid, str(e)))
            if verbose:
                print(f"  {wid}: {e}")
            gc.collect()  # free any partial state from failed run
            continue

    if not v_list:
        raise ValueError("No successful runs; cannot compute epsilon lower bound.")
    if N_batches is None:
        raise ValueError("Could not determine number of batches.")

    eps_lb = epsilon_lower_bound_from_vs(
        v_list, k=k, N=N_batches, confidence_level=confidence_level, delta=delta
    )
    print(f"ε lower bound (batch-pointwise, {len(v_list)} runs): {eps_lb}")

    # Free all references that might remain (e.g. from last run on exception)
    try:
        del params
    except NameError:
        pass
    try:
        del forget_batches
    except NameError:
        pass
    try:
        del observed_map
    except NameError:
        pass
    try:
        del res
    except NameError:
        pass
    del model, eval_step
    gc.collect()
    jax.clear_caches()
    result = dict(eps_lb) if isinstance(eps_lb, dict) else {"eps_lb": eps_lb}
    result["run_ids"] = run_ids
    result["failed_runs"] = failed_runs
    return result