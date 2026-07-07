# audit_utils.py
"""
Utilities for auditing unlearning experiments:
 - Batch I/O and loaders
 - Rebatching
 - Eval step builders
 - Likelihood-based batch selection
 - Batch-level predictions from pointwise stats (cumulative LLR; see evaluate_predictions_from_phi_stats)
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

def make_eval_step(model):
    """Return eval_step(params, batch)."""
    @jax.jit
    def _eval_step(params, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        images, labels = batch
        logits = model.apply({"params": params}, images, train=False)
        one_hot = jax.nn.one_hot(labels, model.num_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        return {"loss": loss, "accuracy": acc}
    return _eval_step


# ---------------------------------------------------------------------
# Likelihood-based Selection
# ---------------------------------------------------------------------
def select_batches_by_likelihood(loss_arr: np.ndarray, stats_json: str, k: int, naming_style: str = "batch_x", eps = 1e-10):
    """
    Select top-k and bottom-k batches by likelihood ratio.
    
    Args:
        loss_arr: Array of losses for each batch
        stats_json: Path to JSON file with statistics
        k: Number of top/bottom batches to select
        naming_style: "numeric" for pure numbers (0, 1, 2, ...), 
                     "batch_x" for batch_X format (batch_0, batch_1, ...) - default
    """
    stats = json.loads(Path(stats_json).read_text())
    scores = []
    ##eps = 1e-8

    for key_str, entry in stats.items():
        if naming_style == "batch_x":
            # Handle batch_X format: extract X as global batch index
            if key_str.startswith("batch_"):
                try:
                    j = int(key_str.split("_")[1])  # Extract X from batch_X
                except (ValueError, IndexError):
                    continue
            else:
                continue
        elif naming_style == "numeric":
            # Handle pure numeric keys (backward compatibility)
            try:
                j = int(key_str)
            except ValueError:
                continue
        else:
            raise ValueError(f"Unknown naming_style: {naming_style}. Use 'batch_x' or 'numeric'.")

        # unpack
        sel, rem = entry["selected"], entry["remaining"]
        n_s, μ_s, σ2_s = sel.get("count"), sel.get("mean"), sel.get("var")
        n_r, μ_r, σ2_r = rem.get("count"), rem.get("mean"), rem.get("var")

        # skip entries with missing or zero counts (as notebook did)
        if n_s is None or n_r is None or n_s == 0 or n_r == 0:
            continue

        # if j >= len(loss_arr):   # extra safeguard
        #     continue

        x = loss_arr[j]

        # clip variances to avoid division by zero
        
        σ2_s = max(σ2_s, eps)
        σ2_r = max(σ2_r, eps)

        lp_s = -0.5 * (((x - μ_s) ** 2) / σ2_s)
        lp_r = -0.5 * (((x - μ_r) ** 2) / σ2_r)
        llr  = lp_s - lp_r

        scores.append((j, llr))

    if not scores:
        raise ValueError("No valid batches found in stats_json")

    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
    ##k = min(k, len(scores_sorted))  # handle case fewer than k valid entries

    print ("scores_sorted [:k]", scores_sorted[:k], "scores_sorted[-k:]", scores_sorted[-k:])

    selected_batches  = np.array([j for j, _ in scores_sorted[:k]], dtype=int)
    remaining_batches = np.array([j for j, _ in scores_sorted[-k:]], dtype=int)
    return selected_batches, remaining_batches


# ---------------------------------------------------------------------
# Batch-level predictions from pointwise stats (cumulative LLR)
# ---------------------------------------------------------------------
# Algorithm from evaluate_predictions_from_phi_stats.py (r2d): for each batch,
# sum per-point LLRs (log p_selected - log p_remaining under Gaussian) to get
# cumulative batch LLR; predict selected (forgotten) iff cumulative_llr > 0.
# Top-k / bottom-k: sort batches by cumulative LLR descending; top-k most
# likely selected, bottom-k most likely remaining.

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
    Compatible with "included"/"excluded" via selected_key="included", remaining_key="excluded".

    Args:
        point_stats: has selected_key and remaining_key, each with mean_phi, var_phi, mean_loss, var_loss
        metric: "phi" or "loss"
        obs_value: observed phi or loss for this point
        selected_key: key for selected/included stats (default "selected")
        remaining_key: key for remaining/excluded stats (default "remaining")
        eps: minimum variance to avoid div by zero

    Returns:
        LLR

    Raises:
        ValueError: if any of mean_phi, var_phi, mean_loss, var_loss is missing
            for selected_key or remaining_key
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
    Uses the same cumulative LLR algorithm as evaluate_predictions_from_phi_stats.py:
    - For each point in a batch: LLR = log p_selected(x) - log p_remaining(x) under Gaussian.
    - Cumulative LLR for batch = sum of point LLRs.
    - Predicted selected (forgotten) iff cumulative_llr > threshold.
    - Raises ValueError if pointwise_stats has no entry for a point in observed_map, or if
      any required stat (mean_phi, var_phi, mean_loss, var_loss) is missing for that point.

    Args:
        pointwise_stats: dict with "points" or the points dict; or path to JSON.
        observed_map: dict "batch_{b}_point_{p}" -> observed phi or loss (float).
        metric: "phi" or "loss"
        selected_key, remaining_key: keys in each point's stats (default "selected"/"remaining")
        eps: min variance for Gaussian log-pdf
        threshold: predict selected when cumulative_llr > threshold (default 0)

    Returns:
        {
            "batch_cumulative_llrs": { batch_idx (int): cumulative_llr (float) },
            "predictions": { batch_idx (int): True if predicted selected },
            "batch_ids_sorted_by_llr": [ batch_idx, ... ]  # descending by LLR
        }
    """
    if isinstance(pointwise_stats, (str, Path)):
        pointwise_stats = json.loads(Path(pointwise_stats).read_text())
    pts = pointwise_stats.get("points", pointwise_stats)
    if not isinstance(pts, dict):
        pts = {}

    # aggregate (batch_idx, point_idx) from point keys and observed_map
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
    accuracy, precision, recall. Mirrors evaluate_predictions_from_phi_stats.py.

    - Batches sorted by cumulative LLR descending. Top-k = most likely selected;
      bottom-k = most likely remaining.
    - top_k_accuracy: fraction of top-k that are in ground_truth_included.
    - bottom_k_accuracy: fraction of bottom-k that are NOT in ground_truth_included.
    - combined_accuracy: (top_k_correct + bottom_k_correct) / (2*k).
    - Confusion metrics: pred = (cumulative_llr > threshold).

    Args:
        batch_cumulative_llrs: batch_idx -> cumulative_llr (int or str keys)
        ground_truth_included: set of batch indices that were in the forget set
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

    gt_arr = np.array([b in gt for b in batch_ids], dtype=bool)
    top_k_correct = sum(1 for b in top_k_ids if b in gt)
    top_k_accuracy = top_k_correct / k_actual if k_actual else 0.0
    bottom_k_correct = sum(1 for b in bottom_k_ids if b not in gt)
    bottom_k_accuracy = bottom_k_correct / k_actual if k_actual else 0.0
    combined_accuracy = (top_k_correct + bottom_k_correct) / (2 * k_actual) if k_actual else 0.0

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

def get_cache_root(main_folder: str, file_batch_size: int, dataset: str = "cifar") -> str:
    return f"{main_folder}/data_split/{dataset}_{file_batch_size}"

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

def epsilon_lower_bound_from_vs(result, k, N, confidence_level=0.95, delta=1e-8, use_median=False):
    """
    Compute epsilon lower bound given overlap result, top-k size, and forget set size.
    Uses functions from cum_runs_eps_lab.py for more accurate computation.
    
    Args:
        result: overlap score (sum of max(0, v*s)) for a single run, or list of results for multiple runs
        k: number of top/bottom batches
        N: total number of forget batches (m)
        confidence_level: float (mapped to ci_delta = 1 - confidence_level)
        delta: Audit noise delta parameter (default: 1e-8)
        use_median: If True, use median-based test; otherwise use average-based test (default: False)
    
    Returns:
        epsilon lower bound (float) for single result, or dict with details for multiple results
    """
    try:
        from cum_runs_eps_lab import compute_avg_v_test_epsilon_lb, compute_median_v_test_epsilon_lb
    except ImportError:
        raise ImportError("cum_runs_eps_lab module is required. Make sure cum_runs_eps_lab.py is available.")
    
    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level
    
    # Handle both single result and list of results
    if isinstance(result, (list, np.ndarray)):
        v_list = result
        T = len(v_list)
    else:
        # Single run: T=1
        v_list = [result]
        T = 1
    
    if use_median:
        # Use median-based test
        result_dict = compute_median_v_test_epsilon_lb(
            m=m,
            r=r,
            T=T,
            v_list=v_list,
            delta=delta,
            ci_delta=ci_delta
        )
        eps_lb = result_dict.get("epsilon_lb", None)
    else:
        # Use average-based test
        result_dict = compute_avg_v_test_epsilon_lb(
            m=m,
            r=r,
            T=T,
            v_list=v_list,
            delta=delta,
            ci_delta=ci_delta,
            direction="ge",  # avg >= a
            theta_max=50.0
        )
        eps_lb = result_dict.get("epsilon_lb", None)
    
    # For single runs, return just the epsilon value for backward compatibility
    if T == 1:
        return eps_lb if eps_lb is not None else 0.0
    else:
        # For multiple runs, return the full result dict
        return result_dict

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
        loss = float(out.get("loss", 0.0))
        losses.append(loss)
        stats.append({"i": i, "n": len(yb), "loss": loss})
    return np.asarray(losses, dtype=float), stats


def select_indices_from_losses(loss_arr, k):
    """Convenience wrapper that builds minimal stats_json for selection."""
    from audit_utils import select_batches_by_likelihood
    stats_json = {"per_batch": [{"i": i, "loss": float(v)} for i, v in enumerate(loss_arr)]}
    return select_batches_by_likelihood(loss_arr, stats_json, k)


def pack_vs_result(selected_idx, N):
    """Bundle v, s, and index sets consistently."""
    from audit_utils import build_v_s_vectors
    selected_idx = sorted(set(int(i) for i in selected_idx))
    non_selected_idx = sorted(set(range(N)) - set(selected_idx))
    v, s = build_v_s_vectors(selected_idx, non_selected_idx, selected_idx, N)
    return {"v": v, "s": s, "selected_idx": selected_idx, "non_selected_idx": non_selected_idx}


# =============================================================
# === PyTorch utilities for audit computation ===
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from contextlib import contextmanager
import sys
import os

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def create_model(model_name: str, num_classes: int, filters: float = 1.0) -> nn.Module:
        """Create a PyTorch model based on model name."""
        try:
            from model import MLP, AllCNN, ResNet18, TinyNet, TinyNetCIFAR100
        except ImportError:
            raise ImportError("Could not import model classes. Make sure model.py is available.")
        
        if model_name == 'mlp':
            return MLP(num_layer=3, num_classes=num_classes, filters_percentage=filters)
        elif model_name in ['cnn', 'allcnn']:
            return AllCNN(num_classes=num_classes, filters_percentage=filters)
        elif model_name == 'resnet18':
            return ResNet18(num_classes=num_classes, filters_percentage=filters)
        elif model_name == 'tinynet':
            return TinyNet(num_classes=num_classes, filters_percentage=filters)
        elif model_name == 'tinynetcifar100':
            return TinyNetCIFAR100(num_classes=num_classes, filters_percentage=filters)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from: mlp, cnn, allcnn, resnet18, tinynet, tinynetcifar100"
            )

def find_model_file(run_dir: Path, use_trained: bool = False, models_subdir_name: str = 'models') -> Path:
        """Find the model file in a run directory (unlearned or trained)."""
        model_keyword = 'trained' if use_trained else 'unlearned'
        model_type = 'trained' if use_trained else 'unlearned'
        
        # Look in models subdirectory first
        models_subdir = run_dir / models_subdir_name
        if models_subdir.exists():
            model_files = [f for f in models_subdir.iterdir() if f.is_file() and model_keyword in f.name.lower() and f.suffix == '.pth']
            if model_files:
                if len(model_files) > 1:
                    print(f"Warning: Multiple {model_type} models found in {models_subdir}, using {model_files[0].name}")
                return model_files[0]
        
        # Fallback: look directly in run_dir
        model_files = [f for f in run_dir.iterdir() if f.is_file() and model_keyword in f.name.lower() and f.suffix == '.pth']
        if not model_files:
            raise FileNotFoundError(f"No {model_type} model found in {run_dir} or {run_dir}/{models_subdir_name}")
        if len(model_files) > 1:
            print(f"Warning: Multiple {model_type} models found in {run_dir}, using {model_files[0].name}")
        return model_files[0]

def _batch_to_xy_tensors(batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch (tuple, list, or dict) to (x, y) tensors on device. Handles unwrap, dict, NHWC->NCHW."""
    while isinstance(batch, (list, tuple)) and len(batch) == 1:
        batch = batch[0]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    elif isinstance(batch, dict):
        x = batch.get("data", batch.get("images", batch.get("x")))
        y = batch.get("labels", batch.get("targets", batch.get("y")))
        if x is None or y is None:
            raise ValueError(f"Batch dict missing 'data'/'x'/'images' or 'labels'/'y'/'targets': {list(batch.keys())}")
    else:
        extra = f", len={len(batch)}" if isinstance(batch, (list, tuple, dict)) else ""
        raise ValueError(f"Unexpected batch type: {type(batch)}{extra}")
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).long()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor([y] if not hasattr(y, "__len__") else list(y)).long()
    if y.dim() == 0:
        y = y.unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] in [1, 3]:
        x = x.permute(0, 3, 1, 2)
    return x.to(device), y.to(device)


@torch.no_grad()
def compute_phi_and_loss_per_point(model: nn.Module, batch: Batch, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point phi and loss for one batch. Returns (phi (N,), loss (N,))."""
    x, y = _batch_to_xy_tensors(batch, device)
    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    bidx = torch.arange(len(y), device=y.device)
    p = probs[bidx, y].double().clamp(min=1e-9, max=1 - 1e-9)
    phi = torch.log(p / (1.0 - p)).cpu().numpy()
    loss = F.nll_loss(F.log_softmax(logits, dim=1), y, reduction="none").cpu().numpy()
    return phi, loss


def _summarize_pointwise_values(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, var) for a list of values; (None, None) if empty."""
    if len(vals) == 0:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    arr = np.array(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.var(arr))


def _resolve_or_compute_pointwise_stats_path(
    main_folder,
    models_dir,
    model_name,
    num_classes,
    filters,
    use_trained_model,
    device,
    metric,
    run_dirs,
    data_folder=None,
    auto_compute_missing_stats=True,
    print_progress=True,
) -> Path:
    """Resolve pointwise stats path, optionally generating it when missing."""
    main_folder = Path(main_folder)
    suf = "_no_unlearn" if use_trained_model else ""
    preferred = main_folder / models_dir / f"forget_stats_pointwise_{metric}{suf}.json"
    fallback = main_folder / f"forget_stats_pointwise_{metric}{suf}.json"

    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    if not auto_compute_missing_stats:
        raise FileNotFoundError(
            f"Pointwise stats file not found: {preferred} (fallback checked: {fallback})"
        )

    if print_progress:
        print(
            f"Pointwise stats not found; generating {preferred.name} "
            f"from {len(run_dirs)} runs in {main_folder / models_dir}"
        )

    data_root = Path(data_folder) if data_folder is not None else (main_folder / "data")
    forget_dir = data_root / "forget"
    if not forget_dir.exists():
        raise FileNotFoundError(f"Forget directory not found: {forget_dir}")
    forget_batches = load_batches(str(data_root), "forget")
    num_batches = len(forget_batches)
    if num_batches == 0:
        raise ValueError(f"No forget batches found under {forget_dir}")

    device_obj = torch.device(device)
    model = create_model(model_name, num_classes, filters).to(device_obj)

    selected_vals: Dict[Tuple[int, int], List[float]] = {}
    remaining_vals: Dict[Tuple[int, int], List[float]] = {}
    valid_run_count = 0

    for run_dir in run_dirs:
        model_path = find_model_file(
            run_dir,
            use_trained=use_trained_model,
            models_subdir_name=models_dir,
        )

        forget_indices_path = None
        for f in run_dir.iterdir():
            if f.is_file() and 'forget' in f.name.lower() and 'indices' in f.name.lower() and f.suffix == '.npy':
                forget_indices_path = f
                break
        if forget_indices_path is None or not forget_indices_path.exists():
            raise FileNotFoundError(f"Forget indices file not found in {run_dir}")

        chosen_set = set(int(x) for x in np.atleast_1d(np.load(forget_indices_path)).tolist())
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        valid_run_count += 1

        for batch_idx, batch in enumerate(forget_batches):
            phi, loss = compute_phi_and_loss_per_point(model, batch, device_obj)
            vals = phi if metric == "phi" else loss
            is_selected = batch_idx in chosen_set
            for point_idx, obs_val in enumerate(vals):
                key = (batch_idx, point_idx)
                if is_selected:
                    selected_vals.setdefault(key, []).append(float(obs_val))
                else:
                    remaining_vals.setdefault(key, []).append(float(obs_val))

    if valid_run_count == 0:
        raise ValueError("No valid runs available to generate pointwise stats.")

    all_keys = set(selected_vals.keys()) | set(remaining_vals.keys())
    points: Dict[str, Dict] = {}
    mean_key = f"mean_{metric}"
    var_key = f"var_{metric}"

    for batch_idx, point_idx in sorted(all_keys):
        sel_list = selected_vals.get((batch_idx, point_idx), [])
        rem_list = remaining_vals.get((batch_idx, point_idx), [])
        sel_mean, sel_var = _summarize_pointwise_values(sel_list)
        rem_mean, rem_var = _summarize_pointwise_values(rem_list)
        pkey = f"batch_{batch_idx}_point_{point_idx}"
        points[pkey] = {
            "batch_idx": int(batch_idx),
            "point_idx": int(point_idx),
            "selected": {
                "count": len(sel_list),
                mean_key: sel_mean,
                var_key: sel_var,
            },
            "remaining": {
                "count": len(rem_list),
                mean_key: rem_mean,
                var_key: rem_var,
            },
        }

    preferred.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "compute_eps_bounds_for_all_runs_batch_pointwise",
        "metric": metric,
        "model_source": "trained" if use_trained_model else "unlearned",
        "runs_root": str(main_folder),
        "models_dir": models_dir,
        "num_runs": valid_run_count,
        "num_batches": num_batches,
        "num_points": len(points),
        "points": points,
    }
    preferred.write_text(json.dumps(payload, indent=2))

    if print_progress:
        print(f"Generated missing pointwise stats at: {preferred}")

    return preferred


def make_eval_step(model, device, metric='phi'):
    """Create evaluation function for PyTorch model.
    
    Args:
        model: PyTorch model
        device: Device to use
        metric: 'phi' for log-odds or 'loss' for cross-entropy loss
    """
    model.eval()
    
    def _eval(batch):
        """Evaluate model on a batch."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).long()
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        
        # Convert from NHWC to NCHW if needed
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            logits = model(x)
            
            if metric == 'phi':
                # Compute phi (log-odds): log(p/(1-p)) where p is probability for true class
                # Using same implementation as evaluate_predictions_from_phi_stats.py
                probs = torch.softmax(logits, dim=1)
                batch_indices = torch.arange(len(y), device=y.device)
                p = probs[batch_indices, y].double().clamp(min=1e-9, max=1 - 1e-9)
                phi_values = torch.log(p / (1.0 - p))
                value = phi_values.mean().item()
            else:  # metric == 'loss'
                # Compute cross-entropy loss
                value = F.cross_entropy(logits, y).item()
            
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y).float().mean().item()
        
        return {"loss": value, "accuracy": acc}
    
    return _eval

def compute_eps_lower_bound(
        main_folder, test_run_id, model_name, num_classes, filters, k, 
        stats_json_path=None, device='cpu', verbose=False, use_trained_model=False, 
        models_dir='models', metric='phi'
    ):
        """
        Compute epsilon lower bound for a test run using PyTorch models.
        
        Args:
            main_folder: Root directory (e.g., 'runs_cifar_1')
            test_run_id: Run identifier (e.g., 'run_0')
            model_name: Model architecture name (e.g., 'tinynet')
            num_classes: Number of classes
            filters: Filter percentage
            k: Number of top/bottom batches to select
            stats_json_path: Path to forget_stats.json (default: auto-determined based on flags)
            device: Device to use for evaluation
            verbose: Whether to print verbose output
            use_trained_model: If True, use trained model instead of unlearned model, and use forget_stats_{metric}_no_unlearn.json
            models_dir: Directory name for models (default: 'models')
            metric: 'phi' for log-odds or 'loss' for cross-entropy loss (default: 'phi')
        """
        main_folder = Path(main_folder)
        
        # Find run directory in {models_dir}/test_run folder
        run_dir = main_folder / models_dir / "test_run" / test_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        if verbose:
            print(f"Using run directory: {run_dir}")
        
        # Find model - look for trained or unlearned based on use_trained_model flag
        model_path = find_model_file(run_dir, use_trained=use_trained_model, models_subdir_name=models_dir)
        model_type = "trained" if use_trained_model else "unlearned"
        if verbose:
            print(f"Loading {model_type} model from: {model_path}")
        
        # Find forget indices - search for files with 'forget' and 'indices' in the name
        forget_indices_path = None
        for f in run_dir.iterdir():
            if f.is_file() and 'forget' in f.name.lower() and 'indices' in f.name.lower() and f.suffix == '.npy':
                forget_indices_path = f
                break
        
        if forget_indices_path is None or not forget_indices_path.exists():
            raise FileNotFoundError(f"Forget indices file not found for {test_run_id} in {run_dir}")
        
        chosen_idx = np.load(forget_indices_path)
        if verbose:
            print(f"Loaded forget indices: {len(chosen_idx)} indices")
            print("chosen_idx", list(chosen_idx))
        
        # Load stats file - use different file based on use_trained_model and metric flags
        if stats_json_path is None:
            if use_trained_model:
                # For trained models, use forget_stats_{metric}_no_unlearn.json
                stats_json_path = main_folder / models_dir / f"forget_stats_{metric}_no_unlearn.json"
                if not stats_json_path.exists():
                    # Fallback to main folder
                    stats_json_path = main_folder / f"forget_stats_{metric}_no_unlearn.json"
            else:
                # For unlearned models, use forget_stats_{metric}.json
                stats_json_path = main_folder / models_dir / f"forget_stats_{metric}.json"
                if not stats_json_path.exists():
                    # Fallback to main folder
                    stats_json_path = main_folder / f"forget_stats_{metric}.json"
        else:
            stats_json_path = Path(stats_json_path)
        
        if not stats_json_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_json_path}")
        
        if verbose:
            print(f"Using stats file: {stats_json_path}")
        
        # Create model and load weights
        device_obj = torch.device(device)
        model = create_model(model_name, num_classes, filters).to(device_obj)
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict)
        if verbose:
            print(f"Model loaded successfully")
        
        # Load forget batches from data directory
        data_dir = main_folder / "data"
        forget_dir = data_dir / "forget"
        if not forget_dir.exists():
            raise FileNotFoundError(f"Forget directory not found: {forget_dir}")
        
        forget_batches = load_batches(str(forget_dir), ".")
        if verbose:
            print(f"Loaded {len(forget_batches)} forget batches")
        
        # Evaluate losses or phi
        eval_step = make_eval_step(model, device_obj, metric=metric)
        metric_name = metric
        if verbose:
            print(f"Computing {metric_name} for {len(forget_batches)} batches...")
        
        loss_arr = np.array([float(eval_step(batch)["loss"]) for batch in forget_batches])
        
        if verbose:
            print(f"{metric_name}_arr", loss_arr)
        
        # Select batches by likelihood
        # Suppress stdout from audit_utils if not verbose
        if verbose:
            selected_k_idx, non_selected_k_idx = select_batches_by_likelihood(
                loss_arr, str(stats_json_path), k=k, naming_style="batch_x"
            )
            print("selected_k_idx", selected_k_idx)
            print("non_selected_k_idx", non_selected_k_idx)
        else:
            with suppress_stdout():
                selected_k_idx, non_selected_k_idx = select_batches_by_likelihood(
                    loss_arr, str(stats_json_path), k=k, naming_style="batch_x"
                )
        
        # Build v, s vectors and compute epsilon bound
        if verbose:
            v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=len(forget_batches))
            result = compute_overlap_score(v, s)
            print("shape", s.shape, v.shape, "result", result)
            eps_lb = epsilon_lower_bound_from_vs(result, k=k, N=len(forget_batches), confidence_level=0.95)
        else:
            with suppress_stdout():
                v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=len(forget_batches))
                result = compute_overlap_score(v, s)
                eps_lb = epsilon_lower_bound_from_vs(result, k=k, N=len(forget_batches), confidence_level=0.95)
        
        if verbose:
            print("ε lower bound:", eps_lb)
        return eps_lb


def compute_eps_lower_bound_batch_pointwise(
        main_folder, test_run_id, model_name, num_classes, filters, k,
        stats_json_path=None, device='cpu', verbose=False, use_trained_model=False,
    models_dir='models', metric='phi', data_folder=None
    ):
        """
        Compute epsilon lower bound for a test run using batch-level pointwise stats
        and cumulative LLR. Uses forget_stats_pointwise_{metric}.json (or _no_unlearn),
        loads the model, builds observed_map from per-point phi/loss on forget batches,
        runs batch_level_predictions_from_pointwise_stats, then build_v_s_vectors +
        compute_overlap_score + epsilon_lower_bound_from_vs (same as compute_eps_lower_bound).

        Args:
            main_folder: Root directory (e.g. 'runs_cifar_bs_10')
            test_run_id: Run identifier (e.g. 'run_0')
            model_name: Model architecture (e.g. 'tinynet')
            num_classes: Number of classes
            filters: Filter percentage
            k: Number of top/bottom batches
            stats_json_path: Path to pointwise stats JSON (default: forget_stats_pointwise_{metric}.json or _no_unlearn)
            device: Device for evaluation
            verbose: Print progress
            use_trained_model: If True, use trained model and forget_stats_pointwise_{metric}_no_unlearn.json
            models_dir: Directory for runs and stats (e.g. 'models_sgd_no_shuffle')
            metric: 'phi' or 'loss'
            data_folder: Optional path to data root containing "forget" batches.
                If None, defaults to {main_folder}/data.

        Returns:
            epsilon lower bound (float)
        """
        main_folder = Path(main_folder)
        run_dir = main_folder / models_dir / "test_run" / test_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        if verbose:
            print(f"Using run directory: {run_dir}")

        model_path = find_model_file(run_dir, use_trained=use_trained_model, models_subdir_name=models_dir)
        if verbose:
            print(f"Loading {'trained' if use_trained_model else 'unlearned'} model from: {model_path}")

        forget_indices_path = None
        for f in run_dir.iterdir():
            if f.is_file() and 'forget' in f.name.lower() and 'indices' in f.name.lower() and f.suffix == '.npy':
                forget_indices_path = f
                break
        if forget_indices_path is None or not forget_indices_path.exists():
            raise FileNotFoundError(f"Forget indices file not found for {test_run_id} in {run_dir}")
        chosen_idx = np.load(forget_indices_path)
        chosen_set = set(int(x) for x in np.atleast_1d(chosen_idx).tolist())
        if verbose:
            print(f"Loaded forget indices: {len(chosen_set)}")

        if stats_json_path is None:
            suf = "_no_unlearn" if use_trained_model else ""
            stats_json_path = main_folder / models_dir / f"forget_stats_pointwise_{metric}{suf}.json"
            if not stats_json_path.exists():
                stats_json_path = main_folder / f"forget_stats_pointwise_{metric}{suf}.json"
        else:
            stats_json_path = Path(stats_json_path)
        if not stats_json_path.exists():
            raise FileNotFoundError(f"Pointwise stats file not found: {stats_json_path}")
        if verbose:
            print(f"Using pointwise stats: {stats_json_path}")

        device_obj = torch.device(device)
        model = create_model(model_name, num_classes, filters).to(device_obj)
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict)

        data_dir = Path(data_folder) if data_folder is not None else (main_folder / "data")
        forget_dir = data_dir / "forget"
        if not forget_dir.exists():
            raise FileNotFoundError(f"Forget directory not found: {forget_dir}")
        forget_batches = load_batches(str(forget_dir), ".")
        if verbose:
            print(f"Loaded {len(forget_batches)} forget batches")
            print(f"Building observed_map from per-point {metric}...")

        observed_map: Dict[str, float] = {}
        for batch_idx, batch in enumerate(forget_batches):
            phi, loss = compute_phi_and_loss_per_point(model, batch, device_obj)
            n = len(phi)
            for i in range(n):
                key = f"batch_{batch_idx}_point_{i}"
                observed_map[key] = float(phi[i]) if metric == "phi" else float(loss[i])

        res = batch_level_predictions_from_pointwise_stats(
            stats_json_path, observed_map, metric=metric
        )
        batch_ids = res["batch_ids_sorted_by_llr"]
        if len(batch_ids) < 2 * k:
            raise ValueError(
                f"Need at least 2*k={2*k} batches for top-k and bottom-k; got {len(batch_ids)}. "
                "Use a smaller k or ensure pointwise stats and forget batches match."
            )
        selected_k_idx = batch_ids[:k]
        non_selected_k_idx = batch_ids[-k:]

        v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, np.atleast_1d(chosen_idx), N=len(forget_batches))
        overlap = compute_overlap_score(v, s)
        eps_lb = epsilon_lower_bound_from_vs(overlap, k=k, N=len(forget_batches), confidence_level=0.95)

        if verbose:
            evals = evaluate_batch_llr_predictions(res["batch_cumulative_llrs"], chosen_set, k)
            print(f"Top-k acc: {evals['top_k_accuracy']:.4f}, bottom-k: {evals['bottom_k_accuracy']:.4f}, combined: {evals['combined_accuracy']:.4f}")
            print("ε lower bound:", eps_lb)
        return eps_lb


def _compute_v_for_single_run(
        main_folder, test_run_id, model_name, num_classes, filters, k,
        stats_json_path=None, device='cpu', verbose=False, use_trained_model=False,
        models_dir='models', metric='phi', data_folder=None
    ):
        """
        Helper function to compute v (overlap score) for a single run.
        Returns (v, m) where v is the overlap score and m is the number of forget batches.
        """
        main_folder = Path(main_folder)
        
        # Find run directory
        run_dir = main_folder / models_dir / "test_run" / test_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Find model
        model_path = find_model_file(run_dir, use_trained=use_trained_model, models_subdir_name=models_dir)
        
        # Find forget indices
        forget_indices_path = None
        for f in run_dir.iterdir():
            if f.is_file() and 'forget' in f.name.lower() and 'indices' in f.name.lower() and f.suffix == '.npy':
                forget_indices_path = f
                break
        
        if forget_indices_path is None or not forget_indices_path.exists():
            raise FileNotFoundError(f"Forget indices file not found for {test_run_id} in {run_dir}")
        
        chosen_idx = np.load(forget_indices_path)
        
        # Load stats file
        if stats_json_path is None:
            if use_trained_model:
                stats_json_path = main_folder / models_dir / f"forget_stats_{metric}_no_unlearn.json"
                if not stats_json_path.exists():
                    stats_json_path = main_folder / f"forget_stats_{metric}_no_unlearn.json"
            else:
                stats_json_path = main_folder / models_dir / f"forget_stats_{metric}.json"
                if not stats_json_path.exists():
                    stats_json_path = main_folder / f"forget_stats_{metric}.json"
        else:
            stats_json_path = Path(stats_json_path)
        
        if not stats_json_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_json_path}")
        
        # Create model and load weights
        device_obj = torch.device(device)
        model = create_model(model_name, num_classes, filters).to(device_obj)
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict)
        
        # Load forget batches
        data_dir = Path(data_folder) if data_folder is not None else (main_folder / "data")
        forget_dir = data_dir / "forget"
        if not forget_dir.exists():
            raise FileNotFoundError(f"Forget directory not found: {forget_dir}")

        forget_batches = load_batches(str(forget_dir), ".")
        m = len(forget_batches)
        
        # Evaluate losses or phi
        eval_step = make_eval_step(model, device_obj, metric=metric)
        loss_arr = np.array([float(eval_step(batch)["loss"]) for batch in forget_batches])
        
        # Select batches by likelihood
        if verbose:
            selected_k_idx, non_selected_k_idx = select_batches_by_likelihood(
                loss_arr, str(stats_json_path), k=k, naming_style="batch_x"
            )
        else:
            with suppress_stdout():
                selected_k_idx, non_selected_k_idx = select_batches_by_likelihood(
                    loss_arr, str(stats_json_path), k=k, naming_style="batch_x"
                )
        
        # Build v, s vectors and compute overlap score
        v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, chosen_idx, N=m)
        result = compute_overlap_score(v, s)
        
        return result, m


def _compute_v_for_single_run_batch_pointwise(
        main_folder, test_run_id, model_name, num_classes, filters, k,
        stats_json_path=None, device='cpu', verbose=False, use_trained_model=False,
    models_dir='models', metric='phi', data_folder=None
    ):
        """
        Compute v (overlap score) for one test run using batch-level pointwise stats
        and cumulative LLR: per-batch cumulative LLR → top-k predicted selected,
        bottom-k predicted remaining → build_v_s_vectors + compute_overlap_score.
        Returns (v, m).
        """
        main_folder = Path(main_folder)
        run_dir = main_folder / models_dir / "test_run" / test_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        model_path = find_model_file(run_dir, use_trained=use_trained_model, models_subdir_name=models_dir)

        forget_indices_path = None
        for f in run_dir.iterdir():
            if f.is_file() and 'forget' in f.name.lower() and 'indices' in f.name.lower() and f.suffix == '.npy':
                forget_indices_path = f
                break
        if forget_indices_path is None or not forget_indices_path.exists():
            raise FileNotFoundError(f"Forget indices file not found for {test_run_id} in {run_dir}")
        chosen_idx = np.load(forget_indices_path)

        if stats_json_path is None:
            suf = "_no_unlearn" if use_trained_model else ""
            stats_json_path = main_folder / models_dir / f"forget_stats_pointwise_{metric}{suf}.json"
            if not stats_json_path.exists():
                stats_json_path = main_folder / f"forget_stats_pointwise_{metric}{suf}.json"
        else:
            stats_json_path = Path(stats_json_path)
        if not stats_json_path.exists():
            raise FileNotFoundError(f"Pointwise stats file not found: {stats_json_path}")

        device_obj = torch.device(device)
        model = create_model(model_name, num_classes, filters).to(device_obj)
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict)

        data_root = Path(data_folder) if data_folder is not None else (main_folder / "data")
        forget_dir = data_root / "forget"
        if not forget_dir.exists():
            raise FileNotFoundError(f"Forget directory not found: {forget_dir}")
        forget_batches = load_batches(str(data_root), "forget")
        m = len(forget_batches)

        observed_map = {}
        for batch_idx, batch in enumerate(forget_batches):
            phi, loss = compute_phi_and_loss_per_point(model, batch, device_obj)
            for i in range(len(phi)):
                key = f"batch_{batch_idx}_point_{i}"
                observed_map[key] = float(phi[i]) if metric == "phi" else float(loss[i])

        res = batch_level_predictions_from_pointwise_stats(stats_json_path, observed_map, metric=metric)
        batch_ids = res["batch_ids_sorted_by_llr"]
        if len(batch_ids) < 2 * k:
            raise ValueError(
                f"Need at least 2*k={2*k} batches for top-k and bottom-k; got {len(batch_ids)}. "
                "Use a smaller k or ensure pointwise stats and forget batches match."
            )
        selected_k_idx = batch_ids[:k]
        non_selected_k_idx = batch_ids[-k:]

        v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, np.atleast_1d(chosen_idx), N=m)
        return compute_overlap_score(v, s), m


def compute_eps_bounds_for_all_runs(
        main_folder,
        model_name,
        num_classes,
        filters,
        k,
        models_dir="models",
        use_trained_model=False,
        device='cpu',
        verbose=False,
        print_progress=True,
        print_summary=True,
        delta=1e-8,
        confidence_level=0.95,
        metric='phi'
    ):
        """
        Iterate over all test_run_id directories, collect v_list (overlap scores), 
        and compute both average-based and median-based epsilon lower bounds using functions from cum_runs_eps_lab.py.
        
        Args:
            main_folder: Root directory (e.g., 'runs_cifar_1')
            model_name: Model architecture name (e.g., 'tinynet')
            num_classes: Number of classes
            filters: Filter percentage
            k: Number of top/bottom batches to select
            models_dir: Directory name for models (default: 'models')
            use_trained_model: If True, use trained model instead of unlearned model
            device: Device to use for evaluation
            verbose: Whether to print verbose output for each run
            print_progress: Whether to print progress messages during processing
            print_summary: Whether to print summary statistics at the end
            delta: Audit noise delta parameter (default: 1e-8)
            confidence_level: Confidence level (default: 0.95, mapped to ci_delta = 1 - confidence_level)
            metric: 'phi' for log-odds or 'loss' for cross-entropy loss (default: 'phi')
            
        Returns:
            dict with keys:
                - 'eps_lb_avg': average-based epsilon lower bound (computed from all runs)
                - 'eps_lb_median': median-based epsilon lower bound (computed from all runs)
                - 'v_list': list of overlap scores (one per run)
                - 'run_ids': list of run IDs that succeeded
                - 'failed_runs': list of (run_id, error_message) tuples for failed runs
                - 'T': number of successful runs
                - 'm': number of forget batches
                - 'r': 2*k
                - 'result_dict_avg': full result dictionary from average-based computation
                - 'result_dict_median': full result dictionary from median-based computation
        """
        # Read data_folder from config.json in models_dir
        data_folder = None
        config_path = Path(main_folder) / models_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            data_folder = cfg.get("data_dir")
            if data_folder is not None and print_progress:
                print(f"Using data_folder from config.json: {data_folder}")

        # Find all test_run directories
        test_run_base = Path(main_folder) / models_dir / "test_run"
        if not test_run_base.exists():
            raise FileNotFoundError(f"Test run directory not found: {test_run_base}")

        # Get all run directories
        run_dirs = sorted([d for d in test_run_base.iterdir() if d.is_dir() and d.name.startswith("run_")])
        total_runs = len(run_dirs)
        
        if print_progress:
            if HAS_TQDM:
                pbar = tqdm(run_dirs, desc="Collecting v values", unit="run")
            else:
                print(f"Processing {total_runs} runs to collect v values...")
                pbar = run_dirs
        else:
            pbar = run_dirs

        # Collect v_list (overlap scores) from all runs
        v_list = []
        run_ids = []
        failed_runs = []
        m = None  # Will be set from first successful run

        for run_dir in pbar:
            test_run_id = run_dir.name
            
            # Update progress bar description if using tqdm
            if print_progress and HAS_TQDM:
                pbar.set_description(f"Processing {test_run_id}")
            
            try:
                v, m_run = _compute_v_for_single_run(
                    main_folder=main_folder,
                    test_run_id=test_run_id,
                    model_name=model_name,
                    num_classes=num_classes,
                    filters=filters,
                    k=k,
                    device=device,
                    verbose=verbose,
                    use_trained_model=use_trained_model,
                    models_dir=models_dir,
                    metric=metric,
                    data_folder=data_folder,
                )
                v_list.append(v)
                run_ids.append(test_run_id)
                if m is None:
                    m = m_run  # Set m from first successful run
                elif m != m_run:
                    if verbose:
                        print(f"Warning: Inconsistent number of forget batches. Expected {m}, got {m_run} for {test_run_id}")
            except Exception as e:
                failed_runs.append((test_run_id, str(e)))
                if print_progress and HAS_TQDM:
                    pbar.write(f"✗ {test_run_id}: Failed - {str(e)}")
                continue
        
        if print_progress and HAS_TQDM:
            pbar.close()

        if len(v_list) == 0:
            raise ValueError("No successful runs found. Cannot compute epsilon lower bound.")

        T = len(v_list)
        r = 2 * k
        
        if print_progress:
            print(f"\nCollected {T} v values from {T} runs")
            print(f"v_list: {v_list}")
            print(f"m (number of forget batches): {m}")
            print(f"r (2*k): {r}")
            print(f"T (number of runs): {T}")

        # Compute both average-based and median-based epsilon lower bounds
        result_dict_avg = epsilon_lower_bound_from_vs(
            result=v_list,
            k=k,
            N=m,
            confidence_level=confidence_level,
            delta=delta,
            use_median=False
        )

        result_dict_median = epsilon_lower_bound_from_vs(
            result=v_list,
            k=k,
            N=m,
            confidence_level=confidence_level,
            delta=delta,
            use_median=True
        )

        # Extract epsilon_lb from result_dicts
        if isinstance(result_dict_avg, dict):
            eps_lb_avg = result_dict_avg.get("epsilon_lb", None)
        else:
            eps_lb_avg = result_dict_avg

        if isinstance(result_dict_median, dict):
            eps_lb_median = result_dict_median.get("epsilon_lb", None)
        else:
            eps_lb_median = result_dict_median

        # Print summary if requested
        if print_summary:
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"\nTotal runs processed: {T}")
            print(f"Failed runs: {len(failed_runs)}")
            print(f"\nEpsilon Lower Bounds (computed from all runs):")
            print(f"  Average-based: {eps_lb_avg}")
            print(f"  Median-based:  {eps_lb_median}")
            print(f"\nv_list: {v_list}")
            
            if isinstance(result_dict_avg, dict):
                print(f"\nAverage-based computation details:")
                for key, value in result_dict_avg.items():
                    if key != "epsilon_lb":
                        print(f"  {key}: {value}")
            
            if isinstance(result_dict_median, dict):
                print(f"\nMedian-based computation details:")
                for key, value in result_dict_median.items():
                    if key != "epsilon_lb":
                        print(f"  {key}: {value}")

            if len(failed_runs) > 0:
                print(f"\n{'='*60}")
                print("FAILED RUNS")
                print(f"{'='*60}")
                for run_id, error in failed_runs:
                    print(f"  {run_id}: {error}")

        return {
            'eps_lb_avg': eps_lb_avg,
            'eps_lb_median': eps_lb_median,
            'v_list': v_list,
            'run_ids': run_ids,
            'failed_runs': failed_runs,
            'T': T,
            'm': m,
            'r': r,
            'result_dict_avg': result_dict_avg,
            'result_dict_median': result_dict_median
        }


def compute_eps_bounds_for_all_runs_batch_pointwise(
        main_folder,
        model_name,
        num_classes,
        filters,
        k,
        models_dir="models",
        use_trained_model=False,
        device='cpu',
        verbose=False,
        print_progress=True,
        print_summary=True,
        delta=1e-8,
        confidence_level=0.95,
        metric='loss',
        auto_compute_missing_stats=True,
    ):
        """
        Like compute_eps_bounds_for_all_runs but uses batch-level pointwise stats and
        cumulative LLR: for each run, predict top-k / bottom-k from per-batch cumulative
        LLR (sum of per-point LLRs), compute overlap v with ground truth, collect v_list,
        then compute average-based and median-based epsilon lower bounds.

        Uses forget_stats_pointwise_{metric}.json (or _no_unlearn). Batch-level prediction:
        for each batch, cumulative LLR = sum over points of (log p_selected - log p_remaining);
        higher LLR → more likely forgotten; top-k = predicted forgotten, bottom-k = predicted
        remaining; overlap with actual forget set gives v.

        Args:
            main_folder: Root (e.g. 'runs_cifar_bs_1')
            model_name, num_classes, filters, k: model and audit params
            models_dir: e.g. 'models_adam_no_shuffle_fixed_lr'
            use_trained_model: use trained model and forget_stats_pointwise_{metric}_no_unlearn.json
            device, verbose, print_progress, print_summary, delta, confidence_level
            metric: 'phi' or 'loss'
            auto_compute_missing_stats: if True, generate missing pointwise stats json
                at {main_folder}/{models_dir}/forget_stats_pointwise_{metric}[...].json

        Returns:
            dict: eps_lb_avg, eps_lb_median, v_list, run_ids, failed_runs, T, m, r,
                  result_dict_avg, result_dict_median
        """
        # Read data_folder from config.json in models_dir
        data_folder = None
        config_path = Path(main_folder) / models_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            data_folder = cfg.get("data_dir")
            if data_folder is not None and print_progress:
                print(f"Using data_folder from config.json: {data_folder}")

        test_run_base = Path(main_folder) / models_dir / "test_run"
        if not test_run_base.exists():
            raise FileNotFoundError(f"Test run directory not found: {test_run_base}")

        run_dirs = sorted([d for d in test_run_base.iterdir() if d.is_dir() and d.name.startswith("run_")])
        total_runs = len(run_dirs)

        # Use the population runs from main models_dir (not test_run) to build pointwise stats
        models_base = Path(main_folder) / models_dir
        stat_run_dirs = sorted([d for d in models_base.iterdir() if d.is_dir() and d.name.startswith("run_")])

        stats_json_path = _resolve_or_compute_pointwise_stats_path(
            main_folder=main_folder,
            models_dir=models_dir,
            model_name=model_name,
            num_classes=num_classes,
            filters=filters,
            use_trained_model=use_trained_model,
            device=device,
            metric=metric,
            run_dirs=stat_run_dirs,
            data_folder=data_folder,
            auto_compute_missing_stats=auto_compute_missing_stats,
            print_progress=print_progress,
        )

        if print_progress:
            if HAS_TQDM:
                pbar = tqdm(run_dirs, desc="Collecting v (batch-pointwise)", unit="run")
            else:
                print(f"Processing {total_runs} runs (batch-level pointwise + cumulative LLR)...")
                pbar = run_dirs
        else:
            pbar = run_dirs

        v_list = []
        run_ids = []
        failed_runs = []
        m = None

        for run_dir in pbar:
            test_run_id = run_dir.name
            if print_progress and HAS_TQDM:
                pbar.set_description(f"Processing {test_run_id}")
            try:
                v, m_run = _compute_v_for_single_run_batch_pointwise(
                    main_folder=main_folder,
                    test_run_id=test_run_id,
                    model_name=model_name,
                    num_classes=num_classes,
                    filters=filters,
                    k=k,
                    device=device,
                    verbose=verbose,
                    use_trained_model=use_trained_model,
                    models_dir=models_dir,
                    metric=metric,
                    stats_json_path=stats_json_path,
                    data_folder=data_folder,
                )
                v_list.append(v)
                run_ids.append(test_run_id)
                if m is None:
                    m = m_run
                elif m != m_run and verbose:
                    print(f"Warning: Inconsistent forget batch count for {test_run_id}")
            except Exception as e:
                failed_runs.append((test_run_id, str(e)))
                if print_progress and HAS_TQDM:
                    pbar.write(f"✗ {test_run_id}: {str(e)}")

        if print_progress and HAS_TQDM:
            pbar.close()

        if len(v_list) == 0:
            raise ValueError("No successful runs. Cannot compute epsilon lower bound.")

        T = len(v_list)
        r = 2 * k
        if print_progress:
            print(f"\nCollected {T} v values; m={m}, r={r}")

        result_dict_avg = epsilon_lower_bound_from_vs(
            result=v_list, k=k, N=m, confidence_level=confidence_level, delta=delta, use_median=False
        )
        result_dict_median = epsilon_lower_bound_from_vs(
            result=v_list, k=k, N=m, confidence_level=confidence_level, delta=delta, use_median=True
        )
        eps_lb_avg = result_dict_avg.get("epsilon_lb", None) if isinstance(result_dict_avg, dict) else result_dict_avg
        eps_lb_median = result_dict_median.get("epsilon_lb", None) if isinstance(result_dict_median, dict) else result_dict_median

        if print_summary:
            print(f"\n{'='*60}\nRESULTS SUMMARY (batch-level pointwise + cumulative LLR)\n{'='*60}")
            print(f"Runs: {T}, failed: {len(failed_runs)}")
            print(f"Epsilon lower bounds — average: {eps_lb_avg}, median: {eps_lb_median}")
            print(f"v_list: {v_list}")
            if failed_runs:
                for run_id, err in failed_runs:
                    print(f"  Failed {run_id}: {err}")

        return {
            "eps_lb_avg": eps_lb_avg,
            "eps_lb_median": eps_lb_median,
            "v_list": v_list,
            "run_ids": run_ids,
            "failed_runs": failed_runs,
            "T": T,
            "m": m,
            "r": r,
            "result_dict_avg": result_dict_avg,
            "result_dict_median": result_dict_median,
        }

