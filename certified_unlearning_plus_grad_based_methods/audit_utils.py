# audit_utils.py
"""
Batch-pointwise audit of unlearning experiments:
 - per-point eval steps (phi / loss)
 - batch ranking by cumulative log-likelihood ratio from pointwise stats
 - checkpoint restore
 - epsilon / rho / mu lower bounds from overlap scores across runs
"""

import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from jax import tree_map

from src.models.model import ModelFactory
from src.utils.data_cache import load_split as load_batches, rebatch  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------
# Eval Step
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Run Setup & Checkpoint Restore
# ---------------------------------------------------------------------
def load_run_vars(run_dir: Path) -> dict:
    vars_file = Path(run_dir) / "run_vars.json"
    with vars_file.open("r") as f:
        return json.load(f)


def resolve_data_dir(run_dir: Path) -> str:
    """The split directory a run was trained on, as recorded in its run_vars.json."""
    run_vars = load_run_vars(run_dir)
    if "data_dir" not in run_vars:
        raise KeyError(f"{Path(run_dir) / 'run_vars.json'} has no 'data_dir'")
    return run_vars["data_dir"]


def load_chosen_idx(run_dir: Path) -> np.ndarray:
    """Indices of the forget batches this run trained on."""
    return np.load(Path(run_dir) / "chosen_forget_batches.npy")


def resolve_config_path(main_folder: str) -> str:
    """The sweep-level config.yaml (copied there by the experiment launchers)."""
    top_level_config = Path(main_folder) / "config.yaml"
    if top_level_config.exists():
        return str(top_level_config)
    raise FileNotFoundError(
        "Missing required top-level config file: "
        f"{top_level_config}. Please ensure the logs folder contains config.yaml."
    )


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


def epsilon_lower_bound_from_vs(result, k, N, confidence_level=0.95, delta=0.0):
    """
    Compute epsilon lower bound given overlap result(s), top-k size, and forget set size.

    This is the (epsilon, 0) audit only. The bound goes through the reduction
    "(eps,0)-certified unlearning => the audit mechanism M is (2 eps,0)-LDP", whose
    triangle-inequality step degrades to (2 eps, (1+e^eps) delta) as soon as delta > 0,
    so delta must be 0; any other value raises ValueError.

    The returned epsilon is a lower bound on the certified-unlearning epsilon, i.e.
    half the rejected LDP parameter of M (cum_runs_eps_lab applies that factor of 2;
    the un-halved value is available as "epsilon_lb_ldp" inside the *_details dicts).

    Args:
        result: overlap score (sum of max(0, v*s)) for single run, or list of overlap scores for multiple runs
        k: number of top/bottom batches
        N: total number of forget batches (m)
        confidence_level: float (maps to ci_delta = 1 - confidence_level)
        delta: kept only for backward compatibility with existing call sites; must be 0.0

    Returns:
        dict with 'mean' / 'median' epsilon lower bounds (and matching *_details),
        'T' and 'v_list'. A bound is None when the observation is consistent with
        epsilon = 0 (or when that test does not apply, e.g. 'mean' for T=1).
    """
    if delta != 0.0:
        raise ValueError(
            f"delta={delta}: the overlap audit only supports delta = 0. The LDP "
            "reduction it relies on does not extend to delta > 0; use "
            "rho_lower_bound_from_vs (zCDP) or mu_lower_bound_from_vs (GDP) if an "
            "approximate-DP-style guarantee is wanted."
        )
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
        
        # For T=1 only the median test applies (there is no mean over runs).
        try:
            result_dict = compute_median_v_test_epsilon_lb(
                m=m,
                r=r,
                T=T,
                v_list=v_list,
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
            
            # epsilon_lb is None when the observation is consistent with epsilon = 0,
            # and inf when it is impossible under any finite epsilon; report neither
            # as a numeric bound.
            print(f"Median test gave no finite bound for T=1 (epsilon_lb={eps_lb!r})")
        except Exception as e:
            raise Exception(f"Median function failed for T=1: {e}")

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


def rho_lower_bound_from_vs(result, k, N, confidence_level=0.95, gamma_max=1e4,
                            conv_delta=1e-3):
    """
    Compute a zero-CDP rho lower bound given overlap result(s), top-k size, and forget set size.
    zCDP analogue of epsilon_lower_bound_from_vs: uses compute_avg_v_test_rho_lb /
    compute_median_v_test_rho_lb from cum_runs_eps_lab.py, which combine the pointwise
    audit bound pi_eps(u) with the RDP order-gamma epsilon implied by rho-zCDP
    (eps_gamma(rho) = 4*rho*gamma), minimized over gamma>1.
    Unlike epsilon_lower_bound_from_vs, there is no audit-noise `delta` term for zCDP.

    Args:
        result: overlap score (sum of max(0, v*s)) for single run, or list of overlap scores for multiple runs
        k: number of top/bottom batches
        N: total number of forget batches (m)
        confidence_level: float (maps to ci_delta = 1 - confidence_level)
        gamma_max: upper bound for the Renyi-order search in the zCDP conversion (default: 1e4)
        conv_delta: delta at which 'eps_estimate_mean'/'eps_estimate_median' report the
            (eps, delta) conversion of rho_lb (default: 1e-3). Those conversions run in
            the forward direction (a rho guarantee implies an (eps, delta) guarantee),
            so applied to an audited lower bound they are estimates, NOT lower bounds
            on eps. See cum_runs_eps_lab.eps_estimate_from_rho.

    Returns:
        dict with 'mean'/'median' rho lower bounds (mirrors epsilon_lower_bound_from_vs's
        return shape: 'mean' is None for a single run T=1, populated for T>1), plus
        'eps_estimate_mean'/'eps_estimate_median' and 'conv_delta'
    """
    try:
        from cum_runs_eps_lab import compute_avg_v_test_rho_lb, compute_median_v_test_rho_lb
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from cum_runs_eps_lab import compute_avg_v_test_rho_lb, compute_median_v_test_rho_lb

    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level

    # Handle both single result and list of results
    if isinstance(result, (int, float, np.integer, np.floating)):
        v_list = [float(result)]
        T = 1
    elif isinstance(result, (list, np.ndarray, tuple)):
        v_list = [float(v) for v in result]
        T = len(v_list)
    else:
        raise TypeError(f"result must be int, float, numpy scalar, list, tuple, or numpy array, got {type(result)}")

    if T == 1:
        v = v_list[0]
        if v / r < 0.5:
            print("Less than half of points rightly identified, returning 0")
            return {
                'mean': None,
                'median': 0.0,
                'mean_details': None,
                'median_details': None,
                'eps_estimate_mean': None,
                'eps_estimate_median': 0.0,
                'conv_delta': conv_delta,
                'T': T,
                'v_list': v_list
            }

        result_dict = compute_median_v_test_rho_lb(
            m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, gamma_max=gamma_max,
            conv_delta=conv_delta
        )
        rho_lb = result_dict.get("rho_lb", None)
        print(f"ρ lower bound (single run, median method): {rho_lb}")
        return {
            'mean': None,  # Mean method not applicable for T=1
            'median': rho_lb,
            'mean_details': None,
            'median_details': result_dict,
            'eps_estimate_mean': None,
            'eps_estimate_median': result_dict.get("eps_estimate"),
            'conv_delta': conv_delta,
            'T': T,
            'v_list': v_list
        }

    # For multiple runs, use both avg and median functions
    result_dict_avg = compute_avg_v_test_rho_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, gamma_max=gamma_max,
        conv_delta=conv_delta
    )
    result_dict_median = compute_median_v_test_rho_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, gamma_max=gamma_max,
        conv_delta=conv_delta
    )
    rho_lb_avg = result_dict_avg.get("rho_lb", None)
    rho_lb_median = result_dict_median.get("rho_lb", None)

    print(f"ρ lower bound (from {T} runs):")
    print(f"  Mean method: {rho_lb_avg}")
    print(f"  Median method: {rho_lb_median}")
    print(f"  (eps, delta={conv_delta}) estimates from rho (NOT lower bounds) — "
          f"mean: {result_dict_avg.get('eps_estimate')}, "
          f"median: {result_dict_median.get('eps_estimate')}")

    return {
        'mean': rho_lb_avg,
        'median': rho_lb_median,
        'mean_details': result_dict_avg,
        'median_details': result_dict_median,
        'eps_estimate_mean': result_dict_avg.get("eps_estimate"),
        'eps_estimate_median': result_dict_median.get("eps_estimate"),
        'conv_delta': conv_delta,
        'T': T,
        'v_list': v_list
    }


def mu_lower_bound_from_vs(result, k, N, confidence_level=0.95, conv_delta=1e-3,
                           theta_max=50.0):
    """
    Compute a Gaussian-DP (mu-GDP) lower bound given overlap result(s), top-k size, and
    forget set size. GDP analogue of epsilon_lower_bound_from_vs / rho_lower_bound_from_vs:
    uses compute_avg_v_test_mu_lb / compute_median_v_test_mu_lb from cum_runs_eps_lab.py,
    which audit the GDP parameter directly from its hypothesis-testing semantics — the
    chance-overlap null tail bound q is turned into
    mu_lb = sup{mu : Phi(Phi^{-1}(q) + 2 sqrt(L) mu) <= 1 - confidence_level} = tau/2,
    using the exact f-DP group operation mu_loc(mu) = 2 mu (Dong et al. Thm 3, k=2) for
    the reduction through the common reference law.

    mu_lb needs no halving (unlike epsilon_lower_bound_from_vs): the factor of 2 is
    already inside mu_loc, so mu_lb bounds the certified-unlearning GDP parameter.

    Args:
        result: overlap score (sum of max(0, v*s)) for single run, or list of overlap scores for multiple runs
        k: number of top/bottom batches
        N: total number of forget batches (m)
        confidence_level: float (maps to ci_delta = 1 - confidence_level)
        conv_delta: delta at which 'eps_estimate_mean'/'eps_estimate_median' report the
            (eps, delta) conversion of mu_lb, obtained by inverting delta = theta_eps(mu)
            (default: 1e-3). That inversion is exact for GDP, but applied to an audited
            lower bound the result is an estimate, NOT a lower bound on eps.
        theta_max: cap for the Chernoff lambda search in the mean test's null tail bound

    Returns:
        dict with 'mean'/'median' mu lower bounds (same shape as
        rho_lower_bound_from_vs: 'mean' is None for a single run T=1), plus
        'eps_estimate_mean'/'eps_estimate_median' and 'conv_delta'
    """
    try:
        from cum_runs_eps_lab import compute_avg_v_test_mu_lb, compute_median_v_test_mu_lb
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from cum_runs_eps_lab import compute_avg_v_test_mu_lb, compute_median_v_test_mu_lb

    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level

    # Handle both single result and list of results
    if isinstance(result, (int, float, np.integer, np.floating)):
        v_list = [float(result)]
        T = 1
    elif isinstance(result, (list, np.ndarray, tuple)):
        v_list = [float(v) for v in result]
        T = len(v_list)
    else:
        raise TypeError(f"result must be int, float, numpy scalar, list, tuple, or numpy array, got {type(result)}")

    if T == 1:
        v = v_list[0]
        if v / r < 0.5:
            print("Less than half of points rightly identified, returning 0")
            return {
                'mean': None,
                'median': 0.0,
                'mean_details': None,
                'median_details': None,
                'eps_estimate_mean': None,
                'eps_estimate_median': 0.0,
                'conv_delta': conv_delta,
                'T': T,
                'v_list': v_list
            }

        result_dict = compute_median_v_test_mu_lb(
            m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, conv_delta=conv_delta
        )
        mu_lb = result_dict.get("mu_lb", None)
        print(f"μ lower bound (single run, median method): {mu_lb}")
        return {
            'mean': None,  # Mean method not applicable for T=1
            'median': mu_lb,
            'mean_details': None,
            'median_details': result_dict,
            'eps_estimate_mean': None,
            'eps_estimate_median': result_dict.get("eps_estimate"),
            'conv_delta': conv_delta,
            'T': T,
            'v_list': v_list
        }

    # For multiple runs, use both avg and median functions
    result_dict_avg = compute_avg_v_test_mu_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, theta_max=theta_max,
        conv_delta=conv_delta
    )
    result_dict_median = compute_median_v_test_mu_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, conv_delta=conv_delta
    )
    mu_lb_avg = result_dict_avg.get("mu_lb", None)
    mu_lb_median = result_dict_median.get("mu_lb", None)

    print(f"μ lower bound (from {T} runs):")
    print(f"  Mean method: {mu_lb_avg}")
    print(f"  Median method: {mu_lb_median}")
    print(f"  (eps, delta={conv_delta}) estimates from mu (NOT lower bounds) — "
          f"mean: {result_dict_avg.get('eps_estimate')}, "
          f"median: {result_dict_median.get('eps_estimate')}")

    return {
        'mean': mu_lb_avg,
        'median': mu_lb_median,
        'mean_details': result_dict_avg,
        'median_details': result_dict_median,
        'eps_estimate_mean': result_dict_avg.get("eps_estimate"),
        'eps_estimate_median': result_dict_median.get("eps_estimate"),
        'conv_delta': conv_delta,
        'T': T,
        'v_list': v_list
    }


def generate_forget_stats_pointwise_if_missing(
    main_folder: str,
    unlearn_style: str,
    unlearn_itr: int,
    config_path: str,
    trained_stats_only: bool = False,
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
        trained_stats_only: Score checkpoint_{unlearn_itr} (the trained model) instead
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
    cmd = [
        sys.executable,
        str(script_path),
        "--runs_root", main_folder,
        "--unlearn_style", unlearn_style,
        "--unlearn_itr", str(unlearn_itr),
        "--config_path", config_path,
    ]
    if trained_stats_only:
        cmd.append("--trained_stats_only")
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


def _collect_batch_llr_rankings_pointwise(
    main_folder,
    unlearn_style,
    unlearn_itr,
    verbose=False,
    use_phi=True,
    trained_stats_only=False,
):
    """
    Score every run once and return its forget batches ranked by cumulative LLR.

    For each run under <main_folder>/test_run: restore the checkpoint, score every
    forget point (phi or loss) and turn those into per-batch cumulative LLRs via
    batch_level_predictions_from_pointwise_stats.

    Uses forget_stats_pointwise_phi_*.json or forget_stats_pointwise_loss_*.json, and
    calls generate_forget_stats_pointwise_if_missing when the needed file is absent.

    This is the expensive half of every batch-pointwise audit, and the ranking it
    produces depends on neither k nor which bound is being computed: k only selects the
    top-k / bottom-k slice (see _v_list_from_rankings), and epsilon / rho / mu are all
    functions of the resulting v_list. So one call feeds any number of k values and all
    three bounds (see compute_all_bounds_for_all_runs_batch_pointwise). The model and
    all per-run data are released before returning, so the caller can compute its
    bounds without holding GPU memory.

    Returns:
        (rankings, failed_runs, N_batches), where rankings is a list of
        {"run_id", "batch_ids_sorted_by_llr", "chosen_idx"} dicts, one per scored run.
    """
    test_run_dir = Path(main_folder) / "test_run"
    if not test_run_dir.exists():
        raise FileNotFoundError(f"test_run directory not found: {test_run_dir}")
    run_dirs = sorted([d for d in test_run_dir.iterdir() if d.is_dir() and d.name != "ignored_runs"])
    if not run_dirs:
        raise ValueError(f"No run directories in {test_run_dir}")

    cfg_file = resolve_config_path(main_folder)
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
        print(f"Pointwise forget stats not found ({stats_file.name}), generating...")
        generate_forget_stats_pointwise_if_missing(
            main_folder=main_folder,
            unlearn_style=unlearn_style,
            unlearn_itr=unlearn_itr,
            config_path=cfg_file,
            trained_stats_only=trained_stats_only,
        )

    eval_step = make_eval_step_per_point(model)
    rankings: List[Dict] = []
    failed_runs: List[Tuple[str, str]] = []
    N_batches = None
    forget_pools: Dict[str, list] = {}  # data_dir -> forget batches, loaded once

    for run_dir in run_dirs:
        wid = run_dir.name
        try:
            data_dir = resolve_data_dir(run_dir)
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

            if data_dir not in forget_pools:
                forget_pools[data_dir] = load_batches(data_dir, "forget")
            forget_batches = forget_pools[data_dir]
            chosen_idx = load_chosen_idx(run_dir)

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
            rankings.append({
                "run_id": wid,
                "batch_ids_sorted_by_llr": [int(b) for b in res["batch_ids_sorted_by_llr"]],
                "chosen_idx": np.asarray(chosen_idx, dtype=int),
            })
            if N_batches is None:
                N_batches = len(forget_batches)
            # Free run-specific data before next run to avoid OOM over many runs
            del params, forget_batches, observed_map, res
            gc.collect()
        except Exception as e:
            failed_runs.append((wid, str(e)))
            if verbose:
                print(f"  {wid}: {e}")
            gc.collect()  # free any partial state from failed run
            continue

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
    del model, eval_step, forget_pools
    gc.collect()
    jax.clear_caches()
    return rankings, failed_runs, N_batches


def _v_list_from_rankings(rankings, k, N_batches, verbose=False):
    """
    Turn the LLR rankings from _collect_batch_llr_rankings_pointwise into overlap
    scores v for one value of k.

    Takes the top-k and bottom-k batches of each run's LLR ranking as its "selected" /
    "non-selected" prediction and overlaps that with chosen_idx. A run with fewer than
    2*k batches is reported in the returned failed list (same message the single-pass
    implementation used to raise), not skipped silently.

    Returns:
        (v_list, run_ids, failed_runs) for this k
    """
    v_list = []
    run_ids = []
    failed_runs: List[Tuple[str, str]] = []

    for ranking in rankings:
        wid = ranking["run_id"]
        batch_ids = ranking["batch_ids_sorted_by_llr"]
        if len(batch_ids) < 2 * k:
            msg = f"Need at least 2*k={2 * k} batches; got {len(batch_ids)}"
            failed_runs.append((wid, msg))
            if verbose:
                print(f"  {wid}: {msg}")
            continue
        selected_k_idx = np.array(batch_ids[:k], dtype=int)
        non_selected_k_idx = np.array(batch_ids[-k:], dtype=int)

        v, s = build_v_s_vectors(selected_k_idx, non_selected_k_idx, ranking["chosen_idx"], N=N_batches)
        ov = compute_overlap_score(v, s)
        v_list.append(ov)
        run_ids.append(wid)
        if verbose:
            print(f"  {wid}: overlap = {ov}")

    return v_list, run_ids, failed_runs


def _collect_v_list_batch_pointwise(
    main_folder,
    unlearn_style,
    unlearn_itr,
    k,
    verbose=False,
    use_phi=True,
    trained_stats_only=False,
    bound_name="audit",
):
    """
    Collect the per-run overlap scores v used by every batch-pointwise audit.

    Thin composition of _collect_batch_llr_rankings_pointwise (restore each checkpoint,
    score the forget points, rank the batches by cumulative LLR) and
    _v_list_from_rankings (top-k / bottom-k slice overlapped with chosen_idx -> v).

    This is the shared front half of compute_eps_bounds_for_all_runs_batch_pointwise
    (epsilon), compute_rho_bounds_for_all_runs_batch_pointwise (zCDP) and
    compute_mu_bounds_for_all_runs_batch_pointwise (GDP); only the bound computed from
    v_list differs between them. To get more than one of those bounds, or more than one
    k, from a single checkpoint pass, call
    compute_all_bounds_for_all_runs_batch_pointwise instead.

    Args:
        bound_name: name used in the "no successful runs" error message

    Returns:
        (v_list, run_ids, failed_runs, N_batches)
    """
    rankings, failed_runs, N_batches = _collect_batch_llr_rankings_pointwise(
        main_folder,
        unlearn_style,
        unlearn_itr,
        verbose=verbose,
        use_phi=use_phi,
        trained_stats_only=trained_stats_only,
    )
    v_list, run_ids, failed_k = _v_list_from_rankings(rankings, k, N_batches, verbose=verbose)
    failed_runs = list(failed_runs) + failed_k

    if not v_list:
        raise ValueError(f"No successful runs; cannot compute {bound_name} lower bound.")
    if N_batches is None:
        raise ValueError("Could not determine number of batches.")
    return v_list, run_ids, failed_runs, N_batches


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
    stats and cumulative LLR. Per-run overlap scores come from
    _collect_v_list_batch_pointwise; the collected v_list is then passed to
    epsilon_lower_bound_from_vs.

    delta must be 0.0: this is the (epsilon, 0) audit (see epsilon_lower_bound_from_vs).
    For an approximate-DP-style number use compute_rho_bounds_for_all_runs_batch_pointwise
    (zCDP) or compute_mu_bounds_for_all_runs_batch_pointwise (GDP).
    """
    v_list, run_ids, failed_runs, N_batches = _collect_v_list_batch_pointwise(
        main_folder,
        unlearn_style,
        unlearn_itr,
        k,
        verbose=verbose,
        use_phi=use_phi,
        trained_stats_only=trained_stats_only,
        bound_name="epsilon",
    )

    eps_lb = epsilon_lower_bound_from_vs(
        v_list, k=k, N=N_batches, confidence_level=confidence_level, delta=delta
    )
    print(f"ε lower bound (batch-pointwise, {len(v_list)} runs): {eps_lb}")

    result = dict(eps_lb) if isinstance(eps_lb, dict) else {"eps_lb": eps_lb}
    result["run_ids"] = run_ids
    result["failed_runs"] = failed_runs
    return result


def compute_rho_bounds_for_all_runs_batch_pointwise(
    main_folder,
    unlearn_style,
    unlearn_itr,
    k,
    verbose=False,
    confidence_level=0.95,
    gamma_max=1e4,
    use_phi=True,
    trained_stats_only=False,
    conv_delta=1e-3
):
    """
    zCDP analogue of compute_eps_bounds_for_all_runs_batch_pointwise. Per-run overlap
    scores come from _collect_v_list_batch_pointwise (identical procedure to the eps-DP
    version); the collected v_list is then passed to rho_lower_bound_from_vs, which
    computes average-based and median-based rho lower bounds.

    conv_delta: delta used only for reporting, to convert rho-zCDP into an (eps, delta)
        pair via cum_runs_eps_lab.eps_estimate_from_rho (default: 1e-3). The conversion
        runs one way -- a rho guarantee implies an (eps, delta) guarantee -- so applied
        to an audited lower bound the number is an estimate, NOT a lower bound on eps.
        Reported as comp_eps_from_rho_avg / comp_eps_from_rho_median.
    """
    v_list, run_ids, failed_runs, N_batches = _collect_v_list_batch_pointwise(
        main_folder,
        unlearn_style,
        unlearn_itr,
        k,
        verbose=verbose,
        use_phi=use_phi,
        trained_stats_only=trained_stats_only,
        bound_name="rho",
    )

    rho_lb = rho_lower_bound_from_vs(
        v_list, k=k, N=N_batches, confidence_level=confidence_level, gamma_max=gamma_max,
        conv_delta=conv_delta
    )
    print(f"ρ lower bound (batch-pointwise, {len(v_list)} runs): {rho_lb}")

    # (eps, conv_delta) estimates computed in cum_runs_eps_lab (eps_estimate_from_rho:
    # the tighter of the Balle et al. and Bun-Steinke conversions). NOT lower bounds.
    computed_eps_from_rho_avg = rho_lb.get("eps_estimate_mean") if isinstance(rho_lb, dict) else None
    computed_eps_from_rho_median = rho_lb.get("eps_estimate_median") if isinstance(rho_lb, dict) else None
    print(f"Computed epsilon from rho at delta={conv_delta} (not a lower bound) — "
          f"average: {computed_eps_from_rho_avg}, median: {computed_eps_from_rho_median}")

    result = dict(rho_lb) if isinstance(rho_lb, dict) else {"rho_lb": rho_lb}
    result["run_ids"] = run_ids
    result["failed_runs"] = failed_runs
    result["comp_eps_from_rho_avg"] = computed_eps_from_rho_avg
    result["comp_eps_from_rho_median"] = computed_eps_from_rho_median
    return result


def compute_mu_bounds_for_all_runs_batch_pointwise(
    main_folder,
    unlearn_style,
    unlearn_itr,
    k,
    verbose=False,
    confidence_level=0.95,
    use_phi=True,
    trained_stats_only=False,
    conv_delta=1e-3
):
    """
    Gaussian-DP (mu-GDP) analogue of compute_eps_bounds_for_all_runs_batch_pointwise.
    Per-run overlap scores come from _collect_v_list_batch_pointwise (identical
    procedure to the eps-DP version); the collected v_list is then passed to
    mu_lower_bound_from_vs, which computes average-based and median-based mu lower
    bounds from the chance-overlap null tail bound and the exact f-DP group operation
    mu_loc(mu) = 2 mu.

    Unlike the epsilon audit, mu_lb needs no halving (the factor of 2 is inside
    mu_loc), and unlike the zCDP audit its (eps, delta) conversion is exact for GDP.

    conv_delta: delta used only for reporting, to convert mu into an (eps, delta) pair
        by inverting delta = theta_eps(mu) (cum_runs_eps_lab.eps_estimate_from_mu,
        default: 1e-3). Applied to an audited lower bound the number is an estimate,
        NOT a lower bound on eps. Reported as comp_eps_from_mu_avg /
        comp_eps_from_mu_median.
    """
    v_list, run_ids, failed_runs, N_batches = _collect_v_list_batch_pointwise(
        main_folder,
        unlearn_style,
        unlearn_itr,
        k,
        verbose=verbose,
        use_phi=use_phi,
        trained_stats_only=trained_stats_only,
        bound_name="mu",
    )

    mu_lb = mu_lower_bound_from_vs(
        v_list, k=k, N=N_batches, confidence_level=confidence_level, conv_delta=conv_delta
    )
    print(f"μ lower bound (batch-pointwise, {len(v_list)} runs): {mu_lb}")

    computed_eps_from_mu_avg = mu_lb.get("eps_estimate_mean") if isinstance(mu_lb, dict) else None
    computed_eps_from_mu_median = mu_lb.get("eps_estimate_median") if isinstance(mu_lb, dict) else None
    print(f"Computed epsilon from mu at delta={conv_delta} (not a lower bound) — "
          f"average: {computed_eps_from_mu_avg}, median: {computed_eps_from_mu_median}")

    result = dict(mu_lb) if isinstance(mu_lb, dict) else {"mu_lb": mu_lb}
    result["run_ids"] = run_ids
    result["failed_runs"] = failed_runs
    result["comp_eps_from_mu_avg"] = computed_eps_from_mu_avg
    result["comp_eps_from_mu_median"] = computed_eps_from_mu_median
    return result


def compute_all_bounds_for_all_runs_batch_pointwise(
    main_folder,
    unlearn_style,
    unlearn_itr,
    k,
    verbose=False,
    confidence_level=0.95,
    gamma_max=1e4,
    use_phi=True,
    trained_stats_only=False,
    conv_delta=1e-3,
):
    """
    Run the epsilon (delta = 0), rho-zCDP and mu-GDP batch-pointwise audits together,
    for one or several values of k, from a single pass over the checkpoints.

    Equivalent to calling compute_eps_bounds_for_all_runs_batch_pointwise,
    compute_rho_bounds_for_all_runs_batch_pointwise and
    compute_mu_bounds_for_all_runs_batch_pointwise once per k, but the expensive part
    (restore every checkpoint, score every forget point, rank the batches by cumulative
    LLR) is shared: it depends on neither k nor the bound, so it runs exactly once.

    Args:
        k: int, or an iterable of ints to sweep. Each k re-slices the same rankings.
        gamma_max: Renyi-order search bound, rho audit only
        conv_delta: delta for the reported (eps, delta) conversions of rho and mu.
            Those are estimates, NOT lower bounds on eps; the audited (eps, 0) bound is
            the epsilon entry.

    Returns:
        dict with the shared run metadata and a "by_k" map from k to
        {"k", "T", "v_list", "run_ids", "failed_runs", "epsilon", "rho", "mu"}, whose
        three bound entries are exactly what the single-bound functions return.
    """
    k_list = [int(k)] if isinstance(k, (int, np.integer)) else [int(x) for x in k]
    if not k_list:
        raise ValueError("k must be an int or a non-empty iterable of ints")

    rankings, scoring_failed_runs, N_batches = _collect_batch_llr_rankings_pointwise(
        main_folder,
        unlearn_style,
        unlearn_itr,
        verbose=verbose,
        use_phi=use_phi,
        trained_stats_only=trained_stats_only,
    )
    if not rankings:
        raise ValueError(
            f"No runs could be scored in {main_folder}; cannot compute any bound. "
            f"Failures: {scoring_failed_runs}"
        )
    if N_batches is None:
        raise ValueError("Could not determine number of batches.")

    result = {
        "main_folder": str(main_folder),
        "unlearn_style": unlearn_style,
        "unlearn_itr": unlearn_itr,
        "metric": "phi" if use_phi else "loss",
        "trained_stats_only": bool(trained_stats_only),
        "confidence_level": confidence_level,
        "conv_delta": conv_delta,
        "gamma_max": gamma_max,
        "epsilon_delta": 0.0,  # the overlap epsilon audit is (eps, 0) only
        "N_batches": N_batches,
        "n_runs_scored": len(rankings),
        "scoring_failed_runs": scoring_failed_runs,
        "by_k": {},
    }

    for k_val in k_list:
        v_list, run_ids, failed_k = _v_list_from_rankings(
            rankings, k_val, N_batches, verbose=verbose
        )
        entry = {
            "k": k_val,
            "T": len(v_list),
            "v_list": [int(v) for v in v_list],
            "run_ids": run_ids,
            "failed_runs": list(scoring_failed_runs) + failed_k,
        }
        if not v_list:
            entry["error"] = f"No usable runs at k={k_val} (need at least 2*k batches per run)"
            print(f"[k={k_val}] {entry['error']}")
            result["by_k"][k_val] = entry
            continue

        print(f"\n=== {main_folder} | k={k_val} | {len(v_list)} runs | "
              f"metric={result['metric']} ===")
        entry["epsilon"] = epsilon_lower_bound_from_vs(
            v_list, k=k_val, N=N_batches, confidence_level=confidence_level, delta=0.0
        )
        entry["rho"] = rho_lower_bound_from_vs(
            v_list, k=k_val, N=N_batches, confidence_level=confidence_level,
            gamma_max=gamma_max, conv_delta=conv_delta,
        )
        entry["mu"] = mu_lower_bound_from_vs(
            v_list, k=k_val, N=N_batches, confidence_level=confidence_level,
            conv_delta=conv_delta,
        )
        result["by_k"][k_val] = entry

    if all("error" in e for e in result["by_k"].values()):
        raise ValueError(f"No k value produced a usable audit for {main_folder}")
    return result
