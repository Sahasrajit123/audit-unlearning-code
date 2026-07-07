#!/usr/bin/env python
"""
Functions for loading noisy models with the new directory structure.

This module provides functions for loading models and computing noise scales
for the new directory structure where runs are organized as:
- run_XXX/unlearn/logs/ and run_XXX/unlearn/checkpoints/
- run_XXX/train/logs/ and run_XXX/train/checkpoints/
"""

import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.optimize import root_scalar

import models
from r2d import add_gaussian_noise_to_weights, h_function


def compute_training_checkpoint_name_new_structure(log_path: Path) -> Path:
    """
    Compute the training model checkpoint path from a log path for the new directory structure.
    
    Given a log path like:
    /path/to/run_001/unlearn/logs/cifar10_tinynet_1_0_forget_4500_lr_0_01_bs_40500_mbs_128_ls_ce_seed_3_scheduler_1_loadedfromfinal.pkl
    
    Returns the training checkpoint path:
    /path/to/run_001/train/checkpoints/cifar10_tinynet_1_0_forget_4500_lr_0_01_bs_42750_mbs_128_ls_ce_seed_3_scheduler_1_34_final.pt
    
    The seed is extracted from the log filename, and the checkpoint is found in the sibling train/checkpoints directory.
    """
    # Get parent directory (one level above logs/)
    base_dir = log_path.parent.parent
    
    # Verify we're in the new structure
    if base_dir.name != "unlearn":
        raise ValueError(f"Expected 'unlearn' directory, got: {base_dir.name}")
    
    # Navigate to train directory
    train_checkpoints_dir = base_dir.parent / "train" / "checkpoints"
    
    # Extract seed from log filename
    log_name = log_path.stem
    seed_match = re.search(r"seed_(\d+)", log_name)
    if not seed_match:
        raise ValueError(f"Cannot extract seed from log filename: {log_name}")
    train_seed = int(seed_match.group(1))
    
    # Extract epoch from "loadedfromfinal" or "loadedfrom<epoch>" pattern
    loadedfrom_match = re.search(r"loadedfrom(\d+|final)", log_name)
    if not loadedfrom_match:
        raise ValueError(f"Cannot extract loadedfrom info from log filename: {log_name}")
    
    loadedfrom_value = loadedfrom_match.group(1)
    
    if loadedfrom_value == "final":
        # Find the final checkpoint (ends with _final.pt)
        candidates = list(train_checkpoints_dir.glob(f"*seed_{train_seed}*scheduler_1_*_final.pt"))
        if not candidates:
            raise FileNotFoundError(f"Cannot find final checkpoint for seed {train_seed} in {train_checkpoints_dir}")
        # Return the final checkpoint (should be only one)
        return candidates[0]
    else:
        # Find checkpoint with specific epoch
        train_epoch = int(loadedfrom_value)
        candidates = list(train_checkpoints_dir.glob(f"*seed_{train_seed}*scheduler_1_{train_epoch}_final.pt"))
        if not candidates:
            raise FileNotFoundError(f"Cannot find checkpoint for seed {train_seed}, epoch {train_epoch} in {train_checkpoints_dir}")
        # Return the matching checkpoint
        return candidates[0]


def count_points_in_batches(batch_dir: Path, batch_indices: set) -> int:
    """Count total number of points in the specified batch files."""
    total_points = 0
    for batch_idx in batch_indices:
        batch_file = batch_dir / f"batch_{batch_idx:05d}.pkl"
        if batch_file.exists():
            batch_data = pickle.load(open(batch_file, "rb"))
            if isinstance(batch_data, tuple):
                x_np = batch_data[0]
            elif isinstance(batch_data, dict):
                x_np = batch_data.get('data', batch_data.get('images', batch_data.get('x')))
            else:
                continue
            
            if isinstance(x_np, np.ndarray):
                if x_np.ndim == 3:
                    total_points += 1
                elif x_np.ndim == 4:
                    total_points += x_np.shape[0]
        else:
            # Try alternative naming
            batch_file = batch_dir / f"batch_{batch_idx}.pkl"
            if batch_file.exists():
                batch_data = pickle.load(open(batch_file, "rb"))
                if isinstance(batch_data, tuple):
                    x_np = batch_data[0]
                elif isinstance(batch_data, dict):
                    x_np = batch_data.get('data', batch_data.get('images', batch_data.get('x')))
                else:
                    continue
                
                if isinstance(x_np, np.ndarray):
                    if x_np.ndim == 3:
                        total_points += 1
                    elif x_np.ndim == 4:
                        total_points += x_np.shape[0]
    return total_points


def count_points_in_directory(data_dir: Path) -> int:
    """Count total number of points in all batch files in a directory."""
    total_points = 0
    batch_files = sorted(data_dir.glob("batch_*.pkl"))
    for batch_file in batch_files:
        batch_data = pickle.load(open(batch_file, "rb"))
        if isinstance(batch_data, tuple):
            x_np = batch_data[0]
        elif isinstance(batch_data, dict):
            x_np = batch_data.get('data', batch_data.get('images', batch_data.get('x')))
        else:
            continue
        
        if isinstance(x_np, np.ndarray):
            if x_np.ndim == 3:
                total_points += 1
            elif x_np.ndim == 4:
                total_points += x_np.shape[0]
    return total_points


def load_forget_ids(run_dir: Path) -> set:
    """Load forget batch indices from a run directory."""
    forget_ids_path_json = run_dir / "forget_ids.json"
    forget_ids_path_npy = run_dir / "forget_ids.npy"
    
    if forget_ids_path_json.exists():
        data = json.load(open(forget_ids_path_json, "r"))
        # Handle structured format with type indicator
        if isinstance(data, dict) and "type" in data and "values" in data:
            if data["type"] == "batch_indices":
                return set(data["values"])
            elif data["type"] == "identity_ids":
                return set(data["values"])
        elif isinstance(data, list):
            # Legacy format: assume batch indices
            return set(data)
    elif forget_ids_path_npy.exists():
        data = np.load(forget_ids_path_npy)
        return set(data.tolist())
    
    return set()


def extract_training_params_from_path(run_dir: Path) -> Tuple[str, int, int]:
    """
    Extract batch_size, train_epochs, and unlearn_epochs from the run directory path.
    
    Path pattern: .../bs_<batch_size>_train_<train_epochs>_unlearn_<unlearn_epochs>_lr_...
    Handles both direct runs and runs inside test_run subdirectory.
    Returns: (batch_size_str, train_epochs, unlearn_epochs)
    """
    # Get the parent directory - might be test_run or the directory with training parameters
    parent = run_dir.parent
    
    # If parent is test_run, go up one more level to get the directory with training parameters
    if parent.name == "test_run":
        parent = parent.parent
    
    parent_name = parent.name
    
    # Extract batch size (either "full" or a number)
    bs_match = re.search(r"bs_(full|\d+)_train_(\d+)_unlearn_(\d+)", parent_name)
    if not bs_match:
        raise ValueError(f"Cannot extract training parameters from path: {parent_name} (from run_dir: {run_dir})")
    
    batch_size_str = bs_match.group(1)  # "full" or a number
    train_epochs = int(bs_match.group(2))
    unlearn_epochs = int(bs_match.group(3))
    
    return batch_size_str, train_epochs, unlearn_epochs


def compute_T_and_K(batch_size_str: str, train_epochs: int, unlearn_epochs: int, n: int, num_forget: int) -> Tuple[int, int]:
    """
    Compute T (total training steps) and K (unlearning steps) from training parameters.
    
    If batch_size_str == "full": one step per epoch
    Otherwise: steps per epoch = n / batch_size (training) or (n - num_forget) / batch_size (unlearning)
    
    Returns: (T, K) where T - K is training steps and K is unlearning steps
    """
    if batch_size_str == "full":
        # One step per epoch
        steps_per_training_epoch = 1
        steps_per_unlearning_epoch = 1
    else:
        batch_size = int(batch_size_str)
        steps_per_training_epoch = math.ceil(n / batch_size)
        steps_per_unlearning_epoch = math.ceil((n - num_forget) / batch_size)
    
    training_steps = train_epochs * steps_per_training_epoch
    unlearning_steps = unlearn_epochs * steps_per_unlearning_epoch
    
    T = training_steps + unlearning_steps
    K = unlearning_steps
    
    return T, K


def compute_n_and_num_forget(log_path: Path, dataroot: Path) -> Tuple[int, int]:
    """
    Compute n (total samples) and num_forget from data folder and forget_ids.json.
    
    n = number of points in retain/train + number of points in forget batches
    num_forget = total number of points in forget batches (from forget_ids.json)
    """
    # Find run directory from log_path
    # log_path is like: .../run_001/unlearn/logs/...
    base_dir = log_path.parent.parent  # e.g., unlearn
    if base_dir.name != "unlearn":
        raise ValueError(f"Expected 'unlearn' directory, got: {base_dir.name}")
    
    run_dir = base_dir.parent  # e.g., run_001
    
    # Load forget batch indices
    forget_batch_indices = load_forget_ids(run_dir)
    if not forget_batch_indices:
        raise ValueError(f"Cannot find forget_ids.json or it's empty in {run_dir}")
    
    # dataroot is assumed to be provided
    if dataroot is None:
        raise ValueError("dataroot must be provided")
    
    # Count points in retain folder
    retain_dir = dataroot / "retain"
    
    if not retain_dir.exists():
        raise ValueError(f"retain/ directory not found in {dataroot}")
    
    retain_points = count_points_in_directory(retain_dir)
    
    # Count points in forget batches
    forget_dir = dataroot / "forget"
    if not forget_dir.exists():
        raise ValueError(f"Forget directory not found: {forget_dir}")
    
    forget_points = count_points_in_batches(forget_dir, forget_batch_indices)
    
    n = retain_points + forget_points
    num_forget = forget_points
    
    return n, num_forget


def compute_sigma_general(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Compute sigma for Gaussian mechanism using the general (ε, δ)-DP formula.
    
    For any ε > 0 and δ ∈ (0, 1), the Gaussian mechanism with variance σ² is (ε, δ)-DP 
    iff σ satisfies:
    
    Φ(- (εσ / Δ) + (Δ / (2σ))) - e^ε Φ(- (εσ / Δ) - (Δ / (2σ))) ≤ δ
    
    where Φ is the CDF of the standard normal distribution, and Δ is the sensitivity.
    
    Args:
        epsilon: Privacy parameter ε > 0
        delta: Privacy parameter δ ∈ (0, 1)
        sensitivity: Sensitivity Δ (GS in the code)
    
    Returns:
        The smallest sigma that satisfies the (ε, δ)-DP condition
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
    
    # Helper function to compute the left-hand side of the inequality
    def dp_condition(sigma_val):
        if sigma_val <= 0:
            return float('inf')
        # Compute the two terms in the formula
        term1_arg = - (epsilon * sigma_val / sensitivity) + (sensitivity / (2 * sigma_val))
        term2_arg = - (epsilon * sigma_val / sensitivity) - (sensitivity / (2 * sigma_val))
        
        # Φ is the CDF of standard normal
        phi_term1 = stats.norm.cdf(term1_arg)
        phi_term2 = stats.norm.cdf(term2_arg)
        
        # Compute: lhs = Φ(term1) - e^ε * Φ(term2)
        # Use log-space computation for numerical stability, especially for large epsilon
        # We compute: log(|lhs|) using logsumexp, then convert back
        
        # Compute log of both terms
        # For phi_term1: log(phi_term1) if > 0, else -inf
        if phi_term1 > 0:
            log_phi1 = np.log(phi_term1)
        else:
            # phi_term1 is 0 or negative (shouldn't happen for CDF, but handle it)
            log_phi1 = -np.inf
        
        # For e^ε * phi_term2: epsilon + log(phi_term2) if phi_term2 > 0, else -inf
        if phi_term2 > 0:
            log_phi2 = np.log(phi_term2)
            log_exp_epsilon_phi2 = epsilon + log_phi2
        else:
            # phi_term2 is 0 or negative (shouldn't happen for CDF, but handle it)
            log_exp_epsilon_phi2 = -np.inf
        
        # Compute lhs = phi_term1 - e^ε * phi_term2 in log space
        # Use stable subtraction: log(a - b) = log(max(a,b)) + log(1 - exp(log(min(a,b)) - log(max(a,b))))
        # This avoids overflow/underflow issues
        
        # Check if both terms are -inf (both are 0)
        if np.isneginf(log_phi1) and np.isneginf(log_exp_epsilon_phi2):
            lhs = 0.0
        elif np.isneginf(log_phi1):
            # phi_term1 is 0, so lhs = -exp_epsilon_phi2
            lhs = -np.exp(log_exp_epsilon_phi2)
        elif np.isneginf(log_exp_epsilon_phi2):
            # exp_epsilon_phi2 is 0, so lhs = phi_term1
            lhs = np.exp(log_phi1)
        else:
            # Both terms are finite, use stable subtraction in log space
            # Determine which term is larger
            if log_phi1 >= log_exp_epsilon_phi2:
                # phi_term1 >= exp_epsilon_phi2, so lhs >= 0
                # log(lhs) = log(phi_term1) + log(1 - exp(log_exp_epsilon_phi2 - log_phi1))
                log_diff = log_exp_epsilon_phi2 - log_phi1
                if log_diff < -50:  # exp(log_diff) is essentially 0
                    lhs = np.exp(log_phi1)
                else:
                    log_lhs = log_phi1 + np.log1p(-np.exp(log_diff))
                    lhs = np.exp(log_lhs)
            else:
                # exp_epsilon_phi2 > phi_term1, so lhs < 0
                # log(|lhs|) = log(exp_epsilon_phi2) + log(1 - exp(log_phi1 - log_exp_epsilon_phi2))
                log_diff = log_phi1 - log_exp_epsilon_phi2
                if log_diff < -50:  # exp(log_diff) is essentially 0
                    lhs = -np.exp(log_exp_epsilon_phi2)
                else:
                    log_abs_lhs = log_exp_epsilon_phi2 + np.log1p(-np.exp(log_diff))
                    lhs = -np.exp(log_abs_lhs)
        
        # Return the difference from delta (we want this to be ≤ 0)
        result = lhs - delta
        
        # Check for inf/nan cases - these indicate numerical issues
        if not np.isfinite(result):
            raise ValueError(
                f"Non-finite value in dp_condition computation: "
                f"result={result}, lhs={lhs}, "
                f"log_phi1={log_phi1}, log_exp_epsilon_phi2={log_exp_epsilon_phi2}, "
                f"phi_term1={phi_term1}, phi_term2={phi_term2}, "
                f"term1_arg={term1_arg}, term2_arg={term2_arg}, "
                f"sigma_val={sigma_val}, epsilon={epsilon}, sensitivity={sensitivity}"
            )
        
        return result
    
    # Use the old formula as an initial guess (works well for small epsilon)
    initial_guess = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    # For the Gaussian mechanism, we know:
    # - For very small sigma, dp_condition > 0 (condition not satisfied)
    # - For very large sigma, dp_condition < 0 (condition satisfied)
    # - The function is monotonic (decreasing) in sigma
    # So we can use a wide bracket and let scipy find the root efficiently
    
    # Define a wide bracket that should work for all reasonable parameter values
    # Lower bound: very small positive value (ensures condition not satisfied)
    # Upper bound: large value (ensures condition is satisfied)
    low = max(initial_guess * 1e-6, sensitivity * 1e-10)  # Very small but positive
    high = initial_guess * 1e6  # Very large
    
    # Ensure we have opposite signs at the bounds
    # If not, expand the range systematically
    max_expansions = 20
    expansion_count = 0
    while expansion_count < max_expansions:
        low_val = dp_condition(low)
        high_val = dp_condition(high)
        
        # Check if we have a valid bracket (opposite signs)
        if low_val > 0 and high_val < 0:
            break
        
        # If low is already satisfied (shouldn't happen for very small sigma), something's wrong
        if low_val <= 0:
            # This is unexpected, but return a small sigma
            return low
        
        # If high is not satisfied yet, expand it
        if high_val >= 0:
            high *= 10
            expansion_count += 1
        else:
            # Both are negative, which shouldn't happen, but try smaller low
            low *= 0.1
            expansion_count += 1
    else:
        raise ValueError(f"Could not find valid bracket for root finding after {max_expansions} expansions. "
                        f"low={low:.6e} (value={dp_condition(low):.6e}), "
                        f"high={high:.6e} (value={dp_condition(high):.6e})")
    
    # Use scipy's robust root finding method
    try:
        result = root_scalar(
            dp_condition,
            bracket=[low, high],
            method='brentq',
            xtol=1e-10
        )
        sigma = result.root
    except ValueError as e:
        raise ValueError(f"Root finding failed: {e}. "
                        f"Bracket: low={low:.6e} (value={dp_condition(low):.6e}), "
                        f"high={high:.6e} (value={dp_condition(high):.6e})") from e
    
    # Verify the result satisfies the condition (with some numerical tolerance)
    # We want dp_condition(sigma) <= 0, meaning lhs <= delta
    if dp_condition(sigma) > 1e-8:  # Allow small numerical error
        # If not satisfied, the root finder might have found a value slightly too small
        # Try to find a slightly larger sigma that satisfies the condition
        test_sigma = sigma * 1.01
        max_iter = 100
        iter_count = 0
        while dp_condition(test_sigma) > 0 and iter_count < max_iter and test_sigma < sigma * 10:
            test_sigma *= 1.01
            iter_count += 1
        if dp_condition(test_sigma) <= 0:
            sigma = test_sigma
        else:
            # If we still can't find a valid sigma, raise an error
            raise ValueError(f"Could not find valid sigma satisfying (ε, δ)-DP condition. "
                           f"Last attempt: sigma={test_sigma:.6e}, condition={dp_condition(test_sigma):.6e}")
    
    return max(sigma, 0)  # Ensure non-negative


def archive_compute_sigma_from_log_new_structure(
    log_path: Path, checkpoint_name: str, epsilon: float, delta: float, dataroot: Path, verbose: bool = False
) -> float:
    """Replicate the noise-scale computation from model_distance_analysis.ipynb for the new directory structure."""
    logs: Dict = pickle.load(open(log_path, "rb"))

    # Compute training checkpoint path from log_path
    train_checkpoint_path = compute_training_checkpoint_name_new_structure(log_path)
    
    # Find and load training log file to get learn_batch_size
    # Training logs are in sibling train/logs directory
    base_dir = log_path.parent.parent  # e.g., unlearn
    if base_dir.name != "unlearn":
        raise ValueError(f"Expected 'unlearn' directory, got: {base_dir.name}")
    
    # Required parameters; raise if missing instead of silently defaulting
    if "Lipschitz" not in logs:
        raise ValueError(f"[sigma] Missing Lipschitz in {log_path}")
    L = logs["Lipschitz"]

    grad_norms = logs.get("grad norm over epochs", None)
    if not (isinstance(grad_norms, Sequence) and len(grad_norms) > 0):
        raise ValueError(f"[sigma] Missing/nonfinite grad norms in {log_path}")
    G = max(grad_norms)

    # Compute n and num_forget from data folder and forget_ids.json
    n, num_forget = compute_n_and_num_forget(log_path, dataroot=dataroot)
    if verbose:
        print(f"[sigma] Computed n={n}, num_forget={num_forget} from data folder")

    # Extract training parameters from path and compute T and K
    run_dir = base_dir.parent  # e.g., run_001
    batch_size_str, train_epochs, unlearn_epochs = extract_training_params_from_path(run_dir)
    T, K = compute_T_and_K(batch_size_str, train_epochs, unlearn_epochs, n, num_forget)
    if verbose:
        print(f"[sigma] Extracted batch_size={batch_size_str}, train_epochs={train_epochs}, unlearn_epochs={unlearn_epochs}")
        print(f"[sigma] Computed T={T}, K={K}")

    # Compute eta (learning rate) from logs
    if "lr" in logs:
        eta = logs["lr"]
    elif "learning_rate" in logs:
        eta = logs["learning_rate"]
    else:
        raise ValueError(f"[sigma] Missing lr/learning_rate in {log_path}")
    if not (isinstance(eta, (int, float)) and eta > 0):
        raise ValueError(f"[sigma] Invalid eta={eta} in {log_path}")

    # Compute h(K) using the h_function
    h_K = h_function(K, eta, L, num_forget, n, T)
    if not np.isfinite(h_K):
        raise ValueError(f"[sigma] Non-finite h(K) for {checkpoint_name}")
    ##print ("h_K", h_K, "G", G)
    GS = (2 * num_forget * G * h_K) / (L * n)
    if not np.isfinite(GS):
        raise ValueError(f"[sigma] Non-finite GS for {checkpoint_name}")

    sigma = GS * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    if verbose:
        print(f"[sigma] h(K)={h_K:.4e}, GS={GS:.4e}, sigma={sigma:.4e}")
    return sigma

def compute_sigma_from_log_new_structure(
    log_path: Path, checkpoint_name: str, epsilon: float, delta: float, dataroot: Path, verbose: bool = False
) -> float:
    """Replicate the noise-scale computation from model_distance_analysis.ipynb for the new directory structure."""
    logs: Dict = pickle.load(open(log_path, "rb"))

    # Compute training checkpoint path from log_path
    train_checkpoint_path = compute_training_checkpoint_name_new_structure(log_path)
    
    # Find and load training log file to get learn_batch_size
    # Training logs are in sibling train/logs directory
    base_dir = log_path.parent.parent  # e.g., unlearn
    if base_dir.name != "unlearn":
        raise ValueError(f"Expected 'unlearn' directory, got: {base_dir.name}")
    
    # Required parameters; raise if missing instead of silently defaulting
    if "Lipschitz" not in logs:
        raise ValueError(f"[sigma] Missing Lipschitz in {log_path}")
    L = logs["Lipschitz"]

    grad_norms = logs.get("grad norm over epochs", None)
    if not (isinstance(grad_norms, Sequence) and len(grad_norms) > 0):
        raise ValueError(f"[sigma] Missing/nonfinite grad norms in {log_path}")
    G = max(grad_norms)

    # Compute n and num_forget from data folder and forget_ids.json
    n, num_forget = compute_n_and_num_forget(log_path, dataroot=dataroot)
    if verbose:
        print(f"[sigma] Computed n={n}, num_forget={num_forget} from data folder")

    # Extract training parameters from path and compute T and K
    run_dir = base_dir.parent  # e.g., run_001
    batch_size_str, train_epochs, unlearn_epochs = extract_training_params_from_path(run_dir)
    T, K = compute_T_and_K(batch_size_str, train_epochs, unlearn_epochs, n, num_forget)
    if verbose:
        print(f"[sigma] Extracted batch_size={batch_size_str}, train_epochs={train_epochs}, unlearn_epochs={unlearn_epochs}")
        print(f"[sigma] Computed T={T}, K={K}")

    # Compute eta (learning rate) from logs
    if "lr" in logs:
        eta = logs["lr"]
    elif "learning_rate" in logs:
        eta = logs["learning_rate"]
    else:
        raise ValueError(f"[sigma] Missing lr/learning_rate in {log_path}")
    if not (isinstance(eta, (int, float)) and eta > 0):
        raise ValueError(f"[sigma] Invalid eta={eta} in {log_path}")

    if verbose:
        print(f"[sigma] Lipschitz L={L:.4e}, max grad norm G={G:.4e}, learning rate eta={eta:.4e}")

    # Compute h(K) using the h_function
    h_K = h_function(K, eta, L, num_forget, n, T)
    if not np.isfinite(h_K):
        raise ValueError(f"[sigma] Non-finite h(K) for {checkpoint_name}")
    ##print ("h_K", h_K, "G", G)
    GS = (2 * num_forget * G * h_K) / (L * n)
    if not np.isfinite(GS):
        raise ValueError(f"[sigma] Non-finite GS for {checkpoint_name}")

    # Use the general (ε, δ)-DP formula that works for any epsilon > 0
    # The formula: Φ(- (εσ / Δ) + (Δ / (2σ))) - e^ε Φ(- (εσ / Δ) - (Δ / (2σ))) ≤ δ
    sigma = compute_sigma_general(epsilon, delta, GS)
    if verbose:
        print(f"[sigma] h(K)={h_K:.4e}, GS={GS:.4e}, sigma={sigma:.4e}")
        print(f"[sigma] GS formula: 2 * num_forget({num_forget}) * G({G:.4e}) * h_K({h_K:.4e}) / (L({L:.4e}) * n({n})) = {GS:.4e}")
    return sigma

def load_noisy_model(
    checkpoint_path: Path,
    log_path: Path,
    device: torch.device,
    epsilon: float,
    delta: float,
    dataroot: Path,
    verbose: bool = False,
    add_noise: bool = True,
) -> Tuple[torch.nn.Module, float]:
    """Load TinyNet checkpoint and optionally add certified Gaussian noise to its weights for the new directory structure."""
    model = models.get_model("tinynet", num_classes=10, filters_percentage=1.0).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()

    if not add_noise:
        return model, None

    sigma = compute_sigma_from_log_new_structure(log_path, checkpoint_path.name, epsilon, delta, dataroot=dataroot, verbose=verbose)
    sigma_old = archive_compute_sigma_from_log_new_structure(log_path, checkpoint_path.name, epsilon, delta, dataroot=dataroot, verbose=verbose)
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        add_gaussian_noise_to_weights(model, sigma, device)
    else:
            raise ValueError("Skipping noise injection (sigma invalid)")
    return model, sigma

