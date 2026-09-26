#!/usr/bin/env python3
"""
Compute epsilon lower bounds by sampling combo_indices and predicting them.

For each run_id in range(num_runs):
1. Sample a combo_index
2. Load a model from eval_folders for that combo_index
3. Compute phi/loss for all points in forget batches
4. Use cumulative log-likelihood to predict combo_index
5. Compute overlap between predicted and true forget indices
6. Use overlap to compute epsilon lower bound
"""

import json
import math
import random
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
import yaml

from src.models.model import ModelFactory
import audit_utils as au

# Import functions from cum_runs_eps_lab
from cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb,
    compute_median_v_test_epsilon_lb,
    compute_avg_v_test_rho_lb,
    compute_median_v_test_rho_lb,
    compute_avg_v_test_mu_lb,
    compute_median_v_test_mu_lb,
    rho_lb_pairwise_from_roc,
    eps_estimate_from_mu,
    eps_estimate_from_rho,
)


def log_pdf_gaussian(x: float, mu: float, var: float, eps: float = 1e-10) -> float:
    """Log probability density of x under Gaussian(mu, var)."""
    var = max(var, eps)
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mu) ** 2) / var


def cumulative_loglik_per_combo(
    stats: Dict,
    obs_values: Dict[str, Tuple[float, float]],
    metric: str = "phi",
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Compute cumulative log-likelihood per combo_index (model) from observed values.
    
    Args:
        stats: Dictionary with "points" key containing per-point stats
        obs_values: Dict mapping "batch_{b}_point_{p}" -> (phi_value, loss_value)
        metric: "phi" or "loss"
    
    Returns:
        (combo_scores, combo_counts) where combo_scores maps "model_{combo_idx}" -> cumulative log-likelihood
    """
    # Handle both formats: stats may have "points" key or point keys directly at root
    points = stats.get("points", stats)
    combo_scores: Dict[str, float] = {}
    combo_counts: Dict[str, int] = {}

    for point_key, (phi_val, loss_val) in obs_values.items():
        point_stats = points.get(point_key)
        if not point_stats:
            continue
        
        # Get the metric value for this point
        x = phi_val if metric == "phi" else loss_val
        if x is None:
            continue

        # Iterate over all models (combo_indices) for this point
        for model_key, model_stats in point_stats.items():
            if not model_key.startswith("model_"):
                continue
            
            if metric == "phi":
                mu = model_stats.get("mean_phi")
                var = model_stats.get("var_phi")
            else:
                mu = model_stats.get("mean_loss")
                var = model_stats.get("var_loss")

            if mu is None or var is None:
                continue

            logp = log_pdf_gaussian(x, mu, var)
            combo_scores[model_key] = combo_scores.get(model_key, 0.0) + logp
            combo_counts[model_key] = combo_counts.get(model_key, 0) + 1

    return combo_scores, combo_counts


def predict_combo_by_loglik(
    stats: Dict,
    obs_values: Dict[str, Tuple[float, float]],
    metric: str = "phi",
) -> Tuple[Optional[str], Dict[str, float], Dict[str, int]]:
    """
    Predict combo_index using cumulative log-likelihood.
    
    Returns:
        (predicted_combo_key, combo_scores, combo_counts)
    """
    combo_scores, combo_counts = cumulative_loglik_per_combo(stats, obs_values, metric)
    if not combo_scores:
        return None, combo_scores, combo_counts

    best_combo = max(combo_scores.items(), key=lambda kv: kv[1])[0]
    return best_combo, combo_scores, combo_counts


def load_model_from_eval_folder(
    main_folder: Path,
    run_file: str,
    combo_idx: int,
    unlearn_style: str,
    unlearn_itr: int,
    eval_trial_idx: int,
) -> Dict:
    """
    Load model from eval_folders for a specific combo_index and trial.
    
    Returns:
        Model parameters
    """
    run_dir = main_folder / run_file
    eval_folder = run_dir / "eval_folders" / f"ckpt_trial_{eval_trial_idx}"
    ckpt_path = eval_folder / f"unlearn_{unlearn_style}_{unlearn_itr}"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt_path_abs = os.path.abspath(str(ckpt_path))
    state = au.restore_orbax_state(ckpt_path_abs)
    params = state["params"] if isinstance(state, dict) else state.params
    return params


def compute_obs_values_for_model(
    params: Dict,
    model,
    forget_batches: List,
    eval_step: callable,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute phi and loss values for all points in forget batches.
    
    Returns:
        Dict mapping "batch_{b}_point_{p}" -> (phi_value, loss_value)
    """
    obs_values = {}
    
    for batch_idx, batch in enumerate(forget_batches):
        x, y = batch[0], batch[1]
        if isinstance(x, np.ndarray):
            x = jnp.array(x)
        if isinstance(y, np.ndarray):
            y = jnp.array(y)
        
        # Evaluate batch
        out = eval_step(params, (x, y))
        phi_arr = np.array(out["phi"])
        loss_arr = np.array(out["loss"])
        
        n_points = phi_arr.shape[0]
        for point_idx in range(n_points):
            key = f"batch_{batch_idx}_point_{point_idx}"
            obs_values[key] = (float(phi_arr[point_idx]), float(loss_arr[point_idx]))
    
    return obs_values


def get_eval_trial_numbers(main_folder: Path, run_file: str) -> List[int]:
    """Get actual trial numbers from eval_folders (e.g., [50, 51, 52, ...])."""
    run_dir = main_folder / run_file
    eval_folders_dir = run_dir / "eval_folders"
    if not eval_folders_dir.exists():
        return []
    
    trial_numbers = []
    for d in eval_folders_dir.iterdir():
        if d.is_dir() and d.name.startswith("ckpt_trial_"):
            try:
                trial_num = int(d.name.replace("ckpt_trial_", ""))
                trial_numbers.append(trial_num)
            except ValueError:
                continue
    
    return sorted(trial_numbers)


def get_forget_indices_from_mapping(
    main_folder: Path,
    combo_idx: int,
    mapping_file: Optional[Path] = None,
) -> List[int]:
    """Get forget indices (chosen_idx) for a combo_index from mapping file."""
    if mapping_file is None:
        mapping_file = main_folder / "combo_idx_to_run_file_mapping.json"
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    combo_key = str(combo_idx)
    if combo_key not in mapping:
        raise ValueError(f"combo_idx {combo_idx} not found in mapping")
    
    return mapping[combo_key]["chosen_idx"]


def compute_eps_bounds_sampled_combos(
    main_folder: str,
    unlearn_style: str,
    unlearn_itr: int,
    num_runs: int,
    metric: str = "phi",
    sampling_seed: int = 123,
    confidence_level: float = 0.95,
    delta: float = 0.0,
    epsilon_delta: float = 0.0,
    ci_delta: float = 0.05,
    conv_delta: float = 1e-3,
    mapping_file: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    skip_missing_trials: bool = True,
):
    """
    Compute epsilon / rho / mu lower bounds by sampling combo_indices and predicting them.

    Three audits are computed from the same per-run overlap statistic v_list:
      - epsilon (pure DP, delta = 0): avg_v_test / median_v_test
      - rho (zCDP):                   rho_avg_v_test / rho_median_v_test
      - mu (Gaussian DP):             mu_avg_v_test / mu_median_v_test
    plus, when m = 2, the direct pairwise Clopper-Pearson auditor (m2_cp_result), which
    reports epsilon, mu AND rho from the attack's single ROC point.

    Every value reported is a bound on the *certified-unlearning* parameter. The audits
    observe distinguishability between two unlearned laws P_s1, P_s2, which the reduction
    through the reference law inflates by 2. For the pairwise epsilon and mu that factor
    is applied here as an explicit /2 (raw values kept under the "*_pairwise" keys); for
    rho, and for the v-list mu/rho audits, the factor already sits inside the local
    profile (eps_gamma^loc(rho) resp. mu_loc(mu) = 2 mu), so those are not halved.

    Args:
        main_folder: Root directory containing runs
        unlearn_style: "epoch" or "step"
        unlearn_itr: Unlearning iteration number
        num_runs: Number of runs to sample
        metric: "phi" or "loss"
        sampling_seed: Random seed for sampling combo_indices
        confidence_level: Confidence level for epsilon bound
        delta: must be 0.0 — every audit here is (epsilon, 0)
        epsilon_delta: must be 0.0 — the delta of the m=2 Clopper-Pearson bound (eq. 4)
        ci_delta: Delta for confidence interval
        conv_delta: delta at which rho/mu are additionally converted to an epsilon
            estimate for reporting (default: 1e-3). Those conversions run forward
            (rho/mu guarantee -> (eps, delta) guarantee), so they are NOT lower bounds.
        mapping_file: Path to combo_idx_to_run_file_mapping.json (default: {main_folder}/combo_idx_to_run_file_mapping.json)
        verbose: Print detailed progress
        skip_missing_trials: when the per-combo distribution stats have to be generated,
            skip individual ckpt_trial_* folders that lack this step's checkpoint instead
            of aborting the step (default: True). The distribution stats only build the
            attack, so dropping a stray incomplete trial does not affect the validity of
            the bound; set False to require every trial to be present.

    Returns:
        Dictionary with epsilon/rho/mu bounds and statistics
    """
    if delta != 0.0:
        raise ValueError(
            f"delta={delta}: the overlap audit only supports delta = 0 "
            "(see audit_utils.epsilon_lower_bound_from_vs)."
        )
    if epsilon_delta != 0.0:
        raise ValueError(
            f"epsilon_delta={epsilon_delta}: these audits are (epsilon, 0) only. The "
            "certified-unlearning reduction they rely on holds at delta = 0; for an "
            "approximate-DP-style number read rho_lb / mu_lb (and their eps_estimate "
            "at conv_delta) instead."
        )

    main_folder = Path(main_folder)

    # Set random seed
    random.seed(sampling_seed)
    np.random.seed(sampling_seed)
    
    # Load mapping file
    if mapping_file is None:
        mapping_file = main_folder / "combo_idx_to_run_file_mapping.json"
    else:
        mapping_file = Path(mapping_file)
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        combo_mapping = json.load(f)
    
    # Get all available combo_indices, skipping those whose run_file was moved to ignored_runs
    ignored_dir = main_folder / "ignored_runs"
    available_combos = sorted([
        int(k) for k, v in combo_mapping.items()
        if (main_folder / v["run_file"]).exists()
        and not (ignored_dir / v["run_file"]).exists()
    ])
    if not available_combos:
        raise ValueError("No combo_indices found in mapping file (all may be ignored)")

    n_ignored = len(combo_mapping) - len(available_combos)
    if n_ignored:
        print(f"Skipping {n_ignored} ignored combo(s) (run_file moved to ignored_runs/)")
    print(f"Found {len(available_combos)} available combo_indices: {available_combos}")
    
    # Load stats file (evaluation_per_combo_*.json)
    stats_file = main_folder / f"evaluation_per_combo_{unlearn_style}_{unlearn_itr}.json"
    if not stats_file.exists():
        print(f"Stats file not found: {stats_file}")
        print("Generating stats file by calling evaluate_models_per_combo...")
        print(f"  skip_missing_trials={skip_missing_trials}: trials without a checkpoint "
              f"at this step are {'skipped with a warning' if skip_missing_trials else 'treated as errors'}")
        
        # Import the function to generate stats
        from evaluate_models_per_combo import evaluate_models_per_combo
        
        # Generate the stats file
        generated_stats_file = evaluate_models_per_combo(
            runs_root=main_folder,
            unlearn_style=unlearn_style,
            unlearn_itr=unlearn_itr,
            mapping_file=mapping_file,
            skip_missing=skip_missing_trials,
            verbose=True,  # keep the per-trial skip warnings visible in the audit log
            progress_callback=progress_callback,
        )
        
        print(f"✅ Generated stats file: {generated_stats_file}")
        stats_file = generated_stats_file
    
    print(f"Loading stats from: {stats_file}")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Load config from first run to get model info
    first_combo = available_combos[0]
    first_run_file = combo_mapping[str(first_combo)]["run_file"]
    first_run_dir = main_folder / first_run_file
    config_file = first_run_dir / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = ModelFactory.create_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["n_classes"],
    )
    num_classes = cfg["model"]["n_classes"]
    
    # The forget pool, from the data_dir the runs were trained on
    forget_batches = au.load_batches(au.resolve_data_dir(first_run_dir), "forget")
    m = len(forget_batches)  # Number of forget batches

    # Set r = m (all forget batches are used per combo)
    r = m  # Number of forget batches per combo
    
    print(f"m (total forget batches): {m}")
    print(f"r (forget batches per combo): {r}")
    print(f"  r is {'even' if r % 2 == 0 else 'ODD'} (must be even for log_f_values)")
    
    # Build eval step
    eval_step = au.make_eval_step_per_point(model)
    
    # Aggregate combo_index counts (how many runs sampled each combo)
    combo_counts: Dict[int, int] = defaultdict(int)
    combo_to_eval_trials: Dict[int, List[int]] = defaultdict(list)
    
    # For each combo, determine which eval trials are available (actual trial numbers)
    for combo_idx in available_combos:
        run_file = combo_mapping[str(combo_idx)]["run_file"]
        trial_numbers = get_eval_trial_numbers(main_folder, run_file)
        if trial_numbers:
            combo_to_eval_trials[combo_idx] = trial_numbers
    
    print(f"Eval trials per combo: {dict(combo_to_eval_trials)}")
    
    # Sample combo_indices for each run
    sampled_combos = []
    for run_id in range(num_runs):
        combo_idx = random.choice(available_combos)
        sampled_combos.append(combo_idx)
        combo_counts[combo_idx] += 1
    
    print(f"Sampled {num_runs} runs:")
    for combo_idx, count in sorted(combo_counts.items()):
        print(f"  combo_idx {combo_idx}: {count} runs")
    
    # Validate that combo counts don't exceed available eval trials
    for combo_idx, count in combo_counts.items():
        available_trials = combo_to_eval_trials.get(combo_idx, [])
        num_trials = len(available_trials)
        if count > num_trials:
            raise ValueError(
                f"combo_idx {combo_idx} was sampled {count} times, but only {num_trials} "
                f"eval trials are available in eval_folders. Cannot use more eval trials than available."
            )
    
    # Process each sampled run
    overlap_sizes = []
    overlap_ratios = []
    jaccard_scores = []
    chosen_combos = []
    predicted_combos = []
    failed_runs = []
    
    # Track which eval_trial to use for each combo_index (cycle through available trials)
    combo_trial_counters: Dict[int, int] = defaultdict(int)
    
    for run_id, combo_idx in enumerate(sampled_combos):
        try:
            if verbose:
                print(f"\n[{run_id+1}/{num_runs}] Processing combo_idx {combo_idx}...")
            
            run_file = combo_mapping[str(combo_idx)]["run_file"]
            
            # Choose an eval trial for this combo (cycle through available trials)
            available_trials = combo_to_eval_trials.get(combo_idx, [])
            if not available_trials:
                raise ValueError(f"No eval trials available for combo_idx {combo_idx}")
            
            # Use the next available trial in order (cycling if needed)
            trial_counter = combo_trial_counters[combo_idx]
            eval_trial_idx = available_trials[trial_counter % len(available_trials)]
            combo_trial_counters[combo_idx] += 1
            
            if verbose:
                print(f"  Using eval_trial {eval_trial_idx} from {run_file}")
            
            # Load model
            params = load_model_from_eval_folder(
                main_folder, run_file, combo_idx, unlearn_style, unlearn_itr, eval_trial_idx
            )

            if progress_callback is not None:
                progress_callback(
                    "evaluation",
                    run_id + 1,
                    num_runs,
                    f"combo_idx {combo_idx}",
                )
            
            # Compute observed values
            obs_values = compute_obs_values_for_model(params, model, forget_batches, eval_step)
            
            # Predict combo_index using cumulative log-likelihood
            predicted_combo_key, combo_scores, combo_counts_pred = predict_combo_by_loglik(
                stats, obs_values, metric=metric
            )
            
            if predicted_combo_key is None:
                raise ValueError("Failed to predict combo_index")
            
            # Extract predicted combo_idx from "model_{combo_idx}"
            predicted_combo_idx = int(predicted_combo_key.replace("model_", ""))
            
            # Get true and predicted forget indices
            true_chosen_idx = get_forget_indices_from_mapping(main_folder, combo_idx, mapping_file)
            pred_chosen_idx = get_forget_indices_from_mapping(main_folder, predicted_combo_idx, mapping_file)
            
            # Compute overlap
            true_set = set(true_chosen_idx)
            pred_set = set(pred_chosen_idx)
            intersection = true_set & pred_set
            union = true_set | pred_set
            overlap = len(intersection)
            chosen_ratio = overlap / len(true_set) if true_set else 0.0
            jaccard = overlap / len(union) if union else 0.0
            
            overlap_sizes.append(overlap)
            overlap_ratios.append(chosen_ratio)
            jaccard_scores.append(jaccard)
            chosen_combos.append(combo_idx)
            predicted_combos.append(predicted_combo_idx)
            
            if verbose:
                print(f"  True combo: {combo_idx}, Predicted: {predicted_combo_idx}")
                print(f"  Overlap: {overlap}/{len(true_set)} (ratio: {chosen_ratio:.4f}, jaccard: {jaccard:.4f})")
            
            # Free memory
            del params, obs_values
            jax.clear_caches()
            
        except Exception as e:
            failed_runs.append((run_id, combo_idx, str(e)))
            if verbose:
                print(f"  Failed: {e}")
            continue
    
    if not overlap_sizes:
        raise ValueError("No successful runs; cannot compute epsilon lower bound.")
    
    print(f"\nProcessed {len(overlap_sizes)} successful runs")
    print(f"Failed runs: {len(failed_runs)}")
    
    # Compute v_list (2 * overlap for each run, as in reference)
    v_list = [2 * v for v in overlap_sizes]
    T = len(v_list)
    
    # Compute epsilon lower bounds
    print ("Overlap array", v_list)
    # Overlap audit: (epsilon, 0) only -- the LDP reduction it relies on does not
    # extend to delta > 0, so epsilon_delta is NOT passed here (it is used only by the
    # m=2 Clopper-Pearson bound below). epsilon_lb is already the certified-unlearning
    # epsilon, i.e. half the rejected LDP parameter ("epsilon_lb_ldp" in the dicts).
    avg_lb = compute_avg_v_test_epsilon_lb(
        m=m,
        r=r,
        T=T,
        v_list=v_list,
        ci_delta=ci_delta,
        direction="ge",
    )
    median_lb = compute_median_v_test_epsilon_lb(
        m=m,
        r=r,
        T=T,
        v_list=v_list,
        ci_delta=ci_delta,
    )
    avg_epsilon_lb = avg_lb.get('epsilon_lb')
    median_epsilon_lb = median_lb.get('epsilon_lb')
    
    print("\nEpsilon lower bounds:")
    if avg_epsilon_lb is not None:
        print(f"  Avg-v test epsilon_lb: {avg_epsilon_lb:.6f}")
    else:
        print(f"  Avg-v test epsilon_lb: None (infeasible)")
    if median_epsilon_lb is not None:
        print(f"  Median-v test epsilon_lb: {median_epsilon_lb:.6f}")
    else:
        print(f"  Median-v test epsilon_lb: None (infeasible)")

    # --- rho-zCDP and mu-GDP lower bounds from the same v_list ---
    # Same overlap statistic, different privacy parameter: the zCDP audit bounds the
    # Renyi/zCDP parameter rho, the GDP audit bounds the Gaussian-DP parameter mu.
    # mu needs no halving (the group factor of 2 is inside mu_loc); rho is reported as
    # audited. eps_estimate_* are forward (rho/mu -> (eps, delta)) conversions at
    # conv_delta and are NOT lower bounds on epsilon.
    avg_rho_lb = compute_avg_v_test_rho_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, conv_delta=conv_delta,
    )
    median_rho_lb = compute_median_v_test_rho_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, conv_delta=conv_delta,
    )
    avg_mu_lb = compute_avg_v_test_mu_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, conv_delta=conv_delta,
    )
    median_mu_lb = compute_median_v_test_mu_lb(
        m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, conv_delta=conv_delta,
    )

    print("\nrho (zCDP) lower bounds:")
    print(f"  Avg-v test rho_lb:    {avg_rho_lb.get('rho_lb')}  "
          f"(eps estimate at delta={conv_delta}: {avg_rho_lb.get('eps_estimate')})")
    print(f"  Median-v test rho_lb: {median_rho_lb.get('rho_lb')}  "
          f"(eps estimate at delta={conv_delta}: {median_rho_lb.get('eps_estimate')})")
    print("mu (GDP) lower bounds:")
    print(f"  Avg-v test mu_lb:    {avg_mu_lb.get('mu_lb')}  "
          f"(eps estimate at delta={conv_delta}: {avg_mu_lb.get('eps_estimate')})")
    print(f"  Median-v test mu_lb: {median_mu_lb.get('mu_lb')}  "
          f"(eps estimate at delta={conv_delta}: {median_mu_lb.get('eps_estimate')})")

    # --- Direct Clopper-Pearson epsilon lower bound (m=2 only) ---
    m2_cp_result = None
    if m == 2 and len(available_combos) == 2:
        from scipy.stats import beta as _beta_dist

        combo_neg, combo_pos = available_combos[0], available_combos[1]

        tp = sum(1 for c, p in zip(chosen_combos, predicted_combos) if c == combo_pos and p == combo_pos)
        fn = sum(1 for c, p in zip(chosen_combos, predicted_combos) if c == combo_pos and p == combo_neg)
        fp = sum(1 for c, p in zip(chosen_combos, predicted_combos) if c == combo_neg and p == combo_pos)
        tn = sum(1 for c, p in zip(chosen_combos, predicted_combos) if c == combo_neg and p == combo_neg)
        n_pos = tp + fn
        n_neg = fp + tn

        # confidence = 1 - ci_delta; divide by 2 so each bound uses alpha = ci_delta/2
        alpha = ci_delta / 2

        # Clopper-Pearson upper bound: Beta.ppf(1-alpha, k+1, n-k)
        fp_high = _beta_dist.ppf(1 - alpha, fp + 1, n_neg - fp) if n_neg > fp else 1.0
        fn_high = _beta_dist.ppf(1 - alpha, fn + 1, n_pos - fn) if n_pos > fn else 1.0
        if n_neg == 0:
            fp_high = 1.0
        if n_pos == 0:
            fn_high = 1.0

        # eq. (4), at delta = 0:
        #   eps_pairwise = max(log((1-FP^high)/FN^high), log((1-FN^high)/FP^high))
        # This is the distinguishability between the two unlearned laws P_s1, P_s2, i.e.
        # the pairwise/LDP parameter. Certified unlearning gives P_s1 ~_{2 eps} P_s2
        # through the reference law, so the certified-unlearning epsilon is half of it
        # (same factor of 2 as the avg/median tests; raw value kept as epsilon_lb_pairwise).
        def _safe_log_ratio(a, b):
            if a <= 0 or b <= 0:
                return None
            return math.log(a / b)

        term1 = _safe_log_ratio(1 - epsilon_delta - fp_high, fn_high)
        term2 = _safe_log_ratio(1 - epsilon_delta - fn_high, fp_high)

        valid_terms = [t for t in (term1, term2) if t is not None]
        m2_eps_pairwise = max(valid_terms) if valid_terms else None
        m2_eps_lb = 0.5 * m2_eps_pairwise if m2_eps_pairwise is not None else None

        # GDP analogue of eq. (4) from the same (FP^high, FN^high) point: mu-GDP means
        # the trade-off dominates G_mu, i.e. FN >= Phi(Phi^{-1}(1-FP) - mu_pairwise), so
        #   mu_pairwise >= Phi^{-1}(1-FP^high) - Phi^{-1}(FN^high),
        # and the group operation mu_loc(mu) = 2 mu makes the certified-unlearning
        # parameter mu_lb = mu_pairwise/2. Clamped at 0 (a test worse than chance
        # certifies nothing).
        from scipy.special import ndtri as _ndtri
        if 0.0 < fp_high < 1.0 and 0.0 < fn_high < 1.0:
            m2_mu_pairwise = float(_ndtri(1.0 - fp_high) - _ndtri(fn_high))
            m2_mu_pairwise = max(0.0, m2_mu_pairwise)
            m2_mu_lb = 0.5 * m2_mu_pairwise
            m2_mu_eps_estimate = eps_estimate_from_mu(m2_mu_lb, conv_delta=conv_delta)
        else:
            m2_mu_pairwise = None
            m2_mu_lb = None
            m2_mu_eps_estimate = None

        # zCDP analogue, eq:pairwise_auditor_lb: the Renyi change-of-measure inequality
        # on a single test draw, applied to A and to A^c, gives
        #   rho >= (gamma/(gamma-1) log TPR^low - log FPR^high) / eps_gamma^loc(1,gamma)
        # and its TNR/FNR mirror; the auditor reports the sup over gamma>1 of the larger
        # (clamped at 0). The local RDP bound eps_gamma^loc(rho) already carries the
        # reference-law factor, so rho_lb is a certified-unlearning rho and is NOT
        # halved -- unlike the epsilon and mu above.
        # TPR^low = 1 - FNR^high and TNR^low = 1 - FPR^high exactly for Clopper-Pearson
        # intervals, so these four rates are the same two confidence statements already
        # used for the epsilon bound (no extra confidence budget).
        m2_tpr_low = 1.0 - fn_high
        m2_tnr_low = 1.0 - fp_high
        m2_rho = rho_lb_pairwise_from_roc(
            tpr_low=m2_tpr_low,
            fpr_high=fp_high,
            tnr_low=m2_tnr_low,
            fnr_high=fn_high,
        )
        m2_rho_lb = m2_rho.get("rho_lb")
        m2_rho_eps_estimate = eps_estimate_from_rho(m2_rho_lb, conv_delta=conv_delta) \
            if (m2_rho_lb is not None and np.isfinite(m2_rho_lb)) else None

        m2_cp_result = {
            "combo_pos": combo_pos,
            "combo_neg": combo_neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_pos": n_pos, "n_neg": n_neg,
            "tpr": tp / n_pos if n_pos > 0 else None,
            "fpr": fp / n_neg if n_neg > 0 else None,
            "fp_high": fp_high,
            "fn_high": fn_high,
            "tpr_low": m2_tpr_low,
            "tnr_low": m2_tnr_low,
            "ci_alpha_each": alpha,
            "epsilon_lb": m2_eps_lb,
            "epsilon_lb_pairwise": m2_eps_pairwise,
            "mu_lb": m2_mu_lb,
            "mu_lb_pairwise": m2_mu_pairwise,
            "mu_eps_estimate": m2_mu_eps_estimate,
            "rho_lb": m2_rho_lb,
            "rho_gamma_star": m2_rho.get("gamma_star"),
            "rho_b_plus": m2_rho.get("b_plus"),
            "rho_b_minus": m2_rho.get("b_minus"),
            "rho_eps_estimate": m2_rho_eps_estimate,
            "rho_note": m2_rho.get("note"),
            "conv_delta": conv_delta,
        }

        print("\n[m=2] Direct Clopper-Pearson pairwise bounds:")
        print(f"  combo_pos={combo_pos}, combo_neg={combo_neg}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}  (n_pos={n_pos}, n_neg={n_neg})")
        if n_pos > 0:
            print(f"  TPR={tp/n_pos:.4f}, FNR={fn/n_pos:.4f}")
        if n_neg > 0:
            print(f"  FPR={fp/n_neg:.4f}")
        print(f"  FP^high={fp_high:.6f}, FN^high={fn_high:.6f}  (each alpha={alpha}, confidence={1-alpha})")
        if m2_eps_lb is not None:
            print(f"  epsilon_lb (m=2 direct CP) = {m2_eps_lb:.6f} "
                  f"(pairwise {m2_eps_pairwise:.6f} / 2)")
        else:
            print(f"  epsilon_lb (m=2 direct CP) = None (infeasible)")
        if m2_mu_lb is not None:
            print(f"  mu_lb      (m=2 direct CP) = {m2_mu_lb:.6f} "
                  f"(pairwise {m2_mu_pairwise:.6f} / 2; eps estimate at "
                  f"delta={conv_delta}: {m2_mu_eps_estimate})")
        else:
            print(f"  mu_lb      (m=2 direct CP) = None (FP^high/FN^high at the boundary)")
        print(f"  TPR^low={m2_tpr_low:.6f}, TNR^low={m2_tnr_low:.6f}")
        if m2_rho_lb is not None:
            print(f"  rho_lb     (m=2 direct CP) = {m2_rho_lb:.6f} at gamma*="
                  f"{m2_rho.get('gamma_star')} "
                  f"(b+={m2_rho.get('b_plus'):.6f}, b-={m2_rho.get('b_minus'):.6f}; "
                  f"eps estimate at delta={conv_delta}: {m2_rho_eps_estimate}) "
                  f"[not halved: factor inside eps_gamma^loc]")
        else:
            print(f"  rho_lb     (m=2 direct CP) = None")

    # Prepare results
    results = {
        "main_folder": str(main_folder),
        "unlearn_style": unlearn_style,
        "unlearn_itr": unlearn_itr,
        "metric": metric,
        "num_runs": num_runs,
        "sampling_seed": sampling_seed,
        "m": m,
        "r": r,
        "T": T,
        "epsilon_delta": epsilon_delta,
        "ci_delta": ci_delta,
        "conv_delta": conv_delta,
        "overlap_sizes": overlap_sizes,
        "overlap_ratios": overlap_ratios,
        "jaccard_scores": jaccard_scores,
        "chosen_combos": chosen_combos,
        "predicted_combos": predicted_combos,
        "v_list": v_list,
        "combo_counts": dict(combo_counts),
        "avg_v_test": avg_lb,
        "median_v_test": median_lb,
        "rho_avg_v_test": avg_rho_lb,
        "rho_median_v_test": median_rho_lb,
        "mu_avg_v_test": avg_mu_lb,
        "mu_median_v_test": median_mu_lb,
        "failed_runs": failed_runs,
        "m2_cp_result": m2_cp_result,
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute epsilon bounds by sampling combo_indices"
    )
    parser.add_argument("--main_folder", type=str, required=True,
                       help="Root directory containing runs")
    parser.add_argument("--unlearn_style", type=str, default="epoch",
                       choices=["epoch", "step"],
                       help="Unlearning style")
    parser.add_argument("--unlearn_itr", type=int, required=True,
                       help="Unlearning iteration number")
    parser.add_argument("--num_runs", type=int, required=True,
                       help="Number of runs to sample")
    parser.add_argument("--metric", type=str, default="loss",
                       choices=["phi", "loss"],
                       help="Metric to use")
    parser.add_argument("--sampling_seed", type=int, default=123,
                       help="Random seed for sampling")
    parser.add_argument("--epsilon_delta", type=float, default=0.0,
                       help="Must be 0.0: these audits are (epsilon, 0) only")
    parser.add_argument("--ci_delta", type=float, default=0.05,
                       help="Delta for confidence interval")
    parser.add_argument("--conv_delta", type=float, default=1e-3,
                       help="Delta for the rho/mu -> epsilon estimates (reporting only)")
    parser.add_argument("--mapping_file", type=str, default=None,
                       help="Path to combo_idx_to_run_file_mapping.json")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    results = compute_eps_bounds_sampled_combos(
        main_folder=args.main_folder,
        unlearn_style=args.unlearn_style,
        unlearn_itr=args.unlearn_itr,
        num_runs=args.num_runs,
        metric=args.metric,
        sampling_seed=args.sampling_seed,
        epsilon_delta=args.epsilon_delta,
        ci_delta=args.ci_delta,
        conv_delta=args.conv_delta,
        mapping_file=args.mapping_file,
        verbose=args.verbose,
    )
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
