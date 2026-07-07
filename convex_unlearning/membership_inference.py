"""
Membership Inference Attack on Unlearning Algorithm

Tests if we can distinguish which forget set was used (θ^{f1} vs θ^{f2})
by analyzing the distribution of unlearned model weights.

Approach:
1. Train model on retain + forget_half1 → get w̄_1 (before noise)
2. Train model on retain + forget_half2 → get w̄_2 (before noise)
3. Add Gaussian noise N(0, σ²I) multiple times to each w̄
4. Estimate mean and covariance for each distribution
5. Use likelihood ratio test to predict which forget set was used
"""
import json
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, beta as beta_dist


def clopper_pearson(x, n, confidence=0.95):
    """
    Clopper-Pearson (exact) confidence interval for binomial proportion p = x/n.
    Returns (lower, upper) for the proportion. Uses Beta quantiles: see e.g. Wikipedia.
    """
    if n <= 0:
        return (np.nan, np.nan)
    x = int(np.clip(x, 0, n))
    alpha = 1.0 - confidence
    if x <= 0:
        lo = 0.0
        hi = 1.0 - beta_dist.ppf(alpha / 2, n, 1)  # upper when x=0
    elif x >= n:
        lo = beta_dist.ppf(alpha / 2, n, 1)  # lower when x=n
        hi = 1.0
    else:
        lo = beta_dist.ppf(alpha / 2, x, n - x + 1)
        hi = beta_dist.ppf(1 - alpha / 2, x + 1, n - x)
    return (float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)))


def epsilon_empirical_lower_bound(tpr_low, fpr_high, delta):
    """
    Lower confidence bound on empirical epsilon from Equation (5):
    ε_emp^lower = max( log((1-δ-FP^high)/FN^high), log((1-δ-FN^high)/FP^high) ).
    Here FP^high = fpr_high (upper bound on FPR), FN^high = 1 - tpr_low (upper bound on FN rate).
    Returns np.nan when the formula is inapplicable (e.g. 1-δ - FP^high <= 0 or 1-δ - FN^high <= 0,
    or division by zero). This is unrelated to the LLR decision rule (choose higher LLR).
    """
    fn_rate_high = 1.0 - tpr_low  # FN^high
    fp_high = fpr_high
    # Valid only when 1-delta - fp_high > 0 and 1-delta - fn_rate_high > 0
    term1_ok = (1.0 - delta - fp_high) > 1e-12 and fn_rate_high > 1e-12
    term2_ok = (1.0 - delta - fn_rate_high) > 1e-12 and fp_high > 1e-12
    vals = []
    if term1_ok:
        vals.append(np.log((1.0 - delta - fp_high) / fn_rate_high))
    if term2_ok:
        vals.append(np.log((1.0 - delta - fn_rate_high) / fp_high))
    if not vals:
        return np.nan
    return float(max(vals))


def compute_w_bar_before_noise(trained_result, X_forget, y_forget,
                                X_retain, y_retain, M, L, epsilon, delta,
                                verbose=False):
    """
    Compute w̄ (intermediate weights BEFORE adding noise).
    Supports both logistic and MSE loss via unlearning.compute_hessian/compute_gradient.
    """
    from gaussian_mechanism import gaussian_mechanism_sigma_general
    from unlearning import compute_hessian, compute_gradient

    n = len(X_retain) + len(X_forget)
    m = len(X_forget)
    loss = trained_result.get('loss', 'logistic')
    include_intercept = trained_result.get('include_intercept', True)
    d_feat = X_retain.shape[1]
    d = len(trained_result['weights'])  # d_feat + 1 if intercept else d_feat
    w_original = trained_result['weights']
    hessian_full = trained_result['hessian']
    Lambda = trained_result['Lambda']
    per_sample_reg = trained_result['per_sample_reg']
    n_retain = len(X_retain)
    reg_forget = np.asarray(per_sample_reg[n_retain:n_retain + m], dtype=float)
    mu = trained_result['mu']

    gamma = (2 * M * m**2 * L**2) / (mu**3 * n**2)
    sigma = gaussian_mechanism_sigma_general(gamma, epsilon, delta)

    if verbose:
        print(f"\nComputing w̄ (before noise):")
        print(f"  γ = {gamma:.6f}")
        print(f"  σ = {sigma:.6f}")

    if include_intercept:
        X_retain_aug = np.column_stack([X_retain, np.ones(len(X_retain))])
        X_forget_aug = np.column_stack([X_forget, np.ones(len(X_forget))])
    else:
        X_retain_aug = X_retain
        X_forget_aug = X_forget

    tr = trained_result
    hessian_forget = compute_hessian(X_forget_aug, y_forget, w_original, per_sample_reg=reg_forget, loss=loss, trained_result=tr)
    hessian_hat = (hessian_full - hessian_forget) / (n - m)
    reg_retain = np.asarray(per_sample_reg[:n_retain], dtype=float)
    hessian_retain = compute_hessian(X_retain_aug, y_retain, w_original, per_sample_reg=reg_retain, loss=loss, trained_result=tr)

    if verbose:
        print(f"  Condition number of Ĥ: {np.linalg.cond(hessian_hat):.2e}")

    grad_forget = compute_gradient(X_forget_aug, y_forget, w_original, per_sample_reg=reg_forget, loss=loss, trained_result=tr)
    try:
        hessian_inv = np.linalg.inv(hessian_hat)
    except np.linalg.LinAlgError:
        hessian_inv = np.linalg.inv(hessian_hat + 1e-6 * np.eye(d))
    w_bar = w_original + (1.0 / (n - m)) * (hessian_inv @ grad_forget)

    if verbose:
        print(f"  ||w̄ - ŵ|| = {np.linalg.norm(w_bar - w_original):.6f}")

    return {
        'w_bar': w_bar,
        'w_original': w_original,
        'gamma': gamma,
        'sigma': sigma,
        'hessian_retain': hessian_retain,
        'hessian_hat': hessian_hat,
        'M': M,
        'L': L,
        'per_sample_reg': per_sample_reg,
    }


def verify_lemma3_bound(trained_result, X_forget, y_forget, X_retain, y_retain,
                        M, L, epsilon=1.0, delta=0.01, w_retrain=None, verbose=True):
    """
    Compare ||w_bar - ŵ'|| to the Lemma 3 bound γ = (2 M L² m²)/(λ³ n²).
    
    Notation:
      w_bar   = one Newton step unlearning (before noise) — algorithm output
      ŵ'      = minimizer on RETAIN only (S\\U)         — true retain optimum = w_retrain
    We report: empirical = ||w_bar - w_retrain|| (distance of unlearning output from true ŵ')
    and compare to the bound γ.
    
    Parameters:
    -----------
    trained_result : dict from train() on full S
    X_forget, y_forget : forget set U (m samples)
    X_retain, y_retain : retain set S\\U
    M, L : Lipschitz constants
    epsilon, delta : for computing γ
    w_retrain : np.ndarray. Weights from retraining on (X_retain, y_retain) only = true ŵ'. Required.
    verbose : bool
        Print step-by-step summary.
    
    Returns:
    --------
    dict with empirical_lemma3, bound_lemma3, ratio_lemma3 (Lemma 3),
    empirical_lemma6, bound_lemma6, ratio_lemma6 (Lemma 6), and M, L, Lambda, lambda, norms.
    """
    result = compute_w_bar_before_noise(
        trained_result, X_forget, y_forget, X_retain, y_retain,
        M=M, L=L, epsilon=epsilon, delta=delta, verbose=False
    )
    w_bar = result['w_bar']            # unlearning output (before noise)
    w_original = result['w_original'] # optimum on full S = w̄
    bound = result['gamma']            # Lemma 3 bound γ = (2 M L² m²)/(λ³ n²)
    
    if w_retrain is None:
        raise ValueError("w_retrain is required (retain-only optimum ŵ'). Retrain on (X_retain, y_retain) and pass .weights")
    w_retrain = np.asarray(w_retrain)
    if len(w_retrain) != len(w_original):
        raise ValueError("w_retrain must have same length as trained weights (incl. intercept)")
    
    empirical = np.linalg.norm(w_bar - w_retrain)
    ratio = empirical / bound if bound > 0 else np.nan
    n = len(X_retain) + len(X_forget)
    m = len(X_forget)
    Lambda = trained_result['Lambda']
    mu = trained_result['mu']  # effective strong convexity

    # Lemma 6: ||ŵ - ŵ'|| ≤ (2 m L)/(μ n); ŵ = w_original, ŵ' = w_retrain
    w_original_retrain_diff = np.linalg.norm(w_original - w_retrain)
    lemma6_bound = (2.0 * m * L) / (mu * n) if (mu * n) > 0 else np.nan
    lemma6_ratio = w_original_retrain_diff / lemma6_bound if lemma6_bound > 0 else np.nan

    out = {
        'empirical_lemma3': empirical,
        'bound_lemma3': bound,
        'ratio_lemma3': ratio,
        'M': M,
        'L': L,
        'Lambda': Lambda,
        'lambda': mu,
        'w_bar_norm': float(np.linalg.norm(w_bar)),
        'w_original_norm': float(np.linalg.norm(w_original)),
        'w_retrain_norm': float(np.linalg.norm(w_retrain)),
        'empirical_lemma6': float(w_original_retrain_diff),
        'bound_lemma6': float(lemma6_bound),
        'ratio_lemma6': float(lemma6_ratio),
        'w_bar': w_bar,
        'w_original': w_original,
        'w_retrain': w_retrain,
    }
    
    if verbose:
        print("\n" + "="*70)
        print("LEMMA 3 CHECK: ||w_bar - ŵ'|| vs bound γ = (2 M L² m²)/(λ³ n²)")
        print("="*70)
        print("  w_bar  = one Newton step unlearning (before noise)")
        print("  ŵ'     = minimizer on RETAIN only (S\\U) = w_retrain")
        print(f"  n = {n},  m = {m},  λ = Λ/n")
        print(f"\n  1. Bound:      γ = {bound:.6g}")
        print(f"  2. Empirical: ||w_bar - ŵ'|| = {empirical:.6g}")
        print(f"  3. Ratio:     empirical / bound = {ratio:.4f}")
        if ratio < 0.1:
            print(f"     → Bound is LOOSE (empirical << bound)")
        elif ratio < 0.5:
            print(f"     → Bound is moderately loose")
        elif ratio <= 1.0:
            print(f"     → Empirical ≤ bound (within guarantee)")
        else:
            print(f"     → WARNING: empirical > bound")
        print("="*70)
        print("\nLEMMA 6 CHECK: ||ŵ - ŵ'|| ≤ (2 m L)/(λ n)")
        print("  ŵ = full-set optimum (w_original),  ŵ' = retain-only optimum (w_retrain)")
        print(f"  Lemma 6 bound:  (2 m L)/(λ n) = {lemma6_bound:.6g}")
        print(f"  Empirical:      ||ŵ - ŵ'|| = {w_original_retrain_diff:.6g}")
        print(f"  Ratio:         empirical / bound = {lemma6_ratio:.4f}")
        if lemma6_ratio <= 1.0:
            print("  → Empirical ≤ bound (Lemma 6 satisfied)")
        else:
            print("  → WARNING: empirical > bound")
        print("="*70)
    
    return out


def _lemma_result_to_json_serializable(lemma_result):
    """Extract JSON-serializable scalars from one lemma verification result."""
    return {
        'bound_lemma3': float(lemma_result['bound_lemma3']),
        'empirical_lemma3': float(lemma_result['empirical_lemma3']),
        'ratio_lemma3': float(lemma_result['ratio_lemma3']),
        'empirical_lemma6': float(lemma_result['empirical_lemma6']),
        'bound_lemma6': float(lemma_result['bound_lemma6']),
        'ratio_lemma6': float(lemma_result['ratio_lemma6']),
        'M': float(lemma_result['M']),
        'L': float(lemma_result['L']),
        'Lambda': float(lemma_result['Lambda']),
        'lambda': float(lemma_result['lambda']),
        'w_bar_norm': float(lemma_result['w_bar_norm']),
        'w_original_norm': float(lemma_result['w_original_norm']),
        'w_retrain_norm': float(lemma_result['w_retrain_norm']),
    }


def save_lemma_verification(lemma_result_f1, lemma_result_f2, n, m, filepath, single_partition=False):
    """
    Save Lemma 3 and Lemma 6 verification to a JSON file.

    When single_partition is False (default): two forget sets (f1 and f2).
      lemma_result_f1: from verify_lemma3_bound with trained_f1 and forget_first_half.
      lemma_result_f2: from verify_lemma3_bound with trained_f2 and forget_second_half.
    When single_partition is True: one forget set only; lemma_result_f2 is ignored.
      Saves a single 'forget' entry.
    """
    import json
    w_retrain_norm = float(lemma_result_f1['w_retrain_norm'])
    if single_partition:
        out = {
            'n': n,
            'm': m,
            'w_retrain_norm': w_retrain_norm,
            'forget': _lemma_result_to_json_serializable(lemma_result_f1),
        }
    else:
        out = {
            'n': n,
            'm': m,
            'w_retrain_norm': w_retrain_norm,
            'forget_f1': _lemma_result_to_json_serializable(lemma_result_f1),
            'forget_f2': _lemma_result_to_json_serializable(lemma_result_f2),
        }
    # For cubic (e1/e2): retain R = all zeros => minimizer on R is w=0
    if w_retrain_norm == 0:
        out['_note'] = 'w_retrain_norm=0 is expected for cubic (e1/e2): retain set R is all zeros, so ERM on R is w=0.'
    with open(filepath, 'w') as f:
        json.dump(out, f, indent=2)
    if single_partition:
        print(f"\nLemma verification (3 & 6) for forget set saved to: {filepath}")
    else:
        print(f"\nLemma verification (3 & 6) for both forget sets saved to: {filepath}")


def generate_unlearned_models(w_bar, sigma, n_samples, random_state=None):
    """
    Generate multiple unlearned models by adding Gaussian noise to w̄.
    
    Parameters:
    -----------
    w_bar : np.ndarray
        Intermediate weights (before noise)
    sigma : float
        Noise standard deviation
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed
        
    Returns:
    --------
    models : np.ndarray, shape (n_samples, d+1)
        Generated unlearned models
    """
    rng = np.random.RandomState(random_state)
    d = len(w_bar)
    
    # Generate noise samples
    noise_samples = rng.normal(0, sigma, size=(n_samples, d))
    
    # Add noise to w̄
    models = w_bar + noise_samples
    
    return models


def estimate_distribution(models, verbose=False, cov_regularization=1e-8):
    """
    Estimate mean and covariance of model distribution.
    
    Parameters:
    -----------
    models : np.ndarray, shape (n_samples, d)
        Sampled models
    verbose : bool
        Print information
    cov_regularization : float
        Added to diagonal of covariance to avoid singular/ill-conditioned matrices
        (prevents NaN from logpdf when using this dist in LLR).
        
    Returns:
    --------
    dict with:
        mean : np.ndarray
            Estimated mean
        cov : np.ndarray
            Estimated covariance (regularized)
        std : np.ndarray
            Standard deviations
    """
    mean = np.atleast_1d(np.mean(models, axis=0))
    cov = np.atleast_2d(np.cov(models, rowvar=False))
    d = cov.shape[0]
    # Avoid singular/ill-conditioned cov so LLR logpdf doesn't produce NaN
    reg = cov_regularization * max(1.0, np.trace(cov) / d)
    cov = cov + reg * np.eye(d)
    std = np.std(models, axis=0)
    
    if verbose:
        print(f"\nDistribution estimation:")
        print(f"  Estimated mean: {mean[:3]}... (showing first 3)")
        print(f"  Mean std: {np.mean(std):.6f}")
        print(f"  Cov condition number: {np.linalg.cond(cov):.2e}")
    
    return {
        'mean': mean,
        'cov': cov,
        'std': std
    }


def log_likelihood_ratio(model, dist1, dist2):
    """
    Compute log-likelihood ratio for a model under two distributions.
    
    LLR = log(P(model | dist1)) - log(P(model | dist2))
    
    If LLR > 0, model is more likely from dist1.
    If LLR < 0, model is more likely from dist2.
    
    Parameters:
    -----------
    model : np.ndarray
        Model weights to test
    dist1 : dict
        Distribution 1 (mean, cov)
    dist2 : dict
        Distribution 2 (mean, cov)
        
    Returns:
    --------
    llr : float
        Log-likelihood ratio (or simplified version if numerical issues occur)
    """
    # Compute log-likelihoods
    try:
        mvn1 = multivariate_normal(mean=dist1['mean'], cov=dist1['cov'], 
                                   allow_singular=True)
        mvn2 = multivariate_normal(mean=dist2['mean'], cov=dist2['cov'],
                                   allow_singular=True)
        
        log_prob1 = mvn1.logpdf(model)
        log_prob2 = mvn2.logpdf(model)
        
        # Check for invalid values (NaN or Inf)
        if not np.isfinite(log_prob1) or not np.isfinite(log_prob2):
            # Use simplified version if we get NaN/Inf
            diff1 = model - dist1['mean']
            diff2 = model - dist2['mean']
            llr = np.linalg.norm(diff2) - np.linalg.norm(diff1)
        else:
            llr = log_prob1 - log_prob2
            # Double-check the result is valid
            if not np.isfinite(llr):
                diff1 = model - dist1['mean']
                diff2 = model - dist2['mean']
                llr = np.linalg.norm(diff2) - np.linalg.norm(diff1)
        
    except Exception as e:
        # If singular or other exception, use simplified version
        diff1 = model - dist1['mean']
        diff2 = model - dist2['mean']
        llr = np.linalg.norm(diff2) - np.linalg.norm(diff1)
    
    return llr


def log_likelihood_under_dist(model, dist):
    """
    Log-likelihood of model under a single Gaussian distribution (mean, cov).
    Used for multi-configuration prediction: choose config with largest log p(model | dist).
    """
    try:
        mvn = multivariate_normal(mean=dist['mean'], cov=dist['cov'], allow_singular=True)
        log_prob = mvn.logpdf(model)
        if np.isfinite(log_prob):
            return float(log_prob)
    except Exception:
        pass
    # Fallback: negative squared L2 distance to mean
    diff = model - dist['mean']
    return float(-np.linalg.norm(diff)**2)


def predict_configuration_by_llr(model, distributions_by_tuple):
    """
    Predict which configuration (tuple) the model came from by maximum log-likelihood.
    distributions_by_tuple : dict mapping tuple (e.g. (0,1,2)) -> dist dict with 'mean', 'cov'.
    Returns the tuple with largest log p(model | dist). Ties broken lexicographically.
    """
    best_logp = -np.inf
    best_tuple = None
    for t, dist in distributions_by_tuple.items():
        logp = log_likelihood_under_dist(model, dist)
        if logp > best_logp or (logp == best_logp and (best_tuple is None or t < best_tuple)):
            best_logp = logp
            best_tuple = t
    return best_tuple


def run_membership_inference_attack_two_partition(data, trained_f1, trained_f2, 
                                    epsilon, delta=0.01, n_samples_per_dist=1000,
                                    n_test=500, random_state=42, confidence=0.95, ci_delta=0.05):
    """
    Run complete membership inference attack.
    
    Parameters:
    -----------
    data : dict
        Dataset with X_retain, X_partitions (list of at least 2 arrays)
    trained_f1 : dict
        Trained model on retain + first_half
    trained_f2 : dict
        Trained model on retain + second_half
    epsilon : float
        Privacy budget (ε)
    delta : float
        Privacy parameter (δ) - probability bound for privacy failure
        Common values: 0.001 (very strict), 0.01 (standard), 0.1 (relaxed)
    n_samples_per_dist : int
        Number of samples to generate for each distribution
    n_test : int
        Number of test samples for attack
    random_state : int
        Random seed
    confidence : float
        Clopper-Pearson confidence level for TPR/FPR bounds (e.g. 0.95)
    ci_delta : float
        Confidence level for ε_lb (avg-v test bound); target probability in M(c)*ratio(eps)^T <= ci_delta (e.g. 0.05)
        
    Returns:
    --------
    dict with attack results
    """
    rng = np.random.RandomState(random_state)
    
    print("\n" + "="*70)
    print("MEMBERSHIP INFERENCE ATTACK")
    print("="*70)
    print(f"Privacy parameters: ε = {epsilon}, δ = {delta}")
    print(f"  → (ε, δ)-differential privacy")
    print(f"  → δ is the probability bound for privacy failure")
    print(f"Samples per distribution: {n_samples_per_dist}")
    print(f"Test samples: {n_test}")
    
    # Step 1: Compute w̄ for both forget sets (everything before noise)
    print("\n### Computing w̄ for First Half (θ^{f1}) ###")
    result_f1 = compute_w_bar_before_noise(
        trained_result=trained_f1,
        X_forget=data['X_partitions'][0],
        y_forget=np.zeros(len(data['X_partitions'][0])),
        X_retain=data['X_retain'],
        y_retain=data['y_retain'],
        M=trained_f1['M'],
        L=trained_f1['L'],
        epsilon=epsilon,
        delta=delta,
        verbose=True
    )
    
    print("\n### Computing w̄ for Second Half (θ^{f2}) ###")
    result_f2 = compute_w_bar_before_noise(
        trained_result=trained_f2,
        X_forget=data['X_partitions'][1],
        y_forget=np.zeros(len(data['X_partitions'][1])),
        X_retain=data['X_retain'],
        y_retain=data['y_retain'],
        M=trained_f2['M'],
        L=trained_f2['L'],
        epsilon=epsilon,
        delta=delta,
        verbose=True
    )
    
    # Step 2: Generate multiple unlearned models for each
    print(f"\n### Generating {n_samples_per_dist} samples from each distribution ###")
    
    models_f1 = generate_unlearned_models(
        result_f1['w_bar'], 
        result_f1['sigma'],
        n_samples_per_dist,
        random_state=rng.randint(0, 10000)
    )
    
    models_f2 = generate_unlearned_models(
        result_f2['w_bar'],
        result_f2['sigma'],
        n_samples_per_dist,
        random_state=rng.randint(0, 10000)
    )
    
    print(f"Generated models from first half:  shape {models_f1.shape}")
    print(f"Generated models from second half: shape {models_f2.shape}")
    
    # Step 3: Estimate distributions
    print("\n### Estimating Distributions ###")
    dist_f1 = estimate_distribution(models_f1, verbose=True)
    print("Distribution for First Half (θ^{f1}):")
    
    dist_f2 = estimate_distribution(models_f2, verbose=True)
    print("Distribution for Second Half (θ^{f2}):")
    
    # Step 4: Generate test samples
    # For each of 2*n_test positions, choose f1 (1) or f2 (0) uniformly at random.
    which_f1 = rng.randint(0, 2, size=2 * n_test)  # 1 = f1, 0 = f2
    n_f1 = int(which_f1.sum())
    n_f2 = 2 * n_test - n_f1
    print(f"\n### Generating test samples: f1/f2 chosen uniformly at random per position → {n_f1} from f1, {n_f2} from f2 ###")
    test_f1 = generate_unlearned_models(
        result_f1['w_bar'],
        result_f1['sigma'],
        n_f1,
        random_state=rng.randint(0, 10000)
    )
    test_f2 = generate_unlearned_models(
        result_f2['w_bar'],
        result_f2['sigma'],
        n_f2,
        random_state=rng.randint(0, 10000)
    )
    # Build ordered list: position i gets next model from f1 or f2 according to which_f1[i]
    test_models_ordered = []
    idx_f1, idx_f2 = 0, 0
    for i in range(2 * n_test):
        if which_f1[i] == 1:
            test_models_ordered.append(test_f1[idx_f1])
            idx_f1 += 1
        else:
            test_models_ordered.append(test_f2[idx_f2])
            idx_f2 += 1
    
    # Step 5: Run attack - compute LLR for each test sample (in order)
    print("\n### Running Attack ###")
    llrs_all = np.array([
        log_likelihood_ratio(model, dist_f1, dist_f2) for model in test_models_ordered
    ])
    all_labels = which_f1.copy()  # 1 = f1, 0 = f2 (true label)
    
    # NaN is not interpretable → raise. ±inf are valid: +inf → predict f1, -inf → predict f2.
    if np.any(np.isnan(llrs_all)):
        n_nan = int(np.sum(np.isnan(llrs_all)))
        raise ValueError(f"Found {n_nan} NaN LLR(s); cannot proceed.")
    
    all_llrs = llrs_all
    # For return/plots: split by true label
    llrs_f1_valid = all_llrs[all_labels == 1]
    llrs_f2_valid = all_llrs[all_labels == 0]
    llrs_f1 = llrs_all[which_f1 == 1]
    llrs_f2 = llrs_all[which_f1 == 0]
    n_invalid_f1 = int(np.sum(np.isnan(llrs_f1)))
    n_invalid_f2 = int(np.sum(np.isnan(llrs_f2)))
    
    # Step 6: Classify using LLR threshold
    # Prediction: LLR > 0 → f1 (1), LLR < 0 → f2 (0). ±inf are valid (+inf → 1, -inf → 0).
    all_predictions = (all_llrs > 0).astype(int)
    
    # Step 7: Compute metrics
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    # Rates
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)
    fpr = fp / (fp + tn)  # False Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate (Specificity)
    fnr = fn / (fn + tp)  # False Negative Rate
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # ROC curve
    fpr_curve, tpr_curve, thresholds = roc_curve(all_labels, all_llrs)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    # Clopper-Pearson confidence bounds for TPR and FPR
    n_pos = int(tp + fn)
    n_neg = int(fp + tn)
    tpr_low, tpr_high = clopper_pearson(int(tp), n_pos, confidence=confidence)
    fpr_low, fpr_high = clopper_pearson(int(fp), n_neg, confidence=confidence)
    epsilon_emp_lower = epsilon_empirical_lower_bound(tpr_low, fpr_high, delta=1e-8) / 2.0
    
    # Avg-v-test epsilon lower bound: v_list[i] = 2 if correct, 0 if wrong (m=2, r=2)
    T = len(all_predictions)  # 2*n_test
    v_list = [2 if all_predictions[i] == all_labels[i] else 0 for i in range(T)]
    try:
        from cum_runs_eps_lab import compute_avg_v_test_epsilon_lb
        avg_v_res = compute_avg_v_test_epsilon_lb(
            m=2, r=2, T=T, v_list=v_list,
            delta=1e-8, ci_delta=ci_delta
        )
        epsilon_lb_avg_v = avg_v_res["epsilon_lb"] / 2.0
    except Exception as e:
        epsilon_lb_avg_v = np.nan
        import sys
        print(f"  Warning: avg-v-test epsilon_lb failed: {e}", file=sys.stderr)

    # Print results
    print("\n" + "="*70)
    print("ATTACK RESULTS")
    print("="*70)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted F1    Predicted F2")
    print(f"  Actual F1     {tp:<15} {fn:<15}")
    print(f"  Actual F2     {fp:<15} {tn:<15}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  TPR (Recall/Sensitivity): {tpr:.4f}  [{tpr_low:.4f}, {tpr_high:.4f}] (CP {confidence:.0%})")
    print(f"  TNR (Specificity):        {tnr:.4f}")
    print(f"  FPR:                      {fpr:.4f}  [{fpr_low:.4f}, {fpr_high:.4f}] (CP {confidence:.0%})")
    print(f"  FNR:                      {fnr:.4f}")
    print(f"  ROC AUC:                  {roc_auc:.4f}")
    if not np.isnan(epsilon_emp_lower):
        print(f"  ε_emp^lower (δ={delta}):     {epsilon_emp_lower:.4f}")
    else:
        print(f"  ε_emp^lower (δ={delta}):     (invalid: 1-δ-FP^high or 1-δ-FN^high <= 0)")
    if not np.isnan(epsilon_lb_avg_v):
        print(f"  ε_lb (avg-v test, m=2,r=2): {epsilon_lb_avg_v:.4f}")
    else:
        print(f"  ε_lb (avg-v test, m=2,r=2): (failed)")
    
    print(f"\nInterpretation:")
    if roc_auc > 0.9:
        print(f"  ❌ SEVERE PRIVACY LEAK - Attack highly successful!")
    elif roc_auc > 0.7:
        print(f"  ⚠  MODERATE PRIVACY LEAK - Attack moderately successful")
    elif roc_auc > 0.6:
        print(f"  ⚠  WEAK PRIVACY LEAK - Attack slightly successful")
    else:
        print(f"  ✓ ATTACK FAILED - Privacy preserved (AUC ≈ 0.5 is random guessing)")
    
    return {
        'result_f1': result_f1,
        'result_f2': result_f2,
        'models_f1': models_f1,
        'models_f2': models_f2,
        'dist_f1': dist_f1,
        'dist_f2': dist_f2,
        'test_f1': test_f1,
        'test_f2': test_f2,
        'llrs_f1': llrs_f1,  # Original (may contain invalid values)
        'llrs_f2': llrs_f2,  # Original (may contain invalid values)
        'llrs_f1_valid': llrs_f1_valid,  # Filtered valid LLRs
        'llrs_f2_valid': llrs_f2_valid,  # Filtered valid LLRs
        'n_invalid_f1': n_invalid_f1,
        'n_invalid_f2': n_invalid_f2,
        'predictions': all_predictions,
        'labels': all_labels,
        'llrs': all_llrs,  # Filtered valid LLRs (used for metrics)
        'confusion_matrix': (tn, fp, fn, tp),
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'accuracy': accuracy,
        'precision': precision,
        'roc_auc': roc_auc,
        'fpr_curve': fpr_curve,
        'tpr_curve': tpr_curve,
        'epsilon': epsilon,
        'tpr_low': tpr_low,
        'tpr_high': tpr_high,
        'fpr_low': fpr_low,
        'fpr_high': fpr_high,
        'epsilon_empirical_lower': epsilon_emp_lower,
        'epsilon_lb_avg_v': epsilon_lb_avg_v,
        'confidence': confidence,
    }


def run_membership_inference_attack_single_partition(
    data, trained_retain, trained_full,
    epsilon, delta=0.01, n_samples_per_dist=1000, n_test=500,
    random_state=42, confidence=0.95, ci_delta=0.05
):
    """
    Single-partition hypothesis test: distinguish
    - Model A: trained on retain only, then add same noise σ → N(w_retain, σ² I)
    - Model B: trained on retain union forget, unlearn on forget → N(w̄, σ² I)

    Both distributions are estimated by sampling n_samples_per_dist times and
    computing mean/covariance. Inference on 2*n_test samples (n_test per class).

    Parameters:
    -----------
    data : dict
        Must have X_retain, y_retain, X_forget, y_forget (single forget set).
    trained_retain : dict
        Result of train() on (X_retain, y_retain) only.
    trained_full : dict
        Result of train() on (X_retain + X_forget, y_retain + y_forget).
    epsilon, delta, n_samples_per_dist, n_test, random_state, confidence, ci_delta
        Same as run_membership_inference_attack_two_partition.

    Returns:
    --------
    dict with same keys as run_membership_inference_attack_two_partition (result_f1/f2 = unlearn/retain,
    dist_f1/f2 estimated from samples) so save_attack_results, plot_roc_curve, plot_llr_distributions work.
    """
    rng = np.random.RandomState(random_state)
    X_retain = data['X_retain']
    y_retain = data['y_retain']
    X_forget = data['X_forget']
    y_forget = data['y_forget']

    print("\n" + "="*70)
    print("SINGLE-PARTITION MEMBERSHIP INFERENCE ATTACK")
    print("  (Retain+noise vs Retain+Forget→Unlearn+noise; same σ)")
    print("="*70)
    print(f"Privacy parameters: ε = {epsilon}, δ = {delta}")
    print(f"Samples per distribution (for mean/cov): {n_samples_per_dist}")
    print(f"Test samples per class: {n_test} (total {2*n_test})")

    w_retain = np.asarray(trained_retain['weights'])
    d = len(w_retain)

    print("\n### Computing w̄ for Unlearn (retain+forget → unlearn forget) ###")
    result_unlearn = compute_w_bar_before_noise(
        trained_result=trained_full,
        X_forget=X_forget,
        y_forget=y_forget,
        X_retain=X_retain,
        y_retain=y_retain,
        M=trained_full['M'],
        L=trained_full['L'],
        epsilon=epsilon,
        delta=delta,
        verbose=True,
    )
    w_bar = result_unlearn['w_bar']
    sigma = result_unlearn['sigma']

    # Same noise σ for both: retain ~ N(w_retain, σ² I), unlearn ~ N(w̄, σ² I)
    print(f"\n### Same σ = {sigma:.6f} for both retain and unlearn ###")

    # Step 1: Sample n_samples_per_dist from each distribution and estimate mean/cov
    print(f"\n### Generating {n_samples_per_dist} samples per distribution (for estimation) ###")
    models_retain = generate_unlearned_models(
        w_retain, sigma, n_samples_per_dist, random_state=rng.randint(0, 10000)
    )
    models_unlearn = generate_unlearned_models(
        w_bar, sigma, n_samples_per_dist, random_state=rng.randint(0, 10000)
    )
    print("Estimating Retain distribution (mean, cov) from samples...")
    dist_retain = estimate_distribution(models_retain, verbose=True)
    print("Estimating Unlearn distribution (mean, cov) from samples...")
    dist_unlearn = estimate_distribution(models_unlearn, verbose=True)

    result_retain = {'w_bar': w_retain.copy(), 'sigma': sigma}

    # Step 2: Generate test set: n_test from retain, n_test from unlearn (random order)
    which_unlearn = rng.randint(0, 2, size=2 * n_test)
    n_unlearn_actual = int(which_unlearn.sum())
    n_retain_actual = 2 * n_test - n_unlearn_actual
    test_retain = generate_unlearned_models(
        w_retain, sigma, n_retain_actual, random_state=rng.randint(0, 10000)
    )
    test_unlearn = generate_unlearned_models(
        w_bar, sigma, n_unlearn_actual, random_state=rng.randint(0, 10000)
    )
    test_models_ordered = []
    idx_r, idx_u = 0, 0
    for i in range(2 * n_test):
        if which_unlearn[i] == 1:
            test_models_ordered.append(test_unlearn[idx_u])
            idx_u += 1
        else:
            test_models_ordered.append(test_retain[idx_r])
            idx_r += 1
    test_models_ordered = np.array(test_models_ordered)
    all_labels = which_unlearn.copy()

    print("\n### Running Attack (LLR: unlearn vs retain, using estimated distributions) ###")
    llrs_all = np.array([
        log_likelihood_ratio(model, dist_unlearn, dist_retain)
        for model in test_models_ordered
    ])
    # NaN can occur if estimated cov is ill-conditioned or logpdf underflows; treat as tie (predict retain)
    n_nan = int(np.sum(np.isnan(llrs_all)))
    if n_nan > 0:
        llrs_all = np.where(np.isfinite(llrs_all), llrs_all, 0.0)
        print(f"  Note: {n_nan} NaN LLR(s) replaced with 0 (random guess).")
    all_llrs = llrs_all
    llrs_retain = all_llrs[all_labels == 0]
    llrs_unlearn = all_llrs[all_labels == 1]
    # Map so F1 = unlearn (LLR > 0), F2 = retain (LLR < 0) for consistent plot legend
    llrs_f1_valid = llrs_unlearn
    llrs_f2_valid = llrs_retain

    all_predictions = (all_llrs > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr_curve, tpr_curve, _ = roc_curve(all_labels, all_llrs)
    roc_auc = auc(fpr_curve, tpr_curve)

    n_pos = int(tp + fn)
    n_neg = int(fp + tn)
    tpr_low, tpr_high = clopper_pearson(int(tp), n_pos, confidence=confidence)
    fpr_low, fpr_high = clopper_pearson(int(fp), n_neg, confidence=confidence)
    epsilon_emp_lower = epsilon_empirical_lower_bound(tpr_low, fpr_high, delta)

    T = len(all_predictions)
    v_list = [2 if all_predictions[i] == all_labels[i] else 0 for i in range(T)]
    try:
        from cum_runs_eps_lab import compute_avg_v_test_epsilon_lb
        avg_v_res = compute_avg_v_test_epsilon_lb(
            m=2, r=2, T=T, v_list=v_list, delta=delta, ci_delta=ci_delta
        )
        raw = avg_v_res["epsilon_lb"]
        # Coerce to scalar float (module may return dict/array)
        epsilon_lb_avg_v = float(np.asarray(raw).ravel()[0]) if np.size(raw) else np.nan
    except Exception as e:
        epsilon_lb_avg_v = np.nan
        import sys
        print(f"  Warning: avg-v-test epsilon_lb failed: {e}", file=sys.stderr)

    def _valid_float(x):
        try:
            v = float(np.asarray(x).ravel()[0]) if np.size(x) else np.nan
            return np.isfinite(v)
        except (TypeError, ValueError, IndexError):
            return False

    print("\n" + "="*70)
    print("ATTACK RESULTS (Single partition)")
    print("="*70)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Retain   Predicted Unlearn")
    print(f"  Actual Retain     {tn:<15} {fp:<15}")
    print(f"  Actual Unlearn    {fn:<15} {tp:<15}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  TPR (Unlearn): {tpr:.4f}  [{tpr_low:.4f}, {tpr_high:.4f}]")
    print(f"  FPR:            {fpr:.4f}  [{fpr_low:.4f}, {fpr_high:.4f}]")
    print(f"  ROC AUC:        {roc_auc:.4f}")
    if _valid_float(epsilon_emp_lower):
        print(f"  ε_emp^lower (δ={delta}): {epsilon_emp_lower:.4f}")
    else:
        print(f"  ε_emp^lower (δ={delta}): — (formula inapplicable for this TPR/FPR)")
    if _valid_float(epsilon_lb_avg_v):
        print(f"  ε_lb (avg-v test):      {epsilon_lb_avg_v:.4f}")
    if roc_auc > 0.6:
        print(f"  ⚠  Leak - can distinguish retain-only vs unlearned")
    else:
        print(f"  ✓ Hard to distinguish retain-only vs unlearned")

    # Return with F1 = unlearn, F2 = retain so plot "F1 (should be > 0)" is correct
    return {
        'result_f1': result_unlearn,
        'result_f2': result_retain,
        'models_f1': models_unlearn,
        'models_f2': models_retain,
        'dist_f1': dist_unlearn,
        'dist_f2': dist_retain,
        'test_f1': test_unlearn,
        'test_f2': test_retain,
        'llrs_f1': llrs_unlearn,
        'llrs_f2': llrs_retain,
        'llrs_f1_valid': llrs_f1_valid,
        'llrs_f2_valid': llrs_f2_valid,
        'n_invalid_f1': 0,
        'n_invalid_f2': 0,
        'predictions': all_predictions,
        'labels': all_labels,
        'llrs': all_llrs,
        'confusion_matrix': (tn, fp, fn, tp),
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'accuracy': accuracy,
        'precision': precision,
        'roc_auc': roc_auc,
        'fpr_curve': fpr_curve,
        'tpr_curve': tpr_curve,
        'epsilon': epsilon,
        'tpr_low': tpr_low,
        'tpr_high': tpr_high,
        'fpr_low': fpr_low,
        'fpr_high': fpr_high,
        'epsilon_empirical_lower': epsilon_emp_lower,
        'epsilon_lb_avg_v': epsilon_lb_avg_v,
        'confidence': confidence,
    }


def run_multi_partition_attack(data, epsilon, delta, n_samples_per_dist, n_test, K, ci_delta,
                              cubic_params, per_sample_reg, max_iter,
                              X_val, y_val, X_test, y_test,
                              random_state=42, verbose=True, n_details=20):
    """
    Multi-partition membership inference: C(K, K/2) configs, predict by max log-likelihood,
    v_list = 2 * overlap, then compute_avg_v_test_epsilon_lb(m=K, r=K, ...).

    data : dict with X_retain, y_retain, X_partitions (list of K arrays), partition_tuples (list of tuples).
    n_details : int. For the first n_details test samples, record full prediction details (true/pred, sigma, w_bar, obs, etc.) for saving.
    """
    from training import train
    rng = np.random.RandomState(random_state)

    X_retain = data['X_retain']
    y_retain = data['y_retain']
    X_partitions = data['X_partitions']
    partition_tuples = data['partition_tuples']

    n_retain = len(X_retain)
    n_forget_per_partition = data['n_forget_per_partition']
    half = K // 2
    n_forget_total = half * n_forget_per_partition

    if verbose:
        print("\n" + "="*70)
        print("MULTI-PARTITION MEMBERSHIP INFERENCE ATTACK")
        print("="*70)
        print(f"  K={K}, forget set size = K/2 = {half}, configurations = {len(partition_tuples)}")
        print(f"  ε={epsilon}, δ={delta}, n_samples_per_dist={n_samples_per_dist}, n_test={n_test}, ci_delta={ci_delta}")

    # Per-configuration: train, w_bar, sigma, sample n_samples_per_dist, estimate distribution
    distributions_by_tuple = {}
    w_bar_by_tuple = {}
    sigma_by_tuple = {}
    for idx, tup in enumerate(partition_tuples):
        if verbose:
            print(f"\n  Configuration {idx+1}/{len(partition_tuples)}: tuple {tup}")
        X_forget = np.vstack([X_partitions[i] for i in tup])
        y_forget = np.zeros(len(X_forget))
        X_train = np.vstack([X_retain, X_forget])
        y_train = np.zeros(len(X_train))
        lam = np.atleast_1d(np.asarray(per_sample_reg, dtype=float)).flat[0]
        per_sample_reg_arr = np.full(len(X_train), lam)

        trained = train(
            X_train, y_train, X_val, y_val, X_test, y_test,
            per_sample_reg=per_sample_reg_arr,
            loss='cubic',
            max_iter=max_iter,
            random_state=int(rng.randint(0, 1e6)),
            verbose=False,
            loss_params=cubic_params
        )
        result = compute_w_bar_before_noise(
            trained_result=trained,
            X_forget=X_forget,
            y_forget=y_forget,
            X_retain=X_retain,
            y_retain=y_retain,
            M=trained['M'],
            L=trained['L'],
            epsilon=epsilon,
            delta=delta,
            verbose=False
        )
        w_bar_by_tuple[tup] = result['w_bar'].copy()
        sigma_by_tuple[tup] = result['sigma']
        models = generate_unlearned_models(
            result['w_bar'], result['sigma'], n_samples_per_dist,
            random_state=int(rng.randint(0, 1e6))
        )
        dist = estimate_distribution(models, verbose=False)
        distributions_by_tuple[tup] = dist

    # Inference: n_test times sample config uniformly, generate test sample as w_bar + N(0, sigma^2 I) for that config, predict by LLR
    v_list = []
    prediction_details = []  # first n_details samples: full info for inspection/saving
    for i in range(n_test):
        true_tuple = partition_tuples[rng.randint(0, len(partition_tuples))]
        w_bar = w_bar_by_tuple[true_tuple]
        sigma = sigma_by_tuple[true_tuple]
        noise = rng.randn(len(w_bar))
        # One unlearned model for this config: w_bar + Gaussian noise (same as in training the attack)
        obs = w_bar + sigma * noise
        predicted_tuple = predict_configuration_by_llr(obs, distributions_by_tuple)
        overlap = len(set(true_tuple) & set(predicted_tuple))
        v_list.append(2 * overlap)

        if i < n_details:
            d = len(w_bar)
            detail = {
                'sample_idx': i,
                'true_tuple': tuple(true_tuple),
                'predicted_tuple': tuple(predicted_tuple),
                'sigma': float(sigma),
                'w_bar_norm': float(np.linalg.norm(w_bar)),
                'w_bar_first3': w_bar[:min(3, d)].tolist(),
                'obs_norm': float(np.linalg.norm(obs)),
                'obs_first3': obs[:min(3, d)].tolist(),
                'overlap': int(overlap),
                'v': int(2 * overlap),
                'correct': (tuple(true_tuple) == tuple(predicted_tuple)),
            }
            detail['logp_true'] = float(log_likelihood_under_dist(obs, distributions_by_tuple[true_tuple]))
            detail['logp_pred'] = float(log_likelihood_under_dist(obs, distributions_by_tuple[predicted_tuple]))
            prediction_details.append(detail)

    T = len(v_list)
    try:
        from cum_runs_eps_lab import compute_avg_v_test_epsilon_lb
        avg_v_res = compute_avg_v_test_epsilon_lb(
            m=K, r=K, T=T, v_list=v_list,
            delta=1e-8, ci_delta=ci_delta
        )
        epsilon_lb = avg_v_res["epsilon_lb"] / 2.0
    except Exception as e:
        epsilon_lb = np.nan
        if verbose:
            import sys
            print(f"  Warning: compute_avg_v_test_epsilon_lb failed: {e}", file=sys.stderr)

    if verbose:
        mean_overlap = np.mean([v // 2 for v in v_list])
        print(f"\n  Mean overlap (of K/2): {mean_overlap:.4f}")
        print(f"  ε_lb (m=K, r=K): {epsilon_lb}")

    return {
        'epsilon_lb': epsilon_lb,
        'v_list': v_list,
        'prediction_details': prediction_details,
        'distributions_by_tuple': distributions_by_tuple,
        'partition_tuples': partition_tuples,
        'K': K,
        'epsilon': epsilon,
        'delta': delta,
    }


def save_attack_results(results, filepath):
    """Save attack results to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nAttack results saved to: {filepath}")


def save_distributions(dist_f1, dist_f2, w_bar_f1, w_bar_f2, sigma, epsilon, delta, filepath):
    """
    Save distributions for later use.
    
    Parameters:
    -----------
    dist_f1 : dict
        Distribution for first half (mean, cov, std)
    dist_f2 : dict
        Distribution for second half (mean, cov, std)
    w_bar_f1 : np.ndarray
        w̄ for first half (before noise)
    w_bar_f2 : np.ndarray
        w̄ for second half (before noise)
    sigma : float
        Noise standard deviation
    epsilon : float
        Privacy budget
    delta : float
        Privacy parameter
    filepath : str
        Path to save distributions
    """
    distributions = {
        'epsilon': epsilon,
        'delta': delta,
        'sigma': sigma,
        'w_bar_f1': w_bar_f1,
        'w_bar_f2': w_bar_f2,
        'dist_f1': {
            'mean': dist_f1['mean'],
            'cov': dist_f1['cov'],
            'std': dist_f1['std']
        },
        'dist_f2': {
            'mean': dist_f2['mean'],
            'cov': dist_f2['cov'],
            'std': dist_f2['std']
        },
        'description': f"Distributions for (ε={epsilon}, δ={delta}), σ={sigma:.6f}"
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(distributions, f)
    
    print(f"Distributions saved to: {filepath}")
    return distributions


def _json_serialize(obj):
    """Convert numpy types and NaN to JSON-serializable Python types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialize(x) for x in obj]
    return obj


def save_attack_metrics_json(results, filepath):
    """Save attack metrics and epsilon lower bounds to JSON (human-readable, quick to open)."""
    r_f1 = results['result_f1']
    metrics = {
        'epsilon': results['epsilon'],
        'delta': results.get('confidence', 0.95),
        'sigma': r_f1['sigma'],
        'M': r_f1.get('M'),
        'L': r_f1.get('L'),
        'lambda': float(np.mean(r_f1['per_sample_reg'])) if r_f1.get('per_sample_reg') is not None else None,
        'accuracy': results['accuracy'],
        'tpr': results['tpr'],
        'fpr': results['fpr'],
        'roc_auc': results['roc_auc'],
        'tpr_low': results['tpr_low'],
        'tpr_high': results['tpr_high'],
        'fpr_low': results['fpr_low'],
        'fpr_high': results['fpr_high'],
        'epsilon_empirical_lower': results['epsilon_empirical_lower'],
        'epsilon_lb_avg_v': results['epsilon_lb_avg_v'],
    }
    with open(filepath, 'w') as f:
        json.dump(_json_serialize(metrics), f, indent=2)
    print(f"Attack metrics (JSON) saved to: {filepath}")


def save_distributions_json(dist_f1, dist_f2, w_bar_f1, w_bar_f2, sigma, epsilon, delta, filepath):
    """Save distribution summary to JSON (no full covariance; quick to read)."""
    summary = {
        'epsilon': epsilon,
        'delta': delta,
        'sigma': sigma,
        'w_bar_f1': w_bar_f1,
        'w_bar_f2': w_bar_f2,
        'mean_f1': dist_f1['mean'],
        'mean_f2': dist_f2['mean'],
        'std_f1': dist_f1['std'],
        'std_f2': dist_f2['std'],
        'description': f"Distributions for (ε={epsilon}, δ={delta}), σ={sigma:.6f}",
    }
    with open(filepath, 'w') as f:
        json.dump(_json_serialize(summary), f, indent=2)
    print(f"Distributions summary (JSON) saved to: {filepath}")


def load_distributions(filepath):
    """Load saved distributions."""
    with open(filepath, 'rb') as f:
        distributions = pickle.load(f)
    print(f"Distributions loaded from: {filepath}")
    print(f"  ε = {distributions['epsilon']}")
    print(f"  δ = {distributions['delta']}")
    print(f"  σ = {distributions['sigma']:.6f}")
    return distributions


def load_attack_results(filepath):
    """Load attack results from pickle file."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"Attack results loaded from: {filepath}")
    return results


def plot_roc_curve(results, save_path='roc_curve.png'):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr_curve'], results['tpr_curve'], 
             lw=2, label=f"ROC (AUC = {results['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve (ε = {results["epsilon"]})', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()


def plot_llr_distributions(results, save_path='llr_distributions.png'):
    """Plot LLR distributions for both classes."""
    plt.figure(figsize=(10, 6))
    
    # Use valid LLRs if available, otherwise use original (may contain invalid values)
    llrs_f1 = results.get('llrs_f1_valid', results['llrs_f1'])
    llrs_f2 = results.get('llrs_f2_valid', results['llrs_f2'])
    
    # Filter out any remaining invalid values for plotting
    llrs_f1_plot = llrs_f1[np.isfinite(llrs_f1)]
    llrs_f2_plot = llrs_f2[np.isfinite(llrs_f2)]
    
    if len(llrs_f1_plot) > 0:
        plt.hist(llrs_f1_plot, bins=50, alpha=0.6, label='From F1 (should be > 0)',
                 color='blue', density=True)
    if len(llrs_f2_plot) > 0:
        plt.hist(llrs_f2_plot, bins=50, alpha=0.6, label='From F2 (should be < 0)',
                 color='red', density=True)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    plt.xlabel('Log-Likelihood Ratio', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'LLR Distributions (ε = {results["epsilon"]})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"LLR distributions saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Membership Inference Attack Module")
    print("Use run_membership_inference_attack_two_partition() to execute attack")