"""
Unlearning algorithm implementation (Algorithm 1 from the paper).
"""
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from gaussian_mechanism import gaussian_mechanism_sigma_general, gaussian_mechanism_sigma_analytic


def compute_gradient(X, y, w, per_sample_reg, loss='logistic', trained_result=None):
    """
    Compute sum of per-sample gradients over (X, y).
    For loss='cubic', trained_result must contain 'cubic_params' (lambda, M); X rows are z_i, no intercept.
    """
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    Lambda = float(np.sum(per_sample_reg))
    if loss == 'cubic':
        from cubic_loss import cubic_gradient_pointwise
        cp = (trained_result or {}).get('cubic_params') or {}
        M = float(cp['M'])
        return sum(cubic_gradient_pointwise(w, X[i], per_sample_reg[i], M) for i in range(len(X)))
    grad_reg = np.zeros_like(w)
    grad_reg[:-1] = Lambda * w[:-1]
    if loss == 'logistic':
        y_signed = 2 * y - 1
        logits = X @ w
        sig = 1 / (1 + np.exp(y_signed * logits))
        grad_data = -X.T @ (y_signed * sig)
    elif loss == 'mse':
        resid = y - (X @ w)
        grad_data = -X.T @ resid
    else:
        raise ValueError(f"loss must be 'logistic', 'mse', or 'cubic', got {loss!r}")
    return grad_data + grad_reg


def compute_hessian(X, y, w, per_sample_reg, loss='logistic', trained_result=None):
    """
    Compute sum of per-sample Hessians over (X, y).
    For loss='cubic', trained_result must contain 'cubic_params'; X rows are z_i, no intercept.
    """
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    Lambda = float(np.sum(per_sample_reg))
    if loss == 'cubic':
        from cubic_loss import cubic_hessian_pointwise
        cp = (trained_result or {}).get('cubic_params') or {}
        M = float(cp['M'])
        return sum(cubic_hessian_pointwise(w, X[i], per_sample_reg[i], M) for i in range(len(X)))
    reg_matrix = np.eye(len(w))
    reg_matrix[-1, -1] = 0
    if loss == 'logistic':
        logits = X @ w
        probs = 1 / (1 + np.exp(-logits))
        weights = probs * (1 - probs)
        weighted_X = X * weights[:, np.newaxis]
        H = X.T @ weighted_X
    elif loss == 'mse':
        H = X.T @ X
    else:
        raise ValueError(f"loss must be 'logistic', 'mse', or 'cubic', got {loss!r}")
    H += Lambda * reg_matrix
    return H


def unlearn(trained_result, X_forget, y_forget, X_retain, y_retain,
            M=1.0, L=1.0, epsilon=1.0, delta=0.01, 
            random_state=42, verbose=True):
    """
    Unlearning algorithm (Ā_sc) from Algorithm 1.
    
    Implements the unlearning procedure that removes the influence of 
    forget set U from trained model.
    
    Parameters:
    -----------
    trained_result : dict
        Output from train() containing model, weights, hessian, per_sample_reg, C (1/Λ)
    X_forget : np.ndarray
        Features of samples to forget (delete requests U)
    y_forget : np.ndarray
        Labels of samples to forget
    X_retain : np.ndarray
        Features of samples to retain (S \ U)
    y_retain : np.ndarray
        Labels of samples to retain
    M : float
        Hessian Lipschitz constant
    L : float
        Smoothness parameter (Lipschitz constant of gradient)
    epsilon : float
        Privacy budget
    delta : float
        Privacy parameter
    random_state : int
        Random seed
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict containing:
        - unlearned_weights: final weights after unlearning
        - intermediate_weights: weights before adding noise
        - metrics: performance metrics
    """
    np.random.seed(random_state)
    
    # Extract from trained result
    w_hat = trained_result['weights']  # ŵ (includes intercept)
    hessian_full = trained_result['hessian']  # H = data_full + Λ I (= n∇²F̂)
    C = trained_result['C']  # 1/Λ for sklearn
    Lambda = trained_result['Lambda']
    per_sample_reg = trained_result['per_sample_reg']  # λ_i, same order as training data
    loss = trained_result.get('loss', 'logistic')
    n_retain = len(X_retain)
    n = len(X_retain) + len(X_forget)  # Total training samples
    m = len(X_forget)  # Number of delete requests
    d = len(w_hat)  # Dimensionality (features + intercept)
    reg_forget = np.asarray(per_sample_reg[n_retain:n_retain + m], dtype=float)
    # mu = effective strong convexity: Λ/n for logistic/MSE, Λ/(2n) for cubic.
    mu = trained_result['mu']

    if verbose:
        print("\n" + "="*60)
        print("UNLEARNING ALGORITHM")
        print("="*60)
        print(f"Total train samples (n): {n}")
        print(f"Forget samples (m): {m}")
        print(f"Retain samples (n-m): {n-m}")
        print(f"Dimensionality (d): {d}")
    
    # Step 1: Set γ and σ using effective strong convexity mu
    gamma = (2 * M * m**2 * L**2) / (mu**3 * n**2)
    
    # Use general Gaussian mechanism (works for all ε, not just small ε)
    sigma = gaussian_mechanism_sigma_general(gamma, epsilon, delta)
    
    # For comparison, also compute analytic approximation
    sigma_analytic = gaussian_mechanism_sigma_analytic(gamma, epsilon, delta)
    
    if verbose:
        print(f"\nHyperparameters:")
        print(f"  M (Hessian Lipschitz): {M}")
        print(f"  L (Smoothness): {L}")
        print(f"  ε (privacy budget): {epsilon}")
        print(f"  δ (privacy parameter): {delta}")
        print(f"\nComputed parameters:")
        print(f"  γ (sensitivity): {gamma:.6f}")
        print(f"  σ (general formula): {sigma:.6f}")
        print(f"  σ (analytic approx): {sigma_analytic:.6f}")
        if epsilon > 1.0:
            diff_percent = abs(sigma - sigma_analytic) / sigma_analytic * 100
            print(f"  Difference: {diff_percent:.1f}% (using general is important for ε > 1)")
        else:
            print(f"  (Analytic approximation valid for ε ≤ 1)")
    
    # Add intercept column to data (cubic loss: no intercept)
    include_intercept = trained_result.get('include_intercept', True)
    if include_intercept:
        X_retain_aug = np.column_stack([X_retain, np.ones(len(X_retain))])
        X_forget_aug = np.column_stack([X_forget, np.ones(len(X_forget))])
    else:
        X_retain_aug = X_retain
        X_forget_aug = X_forget
    
    # Step 2: Compute Ĥ (Equation 7)
    # Ĥ = (1/(n-m)) * (n∇²F̂(ŵ) - Σ_{z∈U} ∇²f(ŵ,z))
    hessian_forget = compute_hessian(X_forget_aug, y_forget, w_hat, per_sample_reg=reg_forget, loss=loss, trained_result=trained_result)
    
    H_hat = (1 / (n - m)) * (hessian_full - hessian_forget)
    
    if verbose:
        print(f"\nStep 2: Computed Ĥ (estimated Hessian on retain set)")
        print(f"  Hessian shape: {H_hat.shape}")
        print(f"  Condition number: {np.linalg.cond(H_hat):.2e}")
    
    # Step 3: Define w̄ (Equation 8)
    # w̄ = ŵ + (1/(n-m)) * Ĥ^{-1} * sum_{z in U} ∇f(ŵ,z)
    
    # Compute sum of gradients on forget set Σ_{z∈U} ∇f(ŵ,z)
    grad_sum_forget = compute_gradient(X_forget_aug, y_forget, w_hat, per_sample_reg=reg_forget, loss=loss, trained_result=trained_result)
    
    # Equation 8
    try:
        H_hat_inv = np.linalg.inv(H_hat)
    except np.linalg.LinAlgError:
        # Add small regularization if singular
        if verbose:
            print("  Warning: Hessian near-singular, adding regularization")
        H_hat_inv = np.linalg.inv(H_hat + 1e-6 * np.eye(d))
    
    w_bar = w_hat + (1 / (n - m)) * (H_hat_inv @ grad_sum_forget)
    
    if verbose:
        print(f"\nStep 3: Computed w̄ (intermediate weights)")
        print(f"  ||w̄ - ŵ||: {np.linalg.norm(w_bar - w_hat):.6f}")
    
    # Step 4: Sample ν from N(0, σ²I_d)
    nu = np.random.normal(0, sigma, size=d)
    
    if verbose:
        print(f"\nStep 4: Sampled noise ν")
        print(f"  ||ν||: {np.linalg.norm(nu):.6f}")
    
    # Step 5: Return w̃ = w̄ + ν
    w_tilde = w_bar + nu
    
    if verbose:
        print(f"\nStep 5: Computed w̃ (final unlearned weights)")
        print(f"  ||w̃ - ŵ||: {np.linalg.norm(w_tilde - w_hat):.6f}")
        print(f"  ||w̃ - w̄||: {np.linalg.norm(w_tilde - w_bar):.6f}")
    
    return {
        'unlearned_weights': w_tilde,
        'intermediate_weights': w_bar,
        'original_weights': w_hat,
        'noise': nu,
        'gamma': gamma,
        'sigma': sigma,
        'H_hat': H_hat
    }


def evaluate_unlearned_model(unlearned_weights, X_val, y_val, X_test, y_test, 
                             X_retain=None, y_retain=None, verbose=True):
    """
    Evaluate the unlearned model on various datasets.
    
    Parameters:
    -----------
    unlearned_weights : np.ndarray
        Weights from unlearning (including intercept)
    X_val, y_val : Validation data
    X_test, y_test : Test data
    X_retain, y_retain : Retain set (optional)
    verbose : bool
        Print metrics
        
    Returns:
    --------
    dict with evaluation metrics
    """
    def predict_with_weights(X, w):
        """Make predictions using weight vector."""
        X_aug = np.column_stack([X, np.ones(len(X))])
        logits = X_aug @ w
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        return preds, probs
    
    metrics = {}
    
    datasets = [('val', X_val, y_val), ('test', X_test, y_test)]
    if X_retain is not None:
        datasets.append(('retain', X_retain, y_retain))
    
    for name, X, y in datasets:
        y_pred, y_proba = predict_with_weights(X, unlearned_weights)
        
        metrics[f'{name}_accuracy'] = accuracy_score(y, y_pred)
        metrics[f'{name}_loss'] = log_loss(y, y_proba)
        metrics[f'{name}_auc'] = roc_auc_score(y, y_proba)
    
    if verbose:
        print("\n" + "="*60)
        print("UNLEARNED MODEL EVALUATION")
        print("="*60)
        for name in ['val', 'test', 'retain']:
            if f'{name}_accuracy' in metrics:
                print(f"{name.capitalize():7s} - Acc: {metrics[f'{name}_accuracy']:.4f}, "
                      f"Loss: {metrics[f'{name}_loss']:.4f}, "
                      f"AUC: {metrics[f'{name}_auc']:.4f}")
        print("="*60)
    
    return metrics
