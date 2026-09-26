"""
Compute Lipschitz constants for L2-regularized logistic regression.

IMPORTANT: Following the paper's Assumption 1, the Lipschitz constants L and M
are defined for the PER-SAMPLE loss f(w, z), NOT the empirical average F̂(w).

For sklearn's logistic regression, the per-sample loss with regularization is:
f(w, z) = (1/2C)||w||² + log(1 + exp(-y w^T x))

Note: The regularization term is included in the per-sample loss.
"""
import numpy as np


def compute_lipschitz_gradient(X, per_sample_reg, include_intercept=True):
    """
    Compute Lipschitz constant L of the gradient for PER-SAMPLE loss f(w,z).
    
    Per-sample loss uses w and x_aug = [x; 1]. Hessian ∇²f = p(1-p) x_aug x_aug^T + λ I,
    so ∇²f ≼ (1/4) x_aug x_aug^T + λ I  =>  λ_max(∇²f) ≤ ||x_aug||²/4 + λ.
    So L_i = ||x_aug_i||²/4 + λ_i = (||x_i||² + 1)/4 + λ_i when include_intercept=True.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n, d)
        Feature matrix (without intercept column)
    per_sample_reg : np.ndarray, shape (n,) or scalar
        Per-sample regularization λ_i.
    include_intercept : bool
        If True, use augmented norm ||x_aug||² = ||x||² + 1 (model has intercept).
        
    Returns:
    --------
    L : float
        Lipschitz constant of gradient (per-sample)
    """
    norms_squared = np.sum(X**2, axis=1)
    if include_intercept:
        norms_squared_aug = norms_squared + 1.0  # ||x_aug||² = ||x||² + 1
    else:
        norms_squared_aug = norms_squared
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    # L = max_i (||x_aug_i||²/4 + λ_i); broadcasting handles scalar or (n,) per_sample_reg
    L = np.max(norms_squared_aug / 4.0 + per_sample_reg)
    return L


def compute_lipschitz_hessian(X, per_sample_reg=None, include_intercept=True):
    """
    Compute Lipschitz constant M of the Hessian for PER-SAMPLE loss f(w,z).
    
    The Hessian ∇²f = p(1-p) x_aug x_aug^T + λ I; its derivative w.r.t. w is bounded
    by |p(1-p)(1-2p)| * ||x_aug||³ ≤ ||x_aug||³/(6√3). So M = max_i ||x_aug_i||³/(6√3).
    With intercept: ||x_aug||² = ||x||² + 1  =>  M_i = (||x_i||²+1)^(3/2)/(6√3).
    
    Parameters:
    -----------
    X : np.ndarray, shape (n, d)
        Feature matrix (without intercept column)
    per_sample_reg : optional
        Unused for M (kept for API consistency)
    include_intercept : bool
        If True, use ||x_aug||³ = (||x||²+1)^(3/2).
        
    Returns:
    --------
    M : float
        Lipschitz constant of Hessian (per-sample)
    """
    norms_squared = np.sum(X**2, axis=1)
    if include_intercept:
        norm_aug_cubed = (norms_squared + 1.0) ** (3.0 / 2.0)  # ||x_aug||³
    else:
        norm_aug_cubed = norms_squared ** (3.0 / 2.0)
    M = np.max(norm_aug_cubed) / (6.0 * np.sqrt(3))
    return M


def compute_lipschitz_constants_cubic(lam, M, B, verbose=True, data_R=1.0):
    """
    Lipschitz constants for cubic loss. With ||w|| ≤ B and ||z|| ≤ data_R:
    L = λB + (M/2)B² + data_R (gradient bound from -<z,w> term). M = M (Hessian-Lipschitz).
    """
    L = lam * B + (M / 2.0) * (B ** 2) + float(data_R)
    if verbose:
        print("\n" + "="*70)
        print("LIPSCHITZ CONSTANTS (cubic loss, from config)")
        print("="*70)
        print(f"λ={lam}, M={M}, B={B}, data_R={data_R}  =>  L = {L:.4f}, M = {M:.4f}")
        print("="*70)
    return {'L': L, 'M': M}


def compute_lipschitz_constants_from_data(X, per_sample_reg, loss='logistic', verbose=True, include_intercept=True):
    """
    Compute both Lipschitz constants from data (per-sample definition).
    
    loss='logistic': L = max_i (||x_aug_i||²/4 + λ_i), M = max_i ||x_aug_i||³/(6√3).
    loss='mse': Hessian is constant (∇²f = x_i x_i^T + λ_i I), so L = max_i (||x_aug_i||² + λ_i), M = 0.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n, d)
    per_sample_reg : np.ndarray (n,) or scalar
    loss : 'logistic' or 'mse'
    verbose : bool
    include_intercept : bool
        
    Returns:
    --------
    dict with L, M, max_norm, mean_norm, max_norm_squared
    """
    n, d = X.shape
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    Lambda = np.sum(per_sample_reg) if per_sample_reg.size > 1 else (n * float(per_sample_reg.flat[0]))
    norms = np.linalg.norm(X, axis=1)
    norms_squared = norms ** 2
    max_norm = np.max(norms)
    mean_norm = np.mean(norms)
    max_norm_squared = np.max(norms_squared)
    if include_intercept:
        norms_squared_aug = norms_squared + 1.0
        max_norm_aug_sq = max_norm_squared + 1.0
        max_norm_aug_cubed = (max_norm_squared + 1.0) ** (3.0 / 2.0)
    else:
        norms_squared_aug = norms_squared
        max_norm_aug_sq = max_norm_squared
        max_norm_aug_cubed = max_norm ** 3

    if loss == 'mse':
        # Per-sample Hessian = x_aug x_aug^T + λ_i I => λ_max = ||x_aug||² + λ_i
        L = np.max(norms_squared_aug + per_sample_reg)
        M = 0.0
    else:
        L = compute_lipschitz_gradient(X, per_sample_reg=per_sample_reg, include_intercept=include_intercept)
        M = compute_lipschitz_hessian(X, include_intercept=include_intercept)

    if verbose:
        reg_desc = f"per_sample_reg (Λ={Lambda:.4f})"
        print("\n" + "="*70)
        print("LIPSCHITZ CONSTANTS COMPUTATION (PER-SAMPLE)")
        print("="*70)
        print(f"Loss: {loss}. Using augmented features ||x_aug||² = ||x||²+1 (fit_intercept=True)")
        print(f"\nDataset shape: n={n}, d={d}")
        print(f"Regularization: {reg_desc}")
        print(f"\nFeature norms: max_i ||x_i|| = {max_norm:.4f}, max_i ||x_aug_i||² = {max_norm_aug_sq:.4f}")
        print(f"\nLipschitz constants (per-sample): L = {L:.4f}, M = {M:.4f}")
        print("="*70)

    return {
        'L': L,
        'M': M,
        'max_norm': max_norm,
        'mean_norm': mean_norm,
        'max_norm_squared': max_norm_squared
    }


def verify_lipschitz_gradient(X, y, per_sample_reg, L, n_samples=100, random_state=42):
    """
    Empirically verify that gradient of PER-SAMPLE loss is L-Lipschitz.
    
    Tests: ||∇f(w,z) - ∇f(w',z)|| ≤ L||w - w'|| for individual samples z.
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    per_sample_reg : np.ndarray or scalar
        Per-sample regularization λ_i
    L : float
        Claimed Lipschitz constant
    n_samples : int
        Number of random pairs to test
    random_state : int
        Random seed
        
    Returns:
    --------
    dict with verification results
    """
    rng = np.random.RandomState(random_state)
    d = X.shape[1]
    
    # Add intercept column
    X_aug = np.column_stack([X, np.ones(len(X))])
    
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    if per_sample_reg.size == 1:
        lam_all = np.full(len(X), float(per_sample_reg.flat[0]))
    else:
        lam_all = per_sample_reg

    def compute_gradient_single(w, x_i, y_i, lam_i):
        """Compute gradient of per-sample loss f(w, z_i) = (λ_i/2)||w||² + log loss."""
        y_signed = 2 * y_i - 1
        logit = x_i @ w
        sig = 1 / (1 + np.exp(y_signed * logit))
        grad_data = -y_signed * sig * x_i
        grad_reg = np.zeros_like(w)
        grad_reg[:-1] = lam_i * w[:-1]
        return grad_data + grad_reg

    ratios = []
    for _ in range(n_samples):
        i = rng.randint(len(X))
        x_i = X_aug[i]
        y_i = y[i]
        lam_i = lam_all[i] if lam_all.size > 1 else lam_all[0]
        
        # Sample two random weight vectors
        w1 = rng.randn(d + 1)
        w2 = rng.randn(d + 1)
        
        # Compute gradients for this specific sample
        grad1 = compute_gradient_single(w1, x_i, y_i, lam_i)
        grad2 = compute_gradient_single(w2, x_i, y_i, lam_i)
        
        # Compute ratio
        grad_diff = np.linalg.norm(grad1 - grad2)
        w_diff = np.linalg.norm(w1 - w2)
        
        if w_diff > 1e-10:
            ratio = grad_diff / w_diff
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    max_ratio = np.max(ratios)
    mean_ratio = np.mean(ratios)
    
    verified = max_ratio <= L * 1.01  # Allow 1% tolerance
    
    return {
        'verified': verified,
        'claimed_L': L,
        'empirical_max': max_ratio,
        'empirical_mean': mean_ratio,
        'margin': L - max_ratio
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    n, d = 1000, 20
    X = np.random.randn(n, d) / np.sqrt(d)
    y = np.random.binomial(1, 0.5, n)
    per_sample_reg = 1.0  # scalar: same λ for all, Λ = n

    constants = compute_lipschitz_constants_from_data(X, per_sample_reg, verbose=True)

    print("\n" + "="*70)
    print("EMPIRICAL VERIFICATION")
    print("="*70)
    verification = verify_lipschitz_gradient(X, y, per_sample_reg, constants['L'], n_samples=100)
    print(f"Claimed L: {verification['claimed_L']:.4f}")
    print(f"Empirical max ratio: {verification['empirical_max']:.4f}")
    print(f"Empirical mean ratio: {verification['empirical_mean']:.4f}")
    print(f"Verified: {verification['verified']}")
    print(f"Margin: {verification['margin']:.4f}")
    print("="*70)
