"""
Cubic loss (all-d): f(w,z) = (λ/2)||w||² + (M/6) Σ_i w_i³ - <z,w>.
∇f = λw + (M/2)(w_1²,...,w_d²) - z,  ∇²f = λI + M·diag(w).
Condition MB ≤ λ/2 for (λ/2)-strong convexity on W = {||w|| ≤ B}.
Optimization via scipy (no sklearn). All parameters driven by config.
"""
import numpy as np
from scipy.optimize import minimize


def cubic_loss_pointwise(w, z, lam, M):
    """Pointwise loss f(w,z) = (λ/2)||w||² + (M/6) Σ_i w_i³ - <z,w>."""
    return 0.5 * lam * np.dot(w, w) + (M / 6.0) * np.sum(w ** 3) - np.dot(z, w)


def cubic_gradient_pointwise(w, z, lam, M):
    """∇f = λw + (M/2)(w_1²,...,w_d²) - z."""
    return lam * w + (M / 2.0) * (w ** 2) - z


def cubic_hessian_pointwise(w, z, lam, M):
    """∇²f = λI + M·diag(w)."""
    d = len(w)
    return lam * np.eye(d) + M * np.diag(w)


def train_cubic(X_train, X_val, X_test, per_sample_reg, cubic_params,
                max_iter=1000, random_state=42, verbose=True):
    """
    Train by minimizing F(w) = (1/n) Σ_i f(w, z_i) over w with ||w|| ≤ B.
    Loss: f(w,z) = (λ/2)||w||² + (M/6)(Σ_i w_i³) - <z,w>. Requires d ≥ 2.
    λ comes from per_sample_reg only (cubic_params.lambda is not used).
    cubic_params: M (Hessian-Lipschitz), B (||w|| ≤ B), optional data_R (default 1).
    Condition MB < λ for strong convexity (checked if verbose)
    Strong convexity parameter μ = λ - MB (used for unlearning bounds).
    """
    from lipschitz_constants import compute_lipschitz_constants_cubic

    n, d = X_train.shape
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    if per_sample_reg.size == 1:
        lam = float(per_sample_reg.flat[0])
        per_sample_reg = np.full(n, lam)
    else:
        lam = float(np.mean(per_sample_reg))
    Lambda = float(np.sum(per_sample_reg))
    M_val = float(cubic_params['M'])
    B = float(cubic_params['B'])
    if M_val * B > lam:
        raise ValueError(f"  Warning: MB = {M_val*B:.4f} > λ = {lam:.4f}; strong convexity may fail on W.")

    def F(w):
        return np.mean([cubic_loss_pointwise(w, X_train[i], per_sample_reg[i], M_val) for i in range(n)])

    def grad_F(w):
        return np.mean([cubic_gradient_pointwise(w, X_train[i], per_sample_reg[i], M_val) for i in range(n)], axis=0)

    w0 = np.zeros(d)
    res = minimize(F, w0, method='L-BFGS-B', jac=grad_F,
                   bounds=[(-B, B)] * d,
                   options={'maxiter': max_iter, 'disp': False})
    w = res.x
    if np.linalg.norm(w) > B:
        w = B * w / np.linalg.norm(w)

    lam_eff = Lambda / n
    H_single = cubic_hessian_pointwise(w, np.zeros(d), lam_eff, M_val)
    hessian = n * H_single

    data_R = float(cubic_params.get('data_R', 1.0))
    lipschitz = compute_lipschitz_constants_cubic(lam=lam, M=M_val, B=B, verbose=verbose, data_R=data_R)
    L, M_out = lipschitz['L'], lipschitz['M']

    train_loss = F(w)
    val_loss = np.mean([cubic_loss_pointwise(w, X_val[i], lam, M_val) for i in range(len(X_val))]) if len(X_val) else np.nan
    test_loss = np.mean([cubic_loss_pointwise(w, X_test[i], lam, M_val) for i in range(len(X_test))]) if len(X_test) else np.nan
    metrics = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss,
               'train_cubic_loss': train_loss, 'val_cubic_loss': val_loss, 'test_cubic_loss': test_loss}

    if verbose:
        print("\n" + "="*60)
        print("TRAINING RESULTS (cubic loss Σ_i w_i³)")
        print("="*60)
        print(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Test loss: {test_loss:.6f}")
        print(f"||w||: {np.linalg.norm(w):.4f}, w[:3]: {w[:min(3,d)]}, L: {L:.4f}, M: {M_out:.4f}")
        print("="*60)

    return {
        'model': None,
        'weights': w,
        'hessian': hessian,
        'metrics': metrics,
        'C': 1.0 / Lambda,
        'Lambda': Lambda,
        'per_sample_reg': per_sample_reg,
        'L': L,
        'M': M_out,
        'loss': 'cubic',
        'include_intercept': False,
        'cubic_params': {'lambda': lam, 'M': M_val, 'B': B},
        'mu': lam - M_val * B,  # effective strong convexity: λ/2 for cubic
    }
