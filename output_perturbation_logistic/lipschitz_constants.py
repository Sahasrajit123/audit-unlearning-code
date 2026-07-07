"""Lipschitz constants for L2-regularized logistic regression (per-sample)."""
import numpy as np


def compute_lipschitz_gradient(X, per_sample_reg, include_intercept=True):
    """L = max_i (||x_aug_i||²/4 + λ_i)."""
    norms_squared = np.sum(X**2, axis=1)
    if include_intercept:
        norms_squared_aug = norms_squared + 1.0
    else:
        norms_squared_aug = norms_squared
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    L = np.max(norms_squared_aug / 4.0 + per_sample_reg)
    return L


def compute_lipschitz_hessian(X, include_intercept=True):
    """M = max_i ||x_aug_i||³/(6√3)."""
    norms_squared = np.sum(X**2, axis=1)
    if include_intercept:
        norm_aug_cubed = (norms_squared + 1.0) ** (3.0 / 2.0)
    else:
        norm_aug_cubed = norms_squared ** (3.0 / 2.0)
    M = np.max(norm_aug_cubed) / (6.0 * np.sqrt(3))
    return M


def compute_lipschitz_constants_from_data(X, per_sample_reg, loss='logistic', verbose=True, include_intercept=True):
    """Return dict with L, M. loss='logistic': L from gradient, M from Hessian. loss='mse': L = max_i(||x_aug_i||^2 + λ_i), M = 0."""
    if loss == 'mse':
        norms_squared = np.sum(X**2, axis=1)
        if include_intercept:
            norms_squared_aug = norms_squared + 1.0
        else:
            norms_squared_aug = norms_squared
        per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
        L = np.max(norms_squared_aug + per_sample_reg)
        M = 0.0
    else:
        L = compute_lipschitz_gradient(X, per_sample_reg=per_sample_reg, include_intercept=include_intercept)
        M = compute_lipschitz_hessian(X, include_intercept=include_intercept)
    if verbose:
        print(f"Lipschitz: L = {L:.4f}, M = {M:.4f}")
    return {'L': L, 'M': M}
