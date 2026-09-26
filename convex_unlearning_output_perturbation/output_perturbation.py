"""
Output perturbation for (ε, δ)-unlearning.

Procedure (Equation 1):
  x_0 = Π_{C_0}(x̂) + ξ_0
  ξ_0 ~ N(0, σ² I_d)

- Π_{C_0}: project (clip) model parameters onto L2 ball of radius C_0.
- Gaussian noise ensures (ε, δ)-DP for the unlearned output.
- σ is calibrated via the exact Gaussian mechanism (valid for all ε > 0).
"""
import numpy as np
from scipy.special import erfc
from scipy.optimize import root_scalar, minimize_scalar


def clip_to_ball(w, C_0):
    """
    Project w onto L2 ball of radius C_0: Π_{C_0}(w).
    If ||w|| <= C_0 return w; else return C_0 * w / ||w||.
    """
    w = np.asarray(w, dtype=float).flatten()
    nrm = np.linalg.norm(w)
    if nrm <= C_0:
        return w.copy()
    return (C_0 / nrm) * w


def gaussian_mechanism_sigma_analytic(sensitivity, epsilon, delta):
    """
    Classical approximation: σ = Δ * sqrt(2 ln(1.25/δ)) / ε.
    Valid / conservative for ε ≤ 1; underestimates noise for ε > 1.
    """
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def gaussian_mechanism_sigma_general(sensitivity, epsilon, delta,
                                     tol=1e-10, max_iter=5000):
    """
    General formula for Gaussian mechanism (works for ALL ε).

    Solves the transcendental equation:
    Φ(Δ/(2σ) - εσ/Δ) - e^ε Φ(-Δ/(2σ) - εσ/Δ) ≤ δ

    where Φ is the standard Gaussian CDF and Δ is the sensitivity.

    We solve for σ such that the above constraint is tight (equality holds).

    Parameters:
    -----------
    sensitivity : float
        Sensitivity Δ
    epsilon : float
        Privacy budget (works for any ε > 0)
    delta : float
        Privacy parameter (typically 1e-5 to 1e-2)
    tol : float
        Tolerance for root finding
    max_iter : int
        Maximum iterations for root finding

    Returns:
    --------
    sigma : float
        Noise standard deviation

    References:
    -----------
    - "The Algorithmic Foundations of Differential Privacy" (Dwork & Roth, 2014)
      Theorem 3.22 (Gaussian Mechanism)
    - "Concentrated Differential Privacy" (Bun & Steinke, 2016)
    """
    # For numerical stability, we solve for σ/Δ (dimensionless)
    # Let t = σ/Δ, then we solve:
    # Φ(1/(2t) - εt) - e^ε Φ(-1/(2t) - εt) = δ

    def equation(t):
        """
        Equation to solve: f(t) = δ where f(t) = Φ(arg1) - e^ε Φ(arg2).
        Uses log-space computation and stable subtraction (as in forget_phi_noisy_loader.compute_sigma_general)
        so that large ε does not cause overflow.
        """
        if t <= 0:
            return np.inf

        arg1 = (1.0 / (2 * t) - epsilon * t)
        arg2 = (-1.0 / (2 * t) - epsilon * t)

        phi_arg1 = 0.5 * erfc(-arg1 / np.sqrt(2))
        phi_arg2 = 0.5 * erfc(-arg2 / np.sqrt(2))

        # Log-space: log_phi1 and log(e^ε * phi_arg2) = epsilon + log(phi_arg2)
        log_phi1 = np.log(phi_arg1) if (phi_arg1 > 0 and np.isfinite(phi_arg1)) else -np.inf
        if phi_arg2 > 0 and np.isfinite(phi_arg2):
            log_exp_epsilon_phi2 = epsilon + np.log(phi_arg2)
        else:
            log_exp_epsilon_phi2 = -np.inf

        # Stable subtraction: lhs = phi_arg1 - e^ε*phi_arg2 in log space
        if np.isneginf(log_phi1) and np.isneginf(log_exp_epsilon_phi2):
            lhs = 0.0
        elif np.isneginf(log_phi1):
            lhs = -np.exp(log_exp_epsilon_phi2) if log_exp_epsilon_phi2 < 700 else -np.inf
        elif np.isneginf(log_exp_epsilon_phi2):
            lhs = np.exp(log_phi1)
        else:
            if log_phi1 >= log_exp_epsilon_phi2:
                log_diff = log_exp_epsilon_phi2 - log_phi1
                if log_diff < -50:
                    lhs = np.exp(log_phi1)
                else:
                    lhs = np.exp(log_phi1 + np.log1p(-np.exp(log_diff)))
            else:
                log_diff = log_phi1 - log_exp_epsilon_phi2
                if log_diff < -50:
                    lhs = -np.exp(log_exp_epsilon_phi2)
                else:
                    lhs = -np.exp(log_exp_epsilon_phi2 + np.log1p(-np.exp(log_diff)))

        return lhs - delta

    # Search range: root is near analytic t (same as r2d forget_phi_noisy_loader)
    sigma_analytic = gaussian_mechanism_sigma_analytic(sensitivity, epsilon, delta)
    t_analytic = sigma_analytic / sensitivity
    t_center = max(t_analytic, 1e-12)
    t_min, t_max = max(t_center / 1e6, 1e-14), min(t_center * 1e6, 1e10)

    # Find bracket [t_low, t_high] with sign change: first try scan, then expand like r2d
    scan_multipliers = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    t_a, t_b = None, None
    f_prev = None
    for mult in scan_multipliers:
        t = t_center * mult
        if t < t_min or t > t_max:
            continue
        f = equation(t)
        if f_prev is not None and np.isfinite(f) and np.isfinite(f_prev) and f * f_prev <= 0:
            t_a = min(t_prev, t)
            t_b = max(t_prev, t)
            break
        t_prev, f_prev = t, f
    else:
        # No sign change in scan: expand bracket like r2d (low/high *= 10 or 0.1)
        t_low = max(t_center * 1e-6, t_min)
        t_high = min(t_center * 1e6, t_max)
        max_expansions = 25
        for _ in range(max_expansions):
            f_lo = equation(t_low)
            f_hi = equation(t_high)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo > 0 and f_hi < 0:
                t_a, t_b = t_low, t_high
                break
            if not np.isfinite(f_lo) or f_lo <= 0:
                t_low *= 0.1
                t_low = max(t_low, t_min)
            if not np.isfinite(f_hi) or f_hi >= 0:
                t_high *= 10
                t_high = min(t_high, t_max)
        else:
            # Last resort: minimize squared residual
            def squared_residual(t):
                val = equation(t)
                return val * val if np.isfinite(val) else np.inf
            sol = minimize_scalar(squared_residual, bounds=(t_min, t_max), method='bounded',
                                  options={'xatol': tol, 'maxiter': max_iter})
            if sol.success and sol.fun <= 1e-4:
                return sol.x * sensitivity
            raise RuntimeError(
                "Root finding failed (no sign change after expansion and minimize_scalar did not find root). "
                "Refusing to fall back to analytic approximation."
            )

    t_lower = max(t_a, t_min)
    t_upper = min(t_b, t_max)

    try:
        sol = root_scalar(equation, bracket=[t_lower, t_upper], method='brentq', xtol=tol, maxiter=max_iter)
        if not sol.converged:
            raise RuntimeError(
                "Root finding did not converge. Refusing to fall back to analytic approximation."
            )
        return sol.root * sensitivity

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Error in root finding ({e}). Refusing to fall back to analytic approximation."
        ) from e


def output_perturbation_sigma(C_0, epsilon, delta):
    """
    Per-coordinate std for ξ_0 ~ N(0, σ² I_d).
    Sensitivity of clipped output is 2 C_0 (L2 ball of radius C_0).
    Uses exact Gaussian mechanism calibration valid for all ε > 0.
    """
    return gaussian_mechanism_sigma_general(2.0 * C_0, epsilon, delta)


def apply_output_perturbation(w_hat, C_0, epsilon, delta, random_state=None):
    """
    Apply output perturbation: x_0 = Π_{C_0}(ŵ) + ξ_0.

    Parameters
    ----------
    w_hat : np.ndarray
        Trained model weights (e.g. on full data), shape (d,) or (d+1,) with intercept.
    C_0 : float
        Clipping radius (sensitivity bound).
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    random_state : int or np.random.RandomState, optional

    Returns
    -------
    x_0 : np.ndarray
        Unlearned model (clipped + noise).
    sigma : float
        Noise std used (for audit: same σ used for retain+noise).
    """
    w_hat = np.asarray(w_hat, dtype=float).flatten()
    clipped = clip_to_ball(w_hat, C_0)
    sigma = output_perturbation_sigma(C_0, epsilon, delta)
    rng = np.random.RandomState(random_state)
    xi = rng.normal(0, sigma, size=clipped.shape)
    x_0 = clipped + xi
    return x_0, sigma


def sample_unlearned_models(w_hat, C_0, epsilon, delta, n_samples, random_state=None):
    """
    Generate n_samples unlearned models from the same ŵ (full-data model):
    each is Π_{C_0}(ŵ) + ξ with ξ ~ N(0, σ² I). Used for attack distribution estimation.
    """
    clipped = clip_to_ball(w_hat, C_0)
    sigma = output_perturbation_sigma(C_0, epsilon, delta)
    rng = np.random.RandomState(random_state)
    d = len(clipped)
    noise = rng.normal(0, sigma, size=(n_samples, d))
    models = clipped + noise
    return models, sigma


def sample_retain_models(w_retain, C_0, epsilon, delta, n_samples, random_state=None):
    """
    Generate n_samples models from retain-only: Π_{C_0}(w_retain) + ξ, same σ.
    Same noise scale as unlearned so the audit is comparable.
    """
    return sample_unlearned_models(w_retain, C_0, epsilon, delta, n_samples, random_state)
