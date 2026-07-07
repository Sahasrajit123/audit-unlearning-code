"""
Data generation for output perturbation unlearning: logistic regression d=2.
Retain + forget so that model on full data differs significantly from model on retain only.
"""
import numpy as np


def f1_link(z, eta=0.8):
    """Link function f1: step at 0. P(y=1) = eta if z>=0 else 1-eta."""
    return np.where(z >= 0, eta, 1 - eta)


def linear_link(z, eta=0.3, range_half=0.6):
    """Linear link for regression: E[y] in [0.5 - range_half, 0.5 + range_half], then y = probs - 0.5.
    Default range_half=0.6 gives y in [-0.6, 0.6] (wider than 0.5). Slope eta."""
    return np.clip(0.5 + eta * z, 0.5 - range_half, 0.5 + range_half)


def _get_rng(random_state):
    """Return a RandomState: use as-is if already RandomState, else RandomState(seed)."""
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def generate_samples(n, d, theta, link_function='f1', eta=0.8, zeta=0.9,
                    feature_distribution='gaussian', loss='logistic', random_state=None,
                    range_half=0.6):
    """Generate (X, y, probs). loss='logistic' -> binary y; loss='mse' -> continuous y = probs - 0.5.
    range_half: for linear_link (MSE), y is in [-range_half, range_half]. Default 0.6."""
    rng = _get_rng(random_state)
    if feature_distribution == 'gaussian':
        X = rng.randn(n, d) / np.sqrt(d)
    else:
        X = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, d))
    z = X @ theta
    if link_function == 'linear':
        probs = linear_link(z, eta=eta, range_half=range_half)
    else:
        probs = f1_link(z, eta=eta)
    if loss == 'logistic':
        y = rng.binomial(1, probs)
    elif loss == 'mse':
        # Regression: y = E[y|X] = probs - 0.5; range ±range_half; add small noise
        y = probs - 0.5 + rng.normal(0, 0.05, size=n)
        y = np.clip(y, -range_half, range_half)
    else:
        raise ValueError(f"loss must be 'logistic' or 'mse', got {loss!r}")
    return X, y, probs


def generate_orthogonal_theta(theta_star, norm=1.0, random_state=None):
    """Return a vector orthogonal to theta_star with L2 norm = norm (for d=2 unique up to sign)."""
    rng = _get_rng(random_state)
    d = len(theta_star)
    v = rng.randn(d)
    theta_star_sq = np.dot(theta_star, theta_star)
    if theta_star_sq < 1e-15:
        return v / np.linalg.norm(v) * norm
    theta_f = v - (np.dot(v, theta_star) / theta_star_sq) * theta_star
    nrm = np.linalg.norm(theta_f)
    if nrm < 1e-15:
        return generate_orthogonal_theta(theta_star, norm=norm, random_state=rng)
    return theta_f / nrm * norm


def generate_strategic_data(n_retain, n_forget, d=2, theta_norm=1.0, forget_norm=8.0,
                            link_function='f1', eta=0.8, zeta=0.9,
                            feature_distribution='gaussian', random_state=42,
                            forget_direction='opposite', loss='logistic', range_half=0.6):
    """
    Generate retain set from θ* and forget set from θ_f so that ERM on full data
    differs significantly from ERM on retain only.

    forget_direction: 'orthogonal' -> θ_f ⊥ θ* (same as before).
                      'opposite'  -> θ_f = -θ* scaled by forget_norm (forget set
                      prefers opposite labels; full model is pulled away from retain).
    """
    rng = _get_rng(random_state)
    theta_star = rng.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star) * theta_norm
    if forget_direction == 'opposite':
        theta_f = -theta_star * (forget_norm / theta_norm)
    else:
        theta_f = generate_orthogonal_theta(theta_star, norm=forget_norm, random_state=rng)

    X_retain, y_retain, p_retain = generate_samples(
        n_retain, d, theta_star, link_function=link_function, eta=eta, zeta=zeta,
        feature_distribution=feature_distribution, loss=loss, range_half=range_half, random_state=rng
    )
    X_forget, y_forget, p_forget = generate_samples(
        n_forget, d, theta_f, link_function=link_function, eta=eta, zeta=zeta,
        feature_distribution=feature_distribution, loss=loss, range_half=range_half, random_state=rng
    )

    X_train = np.vstack([X_retain, X_forget])
    y_train = np.concatenate([y_retain, y_forget])
    p_train = np.concatenate([p_retain, p_forget])

    print("\n" + "="*70)
    print(f"STRATEGIC DATA (d=2, loss={loss}: full vs retain differ)")
    print("="*70)
    print(f"  θ* (retain): ||θ*|| = {np.linalg.norm(theta_star):.4f}")
    print(f"  θ_f (forget): ||θ_f|| = {np.linalg.norm(theta_f):.4f}, θ_f·θ* = {np.dot(theta_f, theta_star):.2e} ({forget_direction})")
    print(f"  Retain: {n_retain} samples, Forget: {n_forget} samples")
    print("="*70)

    return {
        'X_retain': X_retain,
        'y_retain': y_retain,
        'p_retain': p_retain,
        'X_forget': X_forget,
        'y_forget': y_forget,
        'p_forget': p_forget,
        'X_train': X_train,
        'y_train': y_train,
        'p_train': p_train,
        'theta_star': theta_star,
        'theta_f': theta_f,
    }


def generate_test_val_sets(d, theta_star, n_val=200, n_test=400,
                           link_function='f1', eta=0.8, zeta=0.9,
                           feature_distribution='gaussian', loss='logistic', random_state=42,
                           range_half=0.6):
    """Generate validation and test sets from θ*."""
    rng = _get_rng(random_state)
    X_val, y_val, p_val = generate_samples(
        n_val, d, theta_star, link_function=link_function, eta=eta, zeta=zeta,
        feature_distribution=feature_distribution, loss=loss, range_half=range_half, random_state=rng
    )
    X_test, y_test, p_test = generate_samples(
        n_test, d, theta_star, link_function=link_function, eta=eta, zeta=zeta,
        feature_distribution=feature_distribution, loss=loss, range_half=range_half, random_state=rng
    )
    return {
        'X_val': X_val, 'y_val': y_val, 'p_val': p_val,
        'X_test': X_test, 'y_test': y_test, 'p_test': p_test,
    }
