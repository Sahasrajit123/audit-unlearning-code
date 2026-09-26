"""
Strategic forget set generation with orthogonal parameters.

Generates forget set using parameters orthogonal to true θ*:
- θ^{f1} orthogonal to θ*
- θ^{f2} = -θ^{f1} (opposite direction, also orthogonal to θ*)
"""
import numpy as np
from sklearn.model_selection import train_test_split


def generate_orthogonal_theta(theta_star, norm=1.0, random_state=None):
    """
    Generate θ^{f1} orthogonal to θ*.
    
    Uses Gram-Schmidt orthogonalization to find a random vector
    orthogonal to θ*.
    
    Parameters:
    -----------
    theta_star : np.ndarray, shape (d,)
        Original true parameter
    norm : float
        Desired L2 norm of orthogonal vector
    random_state : int or RandomState
        Random seed
        
    Returns:
    --------
    theta_f1 : np.ndarray, shape (d,)
        Vector orthogonal to θ* with specified norm
    """
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)
    d = len(theta_star)

    if d < 2:
        raise ValueError(
            f"generate_orthogonal_theta requires d >= 2 (got d={d}). "
            "No non-zero vector orthogonal to theta_star exists in 1D. "
            "Use d >= 2 for logistic loss with orthogonal forget sets."
        )

    # Generate random vector
    v = rng.randn(d)

    # Remove component parallel to θ* (Gram-Schmidt): θ_f1 = v - (v·θ*/||θ*||²) θ*
    theta_star_sq = np.dot(theta_star, theta_star)
    if theta_star_sq < 1e-15:
        # θ* is (nearly) zero: any unit vector is "orthogonal"
        theta_f1 = v / np.linalg.norm(v) * norm
    else:
        theta_f1 = v - (np.dot(v, theta_star) / theta_star_sq) * theta_star
        theta_f1_norm = np.linalg.norm(theta_f1)
        if theta_f1_norm < 1e-15:
            # v was parallel to θ*; retry with fresh random (iterative, not recursive)
            for _ in range(100):
                v = rng.randn(d)
                theta_f1 = v - (np.dot(v, theta_star) / theta_star_sq) * theta_star
                theta_f1_norm = np.linalg.norm(theta_f1)
                if theta_f1_norm >= 1e-15:
                    break
            else:
                raise RuntimeError("Failed to find a vector orthogonal to theta_star after 100 attempts.")
        theta_f1 = theta_f1 / theta_f1_norm * norm

    dot_product = np.dot(theta_f1, theta_star)
    assert np.abs(dot_product) < 1e-10, f"Not orthogonal: dot product = {dot_product}"

    return theta_f1


def f1_link(z, eta=0.8):
    """Link function f1: step at 0."""
    return np.where(z >= 0, eta, 1 - eta)


def linear_link(z, eta=0.1):
    """
    Linear link: P(y=1) = 0.5 + eta * z, clipped to [0, 1].
    At z=0 → 0.5; z<0 → <0.5; z>0 → >0.5. eta controls the slope (sensitivity to z).
    """
    return np.clip(0.5 + eta * z, 0.0, 1.0)


def f2_link(z, eta=0.8, zeta=0.9):
    """Link function f2: piecewise constant."""
    probs = np.zeros_like(z)
    probs[z < -0.5] = 1 - zeta
    probs[(z >= -0.5) & (z < 0)] = 1 - eta
    probs[(z >= 0) & (z < 0.5)] = eta
    probs[z >= 0.5] = zeta
    return probs


def tanh_link(z, eta=0.5):
    """
    Tanh link function for MSE loss: smooth, non-linear, centered at 0.5.
    
    Output: 0.5 + eta * tanh(z)
    - At z=0: output = 0.5 (centered)
    - As z → +∞: output → 0.5 + eta
    - As z → -∞: output → 0.5 - eta
    - Smooth and differentiable everywhere
    - Not hard-capped (can go slightly beyond [0.5 - eta, 0.5 + eta] due to tanh range)
    
    Parameters:
    -----------
    z : np.ndarray
        Linear predictor (X @ theta)
    eta : float
        Scale parameter controlling the range of outputs
        Larger eta → wider range of possible values
        
    Returns:
    --------
    np.ndarray : Continuous values centered at 0.5, typically in [0.5 - eta, 0.5 + eta]
    """
    return 0.5 + eta * np.tanh(z)



def generate_samples(n, d, theta, link_function='f1', eta=0.8, zeta=0.9, 
                    feature_distribution='gaussian', loss='logistic', random_state=None):
    """
    Generate samples using a specific parameter vector.
    
    Parameters:
    -----------
    n : int
        Number of samples
    d : int
        Dimensionality
    theta : np.ndarray, shape (d,)
        Parameter vector
    link_function : str
        'f1', 'f2', 'linear', or 'tanh' (tanh recommended for MSE loss)
    eta, zeta : float
        Link function parameters (eta = slope for 'linear')
    feature_distribution : str
        'gaussian' or 'uniform'
    loss : str
        'logistic' for binary classification (y ∈ {0, 1})
        'mse' for regression (y ∈ ℝ, mapped from probability)
    random_state : int or RandomState
        Random seed
        
    Returns:
    --------
    X : np.ndarray, shape (n, d)
        Features
    y : np.ndarray, shape (n,)
        Labels (binary for 'logistic', continuous for 'mse')
    probs : np.ndarray, shape (n,)
        P(y=1) for each data point (for 'logistic')
        or probability value used to generate continuous label (for 'mse')
    """
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)
    
    # Generate features
    if feature_distribution == 'gaussian':
        X = rng.randn(n, d) / np.sqrt(d)
    elif feature_distribution == 'uniform':
        X = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, d))
    else:
        raise ValueError(f"Unknown distribution: {feature_distribution}")
    
    # Compute linear predictor
    z = X @ theta
    
    # Apply link function
    if link_function == 'f1':
        probs = f1_link(z, eta=eta)
    elif link_function == 'f2':
        probs = f2_link(z, eta=eta, zeta=zeta)
    elif link_function == 'linear':
        probs = linear_link(z, eta=eta)
    elif link_function == 'tanh':
        probs = tanh_link(z, eta=eta)
    else:
        raise ValueError(f"Unknown link function: {link_function}")
    
    # Generate labels based on loss type
    if loss == 'logistic':
        # Binary classification: sample from Bernoulli distribution
        y = rng.binomial(1, probs)
    elif loss == 'mse':
        # Regression: map probability p to continuous value (p - 0.5)
        # This maps p ∈ [0, 1] to y ∈ [-0.5, 0.5]
        # p=0 → y=-0.5, p=0.5 → y=0, p=1 → y=0.5
        y = probs - 0.5
    else:
        raise ValueError(f"loss must be 'logistic' or 'mse', got {loss!r}")
    
    return X, y, probs


def _parse_unit_vector_a(a, d):
    """Parse config 'a' to unit vector of length d. a can be 'e1'/'e_1' or list/array."""
    if a == 'e1' or a == 'e_1':
        out = np.zeros(d)
        out[0] = 1.0
        return out
    a = np.asarray(a, dtype=float).flatten()
    if len(a) != d:
        raise ValueError(f"cubic a must have length d={d}, got {len(a)}")
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("cubic a must not be zero")
    return a / n


def generate_strategic_data_logistic(n_retain, n_forget, d,
                                     n_partitions=2,
                                     theta_star=None, theta_norm=1.0,
                                     forget_norm=1.0,
                                     link_function='f1', eta=0.8, zeta=0.9,
                                     feature_distribution='gaussian',
                                     loss='logistic',
                                     forget_point_boundary=False,
                                     forget_boundary_label=1,
                                     random_state=42):
    """
    Generate data with strategic forget set for logistic / MSE loss.

    n_partitions=1 : single forget set — all n_forget samples from θ^{f1}.
    n_partitions=2 : two forget sets — half from θ^{f1} ⊥ θ*, half from θ^{f2} = -θ^{f1}.

    forget_point_boundary : bool
        If True, all forget points are placed at the same fixed feature vector
        (theta_f1 / ||theta_f1|| * forget_norm) with the same fixed label
        (forget_boundary_label). Mirrors the cubic construction where all
        forget points sit at a single boundary location.
    forget_boundary_label : int (0 or 1)
        Label assigned to all forget points when forget_point_boundary=True.

    For cubic loss use generate_strategic_data_multipartition instead.
    """
    if loss == 'cubic':
        raise ValueError("Use generate_strategic_data_multipartition for cubic loss.")

    rng = np.random.RandomState(random_state)

    if theta_star is None:
        theta = rng.randn(d)
        theta_star = theta / np.linalg.norm(theta) * theta_norm

    if d == 1:
        # No orthogonal direction exists in 1D: use -theta_star as forget direction.
        theta_f1 = -theta_star / np.linalg.norm(theta_star) * forget_norm
    else:
        theta_f1 = generate_orthogonal_theta(theta_star, norm=forget_norm, random_state=rng)
    theta_f2 = -theta_f1

    # Boundary feature vector: unit direction of theta_f1 scaled to forget_norm
    x_boundary_f1 = theta_f1 / np.linalg.norm(theta_f1) * forget_norm
    x_boundary_f2 = theta_f2 / np.linalg.norm(theta_f2) * forget_norm

    print("\n" + "="*70)
    print("STRATEGIC FORGET SET GENERATION (logistic/MSE)")
    print("="*70)
    print(f"  n_partitions={n_partitions}, ||θ*||={np.linalg.norm(theta_star):.4f}")
    if d == 1:
        print(f"  d=1: θ^{{f1}} = -θ* (opposite direction, no orthogonal exists)")
    if forget_point_boundary:
        print(f"  forget_point_boundary=True: all forget points at fixed x, label={forget_boundary_label}")
        print(f"  x_boundary_f1 = {x_boundary_f1[:min(3,d)]}{'...' if d>3 else ''}")
    print(f"  θ^{{f1}}·θ* = {np.dot(theta_f1, theta_star):.2e}  "
          f"θ^{{f1}}·θ^{{f2}} = {np.dot(theta_f1, theta_f2):.4f}")

    X_retain, y_retain, p_retain = generate_samples(
        n_retain, d, theta_star,
        link_function=link_function, eta=eta, zeta=zeta,
        feature_distribution=feature_distribution, loss=loss, random_state=rng,
    )

    if n_partitions == 1:
        n_f1, n_f2 = n_forget, 0
    else:
        n_f1 = n_forget // 2
        n_f2 = n_forget - n_f1

    if forget_point_boundary:
        X_f1 = np.tile(x_boundary_f1, (n_f1, 1))
        y_f1 = np.full(n_f1, float(forget_boundary_label))
        p_f1 = np.full(n_f1, float(forget_boundary_label))
    else:
        X_f1, y_f1, p_f1 = generate_samples(
            n_f1, d, theta_f1,
            link_function=link_function, eta=eta, zeta=zeta,
            feature_distribution=feature_distribution, loss=loss, random_state=rng,
        )

    if n_f2 > 0:
        if forget_point_boundary:
            X_f2 = np.tile(x_boundary_f2, (n_f2, 1))
            y_f2 = np.full(n_f2, float(forget_boundary_label))
            p_f2 = np.full(n_f2, float(forget_boundary_label))
        else:
            X_f2, y_f2, p_f2 = generate_samples(
                n_f2, d, theta_f2,
                link_function=link_function, eta=eta, zeta=zeta,
                feature_distribution=feature_distribution, loss=loss, random_state=rng,
            )
    else:
        X_f2 = np.zeros((0, d))
        y_f2 = np.zeros(0)
        p_f2 = np.zeros(0)

    X_forget = np.vstack([X_f1, X_f2]) if n_f2 > 0 else X_f1
    y_forget = np.concatenate([y_f1, y_f2])
    p_forget = np.concatenate([p_f1, p_f2])
    forget_half_indices = np.concatenate([np.zeros(n_f1, dtype=int), np.ones(n_f2, dtype=int)])

    X_train = np.vstack([X_retain, X_forget])
    y_train = np.concatenate([y_retain, y_forget])
    p_train = np.concatenate([p_retain, p_forget])

    print(f"  Retain: {len(X_retain)},  Forget: {n_f1} (f1) + {n_f2} (f2) = {len(X_forget)}")
    print("="*70)

    X_partitions = [X_f1] if n_partitions == 1 else [X_f1, X_f2]
    return {
        'X_retain': X_retain, 'y_retain': y_retain, 'p_retain': p_retain,
        'X_forget': X_forget, 'y_forget': y_forget, 'p_forget': p_forget,
        'X_forget_first_half': X_f1, 'y_forget_first_half': y_f1, 'p_forget_first_half': p_f1,
        'X_forget_second_half': X_f2, 'y_forget_second_half': y_f2, 'p_forget_second_half': p_f2,
        'forget_half_indices': forget_half_indices,
        'X_train': X_train, 'y_train': y_train, 'p_train': p_train,
        'theta_star': theta_star, 'theta_f1': theta_f1, 'theta_f2': theta_f2,
        'forget_split': (n_f1, n_f2),
        'X_partitions': X_partitions,
    }



def generate_strategic_data_multipartition(n_retain, n_forget_per_partition, d,
                                           n_partitions=1,
                                           data_R=20.0,
                                           retain_coord=None,
                                           forget_coord=None,
                                           random_state=42):
    """
    Unified cubic data generation for any number of partitions.

    Retain set: n_retain points, each at (retain_coord, 0, ..., 0).
      retain_coord defaults to 0 (all-zeros, original behaviour).

    Forget partition i: n_forget_per_partition points, each at a vector that is
      zero everywhere except index i, where it equals forget_coord.
      forget_coord defaults to -data_R (original behaviour).

    n_partitions=1 : single forget set along e1.
    n_partitions=2 : two forget sets along e1 and e2.
    n_partitions=K : K partitions; forget set = any K/2 → C(K,K/2) configurations.
                     K must be even and d >= K.
    """
    import itertools

    K = n_partitions
    if d < K:
        raise ValueError(f"d must be >= n_partitions, got d={d}, n_partitions={K}")
    if K > 2 and K % 2 != 0:
        raise ValueError(f"n_partitions must be even for K>2, got {K}")

    # Resolve defaults
    retain_val = 0.0 if retain_coord is None else float(retain_coord)
    forget_val = -float(data_R) if forget_coord is None else float(forget_coord)

    # --- retain ---
    X_retain = np.zeros((n_retain, d))
    X_retain[:, 0] = retain_val
    y_retain = np.zeros(n_retain)

    # --- forget partitions ---
    n_per = [n_forget_per_partition] * K

    X_partitions = []
    for i in range(K):
        X_i = np.zeros((n_per[i], d))
        X_i[:, i] = forget_val
        X_partitions.append(X_i)

    X_forget = np.vstack(X_partitions)
    y_forget = np.zeros(len(X_forget))
    X_train = np.vstack([X_retain, X_forget])
    y_train = np.zeros(len(X_train))

    print("\n" + "="*70)
    print("STRATEGIC FORGET SET GENERATION (multipartition)")
    print("="*70)
    print(f"  n_partitions={K}, d={d}")
    print(f"  RETAIN:  {n_retain} × retain_coord={retain_val} at index 0")
    for i in range(K):
        print(f"  FORGET partition {i}: {n_per[i]} × forget_coord={forget_val} at index {i}")
    if K > 2:
        half = K // 2
        partition_tuples = list(itertools.combinations(range(K), half))
        print(f"  Configurations (C({K},{half})): {len(partition_tuples)}")
    print("="*70)

    result = {
        'X_retain': X_retain, 'y_retain': y_retain,
        'X_forget': X_forget, 'y_forget': y_forget,
        'X_partitions': X_partitions,
        'X_train': X_train, 'y_train': y_train,
        'n_partitions': K,
        'n_forget_per_partition': n_per,
        'data_R': data_R,
        'retain_coord': retain_val,
        'forget_coord': forget_val,
    }

    if K > 2:
        result['partition_tuples'] = partition_tuples
        result['K'] = K

    return result


def generate_test_val_sets(d, theta_star, n_val=1000, n_test=2000,
                           link_function='f1', eta=0.8, zeta=0.9,
                           feature_distribution='gaussian',
                           loss='logistic',
                           cubic_params=None,
                           random_state=42):
    """
    Generate validation and test sets using θ* (or for cubic: 50-50 ±a).
    """
    rng = np.random.RandomState(random_state)

    if loss == 'cubic':
        # Val/test: points in Z (||z||<=1). Use zeros to match retain R.
        X_val = np.zeros((n_val, d))
        X_test = np.zeros((n_test, d))
        return {
            'X_val': X_val, 'y_val': np.zeros(n_val), 'p_val': np.full(n_val, 0.5),
            'X_test': X_test, 'y_test': np.zeros(n_test), 'p_test': np.full(n_test, 0.5)
        }

    print(f"\nGenerating validation set ({n_val} samples) using θ*...")
    X_val, y_val, p_val = generate_samples(
        n_val, d, theta_star,
        link_function=link_function,
        eta=eta, zeta=zeta,
        feature_distribution=feature_distribution,
        loss=loss,
        random_state=rng
    )
    
    print(f"Generating test set ({n_test} samples) using θ*...")
    X_test, y_test, p_test = generate_samples(
        n_test, d, theta_star,
        link_function=link_function,
        eta=eta, zeta=zeta,
        feature_distribution=feature_distribution,
        loss=loss,
        random_state=rng
    )
    
    return {
        'X_val': X_val,
        'y_val': y_val,
        'p_val': p_val,
        'X_test': X_test,
        'y_test': y_test,
        'p_test': p_test
    }


def compute_bayes_error(theta, link_function='f1', eta=0.8, zeta=0.9, 
                       n_samples=100000, random_state=42):
    """
    Estimate Bayes error for a given parameter vector.
    
    Parameters:
    -----------
    theta : np.ndarray
        Parameter vector
    link_function : str
        'f1', 'f2', 'linear', or 'tanh' (tanh recommended for MSE loss)
    eta, zeta : float
        Link function parameters
    n_samples : int
        Number of Monte Carlo samples
    random_state : int
        Random seed
        
    Returns:
    --------
    bayes_error : float
        Estimated Bayes error rate
    """
    rng = np.random.RandomState(random_state)
    d = len(theta)
    
    # Generate test data
    X = rng.randn(n_samples, d) / np.sqrt(d)
    z = X @ theta
    
    # Get probabilities
    if link_function == 'f1':
        probs = f1_link(z, eta=eta)
    elif link_function == 'f2':
        probs = f2_link(z, eta=eta, zeta=zeta)
    elif link_function == 'linear':
        probs = linear_link(z, eta=eta)
    elif link_function == 'tanh':
        probs = tanh_link(z, eta=eta)
    else:
        raise ValueError(f"Unknown link function: {link_function}")
    
    # Bayes optimal prediction
    optimal_pred = (probs >= 0.5).astype(int)
    
    # Generate true labels
    y_true = rng.binomial(1, probs)
    
    # Compute error
    bayes_error = np.mean(optimal_pred != y_true)
    
    return bayes_error


if __name__ == "__main__":
    # Test strategic forget set generation
    np.random.seed(42)
    
    # Generate strategic data with new defaults
    data = generate_strategic_data_logistic(
        n_retain=40500,
        n_forget=4500,
        d=20,
        theta_norm=1.0,
        forget_norm=10.0,  # 10x stronger signal!
        link_function='f1',
        eta=0.8,
        random_state=42
    )
    
    # Generate val/test sets
    eval_data = generate_test_val_sets(
        d=20,
        theta_star=data['theta_star'],
        n_val=5000,
        n_test=10000,
        link_function='f1',
        eta=0.8,
        random_state=42
    )
    
    # Compute Bayes errors
    bayes_star = compute_bayes_error(data['theta_star'], 'f1', eta=0.8)
    bayes_f1 = compute_bayes_error(data['theta_f1'], 'f1', eta=0.8)
    
    print(f"\n" + "="*70)
    print("BAYES ERRORS")
    print("="*70)
    print(f"Bayes error (θ*):      {bayes_star:.4f} -> Accuracy: {1-bayes_star:.4f}")
    print(f"Bayes error (θ^{{f1}}):  {bayes_f1:.4f} -> Accuracy: {1-bayes_f1:.4f}")
    print("="*70)
