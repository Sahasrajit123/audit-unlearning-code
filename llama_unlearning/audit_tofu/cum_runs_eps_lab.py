import math
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar


# ----------------------------
# Basic stable helpers
# ----------------------------

def log_binom(n: int, k: int) -> float:
    """log( n choose k ), with invalid k -> -inf."""
    if k < 0 or k > n:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def log1mexp(logx: float) -> float:
    """
    Compute log(1 - exp(logx)) stably for logx <= 0.
    Used with logx = -logZ (tiny).
    """
    if logx > 0:
        raise ValueError("log1mexp expects logx <= 0.")
    # If exp(logx) is extremely small, 1-exp(logx) ~ 1
    if logx < -50:
        return 0.0
    return float(math.log1p(-math.exp(logx)))


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a)+exp(b))."""
    return float(np.logaddexp(a, b))


# ----------------------------
# Compute log f(v) table exactly (log-space)
# ----------------------------

def log_f_values(m: int, r: int) -> np.ndarray:
    """
    Compute log f(v) for v=0..r (inclusive) in log-space.

    f(v) = sum_{a1+a2=v, a1,a2 in [0,r/2]}
           C(n, floor(n/2) - (a1-a2)) * C(r/2,a1)*C(r/2,a2)
    where n = m-r, r even.
    """
    if r % 2 != 0:
        raise ValueError("Assumes r is even so r/2 is integer.")
    if m < r:
        raise ValueError("Need m >= r so n=m-r >= 0.")

    n = m - r
    half_r = r // 2
    center = n // 2  # floor(n/2)

    logf = np.full(r + 1, -np.inf, dtype=float)

    for v in range(r + 1):
        lo = max(0, v - half_r)
        hi = min(half_r, v)
        terms = []
        for a1 in range(lo, hi + 1):
            a2 = v - a1
            d = a1 - a2          # = 2*a1 - v
            k_idx = center - d   # floor(n/2) - (a1-a2)

            lb = log_binom(n, k_idx)
            if not np.isfinite(lb):
                continue

            terms.append(lb + log_binom(half_r, a1) + log_binom(half_r, a2))

        if terms:
            logf[v] = logsumexp(np.array(terms, dtype=float))

    return logf


# ----------------------------
# Z in closed form
# ----------------------------

def log_Z_closed_form(m: int) -> float:
    """Z = C(m, floor(m/2)). Return logZ."""
    return log_binom(m, m // 2)


# ----------------------------
# Threshold c from a list of v values
# ----------------------------

def threshold_c_from_v_list(v_list, logf: np.ndarray, T: int) -> float:
    """
    c = (1/T) * sum_{i=1}^T log f(v_i),
    enforcing len(v_list) == T.
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list of length T={T}, got length {len(v_list)}.")

    r = len(logf) - 1
    s = 0.0
    for i, v in enumerate(v_list):
        if not (0 <= v <= r):
            raise ValueError(f"v_list[{i}]={v} outside [0,{r}].")
        lf = float(logf[v])
        if not np.isfinite(lf):
            raise ValueError(f"f(v)=0 or undefined at v_list[{i}]={v}.")
        s += lf

    return s / T



# ----------------------------
# Chernoff upper bound on log M(c)
# ----------------------------

def logM_chernoff_upper(logf: np.ndarray, T: int, c_avglog: float, theta_max: float = 50.0):
    """
    log M(c) <= inf_{theta>=0} [ T*log sum_v f(v)^{1+theta} - theta*T*c ].
    Works purely in log-space.
    """
    if T <= 0:
        raise ValueError("T must be positive.")

    mask = np.isfinite(logf)
    lf = logf[mask]
    if lf.size == 0:
        raise ValueError("All logf are -inf; f(v)=0 everywhere?")

    def obj(theta: float) -> float:
        log_sum = logsumexp((1.0 + theta) * lf)  # log sum_v f(v)^{1+theta}
        return float(T * log_sum - theta * T * c_avglog)

    res = minimize_scalar(obj, bounds=(0.0, theta_max), method="bounded")
    theta_star = float(res.x)
    logM_bound = float(res.fun)
    return logM_bound, theta_star


# ----------------------------
# Solve for largest epsilon satisfying:
# logM + T*log_ratio(eps) <= log(ci_delta)
# ratio(eps) = (e^eps + ((Z-1)/Z)*delta) / (e^eps + (Z-1))
# ----------------------------

def epsilon_lb_from_logM(
    logM: float,
    logZ: float,
    T: int,
    delta: float,
    ci_delta: float,
    eps_hi_init: float = 50.0,
    tol: float = 1e-10,
    max_iter: int = 200
):
    """
    Returns the largest epsilon >= 0 such that:
      M(c) * ratio(eps)^T <= ci_delta
    using log-space with logM provided.

    If infeasible even at eps=0, returns None.
    """
    if ci_delta <= 0:
        raise ValueError("ci_delta must be positive.")
    if delta < 0:
        raise ValueError("delta must be non-negative.")
    if T <= 0:
        raise ValueError("T must be positive.")

    log_ci = math.log(ci_delta)
    # When delta=0, ((Z-1)/Z)*delta=0 so log_a=-inf; logaddexp(eps,-inf)=eps. Avoid math.log(0).
    log_delta = math.log(delta) if delta > 0 else -np.inf

    # log(Z-1) and log(1 - 1/Z) robustly.
    # log(1/Z) = -logZ
    log_one_minus_1_over_Z = log1mexp(-logZ)   # log(1 - exp(-logZ))
    log_b = logZ + log_one_minus_1_over_Z      # log(Z-1) = logZ + log(1-1/Z)
    log_a = log_delta + log_one_minus_1_over_Z # log(((Z-1)/Z)*delta)

    def log_ratio(eps: float) -> float:
        # log( exp(eps)+a ) - log( exp(eps)+b )
        log_num = logaddexp(eps, log_a)
        log_den = logaddexp(eps, log_b)
        return log_num - log_den

    def lhs(eps: float) -> float:
        # log( M(c) * ratio(eps)^T ) = logM + T*log_ratio
        return logM + T * log_ratio(eps)

    # Check feasibility at eps=0
    if lhs(0.0) > log_ci:
        return None

    # Find an upper bracket where it fails (lhs > log_ci), since lhs increases in eps.
    hi = eps_hi_init
    while lhs(hi) <= log_ci:
        hi *= 2.0
        if hi > 1e6:  # extremely conservative cap
            return hi  # effectively unbounded in this numeric sense

    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if lhs(mid) <= log_ci:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, lo):
            break
    return lo


from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import numpy as np

def logM_lower_chernoff_upper(logf: np.ndarray, T: int, c_avglog: float, theta_max: float = 50.0):
    """
    Upper bound for lower-tail mass:
      M_≤(c) = sum_{v_vec: (1/T) sum log f(v_i) <= c} prod_i f(v_i)

    Bound:
      log M_≤(c) <= inf_{theta>=0} [ T * log sum_v f(v)^{1-theta} + theta*T*c ].

    Works in log-space:
      log sum_v f(v)^{1-theta} = logsumexp((1-theta)*logf[v]).
    """
    if T <= 0:
        raise ValueError("T must be positive.")

    mask = np.isfinite(logf)
    lf = logf[mask]
    if lf.size == 0:
        raise ValueError("All logf are -inf; f(v)=0 everywhere?")

    def obj(theta: float) -> float:
        log_sum = logsumexp((1.0 - theta) * lf)  # log sum_v f(v)^{1-theta}
        return float(T * log_sum + theta * T * c_avglog)

    res = minimize_scalar(obj, bounds=(0.0, theta_max), method="bounded")
    theta_star = float(res.x)
    logM_bound = float(res.fun)
    return logM_bound, theta_star


# ----------------------------
# Main wrapper: from v_list to (c, logM_bound, epsilon_lb)
# ----------------------------

def compute_c_logM_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    delta: float,
    ci_delta: float,
    theta_max: float = 50.0
):
    """
    1) builds logf(v) table
    2) computes c from the provided v_list
    3) computes Chernoff upper bound logM(c)
    4) computes epsilon_lb (largest eps s.t. inequality holds)

    Returns a dict with everything in log-space + epsilon.
    """
    logf = log_f_values(m, r)
    c = threshold_c_from_v_list(v_list, logf, T=T)

    logM_bound, theta_star = logM_chernoff_upper(logf, T=T, c_avglog=c, theta_max=theta_max)

    logZ = log_Z_closed_form(m)
    eps_star = epsilon_lb_from_logM(
        logM=logM_bound,
        logZ=logZ,
        T=T,
        delta=delta,
        ci_delta=ci_delta
    )

    return {
        "c": c,
        "logM_bound": logM_bound,
        "theta_star": theta_star,
        "logZ": logZ,
        "epsilon_lb": eps_star,
        "note": "epsilon_lb is the largest epsilon satisfying the inequality using the Chernoff upper bound on M(c)."
    }

def compute_c_logM_lower_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    delta: float,
    ci_delta: float,
    theta_max: float = 50.0
):
    """
    Uses lower-tail test: (1/T) sum log f(v_i) <= c,
    where c is computed from the provided v_list (len must equal T).

    Returns:
      c, logM_lower_bound, theta_star, logZ, epsilon_lb
    """
    logf = log_f_values(m, r)

    # c computed from the provided vector (len(v_list)=T)
    c = threshold_c_from_v_list(v_list, logf, T=T)

    # lower-tail Chernoff upper bound on log M_≤(c)
    logM_bound, theta_star = logM_lower_chernoff_upper(logf, T=T, c_avglog=c, theta_max=theta_max)

    # closed-form logZ
    logZ = log_Z_closed_form(m)

    # solve for largest epsilon satisfying:
    #   M(c) * ratio(eps)^T <= ci_delta
    eps_star = epsilon_lb_from_logM(
        logM=logM_bound,
        logZ=logZ,
        T=T,
        delta=delta,
        ci_delta=ci_delta
    )

    return {
        "c": c,
        "logM_lower_bound": logM_bound,
        "theta_star": theta_star,
        "logZ": logZ,
        "epsilon_lb": eps_star,
        "note": "Lower-tail: M is over vectors with average log-score <= c."
    }

import numpy as np
from scipy.special import logsumexp

def log_g_from_logf(logf: np.ndarray) -> np.ndarray:
    """
    log g(v) where g(v) = sum_{s=v}^r f(s).
    """
    r = len(logf) - 1
    logg = np.full(r + 1, -np.inf)
    acc = -np.inf
    for v in range(r, -1, -1):
        acc = logsumexp([acc, logf[v]])
        logg[v] = acc
    return logg

def logc_from_v_list_using_g(v_list, logg: np.ndarray, T: int) -> float:
    """
    log c = sum_{i=1}^T log g(v_i)
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list length T={T}, got {len(v_list)}")

    r = len(logg) - 1
    total = 0.0
    for i, v in enumerate(v_list):
        if not (0 <= v <= r):
            raise ValueError(f"v_list[{i}]={v} outside [0,{r}]")
        if not np.isfinite(logg[v]):
            raise ValueError(f"log g(v) is -inf at v={v}")
        total += logg[v]

    return total


def logM_bound_prod_g_le_c(
    logf: np.ndarray,
    logg: np.ndarray,
    T: int,
    logc: float,
    theta_max: float = 50.0
):
    """
    Upper bound on log M(c) for the test:
      prod_i g(v_i) <= c
    with weight prod_i f(v_i).
    """
    mask = np.isfinite(logf) & np.isfinite(logg)
    lf = logf[mask]
    lg = logg[mask]

    def obj(theta: float) -> float:
        # log sum_v f(v) * g(v)^(-theta)
        log_sum = logsumexp(lf - theta * lg)
        return theta * logc + T * log_sum

    res = minimize_scalar(obj, bounds=(0.0, theta_max), method="bounded")
    return float(res.fun), float(res.x)

def compute_logc_logM_gtest_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    delta: float,
    ci_delta: float,
    theta_max: float = 50.0
):
    """
    g-test wrapper.

    Definitions:
      f(v): equality version (alpha1+alpha2 = v)
      g(v): tail sum g(v)=sum_{s=v}^r f(s)

    Threshold (derived from v_list):
      c = prod_{i=1}^T g(v_i)   =>  logc = sum_i log g(v_i)

    Test:
      prod_i g(V_i) <= c

    Cumulative mass:
      M(c) = sum_{v_vec: prod g(v_i) <= c} prod f(v_i)

    Upper bound used:
      log M(c) <= inf_{theta>=0} [ theta*logc + T*log sum_v f(v)*g(v)^(-theta) ].

    Then solve for largest epsilon (epsilon_lb) s.t.
      M(c) * ((e^eps + (Z-1)/Z * delta)/(e^eps + Z-1))^T <= ci_delta,
    where Z = sum_v f(v) = C(m, floor(m/2)).
    """
    # requires: log_f_values, log_Z_closed_form, epsilon_lb_from_logM
    # requires: log_g_from_logf, logc_from_v_list_using_g, logM_bound_prod_g_le_c

    logf = log_f_values(m, r)
    logg = log_g_from_logf(logf)

    # logc computed from v_list (len must equal T)
    logc = logc_from_v_list_using_g(v_list, logg, T=T)

    # bound logM for the g-test
    logM_bound, theta_star = logM_bound_prod_g_le_c(
        logf=logf,
        logg=logg,
        T=T,
        logc=logc,
        theta_max=theta_max,
    )

    # closed-form Z from m (works because Z = sum_v f(v))
    logZ = log_Z_closed_form(m)

    epsilon_lb = epsilon_lb_from_logM(
        logM=logM_bound,
        logZ=logZ,
        T=T,
        delta=delta,
        ci_delta=ci_delta,
    )

    return {
        "logc": float(logc),
        "logM_bound": float(logM_bound),
        "theta_star": float(theta_star),
        "logZ": float(logZ),
        "epsilon_lb": epsilon_lb,
        "note": "g-test: constraint prod g(v_i) <= c, threshold c derived from v_list."
    }

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar


def a_from_v_list(v_list, T: int) -> float:
    """a = average of v_i; enforce len(v_list)=T."""
    if len(v_list) != T:
        raise ValueError(f"Expected v_list length T={T}, got {len(v_list)}.")
    return float(np.mean(v_list))


def logM_bound_avg_v_ge_a(logf: np.ndarray, T: int, a: float, theta_max: float = 50.0):
    """
    Bound log M_avg>= (a) where
      M = sum_{avg(v_i) >= a} prod f(v_i).

    log M <= inf_{theta>=0} [ T*log sum_v f(v) e^{theta v} - theta*T*a ].
    """
    lf = np.asarray(logf, dtype=float)
    mask = np.isfinite(lf)
    lf = lf[mask]
    vgrid = np.arange(len(logf), dtype=float)[mask]

    def obj(theta: float) -> float:
        log_sum = logsumexp(lf + theta * vgrid)  # log sum f(v)*e^{theta v}
        return float(T * log_sum - theta * T * a)

    res = minimize_scalar(obj, bounds=(0.0, theta_max), method="bounded")
    return float(res.fun), float(res.x)


def logM_bound_avg_v_le_a(logf: np.ndarray, T: int, a: float, theta_max: float = 50.0):
    """
    Bound log M_avg<= (a) where
      M = sum_{avg(v_i) <= a} prod f(v_i).

    log M <= inf_{theta>=0} [ T*log sum_v f(v) e^{-theta v} + theta*T*a ].
    """
    lf = np.asarray(logf, dtype=float)
    mask = np.isfinite(lf)
    lf = lf[mask]
    vgrid = np.arange(len(logf), dtype=float)[mask]

    def obj(theta: float) -> float:
        log_sum = logsumexp(lf - theta * vgrid)  # log sum f(v)*e^{-theta v}
        return float(T * log_sum + theta * T * a)

    res = minimize_scalar(obj, bounds=(0.0, theta_max), method="bounded")
    return float(res.fun), float(res.x)


def compute_avg_v_test_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    delta: float,
    ci_delta: float,
    direction: str = "ge",     # "ge" for avg >= a, "le" for avg <= a
    theta_max: float = 50.0
):
    """
    Wrapper analogous to your other ones:
    - threshold a taken from v_list: a = average(v_list)
    - compute Chernoff upper bound on log M for the avg-v test
    - compute epsilon_lb using epsilon_lb_from_logM
    """
    logf = log_f_values(m, r)
    a = a_from_v_list(v_list, T=T)

    if direction == "ge":
        logM_bound, theta_star = logM_bound_avg_v_ge_a(logf, T=T, a=a, theta_max=theta_max)
    elif direction == "le":
        logM_bound, theta_star = logM_bound_avg_v_le_a(logf, T=T, a=a, theta_max=theta_max)
    else:
        raise ValueError("direction must be 'ge' or 'le'.")

    logZ = log_Z_closed_form(m)

    epsilon_lb = epsilon_lb_from_logM(
        logM=logM_bound,
        logZ=logZ,
        T=T,
        delta=delta,
        ci_delta=ci_delta
    )

    return {
        "a": a,
        "direction": direction,
        "logM_bound": logM_bound,
        "theta_star": theta_star,
        "logZ": float(logZ),
        "epsilon_lb": epsilon_lb,
        "note": "avg-v test with f-weight; threshold a is mean(v_list)."
    }


# ----------------------------
# Median-based epsilon lower bound computation
# ----------------------------

def compute_summation(m: int, r: int, v: float):
    """
    Compute the summation:
        Σ
    (α₁,α₂)∈[0,r/2]
      α₁+α₂≥v
    C(m-r, ⌊(m-r)/2⌋ - (α₁-α₂)) * C(r/2, α₁) * C(r/2, α₂)
    
    Where C(n, k) is the binomial coefficient "n choose k".
    
    Uses log-space computations to avoid overflow for large values.
    
    Args:
        m: Parameter m
        r: Parameter r (typically 100-200, but supports up to ~1000)
        v: Lower bound for α₁ + α₂ (can be float)
    
    Returns:
        Tuple of (linear_value, log_value):
        - linear_value: Sum in linear space (may be inf if too large)
        - log_value: Sum in log space (always finite if valid)
    
    Note: α₁ and α₂ are treated as integers in [0, floor(r/2)].
    """
    # Compute bounds
    r_half = math.floor(r / 2)
    m_minus_r = m - r
    floor_expr = math.floor(m_minus_r / 2)
    
    # Use log-space to avoid overflow
    log_terms = []
    
    # Iterate over all integer pairs (α₁, α₂) in [0, r/2]
    # Optimize by only iterating over valid pairs where α₁ + α₂ ≥ v
    for alpha1 in range(r_half + 1):
        # For each α₁, find the minimum α₂ such that α₁ + α₂ ≥ v
        min_alpha2 = max(0, math.ceil(v - alpha1))
        
        # Only iterate if min_alpha2 is within bounds
        if min_alpha2 <= r_half:
            for alpha2 in range(min_alpha2, r_half + 1):
                # Check constraint: α₁ + α₂ ≥ v
                if alpha1 + alpha2 >= v:
                    # Compute k = ⌊(m-r)/2⌋ - (α₁-α₂)
                    k = floor_expr - (alpha1 - alpha2)
                    
                    # Compute binomial coefficient C(m-r, k)
                    # Ensure k is within valid range [0, m-r]
                    if 0 <= k <= m_minus_r:
                        # Use log-space to compute: log(C(m-r, k) * C(r/2, α₁) * C(r/2, α₂))
                        log_term = (
                            log_binom(m_minus_r, k) +
                            log_binom(r_half, alpha1) +
                            log_binom(r_half, alpha2)
                        )
                        if np.isfinite(log_term):
                            log_terms.append(log_term)
    
    # Sum in log-space using logsumexp to avoid overflow
    if len(log_terms) == 0:
        return 0.0, -np.inf  # Return both linear (0) and log (-inf) values
    
    log_terms_array = np.array(log_terms, dtype=float)
    log_total = logsumexp(log_terms_array)
    
    # Always return log value for computation. Only compute linear value if needed for display/debugging
    # and if it's safe to do so (log_total < 700 to avoid overflow)
    if log_total > 700 or not np.isfinite(log_total):
        linear_value = float('inf')
    else:
        try:
            linear_value = float(np.exp(log_total))
            if not np.isfinite(linear_value):
                linear_value = float('inf')
        except (OverflowError, RuntimeWarning):
            linear_value = float('inf')
    
    return linear_value, log_total


def compute_median_v_test_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    delta: float,
    ci_delta: float = 0.05
):
    """
    Compute epsilon lower bound from evaluation statistics using the inequality:
    
    Pr[Median(...) ≥ v] ≤ [summation] × [fraction]^⌈L/2⌉ × (L choose ⌈L/2⌉)
    
    Where:
    - m = total number of batches
    - r = 2*k (where k is top/bottom k batches)
    - T = total number of runs (L)
    - v = median of v_list
    - ci_delta = upper bound probability (default 0.05)
    
    We solve for epsilon such that the right-hand side equals ci_delta.
    
    Args:
        m: Total number of batches
        r: Parameter r (typically 2*k)
        T: Total number of runs (L)
        v_list: List of v values (one per run), length must equal T
        delta: Audit Noise delta parameter
        ci_delta: Upper bound probability (default: 0.05)
    
    Returns:
        Dictionary with epsilon_lb and computation details
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list of length T={T}, got length {len(v_list)}.")
    
    # Compute median of v_list
    v_median = float(np.median(v_list))
    v = math.ceil(v_median)  # Use ceiling of median
    
    # Compute the summation term using the helper function
    # Returns both linear value (for display) and log value (for computation)
    summation_term, log_summation_term = compute_summation(m, r, v)
    
    # Compute binomial coefficients in log-space to avoid overflow
    T_half = math.ceil(T / 2)
    log_T_choose_T_half = log_binom(T, T_half)
    
    # Compute binomial coefficient C(m, ⌊m/2⌋) in log-space
    m_half = math.floor(m / 2)
    log_m_choose_m_half = log_binom(m, m_half)
    
    # Only convert T_choose_T_half if needed for display/debugging, but work in log-space for computation
    # Check if we can safely convert to float without overflow
    if np.isfinite(log_T_choose_T_half) and log_T_choose_T_half < 700:
        try:
            T_choose_T_half = float(np.exp(log_T_choose_T_half))
        except (OverflowError, RuntimeWarning):
            T_choose_T_half = float('inf')
    else:
        T_choose_T_half = float('inf')
    
    # Never exponentiate m_choose_m_half - work entirely in log-space
    # Only compute m_choose_m_half for display/debugging if it's small enough
    if np.isfinite(log_m_choose_m_half) and log_m_choose_m_half < 700:
        try:
            m_choose_m_half = float(np.exp(log_m_choose_m_half))
        except (OverflowError, RuntimeWarning):
            m_choose_m_half = float('inf')
    else:
        m_choose_m_half = float('inf')  # Mark as too large, but we'll use log-space
    
    # The inequality is:
    # ci_delta ≥ [summation_term × fraction]^⌈T/2⌉ × T_choose_T_half
    # 
    # Where fraction = (e^ε + (m_choose_m_half - 1)/m_choose_m_half * delta) / (e^ε + (m_choose_m_half - 1))
    # Note: The entire product (summation_term × fraction) is raised to the power of ⌈T/2⌉
    #
    # Rearranging:
    # [summation_term × fraction]^⌈T/2⌉ ≤ ci_delta / T_choose_T_half
    #
    # Let target = ci_delta / T_choose_T_half
    # Then: [summation_term × fraction]^⌈T/2⌉ ≤ target
    # Taking the ⌈T/2⌉-th root of both sides:
    # summation_term × fraction ≤ target^(1/⌈T/2⌉)
    # Therefore: fraction ≤ target^(1/⌈T/2⌉) / summation_term
    
    # Work in log-space to avoid overflow
    if not np.isfinite(log_T_choose_T_half) or log_T_choose_T_half == -np.inf:
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": 0.0,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "epsilon_lb": float('inf'),
            "note": "Cannot compute bound: T_choose_T_half is 0 or invalid"
        }
    
    # Compute target in log-space: log(ci_delta) - log(T_choose_T_half)
    log_target = math.log(ci_delta) - log_T_choose_T_half
    if log_target >= 0 or not np.isfinite(log_target):
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": T_choose_T_half,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "epsilon_lb": float('inf'),
            "note": "Cannot compute bound: Invalid log_target"
        }
    
    # Compute target_product in log-space: log(target) / T_half
    # This extracts (summation_term × fraction) from [summation_term × fraction]^⌈T/2⌉ ≤ target
    if T_half == 0:
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": T_choose_T_half,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "epsilon_lb": float('inf'),
            "note": "Cannot compute bound: T < 2"
        }
    
    log_target_product = log_target / T_half  # log(target_product) = log(target) / T_half
    
    # Now extract fraction from: summation_term × fraction ≤ target_product
    # Work entirely in log-space using log_summation_term
    if not np.isfinite(log_summation_term) or log_summation_term == -np.inf:
        # Invalid or zero summation term
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": T_choose_T_half,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "epsilon_lb": float('inf'),
            "note": "Cannot compute bound: log_summation_term is invalid or zero"
        }
    
    log_target_fraction = log_target_product - log_summation_term  # log(fraction) = log(target_product) - log(summation_term)
    
    # Now solve for epsilon in:
    # (e^ε + (m_choose_m_half - 1)/m_choose_m_half * delta) / (e^ε + (m_choose_m_half - 1)) ≤ target_fraction
    #
    # Rearranging:
    # e^ε + (m_choose_m_half - 1)/m_choose_m_half * delta ≤ target_fraction * (e^ε + (m_choose_m_half - 1))
    # e^ε + (m_choose_m_half - 1)/m_choose_m_half * delta ≤ target_fraction * e^ε + target_fraction * (m_choose_m_half - 1)
    # e^ε - target_fraction * e^ε ≤ target_fraction * (m_choose_m_half - 1) - (m_choose_m_half - 1)/m_choose_m_half * delta
    # e^ε * (1 - target_fraction) ≤ target_fraction * (m_choose_m_half - 1) - delta * (m_choose_m_half - 1)/m_choose_m_half 
    #
    # If (1 - target_fraction) > 0:
    #   e^ε ≤ (m_choose_m_half - 1) * (target_fraction - delta) / (1 - target_fraction)
    #   ε ≤ ln((m_choose_m_half - 1) * (target_fraction - delta) / (1 - target_fraction))
    
    # Compute m_term in log-space: log(m_choose_m_half - 1) ≈ log(m_choose_m_half) for large values
    # For very large m_choose_m_half, m_choose_m_half - 1 ≈ m_choose_m_half
    # Work entirely in log-space - never exponentiate m_choose_m_half
    if not np.isfinite(log_m_choose_m_half):
        # Invalid log value
        log_m_term = -np.inf
        m_term = 0.0  # For display only
    elif log_m_choose_m_half > 700:
        # Too large to exponentiate - use approximation: log(m_choose_m_half - 1) ≈ log(m_choose_m_half)
        log_m_term = log_m_choose_m_half
        m_term = float('inf')  # For display only
    else:
        # Small enough to compute m_choose_m_half - 1
        try:
            m_choose_m_half_val = float(np.exp(log_m_choose_m_half))
            m_term = m_choose_m_half_val - 1
            log_m_term = math.log(m_term) if m_term > 0 else -np.inf
        except (OverflowError, RuntimeWarning):
            # Fallback to approximation
            log_m_term = log_m_choose_m_half
            m_term = float('inf')  # For display only
    
    # Check if target_fraction >= 1.0 in log space: log_target_fraction >= 0
    if log_target_fraction >= 0:
        # If target_fraction >= 1, the inequality is trivially satisfied for any epsilon
        # Convert to linear only for display
        target_fraction = float(np.exp(log_target_fraction)) if log_target_fraction < 700 else float('inf')
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": T_choose_T_half,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "target_fraction": target_fraction,
            "epsilon_lb": 0.0,
            "note": "Minimum epsilon (target_fraction >= 1)"
        }
    
    # Compute denominator_term = 1.0 - target_fraction in log space
    # log(1 - target_fraction) = log(1 - exp(log_target_fraction))
    # Since log_target_fraction < 0, we can use log1mexp directly
    # log1mexp(logx) computes log(1 - exp(logx)) for logx <= 0
    log_denominator = log1mexp(log_target_fraction)  # log(1 - exp(log_target_fraction))
    
    # Convert denominator to linear space for the division (it should be small since target_fraction < 1)
    try:
        denominator_term = float(np.exp(log_denominator))
        if not np.isfinite(denominator_term) or denominator_term <= 0:
            # Convert target_fraction for display only
            target_fraction = float(np.exp(log_target_fraction)) if log_target_fraction < 700 and np.isfinite(log_target_fraction) else float('inf')
            return {
                "v_median": v_median,
                "v": v,
                "summation_term": summation_term,
                "T_half": T_half,
                "T_choose_T_half": T_choose_T_half,
                "m_half": m_half,
                "m_choose_m_half": m_choose_m_half,
                "target_fraction": target_fraction,
                "epsilon_lb": float('inf'),
                "note": "Cannot solve: denominator_term <= 0"
            }
    except (OverflowError, RuntimeWarning):
        # Convert target_fraction for display only
        target_fraction = float(np.exp(log_target_fraction)) if log_target_fraction < 700 and np.isfinite(log_target_fraction) else float('inf')
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": T_choose_T_half,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "target_fraction": target_fraction,
            "epsilon_lb": float('inf'),
            "note": "Cannot solve: denominator_term overflow"
        }
    
    # Compute numerator_term in log-space: numerator_term = m_term * target_fraction - delta * m_term / m_choose_m_half
    # For large m_choose_m_half, m_term ≈ m_choose_m_half, so m_term / m_choose_m_half ≈ 1
    # Therefore: second_term = delta * m_term / m_choose_m_half ≈ delta
    # So: numerator_term ≈ m_term * target_fraction - delta
    
    if not np.isfinite(log_m_term) or not np.isfinite(log_target_fraction):
        numerator_term = float('inf')
        log_numerator = -np.inf
    else:
        # Compute log(first_term) = log(m_term * target_fraction) = log_m_term + log_target_fraction
        log_first_term = log_m_term + log_target_fraction
        log_delta = math.log(delta) if delta > 0 else -np.inf
        
        # If first_term >> delta (i.e., log_first_term >> log_delta), then numerator_term ≈ first_term
        # Use threshold of ~7 orders of magnitude (log(1e7) ≈ 16, but we use 7 for safety)
        if log_first_term > log_delta + 7:
            # First term dominates, numerator_term ≈ first_term
            log_numerator = log_first_term
            # Convert to linear only if needed and safe
            if log_numerator > 700:
                numerator_term = float('inf')
            else:
                try:
                    numerator_term = float(np.exp(log_numerator))
                except (OverflowError, RuntimeWarning):
                    numerator_term = float('inf')
        else:
            # Need to compute subtraction: numerator_term = first_term - delta
            # Convert to linear space for subtraction (should be safe since first_term is not huge)
            if log_first_term > 700:
                # Can't exponentiate safely, but first_term is huge, so numerator_term ≈ first_term
                log_numerator = log_first_term
                numerator_term = float('inf')
            else:
                try:
                    first_term = float(np.exp(log_first_term))
                    numerator_term = first_term - delta
                    if numerator_term <= 0 or not np.isfinite(numerator_term):
                        numerator_term = float('inf')
                        log_numerator = -np.inf
                    else:
                        log_numerator = math.log(numerator_term)
                except (OverflowError, RuntimeWarning):
                    numerator_term = float('inf')
                    log_numerator = log_first_term  # Approximation
    
    if numerator_term <= 0 or not np.isfinite(numerator_term) or not np.isfinite(log_numerator):
        # Convert target_fraction for display only
        target_fraction = float(np.exp(log_target_fraction)) if log_target_fraction < 700 and np.isfinite(log_target_fraction) else float('inf')
        return {
            "v_median": v_median,
            "v": v,
            "summation_term": summation_term,
            "T_half": T_half,
            "T_choose_T_half": T_choose_T_half,
            "m_half": m_half,
            "m_choose_m_half": m_choose_m_half,
            "target_fraction": target_fraction,
            "numerator_term": numerator_term,
            "epsilon_lb": float('inf'),
            "note": f"Cannot solve: numerator_term <= 0 or invalid: {numerator_term}"
        }
    
    # Compute epsilon entirely in log space: epsilon = log(numerator_term / denominator_term)
    # = log_numerator - log_denominator
    epsilon_lb = log_numerator - log_denominator
    
    # Convert values to linear space only for display/debugging in return dict
    target_fraction = float(np.exp(log_target_fraction)) if log_target_fraction < 700 and np.isfinite(log_target_fraction) else float('inf')
    e_epsilon_bound = float(np.exp(epsilon_lb)) if epsilon_lb < 700 and np.isfinite(epsilon_lb) else float('inf')
    
    return {
        "v_median": v_median,
        "v": v,
        "summation_term": summation_term,
        "T_half": T_half,
        "T_choose_T_half": T_choose_T_half,
        "m_half": m_half,
        "m_choose_m_half": m_choose_m_half,
        "target_fraction": target_fraction,
        "numerator_term": numerator_term,
        "denominator_term": denominator_term,
        "e_epsilon_bound": e_epsilon_bound,
        "epsilon_lb": epsilon_lb,
        "note": "Median-based epsilon lower bound using summation method (computed entirely in log-space)"
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    m, r, T = 200, 100, 25
    v_list = [50, 52, 49, 51, 50, 50, 48, 53, 50, 47]  # example list of v's
    delta = 1e-6
    ci_delta = 1e-8

    out = compute_c_logM_epsilon_lb(m, r, T, v_list, delta, ci_delta)
    print("c =", out["c"])
    print("logM_bound =", out["logM_bound"])
    print("theta_star =", out["theta_star"])
    print("logZ =", out["logZ"])
    print("epsilon_lb =", out["epsilon_lb"])
