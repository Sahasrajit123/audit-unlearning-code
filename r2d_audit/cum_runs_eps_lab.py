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


def _unlearning_eps_from_ldp(eps_ldp):
    """
    Convert an LDP-parameter lower bound for the audit mechanism M into a
    certified-unlearning epsilon lower bound.

    Lemma: (eps,0)-certified unlearning => M is (2 eps,0)-LDP. So if the audit
    rejects every LDP parameter <= eps_ldp, the certified-unlearning epsilon
    satisfies eps >= eps_ldp/2. Passes None (infeasible at eps=0) through.
    """
    return None if eps_ldp is None else 0.5 * eps_ldp


def _safe_exp(logx: float) -> float:
    """
    exp(logx) for display/debugging only: saturates to inf instead of
    overflowing (the audit's binomials routinely exceed float range).
    """
    if not np.isfinite(logx):
        return 0.0 if logx < 0 else float("inf")
    if logx > 700.0:
        return float("inf")
    return float(np.exp(logx))


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
# ratio(eps) = e^eps / (e^eps + (Z-1))
#
# NOTE: delta = 0 only. The audit lower bound goes through the reduction
# "(eps,0)-certified unlearning => M is (2 eps,0)-LDP", which relies on the
# triangle inequality for indistinguishability; that reduction does not extend
# usefully to delta > 0 (X ~_{eps,delta} Y ~_{eps,delta} Z only gives
# X ~_{2 eps, (1+e^eps) delta} Z). So the pointwise bound is
#   Pr[M(S) = s_hat] <= e^eps / (e^eps + Z - 1),
# with no delta slack, and every wrapper below is pure-eps.
# ----------------------------

def epsilon_lb_from_logM(
    logM: float,
    logZ: float,
    T: int,
    ci_delta: float,
    eps_hi_init: float = 50.0,
    tol: float = 1e-10,
    max_iter: int = 200
):
    """
    Returns the largest epsilon >= 0 such that:
      M(c) * ratio(eps)^T <= ci_delta,   ratio(eps) = e^eps / (e^eps + Z - 1)
    using log-space with logM provided.

    This is the (eps, 0) case; there is no delta parameter (see the module note
    above). If infeasible even at eps=0, returns None.

    The value returned is the LDP parameter of the audit mechanism M. The
    wrappers halve it via _unlearning_eps_from_ldp before reporting
    "epsilon_lb", so that every epsilon_lb in this module is a bound on the
    certified-unlearning epsilon.
    """
    if not (0.0 < ci_delta < 1.0):
        raise ValueError("ci_delta must lie in (0,1).")
    if T <= 0:
        raise ValueError("T must be positive.")

    log_ci = math.log(ci_delta)

    # log(Z-1) = logZ + log(1 - 1/Z), computed robustly (log(1/Z) = -logZ).
    log_b = logZ + log1mexp(-logZ)

    def log_ratio(eps: float) -> float:
        # log( exp(eps) ) - log( exp(eps)+b ) = eps - log(e^eps + Z - 1)
        return eps - logaddexp(eps, log_b)

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
        ci_delta=ci_delta
    )

    return {
        "c": c,
        "logM_bound": logM_bound,
        "theta_star": theta_star,
        "logZ": logZ,
        "epsilon_lb": _unlearning_eps_from_ldp(eps_star),
        "note": "epsilon_lb is half the largest LDP epsilon satisfying the inequality using "
                "the Chernoff upper bound on M(c), i.e. a certified-unlearning epsilon."
    }

def compute_c_logM_lower_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
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
        ci_delta=ci_delta
    )

    return {
        "c": c,
        "logM_lower_bound": logM_bound,
        "theta_star": theta_star,
        "logZ": logZ,
        "epsilon_lb": _unlearning_eps_from_ldp(eps_star),
        "note": "Lower-tail: M is over vectors with average log-score <= c. "
                "epsilon_lb is a certified-unlearning epsilon (half the LDP bound)."
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
      M(c) * (e^eps/(e^eps + Z-1))^T <= ci_delta,
    where Z = sum_v f(v) = C(m, floor(m/2)). (delta = 0 only.)
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
        ci_delta=ci_delta,
    )

    return {
        "logc": float(logc),
        "logM_bound": float(logM_bound),
        "theta_star": float(theta_star),
        "logZ": float(logZ),
        "epsilon_lb": _unlearning_eps_from_ldp(epsilon_lb),
        "note": "g-test: constraint prod g(v_i) <= c, threshold c derived from v_list. "
                "epsilon_lb is a certified-unlearning epsilon (half the LDP bound)."
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
    ci_delta: float,
    direction: str = "ge",     # "ge" for avg >= a, "le" for avg <= a
    theta_max: float = 50.0
):
    """
    Mean-test epsilon lower bound (delta = 0 only), i.e. the "mean" inequality of
    the audit lemma:
      Pr[(1/L) sum_l V^(l) >= v] <= inf_{lambda>=0} exp(L log(sum_u e^{lambda u} pi_eps(u)) - lambda L v),
    where pi_eps(u) = f(u) * e^eps/(e^eps + Z - 1) with Z = C(m, floor(m/2)).
    The eps-dependent factor is lambda-free, so the inf factors as
      (e^eps/(e^eps+Z-1))^L * inf_lambda exp(L log(sum_u e^{lambda u} f(u)) - lambda L v),
    which is exactly logM_bound_avg_v_* plus epsilon_lb_from_logM below.

    - threshold a taken from v_list: a = average(v_list)
    - compute Chernoff upper bound on log M for the avg-v test
    - compute the LDP epsilon via epsilon_lb_from_logM, then halve it

    epsilon_lb is a lower bound on the certified-unlearning epsilon: the solver
    bounds the LDP parameter of M, and the lemma's factor of 2 is applied here
    (see _unlearning_eps_from_ldp). Returns None when the observation is
    consistent with eps = 0.
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
        ci_delta=ci_delta
    )

    return {
        "a": a,
        "direction": direction,
        "logM_bound": logM_bound,
        "theta_star": theta_star,
        "logZ": float(logZ),
        "epsilon_lb_ldp": epsilon_lb,
        "epsilon_lb": _unlearning_eps_from_ldp(epsilon_lb),
        "note": "avg-v test with f-weight (delta=0); threshold a is mean(v_list). "
                "epsilon_lb is the certified-unlearning epsilon = epsilon_lb_ldp/2."
    }


# ----------------------------
# Median-based epsilon lower bound computation
# ----------------------------

def compute_summation(m: int, r: int, v: float) -> tuple[float, float]:
    """
    Compute the summation:
        Σ
    (α₁,α₂)∈[0,r/2]
      α₁+α₂≥v
    C(m-r, ⌈(m-r)/2⌉ - (α₁-α₂)) * C(r/2, α₁) * C(r/2, α₂)

    i.e. the unnormalized upper tail Σ_{u≥v} f(u) appearing in P_eps(v) of the
    audit lemma (P_eps(v) = this sum * e^eps/(e^eps + C(m,⌊m/2⌋) - 1)).

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
    
    Note: α₁ and α₂ are treated as integers in [0, r/2], and the constraint
    α₁ + α₂ ≥ v only depends on ⌈v⌉, so v may be a non-integer median.

    Implemented as the upper-tail sum of the f(u) table (log_f_values +
    log_g_from_logf), which is the same quantity: f(u) is the u-th term of this
    sum and g(v) = Σ_{u≥v} f(u). The lemma writes the centre binomial as
    C(m-r, ⌈(m-r)/2⌉ - (α₁-α₂)) while log_f_values uses ⌊(m-r)/2⌋; the two agree
    term-by-term after summing over the (α₁,α₂) ↔ (α₂,α₁) pairs, since that swap
    flips the sign of α₁-α₂ and C(n, ⌈n/2⌉+d) = C(n, ⌊n/2⌋-d).
    """
    logf = log_f_values(m, r)          # validates r even, m >= r
    logg = log_g_from_logf(logf)       # logg[u] = log Σ_{s>=u} f(s)

    v_ceil = math.ceil(v)
    if v_ceil > r:
        return 0.0, -np.inf            # empty tail
    log_total = float(logg[max(0, v_ceil)])

    if not np.isfinite(log_total):
        return 0.0, -np.inf

    # Check if the result is too large to convert to linear space
    # Python's float max is around 1.7e308, so log_max ≈ 709
    # If log_total > 700, we'll get overflow, so return inf for linear value
    if log_total > 700:
        return float('inf'), log_total  # Return inf for display, but keep log value
    
    # Convert back to linear space only if safe
    try:
        total_sum = float(np.exp(log_total))
        if not np.isfinite(total_sum):
            return float('inf'), log_total
        return total_sum, log_total
    except (OverflowError, RuntimeWarning):
        # If still overflow, return inf for linear but keep log value
        return float('inf'), log_total


def compute_median_v_test_epsilon_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float = 0.05
):
    """
    Median-test epsilon lower bound (delta = 0 only).

    Audit lemma, median inequality: if M is (eps, 0)-LDP and the per-run overlap
    scores V^(1),...,V^(L) are i.i.d., then for every v

        Pr[Median(V^(1..L)) >= v] <= C(L, ceil(L/2)) * P_eps(v)^ceil(L/2),
          P_eps(v)  = sum_{u >= v} pi_eps(u),
          pi_eps(u) = f(u) * e^eps / (e^eps + M' - 1),
          M'        = C(m, floor(m/2)),

    where f(u) is the combinatorial count of log_f_values (equivalently, the
    u-th term of compute_summation). There is no (eps, delta) version of this
    bound: it rests on the reduction "(eps,0)-certified unlearning => M is
    (2 eps,0)-LDP", whose triangle-inequality step degrades to
    (2 eps, (1+e^eps) delta) as soon as delta > 0 (see the module note above
    epsilon_lb_from_logM). Hence there is no delta argument.

    The solver finds eps_ldp, the largest eps for which the right-hand side is
    still <= ci_delta, i.e. every LDP parameter <= eps_ldp is rejected at
    confidence 1 - ci_delta. Since Sigma(v) := sum_{u >= v} f(u) carries no eps,
    with L_half := ceil(L/2),

        C(L, L_half) * (Sigma(v) * e^eps/(e^eps + M' - 1))^L_half <= ci_delta
      <=> e^eps/(e^eps + M' - 1) <= q,
            q := (ci_delta / C(L, L_half))^(1/L_half) / Sigma(v)
      <=> eps <= log( q * (M' - 1) / (1 - q) )                     [q < 1]

    which is evaluated in log-space throughout: q is typically astronomically
    small, so -log(1-q) ~ 0 and eps_ldp ~ log q + log M'.

    The reported epsilon_lb is eps_ldp/2, the certified-unlearning epsilon: the
    lemma above gives "(eps,0)-certified unlearning => M is (2 eps,0)-LDP", so
    rejecting every LDP parameter <= eps_ldp rejects every unlearning epsilon
    <= eps_ldp/2. The raw LDP value is also returned as "epsilon_lb_ldp".

    Args:
        m: Total number of batches (forget-set size)
        r: Parameter r = 2*k; must be even with m >= r
        T: Total number of runs (L)
        v_list: List of v values (one per run), length must equal T
        ci_delta: Confidence slack in (0,1) (default: 0.05)

    Returns:
        Dict with epsilon_lb and computation details, where epsilon_lb is
          - None if the observation is consistent with eps = 0 (the closed form
            above is negative, so no positive eps is rejected);
          - inf if the observed median is impossible under any finite eps
            (Sigma(v) = 0);
          - otherwise eps_ldp/2, a lower bound on the certified-unlearning
            epsilon (the un-halved LDP value is in "epsilon_lb_ldp").
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list of length T={T}, got length {len(v_list)}.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (0.0 < ci_delta < 1.0):
        raise ValueError("ci_delta must lie in (0,1).")

    # Threshold = the observed median, which is the event we actually saw. Only
    # ceil(median) matters for the tail sum, since u is an integer and
    # u >= median <=> u >= ceil(median) (relevant when L is even and the median
    # falls between two order statistics).
    v_median = float(np.median(v_list))
    v = math.ceil(v_median)
    if not (0 <= v <= r):
        raise ValueError(f"v={v} (ceil of median of v_list) outside [0,{r}].")

    # Sigma(v) = sum_{u>=v} f(u): the eps-free part of P_eps(v).
    summation_term, log_summation_term = compute_summation(m, r, v)

    # L_half = ceil(L/2): Median >= v forces at least this many runs to have V >= v.
    T_half = math.ceil(T / 2)
    log_T_choose_T_half = log_binom(T, T_half)

    # M' = C(m, floor(m/2)); only ever used in log-space (it overflows for m >~ 1000).
    m_half = math.floor(m / 2)
    log_m_choose_m_half = log_Z_closed_form(m)
    # log(M' - 1) = log M' + log(1 - 1/M'), stable for huge M'.
    log_m_term = (
        log_m_choose_m_half + log1mexp(-log_m_choose_m_half)
        if log_m_choose_m_half > 0.0 else -np.inf
    )

    details = {
        "v_median": v_median,
        "v": v,
        "summation_term": summation_term,
        "log_summation_term": log_summation_term,
        "T_half": T_half,
        "T_choose_T_half": _safe_exp(log_T_choose_T_half),
        "m_half": m_half,
        "m_choose_m_half": _safe_exp(log_m_choose_m_half),
        "log_m_choose_m_half": log_m_choose_m_half,
    }

    # Sigma(v) = 0: the observed median cannot occur under any (eps,0)-LDP M.
    if not np.isfinite(log_summation_term):
        return {
            **details,
            "target_fraction": float("inf"),
            "epsilon_lb_ldp": float("inf"),
            "epsilon_lb": float("inf"),
            "note": "Sigma(v)=0: observation impossible under any finite epsilon.",
        }

    # q = (ci_delta / C(L,L_half))^(1/L_half) / Sigma(v), in log-space.
    log_target = math.log(ci_delta) - log_T_choose_T_half   # < 0 since ci_delta < 1 <= C(L,L_half)
    log_target_product = log_target / T_half
    log_q = log_target_product - log_summation_term
    details["log_target"] = log_target
    details["log_target_product"] = log_target_product
    details["log_target_fraction"] = log_q
    details["target_fraction"] = _safe_exp(log_q)

    # e^eps/(e^eps+M'-1) < 1 for every eps, so q >= 1 makes the test vacuous.
    if log_q >= 0.0:
        return {
            **details,
            "epsilon_lb_ldp": float("inf"),
            "epsilon_lb": float("inf"),
            "note": "target_fraction >= 1: bound holds for every epsilon.",
        }

    if not np.isfinite(log_m_term):
        return {
            **details,
            "epsilon_lb_ldp": None,
            "epsilon_lb": None,
            "note": "M' <= 1: no epsilon can be certified.",
        }

    # log(1 - q); q is usually small enough that this is 0 to machine precision.
    log_one_minus_q = 0.0 if log_q < -50.0 else math.log1p(-math.exp(log_q))

    # Largest LDP parameter of M that the observation rejects.
    epsilon_lb_ldp = log_q + log_m_term - log_one_minus_q

    if epsilon_lb_ldp < 0.0:
        return {
            **details,
            "epsilon_lb_ldp_raw": epsilon_lb_ldp,
            "epsilon_lb_ldp": None,
            "epsilon_lb": None,
            "note": "Infeasible at epsilon=0: observed median is consistent with a "
                    "perfectly private mechanism, so no positive epsilon is rejected.",
        }

    return {
        **details,
        "epsilon_lb_ldp": epsilon_lb_ldp,
        "epsilon_lb": _unlearning_eps_from_ldp(epsilon_lb_ldp),
        "note": "Median-v test (delta=0). epsilon_lb is the certified-unlearning "
                "epsilon = epsilon_lb_ldp/2.",
    }


# ----------------------------
# zCDP-based rho lower bound computation
#
# Uses the pointwise bound
#   pi_eps(u) = (1/M') * sum_{a1+a2=u, a1,a2 in [0,r/2]}
#                   C(m-r, ceil((m-r)/2) - (a1-a2)) * C(r/2,a1) * C(r/2,a2)
# (M' = C(m, floor(m/2)) normalizes pi_eps to a distribution over u in [0,r]; note
# the ceil((m-r)/2) center, vs. the floor((m-r)/2) center used for the unnormalized
# f(v) in log_f_values above), together with the RDP order-gamma epsilon implied by
# rho-zCDP: eps_gamma(rho) = 4*rho*gamma. rho_lb is the
# largest rho consistent with the observed audit statistic (mean or median of v_list)
# at confidence level 1-ci_delta.
# ----------------------------

def log_pi_values(m: int, r: int) -> np.ndarray:
    """
    Compute log pi_eps(u) for u=0..r, normalized so that sum_u pi_eps(u) = 1.
    """
    if r % 2 != 0:
        raise ValueError("Assumes r is even so r/2 is integer.")
    if m < r:
        raise ValueError("Need m >= r so n=m-r >= 0.")

    n = m - r
    half_r = r // 2
    center = math.ceil(n / 2)  # ceil(n/2), per the zCDP pointwise-bound definition

    logf = np.full(r + 1, -np.inf, dtype=float)
    for u in range(r + 1):
        lo = max(0, u - half_r)
        hi = min(half_r, u)
        terms = []
        for a1 in range(lo, hi + 1):
            a2 = u - a1
            d = a1 - a2          # = 2*a1 - u
            k_idx = center - d   # ceil(n/2) - (a1-a2)

            lb = log_binom(n, k_idx)
            if not np.isfinite(lb):
                continue

            terms.append(lb + log_binom(half_r, a1) + log_binom(half_r, a2))

        if terms:
            logf[u] = logsumexp(np.array(terms, dtype=float))

    logZ = log_Z_closed_form(m)  # M' = C(m, floor(m/2)) = C(m, ceil(m/2))
    return logf - logZ


def eps_gamma_zcdp(rho: float, gamma: float) -> float:
    """RDP order-gamma epsilon implied by rho-zCDP: 4*rho*gamma."""
    if gamma <= 1:
        raise ValueError("gamma must be > 1.")
    return 4.0 * rho * gamma


def _min_log_rhs_zcdp(rho: float, L: int, logM_val: float, gamma_max: float = 1e4):
    """
    inf_{gamma>1} (gamma-1)/gamma * (L*4*rho*gamma + logM_val), in log-space.

    logM_val is the log of the gamma-independent tail-mass bound: the average-test
    Chernoff bound log M_mean(v), or log(C(L,ceil(L/2)) * P_eps(v)^ceil(L/2)) for the
    median test.
    """
    def obj(gamma: float) -> float:
        ratio = (gamma - 1.0) / gamma
        eps_g = eps_gamma_zcdp(rho, gamma)
        return ratio * (L * eps_g + logM_val)

    res = minimize_scalar(obj, bounds=(1.0 + 1e-9, gamma_max), method="bounded")
    return float(res.fun), float(res.x)


def _rho_lb_from_logM_zcdp(
    logM_val: float,
    L: int,
    ci_delta: float,
    gamma_max: float = 1e4,
    rho_hi_init: float = 10.0,
    tol: float = 1e-10,
    max_iter: int = 200,
):
    """
    Largest rho >= 0 such that inf_gamma (gamma-1)/gamma*(L*eps_gamma(rho)+logM_val)
    <= log(ci_delta). Returns None if infeasible even at rho=0 (the observation is
    already too improbable for a perfectly private, rho=0 mechanism).
    """
    if ci_delta <= 0:
        raise ValueError("ci_delta must be positive.")
    if L <= 0:
        raise ValueError("L must be positive.")

    log_ci = math.log(ci_delta)

    val0, _ = _min_log_rhs_zcdp(0.0, L, logM_val, gamma_max)
    if val0 > log_ci:
        return None

    # Find an upper bracket where it fails, since the bound is non-decreasing in rho.
    hi = rho_hi_init
    val_hi, _ = _min_log_rhs_zcdp(hi, L, logM_val, gamma_max)
    while val_hi <= log_ci:
        hi *= 2.0
        if hi > 1e6:  # extremely conservative cap
            return hi
        val_hi, _ = _min_log_rhs_zcdp(hi, L, logM_val, gamma_max)

    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val_mid, _ = _min_log_rhs_zcdp(mid, L, logM_val, gamma_max)
        if val_mid <= log_ci:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, lo):
            break
    return lo


def compute_avg_v_test_rho_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    gamma_max: float = 1e4,
    theta_max: float = 50.0,
    conv_delta: float = 1e-3,
):
    """
    zCDP analogue of compute_avg_v_test_epsilon_lb (mean/"ge" test only, matching the
    audit's mean inequality): threshold v = mean(v_list), Chernoff bound on
    Pr[avg >= v] under the normalized pointwise distribution pi_eps, combined with the
    RDP->zCDP conversion eps_gamma(rho) minimized over gamma>1.

    Also reports "eps_estimate", the (eps, conv_delta) conversion of rho_lb via
    eps_estimate_from_rho. That is NOT a lower bound on eps -- see the
    (eps, delta) estimates section.
    """
    logpi = log_pi_values(m, r)
    v = a_from_v_list(v_list, T=T)

    logM_bound, theta_star = logM_bound_avg_v_ge_a(logpi, T=T, a=v, theta_max=theta_max)
    rho_lb = _rho_lb_from_logM_zcdp(logM_bound, L=T, ci_delta=ci_delta, gamma_max=gamma_max)

    return {
        "v": v,
        "logM_bound": logM_bound,
        "theta_star": theta_star,
        "rho_lb": rho_lb,
        "eps_estimate": eps_estimate_from_rho(rho_lb, conv_delta=conv_delta),
        "conv_delta": conv_delta,
        "note": "zCDP avg-v test; rho_lb is the largest rho consistent with the observed mean "
                "at this confidence level. eps_estimate is its (eps, conv_delta) conversion, "
                "not a lower bound on eps.",
    }


def compute_median_v_test_rho_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    gamma_max: float = 1e4,
    conv_delta: float = 1e-3,
):
    """
    zCDP analogue of compute_median_v_test_epsilon_lb: threshold v = ceil(median(v_list)),
    tail-mass bound C(T,ceil(T/2)) * P_eps(v)^ceil(T/2) where P_eps(v) = sum_{u=v}^r pi_eps(u),
    combined with the RDP->zCDP conversion eps_gamma(rho) minimized over gamma>1.

    Also reports "eps_estimate", the (eps, conv_delta) conversion of rho_lb via
    eps_estimate_from_rho. That is NOT a lower bound on eps -- see the
    (eps, delta) estimates section.
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list of length T={T}, got length {len(v_list)}.")

    logpi = log_pi_values(m, r)
    logP = log_g_from_logf(logpi)  # upper tail sums P_eps(v) = sum_{u=v}^r pi_eps(u)

    v_median = float(np.median(v_list))
    v = math.ceil(v_median)
    if not (0 <= v <= r):
        raise ValueError(f"v={v} outside [0,{r}].")

    log_P_v = float(logP[v])
    T_half = math.ceil(T / 2)
    logM_median = log_binom(T, T_half) + T_half * log_P_v

    rho_lb = _rho_lb_from_logM_zcdp(logM_median, L=T, ci_delta=ci_delta, gamma_max=gamma_max)

    return {
        "v_median": v_median,
        "v": v,
        "log_P_v": log_P_v,
        "T_half": T_half,
        "logM_median": logM_median,
        "rho_lb": rho_lb,
        "eps_estimate": eps_estimate_from_rho(rho_lb, conv_delta=conv_delta),
        "conv_delta": conv_delta,
        "note": "zCDP median-v test; rho_lb is the largest rho consistent with the observed "
                "median at this confidence level. eps_estimate is its (eps, conv_delta) "
                "conversion, not a lower bound on eps.",
    }


# ----------------------------
# (eps, delta) estimates from the rho-zCDP audit
#
# These are NOT lower bounds. The audit certifies a lower bound on rho; the
# conversion below runs in the forward direction (a rho guarantee implies an
# (eps, delta) guarantee), so applying it to an audited lower bound yields a
# number that is comparable with epsilon_lb on the same axis but is not itself a
# certified lower bound on eps at that delta. Every name and dict key in this
# section says "estimate" for that reason, and conv_delta travels alongside the
# value so a number can never be read without its delta.
#
# The zCDP conversion is lossy by nature: rho-zCDP is the family
# (alpha, rho*alpha)-RDP for all alpha > 1, and a conversion must pick an alpha.
# See eps_estimate_from_rho.
# ----------------------------

def eps_estimate_from_rho(rho, conv_delta: float = 1e-3, method: str = "tight"):
    """
    (eps, conv_delta) estimate implied by rho-zCDP. NOT a lower bound on eps.

    rho-zCDP means (alpha, rho*alpha)-RDP for every alpha > 1, so any RDP -> DP
    conversion applies and the result depends on which alpha is chosen:

      method="tight" (default) -- the Balle, Barthe, Gaboardi, Hsu & Sato
        conversion (AISTATS 2020, Thm 21; the form used by Opacus and
        dp_accounting), minimized over the order:
          eps = min_{alpha>1} [ rho*alpha + log1p(-1/alpha)
                                - (log conv_delta + log alpha)/(alpha - 1) ].
      method="classic" -- Bun & Steinke (2016) Prop 1.3,
          eps = rho + 2 sqrt(rho log(1/conv_delta)),
        which is exactly the minimum of the simpler conversion
        inf_{alpha>1}[rho*alpha + log(1/delta)/(alpha-1)] (attained at
        alpha - 1 = sqrt(log(1/delta)/rho)). Kept because it is what the audit
        drivers reported historically.

    Both are valid conversions, so the returned value is min(tight, classic):
    satisfying DP at the smaller eps is the stronger statement, and if both
    conversions hold then so does the minimum. Passes None through.
    """
    if rho is None:
        return None
    if not (0.0 < conv_delta < 1.0):
        raise ValueError("conv_delta must lie in (0,1).")
    rho = float(rho)
    if rho <= 0.0:
        return 0.0
    if method not in ("tight", "classic"):
        raise ValueError("method must be 'tight' or 'classic'.")

    log_inv_delta = -math.log(conv_delta)
    eps_classic = rho + 2.0 * math.sqrt(rho * log_inv_delta)
    if method == "classic":
        return float(eps_classic)

    def obj(alpha: float) -> float:
        return (rho * alpha + math.log1p(-1.0 / alpha)
                - (math.log(conv_delta) + math.log(alpha)) / (alpha - 1.0))

    res = minimize_scalar(obj, bounds=(1.0 + 1e-9, 1e6), method="bounded")
    return float(min(float(res.fun), eps_classic))


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    m, r, T = 200, 100, 10
    v_list = [50, 52, 49, 51, 50, 50, 48, 53, 50, 47]  # example list of v's
    ci_delta = 1e-8

    out = compute_c_logM_epsilon_lb(m, r, T, v_list, ci_delta)
    print("c =", out["c"])
    print("logM_bound =", out["logM_bound"])
    print("theta_star =", out["theta_star"])
    print("logZ =", out["logZ"])
    print("epsilon_lb =", out["epsilon_lb"])
