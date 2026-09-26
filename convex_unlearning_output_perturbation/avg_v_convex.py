"""
Convex-setting avg-v audit tests: epsilon and rho lower bounds with no
transitivity loss.

Both wrappers reuse the combinatorial machinery in this project's
cum_runs_eps_lab.py (log_f_values, log_pi_values, logM_bound_avg_v_ge_a,
epsilon_lb_from_logM, ...) but drop the two places where the *general*
unlearning audit pays for comparing two unlearned outputs through a common
reference:

1. epsilon. The general audit bounds the LDP parameter of the audit mechanism M
   and then halves it, because (eps,0)-certified unlearning only implies M is
   (2 eps,0)-LDP. In the convex setting def:zcdp_convex_unlearning compares the
   two distributions the auditor actually observes, so the LDP parameter *is*
   the epsilon and the factor of 2 is not incurred.
   `compute_avg_v_test_epsilon_lb_convex` therefore returns
   epsilon_lb_from_logM un-halved.

2. rho. The general audit converts rho-zCDP to an order-gamma RDP epsilon via
   the weak-triangle bound of lemma:zcdp_to_local_rdp,
       eps_gamma(rho) = 2 rho gamma (1 + sqrt(gamma/(gamma-1))).
   Here the direct curve of eq:convex_direct_rdp_curve applies instead,
       eps_gamma^conv(rho) = rho gamma,
   with no transitivity loss. `compute_avg_v_test_rho_lb_convex` uses that.

Everything is local to this project: there is no dependency on any
cum_runs_eps_lab.py outside this directory.

The rho bound has a closed form (see `_rho_lb_from_logM_conv`): with
Q = -logM_bound and C = -log(ci_delta),
    rho_lb = (sqrt(Q) - sqrt(C))^2 / L,   valid when Q > C,
which is used to cross-check the bisection.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from cum_runs_eps_lab import (
    a_from_v_list,
    epsilon_lb_from_logM,
    logM_bound_avg_v_ge_a,
    logM_bound_avg_v_le_a,
    log_f_values,
    log_pi_values,
    log_Z_closed_form,
)
from zcdp_pairwise import eps_estimate_from_rho


# ---------------------------------------------------------------------------
# epsilon: LDP parameter, un-halved
# ---------------------------------------------------------------------------

def compute_avg_v_test_epsilon_lb_convex(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    direction: str = "ge",
    theta_max: float = 50.0,
):
    """
    Convex-setting (eps, 0) lower bound from the avg-v test.

    Identical to the general-audit avg-v epsilon test except that the returned
    epsilon_lb is the LDP parameter itself, NOT half of it: in the convex
    setting the auditor compares P_f and P_r directly, so the
    (eps,0)-unlearning => (2 eps,0)-LDP lemma is not invoked.

    Returns None for epsilon_lb when the observation is consistent with eps = 0.
    """
    logf = log_f_values(m, r)
    a = a_from_v_list(v_list, T=T)

    if direction == "ge":
        logM_bound, theta_star = logM_bound_avg_v_ge_a(
            logf, T=T, a=a, theta_max=theta_max)
    elif direction == "le":
        logM_bound, theta_star = logM_bound_avg_v_le_a(
            logf, T=T, a=a, theta_max=theta_max)
    else:
        raise ValueError("direction must be 'ge' or 'le'.")

    logZ = log_Z_closed_form(m)
    # delta=0: the avg-v epsilon test is the (eps, 0) case.
    eps_ldp = epsilon_lb_from_logM(
        logM=logM_bound, logZ=logZ, T=T, delta=0.0, ci_delta=ci_delta)

    return {
        "a": a,
        "direction": direction,
        "logM_bound": float(logM_bound),
        "theta_star": float(theta_star),
        "logZ": float(logZ),
        # In the convex setting these coincide -- no factor of 2.
        "epsilon_lb_ldp": eps_ldp,
        "epsilon_lb": eps_ldp,
        "note": "convex avg-v test (delta=0): epsilon_lb is the LDP parameter, "
                "NOT halved (no (eps,0)->(2eps,0)-LDP transitivity loss).",
    }


# ---------------------------------------------------------------------------
# rho: direct curve eps_gamma = rho * gamma
# ---------------------------------------------------------------------------

def eps_gamma_conv(rho: float, gamma: float) -> float:
    """
    Order-gamma RDP epsilon implied by rho-zCDP in the convex setting:
    eps_gamma^conv(rho) = rho * gamma (eq:convex_direct_rdp_curve).

    Contrast with the lab's eps_gamma_zcdp, which carries the weak-triangle
    factor 2*(1 + sqrt(gamma/(gamma-1))) needed only when two unlearned outputs
    are compared through a common reference.
    """
    if gamma <= 1:
        raise ValueError("gamma must be > 1.")
    return float(rho * gamma)


def _min_log_rhs_conv(rho: float, L: int, logM_val: float, gamma_max: float = 1e4):
    """
    inf_{gamma>1} (gamma-1)/gamma * (L*eps_gamma^conv(rho) + logM_val)
      = inf_{gamma>1} [ L*rho*(gamma-1) - Q*(1 - 1/gamma) ],   Q = -logM_val.

    Stationary point: gamma* = sqrt(Q/(L rho)), giving the value
    -(sqrt(Q) - sqrt(L rho))^2 when gamma* > 1. Computed numerically here; the
    closed form is used in _rho_lb_from_logM_conv.
    """
    def obj(gamma: float) -> float:
        ratio = (gamma - 1.0) / gamma
        return ratio * (L * eps_gamma_conv(rho, gamma) + logM_val)

    res = minimize_scalar(obj, bounds=(1.0 + 1e-9, gamma_max), method="bounded")
    return float(res.fun), float(res.x)


def _rho_lb_from_logM_conv(
    logM_val: float,
    L: int,
    ci_delta: float,
    gamma_max: float = 1e4,
    rho_hi_init: float = 10.0,
    tol: float = 1e-12,
    max_iter: int = 300,
    verify: bool = True,
):
    """
    Largest rho >= 0 with
      inf_{gamma>1} (gamma-1)/gamma * (L*rho*gamma + logM_val) <= log(ci_delta).

    Closed form: with Q = -logM_val and C = -log(ci_delta), the inf equals
    -(sqrt(Q) - sqrt(L rho))^2, so the constraint is
    (sqrt(Q) - sqrt(L rho))^2 >= C, i.e.

        rho <= (sqrt(Q) - sqrt(C))^2 / L,   valid when Q > C.

    Returns (rho_lb, gamma_star). rho_lb is None when even rho = 0 is
    inconsistent with the observation at this confidence level (Q <= C).
    """
    if not (0.0 < ci_delta < 1.0):
        raise ValueError("ci_delta must lie in (0,1).")
    if L <= 0:
        raise ValueError("L must be positive.")

    Q = -float(logM_val)
    C = -math.log(ci_delta)

    if not np.isfinite(Q) or Q <= C:
        return None, float("nan")

    rho_closed = (math.sqrt(Q) - math.sqrt(C)) ** 2 / L
    gamma_star = math.sqrt(Q / (L * rho_closed)) if rho_closed > 0 else float("nan")

    if not verify:
        return float(rho_closed), float(gamma_star)

    # Bisection on the numerically-minimized bound, mirroring the lab's solver.
    log_ci = math.log(ci_delta)
    val0, _ = _min_log_rhs_conv(0.0, L, logM_val, gamma_max)
    if val0 > log_ci:
        return None, float("nan")

    hi = rho_hi_init
    val_hi, _ = _min_log_rhs_conv(hi, L, logM_val, gamma_max)
    while val_hi <= log_ci:
        hi *= 2.0
        if hi > 1e6:
            return float(hi), float(gamma_star)
        val_hi, _ = _min_log_rhs_conv(hi, L, logM_val, gamma_max)

    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val_mid, _ = _min_log_rhs_conv(mid, L, logM_val, gamma_max)
        if val_mid <= log_ci:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, lo):
            break

    # The closed form is exact; flag a real disagreement rather than hiding it.
    if abs(lo - rho_closed) > 1e-4 * max(1.0, rho_closed):
        import sys
        print(f"  Warning: convex rho bisection ({lo:.9f}) disagrees with closed "
              f"form ({rho_closed:.9f}).", file=sys.stderr)
    return float(rho_closed), float(gamma_star)


def compute_avg_v_test_rho_lb_convex(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    gamma_max: float = 1e4,
    theta_max: float = 50.0,
    conv_delta: float = 1e-3,
    conv_method: str = "tight",
):
    """
    Convex-setting rho-zCDP lower bound from the avg-v ("ge") test.

    Same as the general-audit avg-v rho test, but the zCDP -> RDP step uses the
    direct curve eps_gamma^conv(rho) = rho*gamma instead of the weak-triangle
    bound, so no transitivity loss is paid.

    Also reports eps_estimate, the (eps, conv_delta) conversion of rho_lb. That
    is an implied-eps estimate, NOT a lower bound on eps.
    """
    logpi = log_pi_values(m, r)
    v = a_from_v_list(v_list, T=T)

    logM_bound, theta_star = logM_bound_avg_v_ge_a(
        logpi, T=T, a=v, theta_max=theta_max)
    rho_lb, gamma_star = _rho_lb_from_logM_conv(
        logM_bound, L=T, ci_delta=ci_delta, gamma_max=gamma_max)

    return {
        "v": v,
        "logM_bound": float(logM_bound),
        "theta_star": float(theta_star),
        "rho_lb": rho_lb,
        "gamma_star": gamma_star,
        # Local clamped conversion (eps >= 0), not lab.eps_estimate_from_rho, so
        # both audit paths report the same eps for a given rho.
        "eps_estimate": eps_estimate_from_rho(
            rho_lb, conv_delta=conv_delta, method=conv_method),
        "conv_delta": conv_delta,
        "conv_method": conv_method,
        "note": "convex zCDP avg-v test using eps_gamma = rho*gamma (direct curve, "
                "no weak-triangle loss). eps_estimate is an implied-eps estimate, "
                "not a lower bound on eps.",
    }
