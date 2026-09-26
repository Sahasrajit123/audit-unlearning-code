"""
Direct convex pairwise zCDP auditor.

Setting
-------
The convex unlearning definition compares the two distributions the auditor
actually observes,

    P_f  (non-empty forget set: full-data model -> clip + noise)
    P_r  (empty forget set:     retain-only model -> clip + noise)

so there is no common reference distribution and hence no weak-triangle /
transitivity loss.  The zCDP guarantee therefore yields the *direct* Renyi
curve

    D_gamma(P_f || P_r), D_gamma(P_r || P_f) <= eps_gamma^conv(rho) = rho * gamma,
    for every gamma > 1.                                          (direct curve)

Population lower bound
----------------------
Let A be the event that the pairwise predictor outputs the non-empty-forget
hypothesis, so P_f(A) = TPR and P_r(A) = FPR.  The Renyi change-of-measure
inequality applied to the direct curve gives, for every gamma > 1,

    TPR <= exp( ((gamma-1)/gamma) * rho * gamma ) * FPR^((gamma-1)/gamma)
         = e^{(gamma-1) rho} * FPR^((gamma-1)/gamma),

and rearranging,

    rho >= log(TPR)/(gamma-1) - log(FPR)/gamma.

Swapping P_f, P_r and repeating the argument on A^c gives the analogous
TNR / FNR expression.

Reported bound
--------------
With simultaneous one-sided Clopper-Pearson endpoints TPR^low, TNR^low and
FPR^high, FNR^high the auditor reports

    rho_LB,conv^(p) := sup_{gamma>1} max{ 0,
        log(TPR^low)/(gamma-1)  - log(FPR^high)/gamma,
        log(TNR^low)/(gamma-1)  - log(FNR^high)/gamma }.

The sup has a closed form (see `rho_lb_branch`): writing a = -log(num_low) and
b = -log(den_high), the branch is maximized at gamma* = 1/(1 - sqrt(a/b)) with
value (sqrt(b) - sqrt(a))^2, valid exactly when a < b.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Simultaneous one-sided Clopper-Pearson endpoints
# ---------------------------------------------------------------------------

def clopper_pearson_one_sided(x, n, side, alpha):
    """
    One-sided exact (Clopper-Pearson) confidence bound for p = x/n at level alpha.

    side="lower": returns p_low with Pr[p >= p_low] >= 1 - alpha.
    side="upper": returns p_high with Pr[p <= p_high] >= 1 - alpha.
    """
    if n <= 0:
        return np.nan
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1).")
    x = int(np.clip(x, 0, n))

    if side == "lower":
        lo = 0.0 if x <= 0 else float(beta_dist.ppf(alpha, x, n - x + 1))
        return float(np.clip(lo, 0.0, 1.0))
    if side == "upper":
        hi = 1.0 if x >= n else float(beta_dist.ppf(1.0 - alpha, x + 1, n - x))
        return float(np.clip(hi, 0.0, 1.0))
    raise ValueError("side must be 'lower' or 'upper'.")


def simultaneous_pairwise_endpoints(tp, n_pos, fp, n_neg, ci_delta=0.05):
    """
    Simultaneous one-sided Clopper-Pearson endpoints for the pairwise auditor.

    Only two one-sided events are needed, because the four endpoints are
    pairwise complementary within each binomial:

        positives (n_pos draws from P_f):  TPR^low,  FNR^high = 1 - TPR^low
        negatives (n_neg draws from P_r):  FPR^high, TNR^low  = 1 - FPR^high

    Each event is taken at level ci_delta/2, so by a union bound all four
    endpoints hold simultaneously with probability at least 1 - ci_delta.

    Returns a dict with tpr_low, fnr_high, fpr_high, tnr_low and the per-event
    level used.
    """
    if not (0.0 < ci_delta < 1.0):
        raise ValueError("ci_delta must lie in (0,1).")
    alpha_each = ci_delta / 2.0

    tpr_low = clopper_pearson_one_sided(tp, n_pos, "lower", alpha_each)
    fpr_high = clopper_pearson_one_sided(fp, n_neg, "upper", alpha_each)

    return {
        "tpr_low": tpr_low,
        "fnr_high": 1.0 - tpr_low,
        "fpr_high": fpr_high,
        "tnr_low": 1.0 - fpr_high,
        "alpha_each": alpha_each,
        "ci_delta": ci_delta,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }


# ---------------------------------------------------------------------------
# rho lower bound
# ---------------------------------------------------------------------------

def rho_lb_branch(num_low, den_high):
    """
    sup_{gamma>1} [ log(num_low)/(gamma-1) - log(den_high)/gamma ], clipped at 0.

    With a = -log(num_low) >= 0 and b = -log(den_high) >= 0 the objective is
    h(gamma) = -a/(gamma-1) + b/gamma.  Setting h'(gamma) = 0 gives
    (gamma-1)/gamma = sqrt(a/b), i.e.

        gamma* = 1 / (1 - sqrt(a/b)),   h(gamma*) = (sqrt(b) - sqrt(a))^2,

    which is an interior maximum precisely when a < b (equivalently
    num_low > den_high, i.e. the auditor separates the two distributions).
    Otherwise h < 0 on (1, inf) and the clipped sup is 0.

    Returns (rho, gamma_star); gamma_star is nan when the bound is vacuous.
    """
    num_low = float(num_low)
    den_high = float(den_high)

    # num_low <= 0 => a = +inf (no evidence); den_high >= 1 => b = 0 (no evidence).
    if not np.isfinite(num_low) or not np.isfinite(den_high):
        return 0.0, float("nan")
    if num_low <= 0.0 or den_high >= 1.0:
        return 0.0, float("nan")
    if den_high <= 0.0:
        # b = +inf: the bound diverges. Cannot happen for finite-sample CP endpoints.
        return float("inf"), float("inf")

    a = -math.log(min(num_low, 1.0))   # >= 0
    b = -math.log(den_high)            # > 0
    if a >= b:
        return 0.0, float("nan")

    s = math.sqrt(a / b)
    gamma_star = 1.0 / (1.0 - s)
    rho = (math.sqrt(b) - math.sqrt(a)) ** 2
    return float(rho), float(gamma_star)


def rho_lb_convex_pairwise(tpr_low, fpr_high, tnr_low=None, fnr_high=None,
                           verify=True):
    """
    rho_LB,conv^(p): the direct convex pairwise zCDP lower bound.

        sup_{gamma>1} max{ 0,
            log(TPR^low)/(gamma-1) - log(FPR^high)/gamma,
            log(TNR^low)/(gamma-1) - log(FNR^high)/gamma }

    tnr_low / fnr_high default to the complements 1 - fpr_high and 1 - tpr_low.
    The closed form is exact and is what gets reported; when verify=True it is
    cross-checked against a numerical sup over gamma (they agree to ~1e-9) and a
    warning is emitted if the numerical sup exceeds it non-trivially.

    Returns a dict with rho_lb, the per-branch values, and the maximizing gamma.
    """
    if tnr_low is None:
        tnr_low = 1.0 - float(fpr_high)
    if fnr_high is None:
        fnr_high = 1.0 - float(tpr_low)

    rho_pos, gamma_pos = rho_lb_branch(tpr_low, fpr_high)   # TPR / FPR branch
    rho_neg, gamma_neg = rho_lb_branch(tnr_low, fnr_high)   # TNR / FNR branch

    if rho_pos >= rho_neg:
        rho_lb, gamma_star, branch = rho_pos, gamma_pos, "tpr_fpr"
    else:
        rho_lb, gamma_star, branch = rho_neg, gamma_neg, "tnr_fnr"

    if verify and np.isfinite(rho_lb) and rho_lb > 0.0:
        rho_numeric = _rho_lb_numeric(tpr_low, fpr_high, tnr_low, fnr_high)
        # The closed form is the exact sup, so it can only be matched, not beaten.
        # Report it regardless; flag a genuine mismatch rather than silently
        # inflating the lower bound by optimizer noise.
        if rho_numeric > rho_lb + 1e-6 * max(1.0, rho_lb):
            import sys
            print(f"  Warning: numerical sup ({rho_numeric:.9f}) exceeds closed-form "
                  f"rho_lb ({rho_lb:.9f}); check rho_lb_branch.", file=sys.stderr)

    return {
        "rho_lb": float(max(rho_lb, 0.0)),
        "gamma_star": float(gamma_star),
        "branch": branch,
        "rho_lb_tpr_fpr": float(rho_pos),
        "rho_lb_tnr_fnr": float(rho_neg),
        "tpr_low": float(tpr_low),
        "fpr_high": float(fpr_high),
        "tnr_low": float(tnr_low),
        "fnr_high": float(fnr_high),
    }


def _rho_lb_numeric(tpr_low, fpr_high, tnr_low, fnr_high, gamma_max=1e7):
    """Numerical sup over gamma > 1, used only to cross-check the closed form."""
    def neg_obj(gamma):
        best = 0.0
        for num_low, den_high in ((tpr_low, fpr_high), (tnr_low, fnr_high)):
            if num_low <= 0.0 or den_high <= 0.0 or den_high > 1.0:
                continue
            val = (math.log(min(num_low, 1.0)) / (gamma - 1.0)
                   - math.log(den_high) / gamma)
            best = max(best, val)
        return -best

    res = minimize_scalar(neg_obj, bounds=(1.0 + 1e-9, gamma_max), method="bounded",
                          options={"xatol": 1e-12})
    return float(max(-res.fun, 0.0))


# ---------------------------------------------------------------------------
# zCDP -> approximate DP
# ---------------------------------------------------------------------------

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
    # Clamped at 0: the "tight" branch dips negative for rho below ~1.36e-4 at
    # conv_delta=1e-2, but eps cannot be negative -- a negative upper bound just
    # means eps = 0 suffices. Only deviation from the cum_runs_eps_lab.py version.
    return float(max(0.0, min(float(res.fun), eps_classic)))


# ---------------------------------------------------------------------------
# rho-zCDP upper bound for the noise actually added
# ---------------------------------------------------------------------------

def rho_zcdp_upper_gaussian(sensitivity, sigma):
    """
    rho-zCDP upper bound for the Gaussian mechanism that was actually run.

    The Gaussian mechanism with L2 sensitivity Delta and per-coordinate noise
    std sigma satisfies rho-zCDP with

        rho = Delta^2 / (2 sigma^2)

    (Bun & Steinke 2016, Prop. 1.6). In this audit sigma is calibrated to the
    target (eps, delta) via the exact Gaussian mechanism and the clipped output
    has sensitivity Delta = 2 C_0, so this converts the eps upper bound that was
    passed in into the zCDP guarantee the injected noise actually provides.
    """
    sensitivity = float(sensitivity)
    sigma = float(sigma)
    if sigma <= 0.0 or not np.isfinite(sigma):
        return float("inf")
    return float(sensitivity ** 2 / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Full auditor
# ---------------------------------------------------------------------------

def convex_pairwise_zcdp_audit(
    tp, n_pos, fp, n_neg,
    sensitivity, sigma,
    ci_delta=0.05,
    conv_delta=1e-3,
    conv_method="tight",
):
    """
    Run the direct convex pairwise zCDP auditor end to end.

    Inputs are the raw confusion counts (tp out of n_pos draws from P_f, fp out
    of n_neg draws from P_r), plus the mechanism's L2 sensitivity and the noise
    std that was used.

    Returns a dict with:
      rho_lb_conv       -- rho_LB,conv^(p), the audited zCDP lower bound
      eps_from_rho_lb   -- (eps, conv_delta) estimate implied by rho_lb_conv
      rho_ub_noise      -- rho-zCDP upper bound of the Gaussian noise added
      eps_from_rho_ub   -- same conversion applied to rho_ub_noise, so the
                           lower and upper sides are directly comparable
      plus the CP endpoints, the maximizing gamma, and the branch that won.
    """
    endpoints = simultaneous_pairwise_endpoints(tp, n_pos, fp, n_neg, ci_delta=ci_delta)
    lb = rho_lb_convex_pairwise(
        tpr_low=endpoints["tpr_low"],
        fpr_high=endpoints["fpr_high"],
        tnr_low=endpoints["tnr_low"],
        fnr_high=endpoints["fnr_high"],
    )
    rho_lb = lb["rho_lb"]
    rho_ub = rho_zcdp_upper_gaussian(sensitivity, sigma)

    return {
        "rho_lb_conv": rho_lb,
        "eps_from_rho_lb": eps_estimate_from_rho(rho_lb, conv_delta=conv_delta,
                                                 method=conv_method),
        "rho_ub_noise": rho_ub,
        "eps_from_rho_ub": (eps_estimate_from_rho(rho_ub, conv_delta=conv_delta,
                                                  method=conv_method)
                            if np.isfinite(rho_ub) else float("inf")),
        "gamma_star": lb["gamma_star"],
        "branch": lb["branch"],
        "rho_lb_tpr_fpr": lb["rho_lb_tpr_fpr"],
        "rho_lb_tnr_fnr": lb["rho_lb_tnr_fnr"],
        "tpr_low_1sided": endpoints["tpr_low"],
        "fpr_high_1sided": endpoints["fpr_high"],
        "tnr_low_1sided": endpoints["tnr_low"],
        "fnr_high_1sided": endpoints["fnr_high"],
        "cp_alpha_each": endpoints["alpha_each"],
        "ci_delta": ci_delta,
        "conv_delta": conv_delta,
        "conv_method": conv_method,
        "sensitivity": float(sensitivity),
        "sigma": float(sigma),
    }
