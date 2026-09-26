"""zCDP (``rho``) and Gaussian-DP (``mu``) lower bounds for the unlearning audit.

Companion to :mod:`audit_tofu.epsilon_bounds`, which audits the pure-``(eps, 0)``
certified-unlearning parameter. The same observed overlap scores ``V^(1..L)`` also
bound the *other* two privacy parametrisations of the same unlearning guarantee:

    rho  -- zCDP-certified unlearning  (Renyi route, lemma "zcdp_to_local_rdp")
    mu   -- GDP-certified unlearning   (f-DP route, lemma "gdp_overlap_audit")

Both live here rather than in :mod:`audit_tofu.cum_runs_eps_lab`, because that file
is vendored **byte-identically** from the released implementation of the paper's
reference code (the Shakespeare experiment), and contains only the pure-eps audit (Lemma 4.2). The zCDP/GDP statements below are
the extended versions of those lemmas; nothing in the vendored file changes. What is
shared with it -- the log-space binomials, the chance-overlap table ``f(v)``, its
upper-tail sums, and the mean-statistic Chernoff minimisation -- is imported, not
reimplemented, so the null model is provably the same one the eps audit uses.

Chance-overlap distribution
---------------------------
With ``M' = C(m, floor(m/2))`` the number of balanced sign vectors and ``r`` even::

    pi(u) = (1/M') * sum_{a1,a2 in {0..r/2}, a1+a2=u}
                C(m-r, ceil((m-r)/2) - (a1-a2)) * C(r/2,a1) * C(r/2,a2)

This is the eps audit's ``f(u)`` normalised to a probability distribution over
``u in {0,...,r}`` (see :func:`log_pi_values` for why the ``ceil`` centre here and
the ``floor`` centre in ``log_f_values`` give the same numbers). Working with the
normalised ``pi`` -- rather than the unnormalised cumulative mass the eps solver
uses -- is what makes the Chernoff/tail quantities below actual *probabilities*,
which is what both the Renyi and the f-DP change-of-measure steps need.

NO HALVING HERE
---------------
:mod:`audit_tofu.epsilon_bounds` divides the solved LDP epsilon by
``LDP_REDUCTION_FACTOR = 2`` (Lemma 4.1: ``(eps,0)``-certified unlearning makes the
auditor ``(2*eps,0)``-LDP). The rho and mu audits must **not** be halved: the cost of
reducing through the common reference law is already inside the local parameter that
the solver inverts, namely

    eps_gamma^loc(rho) = 4 * rho * gamma                                 [zCDP]
    mu_loc(mu)         = 2 * mu                                          [GDP]

so ``rho_lb`` and ``mu_lb`` are directly bounds on the certified-unlearning
parameters. Halving them again would be wrong, which is why these wrappers do not
import :data:`audit_tofu.epsilon_bounds.LDP_REDUCTION_FACTOR`.

Reported ``eps_estimate``
-------------------------
Each report also carries ``eps_estimate``: the ``(eps, conv_delta)``-DP conversion of
``rho_lb`` / ``mu_lb``, purely so the numbers can be read on the same axis as
``epsilon_lb``. It is **not** a lower bound on eps -- converting a lower bound on rho
(or mu) through an upper-bound-direction conversion does not preserve the bound
direction. Every function that returns it says so, and
:func:`rho_lb_report` / :func:`mu_lb_report` repeat it in ``note``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, logsumexp, ndtri, ndtri_exp

from .cum_runs_eps_lab import (
    a_from_v_list,
    log_binom,
    log_g_from_logf,
    log_Z_closed_form,
    log1mexp,
    logM_bound_avg_v_ge_a,
)
from .epsilon_bounds import DEFAULT_ZETA, _validate

__all__ = [
    "DEFAULT_CONV_DELTA",
    "DEFAULT_ZETA",
    # null model
    "log_pi_values",
    # zCDP
    "eps_gamma_zcdp",
    "compute_avg_v_test_rho_lb",
    "compute_median_v_test_rho_lb",
    "rho_lb_pairwise_from_roc",
    # GDP
    "mu_loc_gdp_group",
    "log_q_mean_gdp",
    "log_q_med_gdp",
    "log_gdp_audit_bound",
    "compute_avg_v_test_mu_lb",
    "compute_median_v_test_mu_lb",
    # (eps, delta) display conversions -- not lower bounds on eps
    "eps_estimate_from_rho",
    "eps_estimate_from_mu",
    "log_theta_gdp",
    # report layer (mirrors epsilon_bounds)
    "rho_lb_mean",
    "rho_lb_median",
    "rho_lb_report",
    "mu_lb_mean",
    "mu_lb_median",
    "mu_lb_report",
]

#: ``delta`` used only for the display conversion of rho/mu into an (eps, delta)
#: pair. It is not part of any audit guarantee (both audits are delta-free).
DEFAULT_CONV_DELTA = 1e-3


# ----------------------------------------------------------------------------
# Chance-overlap distribution pi(u)
# ----------------------------------------------------------------------------

def log_pi_values(m: int, r: int) -> np.ndarray:
    """``log pi(u)`` for ``u = 0..r``, normalised so ``sum_u pi(u) = 1``.

    Built with the ``ceil((m-r)/2)`` centre of the zCDP/GDP lemma statements, whereas
    ``cum_runs_eps_lab.log_f_values`` uses ``floor((m-r)/2)``. The two agree *summand
    by summand after pairing*: swapping ``(a1,a2) -> (a2,a1)`` flips the sign of
    ``a1-a2`` and ``C(n, ceil(n/2)+d) = C(n, floor(n/2)-d)``, so each ``u`` gets the
    same total. Implemented with the ``ceil`` centre anyway, to stay literally
    readable against the lemma; ``tests/test_rho_mu_bounds.py`` pins the agreement
    with ``log_f_values - log M'`` for odd and even ``m-r``.
    """
    if r % 2 != 0:
        raise ValueError("Assumes r is even so r/2 is integer.")
    if m < r:
        raise ValueError("Need m >= r so n = m-r >= 0.")

    n = m - r
    half_r = r // 2
    center = math.ceil(n / 2)

    logf = np.full(r + 1, -np.inf, dtype=float)
    for u in range(r + 1):
        terms = []
        for a1 in range(max(0, u - half_r), min(half_r, u) + 1):
            a2 = u - a1
            k_idx = center - (a1 - a2)
            lb = log_binom(n, k_idx)
            if not np.isfinite(lb):
                continue
            terms.append(lb + log_binom(half_r, a1) + log_binom(half_r, a2))
        if terms:
            logf[u] = logsumexp(np.array(terms, dtype=float))

    # M' = C(m, floor(m/2)) = sum_u f(u): the number of balanced sign vectors.
    return logf - log_Z_closed_form(m)


# ----------------------------------------------------------------------------
# zCDP (rho) audit
#
# Reduction through the common reference law R. If unlearning is rho-zCDP certified
# then for the two candidate forget sets s1, s2 the local Renyi divergence obeys
# (lemma "zcdp_to_local_rdp")
#
#   D_gamma(P_{s1} || P_{s2}) <= eps_gamma^loc(rho)
#                              = 4 rho gamma,
#
# the factor 4 being the price of going through R -- which is why a rho obtained by
# inverting this is already a certified-unlearning rho (see the module note
# "NO HALVING HERE").
#
# The Renyi change-of-measure inequality over L independent runs then turns any
# null tail bound into a bound on the matched-hypothesis probability:
#
#   log Pr_matched[E] <= (gamma-1)/gamma * ( L * eps_gamma^loc(rho) + log Pr_null[E] ),
#
# and rho_lb is the largest rho for which the right-hand side, minimised over
# gamma > 1, still sits at or below the confidence budget log(zeta).
# ----------------------------------------------------------------------------

def eps_gamma_zcdp(rho: float, gamma: float) -> float:
    """``eps_gamma^loc(rho) = 4 rho gamma``.

    The local RDP-order-``gamma`` epsilon implied by rho-zCDP certified unlearning.
    Shared by the v-list audits and :func:`rho_lb_pairwise_from_roc` so there is one
    definition of the reduction in the codebase.

    This replaces the earlier ``2 rho gamma (1 + sqrt(gamma/(gamma-1)))``, which is
    the same expression in the ``gamma -> inf`` limit and strictly larger at every
    finite ``gamma``. Because the solver inverts this quantity, the smaller local
    epsilon yields a LARGER ``rho_lb``: results computed under the two forms are not
    comparable, and any stored bound must be recomputed after changing it.
    """
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1.")
    return 4.0 * float(rho) * gamma


def _min_log_rhs_zcdp(rho: float, L: int, logM_val: float, gamma_max: float = 1e4):
    """``inf_{gamma>1} (gamma-1)/gamma * (L*eps_gamma^loc(rho) + logM_val)``.

    ``logM_val`` is the log of the gamma-free null tail bound: the mean test's
    Chernoff bound, or ``log(C(L,ceil(L/2)) * Pi_bar(v)^ceil(L/2))`` for the median
    test. Both are <= 0, so at ``rho = 0`` the infimum is attained as
    ``gamma -> inf`` and equals ``logM_val`` -- i.e. the audit degenerates to "is the
    observation itself improbable under chance", exactly as it should.
    """
    def obj(gamma: float) -> float:
        return ((gamma - 1.0) / gamma) * (L * eps_gamma_zcdp(rho, gamma) + logM_val)

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
    """Largest ``rho >= 0`` whose zCDP bound still clears the confidence budget.

    Solves ``inf_gamma (gamma-1)/gamma * (L*eps_gamma^loc(rho) + logM_val) <=
    log(ci_delta)`` by bisection; the left-hand side is non-decreasing in rho, so the
    solution set is an interval ``[0, rho_lb]``. Returns ``None`` when even
    ``rho = 0`` fails, which means the observation is *not* significant at this
    confidence level (a perfectly private mechanism already explains it).
    """
    if not (0.0 < ci_delta < 1.0):
        raise ValueError("ci_delta must lie in (0,1).")
    if L <= 0:
        raise ValueError("L must be positive.")

    log_ci = math.log(ci_delta)

    val0, _ = _min_log_rhs_zcdp(0.0, L, logM_val, gamma_max)
    if val0 > log_ci:
        return None

    # Grow an upper bracket where the bound fails.
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
    conv_delta: float = DEFAULT_CONV_DELTA,
) -> dict:
    """zCDP analogue of ``compute_avg_v_test_epsilon_lb`` (mean statistic).

    Threshold ``v = mean(v_list)``; Chernoff bound on ``Pr_null[avg V >= v]`` under
    the chance distribution ``pi``; then the Renyi reduction minimised over
    ``gamma > 1``.

    ``eps_estimate`` is the ``(eps, conv_delta)`` conversion of ``rho_lb`` and is not
    a lower bound on eps (see the module docstring).
    """
    logpi = log_pi_values(m, r)
    v = a_from_v_list(v_list, T=T)

    # Same Chernoff minimisation the eps audit uses, but fed the normalised pi, so
    # the result is log of a probability rather than log of a cumulative mass.
    logM_bound, theta_star = logM_bound_avg_v_ge_a(logpi, T=T, a=v, theta_max=theta_max)
    logM_bound = min(0.0, float(logM_bound))

    rho_lb = _rho_lb_from_logM_zcdp(logM_bound, L=T, ci_delta=ci_delta, gamma_max=gamma_max)
    _, gamma_star = _min_log_rhs_zcdp(rho_lb or 0.0, T, logM_bound, gamma_max)

    return {
        "statistic": "mean",
        "v": v,
        "logM_bound": logM_bound,
        "theta_star": theta_star,
        "gamma_star": gamma_star,
        "rho_lb": rho_lb,
        "eps_estimate": eps_estimate_from_rho(rho_lb, conv_delta=conv_delta),
        "conv_delta": float(conv_delta),
        "note": "zCDP mean-v test; rho_lb is the largest rho the observed mean "
                "rejects at this confidence level. eps_estimate is its "
                "(eps, conv_delta) conversion, not a lower bound on eps.",
    }


def compute_median_v_test_rho_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    gamma_max: float = 1e4,
    conv_delta: float = DEFAULT_CONV_DELTA,
) -> dict:
    """zCDP analogue of ``compute_median_v_test_epsilon_lb`` (median statistic).

    Threshold ``v = ceil(median(v_list))`` -- only the ceiling matters, since ``V`` is
    integer-valued and ``V >= median <=> V >= ceil(median)`` (which bites when ``L``
    is even and the median falls between two order statistics). Null tail bound
    ``C(L, ceil(L/2)) * Pi_bar(v)^ceil(L/2)`` with
    ``Pi_bar(v) = sum_{u>=v} pi(u)``, then the same Renyi reduction.
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list of length T={T}, got length {len(v_list)}.")

    logPi_bar = log_g_from_logf(log_pi_values(m, r))

    v_median = float(np.median(v_list))
    v = math.ceil(v_median)
    if not (0 <= v <= r):
        raise ValueError(f"v={v} (ceil of median of v_list) outside [0,{r}].")

    log_Pi_bar_v = float(logPi_bar[v])
    T_half = math.ceil(T / 2)
    logM_median = min(0.0, log_binom(T, T_half) + T_half * log_Pi_bar_v)

    rho_lb = _rho_lb_from_logM_zcdp(logM_median, L=T, ci_delta=ci_delta, gamma_max=gamma_max)
    _, gamma_star = _min_log_rhs_zcdp(rho_lb or 0.0, T, logM_median, gamma_max)

    return {
        "statistic": "median",
        "v_median": v_median,
        "v": v,
        "log_Pi_bar_v": log_Pi_bar_v,
        "T_half": T_half,
        "logM_bound": logM_median,
        "gamma_star": gamma_star,
        "rho_lb": rho_lb,
        "eps_estimate": eps_estimate_from_rho(rho_lb, conv_delta=conv_delta),
        "conv_delta": float(conv_delta),
        "note": "zCDP median-v test; rho_lb is the largest rho the observed median "
                "rejects at this confidence level. eps_estimate is its "
                "(eps, conv_delta) conversion, not a lower bound on eps.",
    }


def rho_lb_pairwise_from_roc(
    tpr_low: float,
    fpr_high: float,
    tnr_low: float,
    fnr_high: float,
    gamma_max: float = 1e4,
    n_grid: int = 4096,
) -> dict:
    """Pairwise auditor's zCDP lower bound from a single ROC point.

    The v-list audits above aggregate a statistic over ``L`` runs. The pairwise
    auditor instead observes one confusion matrix for the two-hypothesis problem
    ``P_0`` vs ``P_1`` (the ``m = 2`` case: which of two forget sets was unlearned)
    and converts its ROC point directly, with no Chernoff step::

        rho_lb^(p) = sup_{gamma>1} max{0, b_gamma^(+), b_gamma^(-)} / (4 gamma),
          b_gamma^(+) = gamma/(gamma-1) * log TPR^low - log FPR^high,
          b_gamma^(-) = gamma/(gamma-1) * log TNR^low - log FNR^high.

    The denominator is ``eps_gamma_zcdp(1, gamma)``, reused verbatim so this bound and
    the v-list audits share one definition of ``eps_gamma^loc``. The two ``b`` terms
    are the same Renyi change-of-measure inequality applied to the acceptance region
    ``A`` and to ``A^c`` with the roles of ``P_0, P_1`` swapped.

    Confidence: ``TPR^low = 1 - FNR^high`` and ``TNR^low = 1 - FPR^high`` hold exactly
    for Clopper-Pearson intervals, so all four rates come out of the same two
    per-class confidence statements -- no extra union-bound budget is needed.

    This function is provided for parity with the pairwise auditor; nothing in this
    repository currently produces ROC points (the TOFU instantiation reports overlap
    scores over ``m = 20`` candidates), so it has no caller here.

    Returns a dict with ``rho_lb >= 0`` (a certified-unlearning rho, not halved), the
    maximising ``gamma_star``, and the two ``b`` terms there. ``rho_lb`` is ``inf``
    when an error-rate upper bound is 0, i.e. no finite rho explains the observation,
    and 0 when the ROC point is consistent with perfect privacy.
    """
    if gamma_max <= 1.0:
        raise ValueError("gamma_max must exceed 1.")

    # A zero upper bound on an error rate means -log(rate) = +inf.
    if (fpr_high is not None and fpr_high <= 0.0) or (fnr_high is not None and fnr_high <= 0.0):
        return {
            "rho_lb": float("inf"),
            "gamma_star": None,
            "b_plus": float("inf"),
            "b_minus": float("inf"),
            "note": "FPR^high or FNR^high is 0: observation impossible under any finite rho.",
        }

    log_tpr_low = math.log(tpr_low) if (tpr_low is not None and tpr_low > 0.0) else -np.inf
    log_tnr_low = math.log(tnr_low) if (tnr_low is not None and tnr_low > 0.0) else -np.inf
    log_fpr_high = math.log(fpr_high)
    log_fnr_high = math.log(fnr_high)

    def objective(gamma: float) -> float:
        ratio = gamma / (gamma - 1.0)
        b_plus = ratio * log_tpr_low - log_fpr_high
        b_minus = ratio * log_tnr_low - log_fnr_high
        best = max(0.0, b_plus, b_minus)
        if best <= 0.0:
            return 0.0
        return best / eps_gamma_zcdp(1.0, gamma)

    # Log-spaced grid in (gamma - 1): the objective vanishes at both ends (gamma -> 1+
    # kills the numerator, gamma -> inf grows the denominator), so the sup is interior.
    gammas = 1.0 + np.logspace(-9.0, math.log10(gamma_max - 1.0), n_grid)
    values = np.array([objective(g) for g in gammas], dtype=float)
    i_best = int(np.argmax(values))
    rho_lb = float(values[i_best])
    gamma_star = float(gammas[i_best])

    # Local refinement between the neighbouring grid points.
    lo = gammas[max(0, i_best - 1)]
    hi = gammas[min(len(gammas) - 1, i_best + 1)]
    if hi > lo:
        res = minimize_scalar(lambda g: -objective(g), bounds=(lo, hi), method="bounded")
        if -float(res.fun) > rho_lb:
            rho_lb = -float(res.fun)
            gamma_star = float(res.x)

    ratio_star = gamma_star / (gamma_star - 1.0)
    return {
        "rho_lb": rho_lb,
        "gamma_star": gamma_star,
        "b_plus": float(ratio_star * log_tpr_low - log_fpr_high),
        "b_minus": float(ratio_star * log_tnr_low - log_fnr_high),
        "note": (
            "Pairwise zCDP lower bound from one ROC point. Already a "
            "certified-unlearning rho: the reference-law factor sits inside "
            "eps_gamma^loc(rho), so it is not halved."
            if rho_lb > 0.0 else
            "max{0, b+, b-} = 0 for every gamma: the ROC point is consistent with rho = 0."
        ),
    }


# ----------------------------------------------------------------------------
# Gaussian-DP (mu) audit
#
# Audits the GDP parameter straight from its hypothesis-testing semantics: no Renyi
# detour, and no assumption that the model laws are Gaussian.
#
# Reduction through the reference law R is the f-DP group operation. Dong, Roth & Su
# (JRSS-B 2022) Thm 3 with k = 2, applied to the chain P_s1 -> R -> P_s2 (each link
# mu-GDP in both directions), gives
#
#   T(P_s1, P_s2) >= 1 - (1 - G_mu)^{o2} = G_{2 mu},
#
# i.e. mu_loc(mu) = 2 mu exactly -- G_mu is a translation by mu in probit
# coordinates and two translations compose to a translation by 2 mu. The paper notes
# the group bound "in general cannot be improved". L independent runs then compose
# exactly to sqrt(L) * mu_loc(mu)-GDP.
#
# With a null tail bound q on the observed statistic, mu-GDP therefore gives
#   Pr_matched[Z >= v] <= Phi(Phi^{-1}(q) + sqrt(L) * 2 mu) =: B_mu(v),
# which inverts in closed form (see _mu_lb_from_logq_gdp). Because mu_loc already
# carries the factor 2 and the audit solves for mu itself, mu_lb needs no halving.
# ----------------------------------------------------------------------------

def mu_loc_gdp_group(mu: float) -> float:
    """Local GDP parameter through the reference law: ``mu_loc(mu) = 2 mu``, exactly.

    Dong, Roth & Su (JRSS-B 2022) Theorem 3: an ``f``-DP mechanism is
    ``[1 - (1-f)^{ok}]``-DP for groups of size ``k``, and in particular a mu-GDP
    mechanism is ``k mu``-GDP. Applied with ``k = 2`` to ``P_s1 -> R -> P_s2``: for
    any test ``phi`` with ``E_{P_s1}[phi] <= a``, the first link gives
    ``E_R[phi] <= 1 - G_mu(a) =: psi(a)``, the second link at level ``psi(a)`` gives
    ``E_{P_s2}[phi] <= psi(psi(a))``, hence ``T(P_s1, P_s2) >= 1 - psi^{o2} = G_{2mu}``.

    So no Gaussian envelope, no hockey-stick profile, no infimum, and -- unlike the
    total-variation triangle-inequality route -- no saturation ceiling on ``mu_lb``.
    """
    if mu < 0.0:
        raise ValueError("mu must be non-negative.")
    return 2.0 * float(mu)


def log_q_mean_gdp(m: int, r: int, L: int, v: float, theta_max: float = 50.0):
    """``log q_mean(v)``: null tail bound for the mean statistic.

        q_mean(v) = min{1, inf_{lam>=0} exp(L log sum_u e^{lam u} pi(u) - lam L v)}

    Returns ``(log_q_mean, lambda_star)``. The minimisation is
    ``logM_bound_avg_v_ge_a`` from the vendored module, which implements exactly this
    expression; handing it the normalised ``pi`` (rather than the eps audit's
    unnormalised ``f``) is what makes the result a probability bound. The ``lambda``
    search is truncated at ``theta_max``, and an infimum over a subset can only be
    larger, so the truncation inflates ``q`` and shrinks ``mu_lb`` -- conservative.
    """
    logpi = log_pi_values(m, r)
    log_q, lambda_star = logM_bound_avg_v_ge_a(logpi, T=L, a=v, theta_max=theta_max)
    return min(0.0, float(log_q)), float(lambda_star)


def log_q_med_gdp(m: int, r: int, L: int, v: int):
    """``log q_med(v)``: null tail bound for the median statistic.

        q_med(v) = min{1, C(L, ceil(L/2)) * Pi_bar(v)^ceil(L/2)},
        Pi_bar(v) = sum_{u>=v} pi(u)

    Returns ``(log_q_med, L_half, log_Pi_bar_v)``.
    """
    logPi_bar = log_g_from_logf(log_pi_values(m, r))
    if not (0 <= v <= r):
        raise ValueError(f"v={v} outside [0,{r}].")
    L_half = math.ceil(L / 2)
    log_Pi_bar_v = float(logPi_bar[v])
    return min(0.0, log_binom(L, L_half) + L_half * log_Pi_bar_v), L_half, log_Pi_bar_v


def log_gdp_audit_bound(mu: float, log_q: float, L: int) -> float:
    """``log B_mu(v)`` where ``B_mu(v) = Phi(Phi^{-1}(q(v)) + sqrt(L) * mu_loc(mu))``.

    ``Phi^{-1}(q)`` is taken straight from ``log q`` via ``ndtri_exp``: ``q`` is
    routinely 1e-150 or smaller, where ``exp`` underflows to 0 and ``ndtri(0) = -inf``
    would destroy the bound.
    """
    z = float(ndtri_exp(log_q)) + math.sqrt(L) * mu_loc_gdp_group(mu)
    return float(log_ndtr(z))


def _mu_lb_from_logq_gdp(log_q: float, L: int, ci_delta: float):
    """``mu_lb = sup({mu >= 0 : B_mu(v_obs) <= ci_delta} u {0})``, in closed form.

    With ``mu_loc(mu) = 2 mu``, monotonicity of ``Phi`` turns ``B_mu <= ci_delta``
    into ``Phi^{-1}(q) + 2 sqrt(L) mu <= Phi^{-1}(ci_delta)``, so

        mu_lb = tau / 2,   tau := (Phi^{-1}(ci_delta) - Phi^{-1}(q)) / sqrt(L),

    and ``mu_lb = 0`` when ``q >= ci_delta`` (there ``B_0 = q > ci_delta`` already, so
    the set is empty and the ``u {0}`` supplies the answer). ``B_mu`` is still
    evaluated at the answer and returned, as a check that it lands on ``ci_delta``.

    Returns ``(mu_lb, tau, log_B_at_mu_lb, note)``.
    """
    if not (0.0 < ci_delta < 1.0):
        raise ValueError("ci_delta must lie in (0,1).")
    if L <= 0:
        raise ValueError("L must be positive.")

    log_ci = math.log(ci_delta)
    if log_q >= log_ci:
        return 0.0, 0.0, log_q, "q_obs >= ci_delta: observation not significant, mu_lb = 0."

    tau = float((ndtri(ci_delta) - ndtri_exp(log_q)) / math.sqrt(L))
    mu_lb = 0.5 * tau
    return mu_lb, tau, log_gdp_audit_bound(mu_lb, log_q, L), (
        "mu_lb = tau/2 in closed form, from the exact f-DP group operation "
        "mu_loc(mu) = 2 mu (Dong et al. Thm 3, k=2)."
    )


def compute_avg_v_test_mu_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    theta_max: float = 50.0,
    conv_delta: float = DEFAULT_CONV_DELTA,
) -> dict:
    """GDP analogue of ``compute_avg_v_test_epsilon_lb`` (mean statistic).

    Threshold ``v = mean(v_list)``, null tail bound ``q_mean(v)``, then
    ``mu_lb = sup{mu : B_mu <= ci_delta} = tau/2``.

    ``mu_lb`` bounds the certified-unlearning GDP parameter directly (no halving).
    ``eps_estimate`` is its ``(eps, conv_delta)`` conversion and is not a lower bound
    on eps.
    """
    v = a_from_v_list(v_list, T=T)
    log_q, lambda_star = log_q_mean_gdp(m, r, L=T, v=v, theta_max=theta_max)
    mu_lb, tau, log_B, note = _mu_lb_from_logq_gdp(log_q, L=T, ci_delta=ci_delta)

    return {
        "statistic": "mean",
        "v": v,
        "log_q": log_q,
        "lambda_star": lambda_star,
        "tau": tau,
        "mu_lb": mu_lb,
        "log_B_at_mu_lb": log_B,
        "eps_estimate": eps_estimate_from_mu(mu_lb, conv_delta=conv_delta),
        "conv_delta": float(conv_delta),
        "note": "GDP mean-v test. " + note,
    }


def compute_median_v_test_mu_lb(
    m: int,
    r: int,
    T: int,
    v_list,
    ci_delta: float,
    conv_delta: float = DEFAULT_CONV_DELTA,
) -> dict:
    """GDP analogue of ``compute_median_v_test_epsilon_lb`` (median statistic).

    Threshold ``v = ceil(median(v_list))``, null tail bound ``q_med(v)``, then the
    same closed-form inversion ``mu_lb = tau/2``.
    """
    if len(v_list) != T:
        raise ValueError(f"Expected v_list of length T={T}, got length {len(v_list)}.")

    v_median = float(np.median(v_list))
    v = math.ceil(v_median)
    if not (0 <= v <= r):
        raise ValueError(f"v={v} (ceil of median of v_list) outside [0,{r}].")

    log_q, T_half, log_Pi_bar_v = log_q_med_gdp(m, r, L=T, v=v)
    mu_lb, tau, log_B, note = _mu_lb_from_logq_gdp(log_q, L=T, ci_delta=ci_delta)

    return {
        "statistic": "median",
        "v_median": v_median,
        "v": v,
        "log_Pi_bar_v": log_Pi_bar_v,
        "T_half": T_half,
        "log_q": log_q,
        "tau": tau,
        "mu_lb": mu_lb,
        "log_B_at_mu_lb": log_B,
        "eps_estimate": eps_estimate_from_mu(mu_lb, conv_delta=conv_delta),
        "conv_delta": float(conv_delta),
        "note": "GDP median-v test. " + note,
    }


# ----------------------------------------------------------------------------
# (eps, delta) display conversions
#
# NOT lower bounds on eps. Both conversions below are of the form "rho-zCDP (resp.
# mu-GDP) implies (eps, delta)-DP", i.e. they map a privacy *guarantee* to a weaker
# guarantee. Applying them to a lower bound gives a number on the epsilon axis with
# no bound semantics attached; it is reported only to make rho_lb / mu_lb legible
# next to epsilon_lb.
# ----------------------------------------------------------------------------

def eps_estimate_from_rho(rho, conv_delta: float = DEFAULT_CONV_DELTA):
    """``eps = rho + 2 sqrt(rho log(1/conv_delta))`` (Bun & Steinke 2016, Prop. 1.3).

    Passes ``None`` (nothing certified) and ``inf`` through unchanged.
    """
    if rho is None:
        return None
    rho = float(rho)
    if not np.isfinite(rho):
        return float("inf")
    if rho <= 0.0:
        return 0.0
    if not (0.0 < conv_delta < 1.0):
        raise ValueError("conv_delta must lie in (0,1).")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / conv_delta))


def log_theta_gdp(eps: float, mu: float) -> float:
    """``log theta_eps(mu)``, the exact privacy profile of ``G_mu``::

        theta_eps(mu) = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2)

    i.e. the smallest ``delta`` for which mu-GDP implies ``(eps, delta)``-DP. Written
    in log-space as ``logA + log(1 - exp(eps + logB - logA))`` so that the
    cancellation between the two terms does not lose the (often tiny) difference.
    """
    if mu < 0.0:
        raise ValueError("mu must be non-negative.")
    if mu == 0.0:
        return -np.inf  # G_0 is perfectly private: delta = 0 for every eps.

    log_a = float(log_ndtr(-eps / mu + mu / 2.0))
    log_b = float(log_ndtr(-eps / mu - mu / 2.0))
    # theta >= 0 always, so the argument is <= 0; clamp against float error at eps=0.
    return log_a + log1mexp(min(0.0, eps + log_b - log_a))


def eps_estimate_from_mu(
    mu,
    conv_delta: float = DEFAULT_CONV_DELTA,
    eps_hi_init: float = 50.0,
    tol: float = 1e-10,
    max_iter: int = 200,
):
    """Smallest ``eps >= 0`` with ``theta_eps(mu) <= conv_delta``, by bisection.

    ``theta_eps(mu)`` is strictly decreasing in ``eps``, so the answer is the unique
    crossing (or 0, when mu-GDP already implies ``(0, conv_delta)``-DP). Passes
    ``None`` and ``inf`` through unchanged.
    """
    if mu is None:
        return None
    mu = float(mu)
    if not np.isfinite(mu):
        return float("inf")
    if mu <= 0.0:
        return 0.0
    if not (0.0 < conv_delta < 1.0):
        raise ValueError("conv_delta must lie in (0,1).")

    log_target = math.log(conv_delta)
    if log_theta_gdp(0.0, mu) <= log_target:
        return 0.0

    hi = eps_hi_init
    while log_theta_gdp(hi, mu) > log_target:
        hi *= 2.0
        if hi > 1e6:  # conservative cap; mu this large is already degenerate
            return hi

    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if log_theta_gdp(mid, mu) <= log_target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return hi  # hi always satisfies the constraint; lo never does


# ----------------------------------------------------------------------------
# Report layer -- mirrors audit_tofu.epsilon_bounds
# ----------------------------------------------------------------------------

def _finalize(
    raw: dict,
    key: str,
    m: int,
    r: int,
    v_list: Sequence[int],
    zeta: float,
    statistic: str,
) -> dict:
    """Normalise a raw rho/mu result into the same report shape as ``epsilon_lb_*``.

    ``key`` is ``"rho_lb"`` or ``"mu_lb"``. Shares the eps audit's contract for a
    non-result: ``None`` means "nothing certified at this confidence level". Unlike
    the eps wrapper there is no halving step here -- see the module docstring.
    """
    value = raw.get(key)

    if value is None:
        certified = False
        note = (
            f"No positive {key} certified: the observed {statistic} overlap is "
            "consistent with random guessing at this confidence level."
        )
    elif value == float("inf"):
        certified = True
        note = f"{key} is unbounded (observation impossible under chance); inspect diagnostics."
    elif value <= 0.0:
        value = None
        certified = False
        note = (
            f"Solved {key} was zero: the observed {statistic} statistic is at or below "
            "chance, so no positive bound is certified."
        )
    else:
        certified = True
        note = (
            f"Reject H0: the certified-unlearning parameter <= {key} at level zeta. "
            "Not halved -- the Lemma 4.1 factor of 2 is already inside the local "
            "parameter this bound inverts (mu_loc(mu) = 2mu; the 2 in "
            "eps_gamma^loc(rho) = 4 rho gamma). 'eps_estimate' is a display conversion, not a "
            "lower bound on eps."
        )

    n = len(v_list)
    return {
        "statistic": statistic,
        key: value,
        f"{key}_raw": raw.get(key),
        "certified": certified,
        # Nothing certified => nothing to convert; the raw layer reports 0.0 here
        # (mu) or None (rho), which would read as a meaningful eps of 0.
        "eps_estimate": raw.get("eps_estimate") if value is not None else None,
        "conv_delta": raw.get("conv_delta"),
        "halved": False,
        "m": int(m),
        "r": int(r),
        "L": n,
        "zeta": float(zeta),
        "delta": 0.0,  # both audits are delta-free; kept for report symmetry
        "v_list": [int(v) for v in v_list],
        "v_mean": sum(v_list) / n,
        "random_guess_baseline": r / 2.0,
        "max_attainable_v": int(r),
        "upstream": raw,
        "note": note,
    }


def rho_lb_mean(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    conv_delta: float = DEFAULT_CONV_DELTA,
    theta_max: float = 50.0,
    gamma_max: float = 1e4,
) -> dict:
    """Mean-based zCDP lower bound. Same arguments as ``epsilon_lb_mean``.

    ``m`` candidate forget batches, ``r`` the TOTAL (even) guess budget with ``r/2``
    guesses per sign, ``v_list`` one overlap in ``[0, r]`` per independent run,
    ``zeta`` the confidence level. Returns ``rho_lb`` plus diagnostics.
    """
    v_list = [int(v) for v in v_list]
    _validate(m, r, v_list)
    raw = compute_avg_v_test_rho_lb(
        m=int(m),
        r=int(r),
        T=len(v_list),
        v_list=v_list,
        ci_delta=float(zeta),
        gamma_max=float(gamma_max),
        theta_max=float(theta_max),
        conv_delta=float(conv_delta),
    )
    return _finalize(raw, "rho_lb", m, r, v_list, zeta, statistic="mean")


def rho_lb_median(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    conv_delta: float = DEFAULT_CONV_DELTA,
    gamma_max: float = 1e4,
) -> dict:
    """Median-based zCDP lower bound, reported alongside the mean as a check."""
    v_list = [int(v) for v in v_list]
    _validate(m, r, v_list)
    raw = compute_median_v_test_rho_lb(
        m=int(m),
        r=int(r),
        T=len(v_list),
        v_list=v_list,
        ci_delta=float(zeta),
        gamma_max=float(gamma_max),
        conv_delta=float(conv_delta),
    )
    return _finalize(raw, "rho_lb", m, r, v_list, zeta, statistic="median")


def rho_lb_report(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    conv_delta: float = DEFAULT_CONV_DELTA,
    theta_max: float = 50.0,
    gamma_max: float = 1e4,
) -> dict:
    """Both zCDP statistics at one ``r``. ``mean`` is the headline number."""
    v_list = [int(v) for v in v_list]
    return {
        "mean": rho_lb_mean(m, r, v_list, zeta, conv_delta, theta_max, gamma_max),
        "median": rho_lb_median(m, r, v_list, zeta, conv_delta, gamma_max),
    }


def mu_lb_mean(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    conv_delta: float = DEFAULT_CONV_DELTA,
    theta_max: float = 50.0,
) -> dict:
    """Mean-based GDP lower bound. Same arguments as ``epsilon_lb_mean``."""
    v_list = [int(v) for v in v_list]
    _validate(m, r, v_list)
    raw = compute_avg_v_test_mu_lb(
        m=int(m),
        r=int(r),
        T=len(v_list),
        v_list=v_list,
        ci_delta=float(zeta),
        theta_max=float(theta_max),
        conv_delta=float(conv_delta),
    )
    return _finalize(raw, "mu_lb", m, r, v_list, zeta, statistic="mean")


def mu_lb_median(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    conv_delta: float = DEFAULT_CONV_DELTA,
) -> dict:
    """Median-based GDP lower bound, reported alongside the mean as a check."""
    v_list = [int(v) for v in v_list]
    _validate(m, r, v_list)
    raw = compute_median_v_test_mu_lb(
        m=int(m),
        r=int(r),
        T=len(v_list),
        v_list=v_list,
        ci_delta=float(zeta),
        conv_delta=float(conv_delta),
    )
    return _finalize(raw, "mu_lb", m, r, v_list, zeta, statistic="median")


def mu_lb_report(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    conv_delta: float = DEFAULT_CONV_DELTA,
    theta_max: float = 50.0,
) -> dict:
    """Both GDP statistics at one ``r``. ``mean`` is the headline number."""
    v_list = [int(v) for v in v_list]
    return {
        "mean": mu_lb_mean(m, r, v_list, zeta, conv_delta, theta_max),
        "median": mu_lb_median(m, r, v_list, zeta, conv_delta),
    }
