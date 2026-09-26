"""Spec validation 8b: the zCDP (``rho``) and Gaussian-DP (``mu``) lower bounds.

Same two-layer structure as :mod:`tests.test_epsilon_bounds`.

1. The **math** in :mod:`audit_tofu.rho_mu_bounds` is checked against its own
   defining statements, recomputed here independently (``math.comb`` integers for the
   chance-overlap distribution, the definition of ``B_mu`` for the GDP inversion, and
   the gamma-minimised Renyi bound for the zCDP bisection). The chance model is also
   pinned to the eps audit's ``f`` table, so the two audits provably share a null.
2. The **report layer** is checked for the contract that differs from the eps
   wrapper: rho and mu are *not* halved, and a non-significant observation yields
   ``None`` rather than a number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import log_ndtr, logsumexp, ndtri, ndtri_exp

from audit_tofu.cum_runs_eps_lab import log_f_values, log_Z_closed_form
from audit_tofu.epsilon_bounds import epsilon_lb_mean
from audit_tofu.rho_mu_bounds import (
    _min_log_rhs_zcdp,
    compute_avg_v_test_mu_lb,
    compute_avg_v_test_rho_lb,
    compute_median_v_test_mu_lb,
    compute_median_v_test_rho_lb,
    eps_estimate_from_mu,
    eps_estimate_from_rho,
    eps_gamma_zcdp,
    log_gdp_audit_bound,
    log_pi_values,
    log_q_mean_gdp,
    log_q_med_gdp,
    log_theta_gdp,
    mu_lb_mean,
    mu_lb_median,
    mu_lb_report,
    mu_loc_gdp_group,
    rho_lb_mean,
    rho_lb_median,
    rho_lb_pairwise_from_roc,
    rho_lb_report,
)

ZETA = 0.05


# --- layer 1a: the chance-overlap distribution --------------------------------

@pytest.mark.parametrize("m,r", [(20, 4), (20, 10), (20, 20), (21, 6), (9, 4), (400, 100)])
def test_log_pi_is_a_probability_distribution(m, r):
    logpi = log_pi_values(m, r)
    assert len(logpi) == r + 1
    assert float(logsumexp(logpi)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("m,r", [(20, 4), (20, 10), (21, 6), (9, 4), (20, 20)])
def test_log_pi_equals_the_eps_audits_f_table_normalised(m, r):
    """pi(u) = f(u)/M'.

    ``log_pi_values`` centres its binomial at ``ceil((m-r)/2)`` and the vendored
    ``log_f_values`` at ``floor((m-r)/2)``; the two agree once the ``(a1,a2)`` swap is
    summed over, including for odd ``m-r``. This is the test that keeps the zCDP/GDP
    audits on the same null model as the eps audit.
    """
    pi = np.exp(log_pi_values(m, r))
    f_over_Z = np.exp(log_f_values(m, r) - log_Z_closed_form(m))
    np.testing.assert_allclose(pi, f_over_Z, rtol=0, atol=1e-14)


@pytest.mark.parametrize("m,r", [(20, 6), (21, 6)])
def test_log_pi_matches_exact_integer_recomputation(m, r):
    """pi recomputed in exact integers straight from the lemma statement."""
    n, half = m - r, r // 2
    center = -(-n // 2)  # ceil(n/2) in integers
    Z = math.comb(m, m // 2)
    for u in range(r + 1):
        total = 0
        for a1 in range(half + 1):
            a2 = u - a1
            if not (0 <= a2 <= half):
                continue
            k = center - (a1 - a2)
            if 0 <= k <= n:
                total += math.comb(n, k) * math.comb(half, a1) * math.comb(half, a2)
        expected = total / Z
        got = float(np.exp(log_pi_values(m, r)[u]))
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15), f"u={u}"


# --- layer 1b: the GDP (mu) audit ---------------------------------------------

@pytest.mark.parametrize("mu", [0.0, 0.25, 1.0, 7.5])
def test_mu_loc_is_exactly_twice_mu(mu):
    """The f-DP group operation with k=2 is exact: no envelope, no saturation."""
    assert mu_loc_gdp_group(mu) == pytest.approx(2.0 * mu)


def test_mu_lb_closed_form_saturates_the_confidence_budget():
    """B_mu(v) == zeta at mu_lb, and > zeta just above it (so the sup is the sup)."""
    m, r, L = 20, 10, 5
    res = compute_avg_v_test_mu_lb(m, r, L, [9] * L, ci_delta=ZETA)
    mu_lb, log_q = res["mu_lb"], res["log_q"]
    assert mu_lb > 0.0

    assert log_gdp_audit_bound(mu_lb, log_q, L) == pytest.approx(math.log(ZETA), abs=1e-9)
    assert log_gdp_audit_bound(mu_lb * 1.001, log_q, L) > math.log(ZETA)


def test_mu_lb_matches_the_tau_over_two_formula():
    m, r, L = 20, 10, 7
    v_list = [8, 9, 10, 9, 8, 10, 9]
    res = compute_avg_v_test_mu_lb(m, r, L, v_list, ci_delta=ZETA)
    tau = (ndtri(ZETA) - ndtri_exp(res["log_q"])) / math.sqrt(L)
    assert res["tau"] == pytest.approx(tau, rel=1e-12)
    assert res["mu_lb"] == pytest.approx(tau / 2.0, rel=1e-12)


def test_gdp_audit_bound_matches_its_definition_in_linear_space():
    """log B_mu = log Phi(Phi^-1(q) + sqrt(L) * 2mu), checked where exp() is safe."""
    from scipy.stats import norm

    log_q, L, mu = math.log(1e-4), 5, 0.3
    expected = norm.cdf(norm.ppf(1e-4) + math.sqrt(L) * 2.0 * mu)
    assert math.exp(log_gdp_audit_bound(mu, log_q, L)) == pytest.approx(expected, rel=1e-9)


def test_median_null_tail_bound_matches_exact_recomputation():
    m, r, L, v = 20, 10, 5, 9
    log_q, L_half, log_tail = log_q_med_gdp(m, r, L, v)
    pi = np.exp(log_pi_values(m, r))
    tail = float(pi[v:].sum())
    assert L_half == 3
    assert math.exp(log_tail) == pytest.approx(tail, rel=1e-12)
    assert math.exp(log_q) == pytest.approx(math.comb(L, L_half) * tail ** L_half, rel=1e-10)


def test_mean_null_tail_bound_is_a_probability_and_chernoff_valid():
    """q_mean is a probability bound: <= 1, and >= the exact tail it dominates."""
    m, r, L = 20, 10, 1
    pi = np.exp(log_pi_values(m, r))
    for v in (6, 7, 8, 9, 10):
        log_q, _ = log_q_mean_gdp(m, r, L, float(v))
        assert log_q <= 0.0
        exact_tail = float(pi[v:].sum())  # L=1: avg(V) >= v is just V >= v
        assert math.exp(log_q) >= exact_tail - 1e-15


# --- layer 1c: the zCDP (rho) audit -------------------------------------------

@pytest.mark.parametrize("gamma", [1.5, 2.0, 10.0, 1e3])
def test_eps_gamma_zcdp_matches_its_definition(gamma):
    rho = 0.37
    expected = 4.0 * rho * gamma
    assert eps_gamma_zcdp(rho, gamma) == pytest.approx(expected, rel=1e-12)
    assert eps_gamma_zcdp(0.0, gamma) == 0.0  # rho=0 is free

    # The previous form, 2 rho gamma (1 + sqrt(gamma/(gamma-1))), is this one's
    # gamma -> inf limit and strictly larger at every finite gamma. Pinned so that a
    # silent revert to it is caught: it would shrink every rho_lb we report.
    old = 2.0 * rho * gamma * (1.0 + math.sqrt(gamma / (gamma - 1.0)))
    assert eps_gamma_zcdp(rho, gamma) < old


def test_eps_gamma_zcdp_rejects_gamma_at_or_below_one():
    with pytest.raises(ValueError):
        eps_gamma_zcdp(1.0, 1.0)


def test_rho_lb_is_the_largest_rho_clearing_the_budget():
    """The bisection is tight: the bound holds at rho_lb and fails just above it."""
    m, r, L = 20, 10, 5
    res = compute_avg_v_test_rho_lb(m, r, L, [10] * L, ci_delta=ZETA)
    rho_lb, logM = res["rho_lb"], res["logM_bound"]
    assert rho_lb > 0.0

    val_at, _ = _min_log_rhs_zcdp(rho_lb, L, logM)
    val_above, _ = _min_log_rhs_zcdp(rho_lb * 1.01, L, logM)
    assert val_at <= math.log(ZETA) + 1e-9
    assert val_above > math.log(ZETA)


def test_rho_bound_holds_against_a_brute_force_gamma_grid():
    """The gamma minimisation is not optimistic: no gamma on a fine grid beats it."""
    m, r, L = 20, 12, 5
    res = compute_avg_v_test_rho_lb(m, r, L, [11] * L, ci_delta=ZETA)
    rho_lb, logM = res["rho_lb"], res["logM_bound"]
    best, _ = _min_log_rhs_zcdp(rho_lb, L, logM)

    gammas = 1.0 + np.logspace(-9.0, 4.0, 20001)
    grid = np.array([
        ((g - 1.0) / g) * (L * eps_gamma_zcdp(rho_lb, g) + logM) for g in gammas
    ])
    assert best <= float(grid.min()) + 1e-8


# --- layer 1d: audit behaviour (both parametrisations) ------------------------

@pytest.mark.parametrize("stat", ["mean", "median"])
def test_chance_level_overlap_certifies_nothing(stat):
    """v = r/2 is exactly random guessing: eps, rho and mu must all abstain."""
    m, r, L = 20, 10, 5
    v_list = [r // 2] * L
    assert epsilon_lb_mean(m, r, v_list, zeta=ZETA)["epsilon_lb"] is None
    assert rho_lb_report(m, r, v_list, zeta=ZETA)[stat]["rho_lb"] is None
    assert mu_lb_report(m, r, v_list, zeta=ZETA)[stat]["mu_lb"] is None


def test_below_chance_overlap_certifies_nothing():
    m, r, L = 20, 10, 5
    v_list = [2] * L
    assert rho_lb_mean(m, r, v_list, zeta=ZETA)["rho_lb"] is None
    assert mu_lb_mean(m, r, v_list, zeta=ZETA)["mu_lb"] is None
    assert mu_lb_mean(m, r, v_list, zeta=ZETA)["eps_estimate"] is None


def test_bounds_increase_with_stronger_evidence():
    m, r, L = 20, 10, 5
    rhos, mus = [], []
    for v in (7, 8, 9, 10):
        rhos.append(rho_lb_mean(m, r, [v] * L, zeta=ZETA)["rho_lb"] or 0.0)
        mus.append(mu_lb_mean(m, r, [v] * L, zeta=ZETA)["mu_lb"] or 0.0)
    assert rhos == sorted(rhos) and rhos[0] < rhos[-1]
    assert mus == sorted(mus) and mus[0] < mus[-1]


def test_bounds_increase_with_more_runs():
    m, r = 20, 10
    rhos, mus = [], []
    for L in (3, 5, 9, 15):
        rhos.append(rho_lb_mean(m, r, [9] * L, zeta=ZETA)["rho_lb"] or 0.0)
        mus.append(mu_lb_mean(m, r, [9] * L, zeta=ZETA)["mu_lb"] or 0.0)
    assert rhos == sorted(rhos) and rhos[0] < rhos[-1]
    assert mus == sorted(mus) and mus[0] < mus[-1]


def test_median_is_never_stronger_than_the_mean_on_constant_v_lists():
    """With every run identical the mean test uses all L runs and the median only
    ceil(L/2), so the mean bound must dominate."""
    m, r, L = 20, 10, 5
    v_list = [10] * L
    assert (compute_median_v_test_rho_lb(m, r, L, v_list, ci_delta=ZETA)["rho_lb"]
            <= compute_avg_v_test_rho_lb(m, r, L, v_list, ci_delta=ZETA)["rho_lb"])
    assert (compute_median_v_test_mu_lb(m, r, L, v_list, ci_delta=ZETA)["mu_lb"]
            <= compute_avg_v_test_mu_lb(m, r, L, v_list, ci_delta=ZETA)["mu_lb"])


def test_median_threshold_uses_the_ceiling_of_an_even_L_median():
    """L even: the median sits between order statistics, and V >= median reduces to
    V >= ceil(median)."""
    m, r, L = 20, 10, 4
    res = compute_median_v_test_mu_lb(m, r, L, [8, 9, 9, 10], ci_delta=ZETA)
    assert res["v_median"] == pytest.approx(9.0)
    assert res["v"] == 9
    res_odd = compute_median_v_test_mu_lb(m, r, L, [8, 8, 9, 10], ci_delta=ZETA)
    assert res_odd["v_median"] == pytest.approx(8.5)
    assert res_odd["v"] == 9


# --- layer 1e: the (eps, delta) display conversions ---------------------------

@pytest.mark.parametrize("rho", [0.01, 0.5, 3.0])
def test_eps_estimate_from_rho_matches_bun_steinke(rho):
    delta = 1e-3
    expected = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    assert eps_estimate_from_rho(rho, delta) == pytest.approx(expected, rel=1e-12)


def test_eps_estimates_pass_through_none_and_infinity():
    assert eps_estimate_from_rho(None) is None
    assert eps_estimate_from_mu(None) is None
    assert eps_estimate_from_rho(float("inf")) == float("inf")
    assert eps_estimate_from_mu(float("inf")) == float("inf")
    assert eps_estimate_from_rho(0.0) == 0.0
    assert eps_estimate_from_mu(0.0) == 0.0


@pytest.mark.parametrize("mu", [0.2, 1.0, 4.0])
def test_log_theta_gdp_matches_the_privacy_profile_of_G_mu(mu):
    from scipy.stats import norm

    for eps in (0.0, 0.5, 2.0):
        expected = norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)
        assert math.exp(log_theta_gdp(eps, mu)) == pytest.approx(expected, rel=1e-8, abs=1e-14)


@pytest.mark.parametrize("mu", [0.2, 1.0, 4.0])
def test_eps_estimate_from_mu_inverts_theta(mu):
    """The returned eps is the crossing: theta(eps) <= delta and theta just below is not."""
    delta = 1e-3
    eps = eps_estimate_from_mu(mu, delta)
    assert log_theta_gdp(eps, mu) <= math.log(delta) + 1e-9
    if eps > 0.0:
        assert log_theta_gdp(eps * (1.0 - 1e-6) - 1e-9, mu) > math.log(delta) - 1e-6


def test_log_theta_gdp_at_zero_eps_is_the_total_variation_distance():
    """theta_0(mu) = 2 Phi(mu/2) - 1 = TV(G_mu)."""
    from scipy.stats import norm

    for mu in (0.3, 1.5, 3.0):
        assert math.exp(log_theta_gdp(0.0, mu)) == pytest.approx(
            2.0 * norm.cdf(mu / 2.0) - 1.0, rel=1e-9
        )


# --- layer 1f: the pairwise ROC zCDP bound ------------------------------------

def test_pairwise_rho_lb_is_zero_at_chance_and_positive_when_separated():
    assert rho_lb_pairwise_from_roc(0.5, 0.5, 0.5, 0.5)["rho_lb"] == 0.0
    strong = rho_lb_pairwise_from_roc(0.95, 0.05, 0.95, 0.05)
    assert strong["rho_lb"] > 0.0
    assert strong["gamma_star"] > 1.0


def test_pairwise_rho_lb_is_symmetric_under_swapping_the_two_directions():
    a = rho_lb_pairwise_from_roc(0.9, 0.2, 0.8, 0.1)
    b = rho_lb_pairwise_from_roc(0.8, 0.1, 0.9, 0.2)
    assert a["rho_lb"] == pytest.approx(b["rho_lb"], rel=1e-9)


def test_pairwise_rho_lb_is_infinite_when_an_error_rate_bound_is_zero():
    assert rho_lb_pairwise_from_roc(0.99, 0.0, 0.99, 0.01)["rho_lb"] == float("inf")
    assert rho_lb_pairwise_from_roc(0.99, 0.01, 0.99, 0.0)["rho_lb"] == float("inf")


def test_pairwise_rho_lb_never_exceeds_the_defining_inequality():
    """rho_lb is a sup over gamma of max{0,b+,b-}/eps_gamma^loc(1,gamma)."""
    tpr, fpr, tnr, fnr = 0.9, 0.1, 0.85, 0.15
    res = rho_lb_pairwise_from_roc(tpr, fpr, tnr, fnr)
    gammas = 1.0 + np.logspace(-9.0, 4.0, 20001)
    best = 0.0
    for g in gammas:
        ratio = g / (g - 1.0)
        b = max(
            ratio * math.log(tpr) - math.log(fpr),
            ratio * math.log(tnr) - math.log(fnr),
        )
        if b > 0.0:
            best = max(best, b / eps_gamma_zcdp(1.0, g))
    assert res["rho_lb"] == pytest.approx(best, rel=1e-6)


# --- layer 2: the report layer -------------------------------------------------

def test_reports_are_not_halved():
    """The Lemma 4.1 factor of 2 sits inside eps_gamma^loc / mu_loc, so the wrapper
    must pass the solved value through unchanged -- unlike epsilon_bounds."""
    m, r, L = 20, 10, 5
    v_list = [9] * L

    raw_rho = compute_avg_v_test_rho_lb(m, r, L, v_list, ci_delta=ZETA)["rho_lb"]
    raw_mu = compute_avg_v_test_mu_lb(m, r, L, v_list, ci_delta=ZETA)["mu_lb"]
    rep_rho = rho_lb_mean(m, r, v_list, zeta=ZETA)
    rep_mu = mu_lb_mean(m, r, v_list, zeta=ZETA)

    assert rep_rho["rho_lb"] == pytest.approx(raw_rho, rel=1e-12)
    assert rep_mu["mu_lb"] == pytest.approx(raw_mu, rel=1e-12)
    assert rep_rho["halved"] is False and rep_mu["halved"] is False

    raw_med_rho = compute_median_v_test_rho_lb(m, r, L, v_list, ci_delta=ZETA)["rho_lb"]
    raw_med_mu = compute_median_v_test_mu_lb(m, r, L, v_list, ci_delta=ZETA)["mu_lb"]
    assert rho_lb_median(m, r, v_list, zeta=ZETA)["rho_lb"] == pytest.approx(raw_med_rho)
    assert mu_lb_median(m, r, v_list, zeta=ZETA)["mu_lb"] == pytest.approx(raw_med_mu)


def test_reports_carry_the_audit_metadata_and_both_statistics():
    m, r, L = 20, 10, 5
    v_list = [7, 8, 9, 10, 9]
    for report, key in ((rho_lb_report(m, r, v_list, zeta=ZETA), "rho_lb"),
                        (mu_lb_report(m, r, v_list, zeta=ZETA), "mu_lb")):
        assert set(report) == {"mean", "median"}
        for stat, block in report.items():
            assert block["statistic"] == stat
            assert block["m"] == m and block["r"] == r and block["L"] == L
            assert block["zeta"] == ZETA
            assert block["delta"] == 0.0  # both audits are delta-free
            assert block["v_list"] == v_list
            assert block["v_mean"] == pytest.approx(sum(v_list) / L)
            assert block["random_guess_baseline"] == r / 2.0
            assert block["max_attainable_v"] == r
            assert key in block and "certified" in block and "note" in block
            assert "upstream" in block


def test_r_is_the_total_guess_budget_not_per_side():
    """Same convention as epsilon_bounds: V ranges over [0, r], so v = r is the
    maximum and an overlap above r is a caller bug, not a doubled r."""
    m, r, L = 20, 10, 3
    assert rho_lb_mean(m, r, [r] * L, zeta=ZETA)["rho_lb"] > 0.0
    with pytest.raises(ValueError):
        rho_lb_mean(m, r, [r + 1] * L, zeta=ZETA)
    with pytest.raises(ValueError):
        mu_lb_mean(m, r, [r + 1] * L, zeta=ZETA)


@pytest.mark.parametrize("bad", [dict(r=9), dict(r=0), dict(r=22)])
def test_reports_validate_r(bad):
    kwargs = dict(m=20, r=10)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        rho_lb_mean(v_list=[5, 5, 5], zeta=ZETA, **kwargs)
    with pytest.raises(ValueError):
        mu_lb_mean(v_list=[5, 5, 5], zeta=ZETA, **kwargs)


def test_empty_v_list_is_rejected():
    with pytest.raises(ValueError):
        rho_lb_mean(20, 10, [], zeta=ZETA)
    with pytest.raises(ValueError):
        mu_lb_mean(20, 10, [], zeta=ZETA)


def test_certified_flag_tracks_the_bound():
    m, r, L = 20, 10, 5
    strong_rho = rho_lb_mean(m, r, [10] * L, zeta=ZETA)
    weak_rho = rho_lb_mean(m, r, [5] * L, zeta=ZETA)
    assert strong_rho["certified"] is True and strong_rho["eps_estimate"] > 0.0
    assert weak_rho["certified"] is False and weak_rho["rho_lb"] is None


def test_bounds_are_conservative_with_a_stricter_confidence_level():
    m, r, L = 20, 10, 7
    loose = mu_lb_mean(m, r, [9] * L, zeta=0.1)["mu_lb"]
    tight = mu_lb_mean(m, r, [9] * L, zeta=0.001)["mu_lb"]
    assert tight < loose
    loose_rho = rho_lb_mean(m, r, [9] * L, zeta=0.1)["rho_lb"]
    tight_rho = rho_lb_mean(m, r, [9] * L, zeta=0.001)["rho_lb"]
    assert tight_rho < loose_rho
