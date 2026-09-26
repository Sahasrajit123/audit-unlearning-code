"""Spec validation 8: the epsilon lower-bound computation.

Two layers are checked.

1. The **vendored** math in :mod:`audit_tofu.cum_runs_eps_lab` is verified against
   Lemma 4.2 of the paper, recomputed independently here with ``math.comb`` in exact
   integer arithmetic. That catches any drift in the file we copied.
2. The **wrapper** in :mod:`audit_tofu.epsilon_bounds` is checked for the two
   convention adapters it exists to provide: the ``r``-is-total convention and the
   Lemma 4.1 halving.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from audit_tofu.cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb,
    compute_median_v_test_epsilon_lb,
    log_binom,
    log_f_values,
    log_Z_closed_form,
)
from audit_tofu.epsilon_bounds import (
    LDP_REDUCTION_FACTOR,
    epsilon_lb_mean,
    epsilon_lb_median,
    epsilon_lb_report,
)


# --- layer 1: the vendored combinatorics match the paper ----------------------

def _f_exact(m: int, r: int, v: int) -> int:
    """f(v) from Lemma 4.2, in exact integers.

        f(v) = sum_{a1+a2=v} C(m-r, floor((m-r)/2) - (a1-a2)) * C(r/2,a1) * C(r/2,a2)

    C(n,k) = C(n,n-k) for even n, so floor and ceil agree for the cases used here.
    """
    n = m - r
    half = r // 2
    total = 0
    for a1 in range(half + 1):
        a2 = v - a1
        if not (0 <= a2 <= half):
            continue
        k = n // 2 - (a1 - a2)
        if 0 <= k <= n:
            total += math.comb(n, k) * math.comb(half, a1) * math.comb(half, a2)
    return total


@pytest.mark.parametrize(
    "m,r",
    [(20, 4), (20, 8), (20, 12), (20, 16), (20, 20), (6, 2), (6, 6), (400, 100)],
)
def test_log_f_values_matches_lemma_4_2(m, r):
    logf = log_f_values(m, r)
    assert len(logf) == r + 1
    for v in range(r + 1):
        expected = _f_exact(m, r, v)
        if expected == 0:
            assert not np.isfinite(logf[v]) or logf[v] == -np.inf
        else:
            assert logf[v] == pytest.approx(math.log(expected), rel=1e-9), (
                f"m={m} r={r} v={v}: f={expected}"
            )


@pytest.mark.parametrize("m,r", [(20, 4), (20, 20), (10, 6)])
def test_f_sums_to_the_number_of_balanced_vectors(m, r):
    """sum_v f(v) = |S_m| = C(m, floor(m/2)) = Z.

    Every balanced sign vector produces exactly one overlap value against a fixed
    guess, so the f table must partition S_m.
    """
    logf = log_f_values(m, r)
    total = sum(_f_exact(m, r, v) for v in range(r + 1))
    assert total == math.comb(m, m // 2)
    from scipy.special import logsumexp

    finite = logf[np.isfinite(logf)]
    assert logsumexp(finite) == pytest.approx(log_Z_closed_form(m), rel=1e-12)


def test_log_Z_is_the_central_binomial():
    for m in (4, 6, 20, 200, 400):
        assert log_Z_closed_form(m) == pytest.approx(
            math.log(math.comb(m, m // 2)), rel=1e-12
        )


def test_log_binom_edge_cases():
    assert log_binom(5, 0) == pytest.approx(0.0)
    assert log_binom(5, 5) == pytest.approx(0.0)
    assert log_binom(5, -1) == -np.inf
    assert log_binom(5, 6) == -np.inf
    assert log_binom(10, 3) == pytest.approx(math.log(120), rel=1e-12)


# --- layer 2: chance-level overlap certifies nothing --------------------------

def test_chance_overlap_yields_no_bound():
    """V = r/2 is exactly random guessing, so no positive epsilon is justified."""
    for r in (4, 8, 12, 16, 20):
        out = epsilon_lb_mean(20, r, [r // 2] * 10)
        assert out["epsilon_lb"] is None, f"r={r} gave {out['epsilon_lb']}"


def test_below_chance_overlap_yields_no_bound():
    out = epsilon_lb_mean(20, 20, [4] * 10)
    assert out["epsilon_lb"] is None


# --- layer 2: monotonicity and the r-sweep -----------------------------------

def test_bound_is_monotone_in_observed_overlap():
    """More leakage must never certify a smaller epsilon."""
    prev = -1.0
    for v in range(10, 21):
        out = epsilon_lb_mean(20, 20, [v] * 10)
        eps = out["epsilon_lb"]
        if eps is None:
            continue
        assert eps >= prev - 1e-9, f"non-monotone at V={v}: {eps} < {prev}"
        prev = eps


def test_perfect_attack_bound_grows_with_r():
    """A perfect attack over a larger support certifies a larger epsilon."""
    vals = []
    for r in (4, 8, 12, 16, 20):
        out = epsilon_lb_mean(20, r, [r] * 10)
        vals.append(out["epsilon_lb"])
    assert all(v is not None for v in vals)
    assert vals == sorted(vals), vals


def test_more_runs_tighten_the_bound():
    """Same per-run overlap, more runs -> more evidence -> larger certified epsilon."""
    e5 = epsilon_lb_mean(20, 20, [18] * 5)["epsilon_lb"]
    e10 = epsilon_lb_mean(20, 20, [18] * 10)["epsilon_lb"]
    e20 = epsilon_lb_mean(20, 20, [18] * 20)["epsilon_lb"]
    assert e5 < e10 < e20


def test_larger_m_raises_the_attainable_bound():
    """Paper Remark 4.3: larger m raises the maximum attainable bound."""
    e_m20 = epsilon_lb_mean(20, 20, [20] * 10)["epsilon_lb"]
    e_m400 = epsilon_lb_mean(400, 20, [20] * 10)["epsilon_lb"]
    assert e_m400 > e_m20


def test_stricter_confidence_gives_a_smaller_bound():
    loose = epsilon_lb_mean(20, 20, [18] * 10, zeta=0.10)["epsilon_lb"]
    tight = epsilon_lb_mean(20, 20, [18] * 10, zeta=0.01)["epsilon_lb"]
    assert tight < loose


# --- layer 2: the Lemma 4.1 halving -----------------------------------------

def test_reported_epsilon_is_the_ldp_solution_halved():
    """The wrapper's whole reason to exist. Upstream omits this division."""
    assert LDP_REDUCTION_FACTOR == 2.0
    out = epsilon_lb_mean(20, 20, [19] * 10)
    assert out["epsilon_lb"] == pytest.approx(out["epsilon_ldp_lb"] / 2.0)


def test_wrapper_ldp_value_equals_the_vendored_result_unchanged():
    """The vendored module must be called with r as-is and its output untouched."""
    m, r, v_list = 20, 12, [9, 10, 11, 8, 12, 10, 9, 11, 10, 10]
    raw = compute_avg_v_test_epsilon_lb(
        m=m, r=r, T=len(v_list), v_list=v_list,
        delta=0.0, ci_delta=0.05, direction="ge", theta_max=50.0,
    )
    wrapped = epsilon_lb_mean(m, r, v_list, zeta=0.05, delta=0.0)
    assert wrapped["epsilon_ldp_lb"] == pytest.approx(raw["epsilon_lb"])
    assert wrapped["epsilon_lb"] == pytest.approx(raw["epsilon_lb"] / 2.0)


def test_no_double_halving_when_the_vendored_module_halves_itself(monkeypatch):
    """The guard: a vendored revision that already applies Lemma 4.1 must not be
    halved a second time.

    The file currently vendored here reports the LDP epsilon (so the wrapper divides).
    A newer revision divides internally and returns BOTH ``epsilon_lb_ldp`` and an
    already-halved ``epsilon_lb``. Dropping that file in unnoticed would understate
    every reported bound by 2x, so both detection signals are exercised: the result
    key, and the module-level ``_unlearning_eps_from_ldp`` helper.
    """
    from audit_tofu import epsilon_bounds as eb

    m, r, v_list = 20, 20, [19] * 10
    ldp = epsilon_lb_mean(m, r, v_list)["epsilon_ldp_lb"]   # as vendored today
    assert ldp > 0

    def fake_halving_entry_point(**kwargs):
        return {
            "epsilon_lb_ldp": ldp,
            "epsilon_lb": ldp / 2.0,     # the newer revision's own division
            "note": "already halved",
        }

    # Signal 1: the result dict advertises the un-halved value separately.
    monkeypatch.setattr(eb, "compute_avg_v_test_epsilon_lb", fake_halving_entry_point)
    out = epsilon_lb_mean(m, r, v_list)
    assert out["halving_applied_by"] == "vendored_module"
    assert out["epsilon_lb"] == pytest.approx(ldp / 2.0), "halved twice"
    assert out["epsilon_ldp_lb"] == pytest.approx(ldp)

    # Signal 2: the module exposes the helper that performs the division, so even a
    # result dict without the extra key is recognised as already halved.
    monkeypatch.setattr(
        eb._vendored, "_unlearning_eps_from_ldp", lambda e: e, raising=False
    )
    monkeypatch.setattr(
        eb,
        "compute_avg_v_test_epsilon_lb",
        lambda **kw: {"epsilon_lb": ldp / 2.0, "note": "already halved, no ldp key"},
    )
    out2 = epsilon_lb_mean(m, r, v_list)
    assert out2["halving_applied_by"] == "vendored_module"
    assert out2["epsilon_lb"] == pytest.approx(ldp / 2.0), "halved twice"
    # The LDP value is reconstructed for the report even when not supplied.
    assert out2["epsilon_ldp_lb"] == pytest.approx(ldp)


def test_delta_is_passed_only_when_the_vendored_signature_accepts_it(monkeypatch):
    """The halving revision dropped ``delta``; passing it would be a TypeError."""
    from audit_tofu import epsilon_bounds as eb

    seen = {}

    def no_delta_entry_point(*, m, r, T, v_list, ci_delta, direction, theta_max):
        seen.update(ci_delta=ci_delta, direction=direction)
        return {"epsilon_lb_ldp": 4.0, "epsilon_lb": 2.0}

    monkeypatch.setattr(eb, "compute_avg_v_test_epsilon_lb", no_delta_entry_point)
    out = epsilon_lb_mean(20, 20, [19] * 10, zeta=0.05, delta=0.0)
    assert out["epsilon_lb"] == pytest.approx(2.0)
    assert seen["ci_delta"] == 0.05

    # A non-zero delta cannot be silently dropped -- the reduction does not support it.
    with pytest.raises(ValueError, match="delta"):
        epsilon_lb_mean(20, 20, [19] * 10, delta=1e-5)


def test_halving_is_recorded_in_every_report():
    for out in (epsilon_lb_mean(20, 20, [19] * 10), epsilon_lb_median(20, 20, [19] * 10)):
        assert out["halving_applied_by"] == "wrapper"   # the file vendored today
        assert out["ldp_reduction_factor"] == 2.0


def test_r_is_total_not_per_side():
    """Guard against reintroducing the upstream caller's per-side convention.

    Upstream `evaluate_llr_predictions.py --r 10` passes `epsilon_r = 20`. Here
    `r=20` means 20 total guesses, so our r=20 must equal upstream's r_value=10.
    """
    v_list = [16] * 10
    ours = epsilon_lb_mean(20, 20, v_list)
    upstream_equivalent = compute_avg_v_test_epsilon_lb(
        m=20, r=2 * 10, T=10, v_list=v_list,
        delta=0.0, ci_delta=0.05, direction="ge", theta_max=50.0,
    )
    assert ours["epsilon_ldp_lb"] == pytest.approx(upstream_equivalent["epsilon_lb"])


# --- layer 2: validation and reporting ---------------------------------------

def test_rejects_odd_r_and_out_of_range_overlap():
    with pytest.raises(ValueError, match="even"):
        epsilon_lb_mean(20, 7, [4] * 10)
    with pytest.raises(ValueError, match=r"r <= m"):
        epsilon_lb_mean(20, 22, [11] * 10)
    with pytest.raises(ValueError, match="outside"):
        epsilon_lb_mean(20, 8, [9] * 10)     # V > r
    with pytest.raises(ValueError, match="outside"):
        epsilon_lb_mean(20, 8, [-1] * 10)
    with pytest.raises(ValueError, match="empty"):
        epsilon_lb_mean(20, 8, [])


def test_report_carries_audit_metadata():
    out = epsilon_lb_mean(20, 16, [14] * 10, zeta=0.05, delta=0.0)
    assert out["m"] == 20
    assert out["r"] == 16
    assert out["L"] == 10
    assert out["zeta"] == 0.05
    assert out["delta"] == 0.0
    assert out["random_guess_baseline"] == 8.0
    assert out["max_attainable_v"] == 16
    assert out["v_mean"] == pytest.approx(14.0)
    assert out["statistic"] == "mean"
    assert "upstream" in out


def test_median_statistic_runs_and_is_labelled():
    out = epsilon_lb_median(20, 20, [18] * 10)
    assert out["statistic"] == "median"
    assert out["epsilon_lb"] is None or out["epsilon_lb"] >= 0


def test_report_returns_both_statistics():
    rep = epsilon_lb_report(20, 20, [18] * 10)
    assert set(rep) == {"mean", "median"}
    assert rep["mean"]["statistic"] == "mean"
    assert rep["median"]["statistic"] == "median"


# --- regression pins ---------------------------------------------------------

@pytest.mark.parametrize(
    "m,r,v,L,expected_ldp",
    [
        (20, 20, 20, 10, 13.178659135155613),
        (20, 16, 16, 10, 10.167297345469706),
        (20, 12, 12, 10, 7.589361514692428),
        (20, 8, 8, 10, 4.999303018121282),
        (20, 4, 4, 10, 2.364615961960226),
    ],
)
def test_regression_pinned_values(m, r, v, L, expected_ldp):
    """Pins the audit's numeric ceiling so a refactor cannot move it silently.

    These are the LDP solutions for a *perfect* attack (V = r on every run) at
    m=20, L=10, zeta=0.05, delta=0. The reported epsilon_LB is half of each.
    """
    out = epsilon_lb_mean(m, r, [v] * L, zeta=0.05, delta=0.0)
    assert out["epsilon_ldp_lb"] == pytest.approx(expected_ldp, rel=1e-9)
    assert out["epsilon_lb"] == pytest.approx(expected_ldp / 2.0, rel=1e-9)


def test_audit_ceiling_at_the_configured_shape():
    """The best epsilon_LB achievable at m=20, L=10 is about 6.59.

    Documented as a test because it is a property of the audit's *shape*, not of any
    unlearning method: no attack on this configuration can certify more.
    """
    best = epsilon_lb_mean(20, 20, [20] * 10)["epsilon_lb"]
    assert best == pytest.approx(6.589329567577806, rel=1e-9)


# --- a negative solve certifies nothing --------------------------------------

def test_negative_median_solution_is_reported_as_none_not_a_negative_bound():
    """Regression: the median test returned a NEGATIVE epsilon at chance overlap.

    eps >= 0 by definition, so "eps > -0.65" is vacuously true for every mechanism.
    The two upstream entry points disagreed: the mean test checks feasibility at
    eps=0 and returns None, while the median test solves an unclamped log difference
    and can go negative. Measured on the real audit, `grad_ascent` produced a
    negative median epsilon at every r while the mean correctly gave None.
    """
    # The actual grad_ascent evaluation overlaps, m=20, chance = 10.
    v_list = [14, 10, 10, 16, 12, 8, 8, 10, 12, 10]

    med = epsilon_lb_median(20, 20, v_list)
    assert med["epsilon_lb"] is None, "a negative solve must not be reported as a bound"
    assert med["certified"] is False
    # The raw value stays inspectable rather than being silently discarded.
    assert med["epsilon_ldp_raw"] is not None and med["epsilon_ldp_raw"] < 0
    assert med["epsilon_ldp_lb"] is None
    assert "certifies nothing" in med["note"]

    mean = epsilon_lb_mean(20, 20, v_list)
    assert mean["epsilon_lb"] is None, "mean already handled this"
    assert mean["certified"] is False

    # Both statistics now share one contract at chance-level overlap.
    assert (mean["epsilon_lb"] is None) == (med["epsilon_lb"] is None)


def test_certified_flag_tracks_a_usable_bound():
    strong = epsilon_lb_mean(20, 20, [20] * 10)
    assert strong["certified"] is True and strong["epsilon_lb"] > 0

    chance = epsilon_lb_mean(20, 20, [10] * 10)
    assert chance["certified"] is False and chance["epsilon_lb"] is None


def test_below_chance_overlap_certifies_nothing_on_either_statistic():
    """Worse than random must never yield a positive bound."""
    for stat in (epsilon_lb_mean, epsilon_lb_median):
        out = stat(20, 20, [4] * 10)          # V=4 vs chance 10
        assert out["epsilon_lb"] is None, (stat.__name__, out["epsilon_lb"])
        assert out["certified"] is False
