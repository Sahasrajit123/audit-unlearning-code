"""Fast unit tests for the privacy-bound building blocks."""
import math

import numpy as np
import pytest

from gaussian_mechanism import gaussian_mechanism_sigma_general as sigma_general_newton
from membership_inference import clopper_pearson, epsilon_empirical_lower_bound
from output_perturbation import clip_to_ball, gaussian_mechanism_sigma_general
from zcdp_pairwise import (
    _rho_lb_numeric,
    eps_estimate_from_rho,
    rho_lb_convex_pairwise,
    rho_zcdp_upper_gaussian,
)


def test_modules_import():
    import avg_v_convex, cubic_loss, cum_runs_eps_lab, data_generation  # noqa: F401
    import data_persistence, lipschitz_constants, op_data_generation  # noqa: F401
    import op_membership_inference, run_output_perturbation_attack  # noqa: F401
    import run_privacy_attack_final, training, unlearning  # noqa: F401


def test_clip_to_ball():
    w = np.array([3.0, 4.0])
    np.testing.assert_allclose(clip_to_ball(w, 1.0), [0.6, 0.8])
    np.testing.assert_allclose(clip_to_ball(w, 10.0), w)


@pytest.mark.parametrize("sigma_fn", [gaussian_mechanism_sigma_general, sigma_general_newton])
def test_gaussian_sigma_decreases_with_epsilon(sigma_fn):
    sigmas = [sigma_fn(2.0, eps, 1e-3) for eps in (0.1, 1.0, 10.0)]
    assert all(s > 0 for s in sigmas)
    assert sigmas[0] > sigmas[1] > sigmas[2]


def test_clopper_pearson_brackets_estimate():
    lo, hi = clopper_pearson(30, 100)
    assert lo < 0.30 < hi
    assert clopper_pearson(0, 50)[0] == 0.0
    assert clopper_pearson(50, 50)[1] == 1.0


def test_eps_emp_lower_no_signal_is_not_positive():
    # A chance-level attack must not certify positive epsilon.
    eps = epsilon_empirical_lower_bound(tpr_low=0.45, fpr_high=0.55, delta=0.01)
    assert not (eps > 0)


def test_rho_lb_closed_form_matches_numeric_sup():
    tpr_low, fpr_high = 0.8, 0.1
    res = rho_lb_convex_pairwise(tpr_low, fpr_high, verify=False)
    a, b = -math.log(tpr_low), -math.log(fpr_high)
    assert res["rho_lb"] == pytest.approx((math.sqrt(b) - math.sqrt(a)) ** 2)
    numeric = _rho_lb_numeric(tpr_low, fpr_high, 1 - fpr_high, 1 - tpr_low)
    assert res["rho_lb"] == pytest.approx(numeric, rel=1e-6)


def test_rho_lb_is_zero_without_separation():
    assert rho_lb_convex_pairwise(0.5, 0.5)["rho_lb"] == 0.0


def test_rho_ub_gaussian():
    assert rho_zcdp_upper_gaussian(2.0, 1.0) == pytest.approx(2.0)


def test_eps_from_rho_is_monotone():
    eps = [eps_estimate_from_rho(r, conv_delta=1e-3) for r in (0.01, 0.1, 1.0)]
    assert eps[0] < eps[1] < eps[2]
