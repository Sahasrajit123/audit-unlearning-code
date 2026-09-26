"""Spec validations 6 and 7: label hygiene, Gaussian likelihoods, variance
flooring, and deterministic tie handling."""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pytest

from audit_tofu import attack as attack_mod
from audit_tofu.attack import (
    Calibration,
    CalibrationWarning,
    QAGaussian,
    fit_calibration,
    log_normal_pdf,
    overlap,
    predict,
)


# --- spec validation 6: the predictor cannot receive evaluation labels --------

def test_predict_signature_has_no_label_parameter():
    """Structural guarantee, so a future refactor cannot quietly add one."""
    sig = inspect.signature(predict)
    names = set(sig.parameters)
    assert names == {"scores", "calibration", "r", "aggregate"}, names

    forbidden = {
        "labels", "label", "sign_vector", "signs", "s", "manifest",
        "truth", "ground_truth", "y", "targets",
    }
    assert not (names & forbidden), f"predict exposes label-like params: {names & forbidden}"


def test_predict_rejects_attempts_to_pass_labels():
    cal = _toy_calibration()
    scores = {("b0", 0): 0.5, ("b1", 0): 1.5, ("b2", 0): 0.6, ("b3", 0): 1.4}
    with pytest.raises(TypeError):
        predict(scores, cal, 2, sign_vector=[1, -1, 1, -1])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        predict(scores, cal, 2, labels=[1, -1, 1, -1])  # type: ignore[call-arg]


def test_attack_module_does_not_import_manifest():
    """The attack must not be able to reach ground truth through the manifest."""
    src = inspect.getsource(attack_mod)
    assert "from .manifest" not in src
    assert "import manifest" not in src


def test_overlap_is_separate_from_prediction():
    sig = inspect.signature(overlap)
    assert list(sig.parameters) == ["guess", "sign_vector"]


# --- spec validation 7a: Gaussian log densities -------------------------------

def test_log_normal_pdf_matches_scipy():
    from scipy.stats import norm

    for x, mu, var in [(0.0, 0.0, 1.0), (1.5, 0.3, 0.25), (-2.0, 1.0, 4.0),
                       (10.0, 0.0, 1e-4)]:
        expected = norm.logpdf(x, loc=mu, scale=math.sqrt(var))
        assert log_normal_pdf(x, mu, var) == pytest.approx(expected, rel=1e-10)


def test_log_normal_pdf_is_stable_for_tiny_variance():
    # A far-out point with a tiny variance must give a large finite negative number,
    # not -inf or nan.
    v = log_normal_pdf(5.0, 0.0, 1e-8)
    assert np.isfinite(v) and v < -1e8


def test_log_normal_pdf_rejects_nonpositive_variance():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            log_normal_pdf(0.0, 0.0, bad)


def test_llr_sign_follows_which_gaussian_is_closer():
    g = QAGaussian("b", 0, mu_in=0.5, var_in=0.01, n_in=5,
                   mu_out=1.5, var_out=0.01, n_out=5)
    assert g.llr(0.5) > 0, "a loss at mu_in should favour the in hypothesis"
    assert g.llr(1.5) < 0, "a loss at mu_out should favour the out hypothesis"
    assert g.llr(1.0) == pytest.approx(0.0, abs=1e-9), "equidistant -> zero LLR"


# --- spec validation 7b: variance flooring -----------------------------------

def _cal_inputs(values_in, values_out, n_batches=2, n_qa=1):
    """Build calibration inputs where every batch/QA sees the given observations."""
    batch_ids = [f"b{i}" for i in range(n_batches)]
    losses, signs = {}, {}
    n = max(len(values_in), len(values_out))
    for k in range(n):
        rid = f"calib_{k:03d}"
        losses[rid], signs[rid] = {}, {}
        for i, b in enumerate(batch_ids):
            # Alternate the sign so each batch accumulates both conditions.
            sj = 1 if (k + i) % 2 == 0 else -1
            signs[rid][b] = sj
            pool = values_in if sj == 1 else values_out
            idx = k // 2
            if idx < len(pool):
                for q in range(n_qa):
                    losses[rid][(b, q)] = pool[idx]
    return losses, signs, batch_ids


def test_variance_is_floored():
    # Zero spread in both conditions -> variance would be 0 without the floor.
    losses, signs, batch_ids = _cal_inputs([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        cal = fit_calibration(losses, signs, batch_ids, var_floor=1e-3, warn=False)
    for g in cal.gaussians.values():
        assert g.var_in >= 1e-3
        assert g.var_out >= 1e-3


def test_degenerate_fit_is_flagged_and_warns():
    losses, signs, batch_ids = _cal_inputs([1.0, 1.0], [2.0, 2.0])
    with pytest.warns(CalibrationWarning):
        cal = fit_calibration(losses, signs, batch_ids, var_floor=1e-6)
    assert cal.diagnostics["num_degenerate"] > 0
    assert any(g.degenerate for g in cal.gaussians.values())
    assert cal.diagnostics["degenerate_examples"]


def test_var_floor_must_be_positive():
    losses, signs, batch_ids = _cal_inputs([1.0, 1.1], [2.0, 2.1])
    with pytest.raises(ValueError, match="var_floor must be positive"):
        fit_calibration(losses, signs, batch_ids, var_floor=0.0, warn=False)


def test_pooled_variance_shares_across_qa_pairs_of_a_batch():
    rng = np.random.default_rng(0)
    batch_ids = ["b0"]
    losses, signs = {}, {}
    for k in range(12):
        rid = f"calib_{k:03d}"
        sj = 1 if k % 2 == 0 else -1
        signs[rid] = {"b0": sj}
        losses[rid] = {}
        for q in range(5):
            # Each QA pair has a different mean but a shared spread.
            mu = (0.5 if sj == 1 else 1.5) + 0.3 * q
            losses[rid][("b0", q)] = mu + rng.normal(0, 0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        per_qa = fit_calibration(losses, signs, batch_ids, var_floor=1e-9,
                                 pool_variance=None, warn=False)
        pooled = fit_calibration(losses, signs, batch_ids, var_floor=1e-9,
                                 pool_variance="author", warn=False)

    pooled_vars = {g.var_in for g in pooled.gaussians.values()}
    assert len(pooled_vars) == 1, "pooling should give one variance per condition"

    # Pooling must not absorb the between-QA mean differences into the variance.
    pooled_var = pooled_vars.pop()
    assert pooled_var < 0.05, pooled_var
    # Means stay per-QA either way.
    assert len({g.mu_in for g in pooled.gaussians.values()}) == 5
    assert len({g.mu_in for g in per_qa.gaussians.values()}) == 5


def test_pool_variance_rejects_unknown_mode():
    losses, signs, batch_ids = _cal_inputs([1.0, 1.1], [2.0, 2.1])
    with pytest.raises(ValueError, match="pool_variance"):
        fit_calibration(losses, signs, batch_ids, pool_variance="global", warn=False)


# --- spec validation 7c: deterministic tie handling ---------------------------

def _toy_calibration(n_batches=4, mu_in=0.5, mu_out=1.5, var=0.01):
    batch_ids = [f"b{i}" for i in range(n_batches)]
    gaussians = {
        (b, 0): QAGaussian(b, 0, mu_in=mu_in, var_in=var, n_in=5,
                           mu_out=mu_out, var_out=var, n_out=5)
        for b in batch_ids
    }
    return Calibration(batch_ids=batch_ids, gaussians=gaussians)


def test_ties_break_deterministically_on_batch_id():
    cal = _toy_calibration(n_batches=6)
    # Every batch gets an identical loss -> every Lambda ties.
    scores = {(b, 0): 1.0 for b in cal.batch_ids}

    first = predict(scores, cal, 4)
    for _ in range(20):
        again = predict(dict(reversed(list(scores.items()))), cal, 4)
        assert again["guess"] == first["guess"], "tie-break is not deterministic"
        assert again["predicted_positive"] == first["predicted_positive"]

    # Ties resolve lexicographically: b0,b1 positive; b5,b4 negative.
    assert first["predicted_positive"] == ["b0", "b1"]
    assert first["predicted_negative"] == ["b4", "b5"]


def test_prediction_budget_is_exactly_r_over_two_per_side():
    cal = _toy_calibration(n_batches=20)
    rng = np.random.default_rng(3)
    scores = {(b, 0): float(rng.normal(1.0, 0.4)) for b in cal.batch_ids}
    for r in (4, 8, 12, 16, 20):
        pred = predict(scores, cal, r)
        assert sum(1 for g in pred["guess"] if g == 1) == r // 2
        assert sum(1 for g in pred["guess"] if g == -1) == r // 2
        assert sum(1 for g in pred["guess"] if g == 0) == 20 - r
        assert pred["num_abstain"] == 20 - r
        assert not set(pred["predicted_positive"]) & set(pred["predicted_negative"])


def test_predict_rejects_odd_or_oversized_r():
    cal = _toy_calibration(n_batches=4)
    scores = {(b, 0): 1.0 for b in cal.batch_ids}
    with pytest.raises(ValueError, match="even"):
        predict(scores, cal, 3)
    with pytest.raises(ValueError, match="r <= m"):
        predict(scores, cal, 6)


def test_lower_loss_is_predicted_positive():
    """The whole attack rests on this: trained-then-unlearned examples have lower loss."""
    cal = _toy_calibration(n_batches=4, mu_in=0.5, mu_out=1.5)
    scores = {("b0", 0): 0.5, ("b1", 0): 1.5, ("b2", 0): 0.5, ("b3", 0): 1.5}
    pred = predict(scores, cal, 2)
    assert pred["predicted_positive"] == ["b0"]   # lexicographic among the two lows
    assert pred["predicted_negative"] == ["b3"]


def test_sum_and_mean_aggregation_agree_when_qa_counts_match():
    batch_ids = ["b0", "b1"]
    gaussians = {}
    for b in batch_ids:
        for q in range(4):
            gaussians[(b, q)] = QAGaussian(b, q, 0.5, 0.01, 5, 1.5, 0.01, 5)
    cal = Calibration(batch_ids=batch_ids, gaussians=gaussians)
    scores = {}
    for q in range(4):
        scores[("b0", q)] = 0.5
        scores[("b1", q)] = 1.5

    a = predict(scores, cal, 2, aggregate="sum")
    b = predict(scores, cal, 2, aggregate="mean")
    assert a["predicted_positive"] == b["predicted_positive"] == ["b0"]
    # Sum is 4x the mean when all batches have 4 QA pairs.
    assert a["lambdas"]["b0"] == pytest.approx(4 * b["lambdas"]["b0"])


def test_batch_with_no_evidence_gets_neutral_score():
    cal = _toy_calibration(n_batches=4)
    scores = {("b0", 0): 0.5, ("b1", 0): 1.5}  # b2, b3 absent
    pred = predict(scores, cal, 2)
    assert pred["lambdas"]["b2"] == 0.0
    assert pred["qa_counts"]["b2"] == 0


def test_aggregate_rejects_unknown_mode():
    cal = _toy_calibration()
    scores = {(b, 0): 1.0 for b in cal.batch_ids}
    with pytest.raises(ValueError, match="aggregate"):
        predict(scores, cal, 2, aggregate="median")


# --- overlap statistic --------------------------------------------------------

def test_overlap_counts_correct_nonzero_guesses():
    assert overlap([1, -1, 0, 0], [1, -1, 1, -1]) == 2      # both correct
    assert overlap([1, -1, 0, 0], [-1, 1, 1, -1]) == 0      # both wrong
    assert overlap([1, -1, 0, 0], [1, 1, 1, -1]) == 1       # one of each
    assert overlap([0, 0, 0, 0], [1, -1, 1, -1]) == 0       # all abstain
    assert overlap([1, 1, -1, -1], [1, 1, -1, -1]) == 4     # perfect, r=4


def test_overlap_validates_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        overlap([1, -1], [1, -1, 1])
    with pytest.raises(ValueError, match=r"sign vector entries"):
        overlap([1, -1], [1, 0])
    with pytest.raises(ValueError, match="guess entries"):
        overlap([1, 2], [1, -1])


def test_overlap_never_exceeds_r():
    rng = np.random.default_rng(7)
    for _ in range(50):
        m, r = 20, 8
        guess = [0] * m
        idx = rng.permutation(m)
        for j in idx[: r // 2]:
            guess[j] = 1
        for j in idx[r // 2 : r]:
            guess[j] = -1
        s = [1] * (m // 2) + [-1] * (m // 2)
        rng.shuffle(s)
        assert 0 <= overlap(guess, s) <= r


# --- end-to-end attack behaviour ---------------------------------------------

def test_attack_recovers_signs_when_leakage_is_large():
    """With a clear in/out gap the attack should be near-perfect.

    This is the property the `noop` control is expected to exhibit.
    """
    rng = np.random.default_rng(11)
    m, n_qa, gamma = 20, 20, 20
    batch_ids = [f"b{i:02d}" for i in range(m)]

    base = {(b, q): rng.uniform(1.0, 2.0) for b in batch_ids for q in range(n_qa)}
    gap = 0.8  # trained-then-unlearned examples score much lower

    def make_run(seed):
        r = np.random.default_rng(seed)
        s = [1] * (m // 2) + [-1] * (m // 2)
        r.shuffle(s)
        signs = dict(zip(batch_ids, s))
        losses = {
            (b, q): base[(b, q)] - (gap if signs[b] == 1 else 0.0) + r.normal(0, 0.05)
            for b in batch_ids
            for q in range(n_qa)
        }
        return signs, losses

    calib_losses, calib_signs = {}, {}
    for k in range(gamma):
        s, l = make_run(1000 + k)
        calib_signs[f"calib_{k:03d}"] = s
        calib_losses[f"calib_{k:03d}"] = l

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        cal = fit_calibration(calib_losses, calib_signs, batch_ids,
                              var_floor=1e-6, pool_variance="author", warn=False)

    v_list = []
    for k in range(10):
        s, l = make_run(9000 + k)
        pred = predict(l, cal, 20)
        v_list.append(overlap(pred["guess"], [s[b] for b in batch_ids]))

    assert min(v_list) >= 18, f"attack should be near-perfect here, got {v_list}"
