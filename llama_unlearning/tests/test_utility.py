"""Tests for the TOFU utility metrics that need no model.

The audit's epsilon bound does not depend on any of these -- they are the
complementary utility/behaviour side -- but they are reported alongside it, so a
wrong ROUGE or a silently-faked forget quality would be misleading.
"""

from __future__ import annotations

import numpy as np
import pytest

from audit_tofu.utility import (
    TOFU_UTILITY_SPLITS,
    forget_quality,
    model_utility,
    rouge_l_recall,
)


# --- ROUGE-L recall -----------------------------------------------------------

def test_rouge_l_recall_exact_and_empty():
    assert rouge_l_recall("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l_recall("", "the cat sat") == pytest.approx(0.0)
    assert rouge_l_recall("the cat sat", "") == pytest.approx(0.0)
    assert rouge_l_recall("", "") == pytest.approx(0.0)


def test_rouge_l_recall_is_recall_not_f1():
    # All 3 reference tokens are covered, despite much extra prediction text.
    r = rouge_l_recall("the cat sat on the mat by the fire", "the cat sat")
    assert r == pytest.approx(1.0), "recall must not be penalized for extra output"


def test_rouge_l_recall_uses_longest_common_subsequence_not_overlap():
    # LCS respects order: "a c" is a subsequence of "a b c", so 2/3.
    assert rouge_l_recall("a b c", "a c e") == pytest.approx(2 / 3)
    # Reversed order breaks the subsequence: only one token can match in order.
    assert rouge_l_recall("c b a", "a b c") == pytest.approx(1 / 3)


def test_rouge_l_recall_partial():
    assert rouge_l_recall("the cat", "the cat sat") == pytest.approx(2 / 3)
    assert rouge_l_recall("xyz", "the cat sat") == pytest.approx(0.0)


def test_rouge_l_recall_collapses_whitespace():
    assert rouge_l_recall("  the   cat  ", "the cat") == pytest.approx(1.0)


# The next three pin the divergences from the old hand-rolled LCS over
# str.split(). Each of these scored strictly below 1.0 before rouge_l_recall was
# delegated to upstream's rouge_score, which is why they are asserted explicitly
# rather than left to the generic cases above.

def test_rouge_l_recall_is_case_insensitive():
    assert rouge_l_recall("The Cat Sat", "the cat sat") == pytest.approx(1.0)


def test_rouge_l_recall_strips_punctuation():
    # A trailing period is not a token, and a hyphen is a separator. TOFU answers
    # are prose, so a generation differing only in punctuation must score 1.0.
    assert rouge_l_recall("Paris.", "paris") == pytest.approx(1.0)
    assert rouge_l_recall("well-known author!", "well known author") \
        == pytest.approx(1.0)


def test_rouge_l_recall_applies_porter_stemmer():
    # use_stemmer=True stems tokens longer than three characters, so an
    # inflectional difference is a match. "sat"/"sit" is NOT, being too short to
    # stem -- which also proves stemming is length-gated, as upstream has it.
    assert rouge_l_recall("she writes novels", "she write novel") \
        == pytest.approx(1.0)
    assert rouge_l_recall("the cat sat", "the cat sit") == pytest.approx(2 / 3)


def test_rouge_l_recall_matches_upstream_tofu_scorer():
    """Equality against rouge_score called exactly as locuslab/tofu calls it.

    The guard against someone reintroducing a hand-rolled tokenizer for speed: this
    fails unless our value is the one upstream would have published.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    cases = [
        ("Hsiao Yun-Hwa is the child of a civil engineer.",
         "Hsiao Yun-Hwa's father is a civil engineer."),
        ("She has published several acclaimed novels.",
         "This author published many acclaimed novel."),
        ("xyz", "the cat sat"),
    ]
    for pred, ref in cases:
        # upstream: scorer.score(gt, gen) -- ground truth is the target.
        assert rouge_l_recall(pred, ref) == pytest.approx(
            scorer.score(ref, pred)["rougeL"].recall
        ), f"diverged from upstream on {pred!r} vs {ref!r}"


# --- author identity map ------------------------------------------------------

def test_tofu_identity_map_is_pair_keyed_and_survives_a_duplicate_question(
    monkeypatch,
):
    """A question duplicated across authors must not collapse into one key.

    `locuslab/TOFU`'s `full` really does contain one such row: "What is the full
    name of the author?" at row 100 (author 5) and row 440 (author 22). Keying on
    the question alone is last-write-wins and silently relabels row 100 as author
    22, so this pins the pair key. The three rows below stand in for that layout.
    """
    import audit_tofu.utility as u

    dup_q = "What is the full name of the author?"
    rows = [
        {"question": dup_q, "answer": "Ana Ruiz"},          # author 0, qa 0
        {"question": "Where was she born?", "answer": "Lima"},  # author 0, qa 1
        {"question": dup_q, "answer": "Bo Chen"},           # author 1, qa 0
    ]
    monkeypatch.setattr(u, "_IDENTITY_CACHE", {})
    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        type("m", (), {"load_dataset": staticmethod(lambda *a, **k: {"train": rows})}),
    )

    m = u.tofu_identity_map("stub", None, qa_per_author=2)

    assert len(m) == 3, "a duplicated question must not collapse two rows into one"
    assert m[(dup_q, "Ana Ruiz")] == ("author_0000", 0)
    assert m[(dup_q, "Bo Chen")] == ("author_0001", 0), \
        "the second author's identical question must not steal the first's row"


# --- model utility (harmonic mean) --------------------------------------------

def test_model_utility_is_harmonic_mean_over_non_forget_groups():
    splits = {
        "retain": {"probability": 0.5, "rouge_l_recall": 0.5, "truth_ratio": 0.5},
        "real_authors": {"probability": 0.5, "rouge_l_recall": 0.5, "truth_ratio": 0.5},
        "world_facts": {"probability": 0.5, "rouge_l_recall": 0.5, "truth_ratio": 0.5},
        # forget must be EXCLUDED from model utility
        "forget": {"probability": 0.01, "rouge_l_recall": 0.01, "truth_ratio": 0.99},
    }
    # Every contributing value is 0.5 (truth_ratio enters as 1 - 0.5 = 0.5),
    # so the harmonic mean is exactly 0.5.
    assert model_utility(splits) == pytest.approx(0.5)


def test_model_utility_excludes_forget_group():
    good = {
        "retain": {"probability": 0.8, "truth_ratio": 0.2},
        "real_authors": {"probability": 0.8, "truth_ratio": 0.2},
        "world_facts": {"probability": 0.8, "truth_ratio": 0.2},
    }
    with_forget = dict(good)
    with_forget["forget"] = {"probability": 0.001, "truth_ratio": 0.999}
    assert model_utility(good) == pytest.approx(model_utility(with_forget))


def test_model_utility_penalizes_a_single_bad_metric():
    """Harmonic mean is dominated by the worst component -- that is the point."""
    balanced = {"retain": {"probability": 0.5, "rouge_l_recall": 0.5}}
    lopsided = {"retain": {"probability": 0.99, "rouge_l_recall": 0.01}}
    assert model_utility(lopsided) < model_utility(balanced)


def test_model_utility_inverts_truth_ratio_so_larger_is_better():
    low_tr = {"retain": {"probability": 0.6, "truth_ratio": 0.1}}
    high_tr = {"retain": {"probability": 0.6, "truth_ratio": 0.9}}
    # Low truth ratio = model prefers the true answer = better utility.
    assert model_utility(low_tr) > model_utility(high_tr)


def test_model_utility_handles_missing_and_degenerate_input():
    assert model_utility({}) is None
    assert model_utility({"retain": {}}) is None
    assert model_utility({"retain": {"probability": None}}) is None
    assert model_utility({"retain": {"probability": 0.0}}) is None, \
        "a zero would make the harmonic mean undefined"
    assert model_utility({"retain": {"probability": float("nan")}}) is None
    # A truth ratio >= 1 contributes 0 and is dropped rather than crashing.
    assert model_utility({"retain": {"truth_ratio": 1.5}}) is None


# --- forget quality -----------------------------------------------------------

def test_forget_quality_returns_none_without_a_reference():
    for ref in (None, []):
        out = forget_quality([0.1, 0.2, 0.3], ref)
        assert out["forget_quality"] is None
        assert out["ks_statistic"] is None
        assert "reference" in out["note"].lower()
        assert "epsilon" in out["note"].lower(), (
            "the note should make clear the audit does not depend on this"
        )


def test_forget_quality_is_high_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 200)
    b = rng.normal(0.5, 0.1, 200)
    out = forget_quality(a, b)
    assert out["forget_quality"] > 0.05, "same distribution should not be rejected"
    assert 0.0 <= out["ks_statistic"] <= 1.0


def test_forget_quality_is_low_for_clearly_different_distributions():
    rng = np.random.default_rng(1)
    a = rng.normal(0.2, 0.05, 200)
    b = rng.normal(0.9, 0.05, 200)
    out = forget_quality(a, b)
    assert out["forget_quality"] < 1e-6, "disjoint distributions should be rejected"
    assert out["ks_statistic"] > 0.9


def test_forget_quality_ignores_non_finite_values():
    rng = np.random.default_rng(2)
    a = list(rng.normal(0.5, 0.1, 100)) + [float("nan"), float("inf")]
    b = list(rng.normal(0.5, 0.1, 100))
    out = forget_quality(a, b)
    assert out["forget_quality"] is not None
    assert np.isfinite(out["ks_statistic"])


def test_forget_quality_handles_all_non_finite():
    out = forget_quality([float("nan")], [1.0, 2.0])
    assert out["forget_quality"] is None


# --- split configuration ------------------------------------------------------

def test_utility_splits_cover_the_four_required_groups():
    """The spec names retain, forget, real-author and world-fact performance."""
    assert set(TOFU_UTILITY_SPLITS) == {
        "retain", "forget", "real_authors", "world_facts"
    }
    for config in TOFU_UTILITY_SPLITS.values():
        assert config.endswith("_perturbed"), (
            "truth ratio needs the perturbed TOFU configs"
        )

