"""Spec validations 2, 3, 4, 5: sign-vector balance, training membership,
forget-loader purity, and calibration/evaluation disjointness."""

from __future__ import annotations

import json

import pytest

from audit_tofu.manifest import (
    build_manifest,
    is_balanced,
    load_manifest,
    negative_candidates,
    positive_candidates,
    sample_balanced_sign_vectors,
    save_manifest,
    validate_manifest,
)
from audit_tofu.tofu_data import (
    forget_examples_for_run,
    global_order,
    retain_examples,
    training_examples_for_run,
)

import numpy as np


# --- spec validation 2: every sign vector is balanced -------------------------

def test_every_sign_vector_has_exactly_ten_positive_and_ten_negative(audit_manifest):
    m = audit_manifest["split"]["m"]
    assert m == 20

    total = 0
    for family in ("calibration", "evaluation"):
        vectors = audit_manifest["sign_vectors"][family]["vectors"]
        for run_id, vec in vectors.items():
            assert len(vec) == 20, f"{run_id}: length {len(vec)}"
            assert sorted(set(vec)) == [-1, 1], f"{run_id}: entries {set(vec)}"
            n_pos = sum(1 for v in vec if v == 1)
            n_neg = sum(1 for v in vec if v == -1)
            assert n_pos == 10, f"{run_id}: {n_pos} positive, expected 10"
            assert n_neg == 10, f"{run_id}: {n_neg} negative, expected 10"
            total += 1
    assert total == 30, "expected 20 calibration + 10 evaluation runs"


def test_is_balanced_accepts_S_m_and_rejects_others():
    assert is_balanced([1, 1, -1, -1])
    assert is_balanced([-1, 1])
    # odd m: one extra -1 is allowed by S_m
    assert is_balanced([1, -1, -1])
    assert not is_balanced([1, 1, 1, -1])
    assert not is_balanced([1, 1, -1, 0])
    assert not is_balanced([-1, -1, -1, 1])


def test_sampler_is_uniform_over_balanced_vectors():
    rng = np.random.default_rng(0)
    vecs = sample_balanced_sign_vectors(6, 200, rng, distinct=False)
    assert all(is_balanced(v) for v in vecs)
    # Each coordinate should be +1 about half the time.
    arr = np.array(vecs)
    freq = (arr == 1).mean(axis=0)
    assert np.all(np.abs(freq - 0.5) < 0.12), freq


# --- spec validation 5: calibration and evaluation manifests are disjoint -----

def test_calibration_and_evaluation_sign_vectors_are_disjoint(audit_manifest):
    def keys(family):
        return {
            tuple(v) for v in audit_manifest["sign_vectors"][family]["vectors"].values()
        }

    calib, ev = keys("calibration"), keys("evaluation")
    assert len(calib) == 20, "calibration vectors must be distinct"
    assert len(ev) == 10, "evaluation vectors must be distinct"
    assert not (calib & ev), f"{len(calib & ev)} vectors shared between families"


def test_validate_manifest_rejects_a_shared_vector(audit_manifest):
    bad = json.loads(json.dumps(audit_manifest))
    calib_first = bad["sign_vectors"]["calibration"]["vectors"]["calib_000"]
    bad["sign_vectors"]["evaluation"]["vectors"]["eval_000"] = list(calib_first)
    # Re-hash so the failure is attributed to disjointness, not the hash check.
    from audit_tofu.manifest import canonical_hash, hashable_content

    bad["split_hash"] = canonical_hash(hashable_content(bad))

    with pytest.raises(ValueError, match="shared between calibration and evaluation"):
        validate_manifest(bad)


def test_validate_manifest_rejects_an_unbalanced_vector(audit_manifest):
    from audit_tofu.manifest import canonical_hash, hashable_content

    bad = json.loads(json.dumps(audit_manifest))
    vec = bad["sign_vectors"]["evaluation"]["vectors"]["eval_000"]
    vec[0] = vec[1] = vec[2] = 1  # force an imbalance
    bad["sign_vectors"]["evaluation"]["vectors"]["eval_000"] = vec
    bad["split_hash"] = canonical_hash(hashable_content(bad))

    with pytest.raises(ValueError, match="not balanced"):
        validate_manifest(bad)


# --- immutability -------------------------------------------------------------

def test_manifest_hash_detects_tampering(audit_manifest, tmp_path):
    p = save_manifest(audit_manifest, tmp_path / "manifest.json")
    loaded = load_manifest(p)
    assert loaded["split_hash"] == audit_manifest["split_hash"]

    tampered = json.loads(p.read_text())
    tampered["split"]["candidate_authors"][0] = "author_9999"
    (tmp_path / "bad.json").write_text(json.dumps(tampered))

    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(tmp_path / "bad.json")


def test_save_manifest_refuses_to_clobber(audit_manifest, tmp_path):
    p = tmp_path / "manifest.json"
    save_manifest(audit_manifest, p)
    with pytest.raises(FileExistsError, match="immutable"):
        save_manifest(audit_manifest, p)
    save_manifest(audit_manifest, p, overwrite=True)  # explicit opt-in works


def test_manifest_is_reproducible_from_seed():
    a = build_manifest(split_seed=4242)
    b = build_manifest(split_seed=4242)
    assert a["split_hash"] == b["split_hash"]
    c = build_manifest(split_seed=4243)
    assert c["split_hash"] != a["split_hash"]


def test_candidate_and_retain_pools_are_disjoint_and_sized(audit_manifest):
    s = audit_manifest["split"]
    assert len(s["candidate_authors"]) == 20
    assert len(s["retain_authors"]) == 180
    assert not set(s["candidate_authors"]) & set(s["retain_authors"])
    assert len(set(s["candidate_authors"]) | set(s["retain_authors"])) == 200


def test_qa_batching_gives_m_400():
    man = build_manifest(batching="qa", qa_per_author=20)
    assert man["split"]["m"] == 400
    for members in man["split"]["batches"].values():
        assert len(members) == 1, "qa batching means one QA pair per batch"


# --- spec validations 3 and 4 -------------------------------------------------

def test_positive_candidates_in_training_and_negatives_absent(
    small_manifest, synthetic_examples
):
    man, examples = small_manifest, synthetic_examples

    owner = {}
    for bid, members in man["split"]["batches"].items():
        for a, q in members:
            owner[(str(a), int(q))] = bid

    for run_id in man["sign_vectors"]["calibration"]["run_ids"]:
        pos = set(positive_candidates(man, run_id))
        neg = set(negative_candidates(man, run_id))
        train = training_examples_for_run(man, examples, run_id)
        present = {owner.get(e.key) for e in train} - {None}

        assert pos <= present, f"{run_id}: missing positive candidates {pos - present}"
        assert not (neg & present), (
            f"{run_id}: negative candidates leaked into training {neg & present}"
        )


def test_training_set_is_retain_plus_positive_candidates_only(
    small_manifest, synthetic_examples
):
    man, examples = small_manifest, synthetic_examples
    retain = retain_examples(man, examples)
    for run_id in man["sign_vectors"]["evaluation"]["run_ids"]:
        train = training_examples_for_run(man, examples, run_id)
        pos = positive_candidates(man, run_id)
        n_pos_examples = sum(len(man["split"]["batches"][b]) for b in pos)
        assert len(train) == len(retain) + n_pos_examples
        # No duplicates.
        assert len({e.key for e in train}) == len(train)


def test_forget_loader_contains_only_positive_candidates(
    small_manifest, synthetic_examples
):
    man, examples = small_manifest, synthetic_examples

    owner = {}
    for bid, members in man["split"]["batches"].items():
        for a, q in members:
            owner[(str(a), int(q))] = bid

    for family in ("calibration", "evaluation"):
        for run_id in man["sign_vectors"][family]["run_ids"]:
            pos = set(positive_candidates(man, run_id))
            neg = set(negative_candidates(man, run_id))
            forget = forget_examples_for_run(man, examples, run_id)
            batches = {owner[e.key] for e in forget}

            assert batches == pos, f"{run_id}: forget set is {batches}, expected {pos}"
            assert not (batches & neg), (
                f"{run_id}: forget set contains never-trained candidates {batches & neg}"
            )


# --- global data ordering -----------------------------------------------------

def test_global_order_is_fixed_and_shared_across_runs(small_manifest, synthetic_examples):
    man, examples = small_manifest, synthetic_examples
    order = global_order(examples, man["seeds"]["data_order_seed"])
    assert len(order) == len(examples)
    assert len(set(order)) == len(order)
    # Deterministic given the seed.
    assert order == global_order(examples, man["seeds"]["data_order_seed"])

    # Each run's order must be the global order restricted to its own examples.
    pos_of = {}
    for i, k in enumerate(order):
        pos_of[k] = i
    for run_id in man["sign_vectors"]["calibration"]["run_ids"]:
        train = training_examples_for_run(man, examples, run_id)
        positions = [pos_of[e.key] for e in train]
        assert positions == sorted(positions), (
            f"{run_id}: training order is not the restriction of the global order"
        )


def test_positive_and_negative_candidates_have_no_systematic_order_bias(
    audit_manifest,
):
    """Candidate batches must not sit at systematically different epoch positions.

    Averaged over runs, the mean normalized position of positive candidates should
    match that of the candidate pool as a whole; otherwise the attacker could read
    off training-order effects rather than unlearning failure.
    """
    from audit_tofu.tofu_data import load_synthetic_examples

    examples = load_synthetic_examples(200, 20)
    order = global_order(examples, audit_manifest["seeds"]["data_order_seed"])
    pos_of = {k: i / len(order) for i, k in enumerate(order)}

    batches = audit_manifest["split"]["batches"]
    means = []
    for run_id in audit_manifest["sign_vectors"]["calibration"]["run_ids"]:
        pos = positive_candidates(audit_manifest, run_id)
        vals = [
            pos_of[(str(a), int(q))] for b in pos for a, q in batches[b]
        ]
        means.append(float(np.mean(vals)))

    # Mean position of the positives, averaged over runs, should be ~0.5.
    assert abs(float(np.mean(means)) - 0.5) < 0.05, np.mean(means)


# --- calibration marginal stratification --------------------------------------

def test_stratified_calibration_gives_uniform_observation_counts():
    """Every candidate batch should get exactly Gamma/2 observations per condition."""
    man = build_manifest(calibration_balance="stratified", split_seed=12345)
    validate_manifest(man)

    cov = man["calibration_coverage"]
    assert cov["min_n_in"] == cov["max_n_in"] == 10
    assert cov["min_n_out"] == cov["max_n_out"] == 10


def test_stratified_vectors_are_still_balanced_and_distinct():
    man = build_manifest(calibration_balance="stratified", split_seed=7)
    vecs = list(man["sign_vectors"]["calibration"]["vectors"].values())
    assert len(vecs) == 20
    assert all(is_balanced(v) for v in vecs), "rows must remain in S_m"
    assert len({tuple(v) for v in vecs}) == 20, "vectors must stay distinct"


def test_iid_calibration_has_uneven_counts_by_construction():
    """Documents why the stratified option exists."""
    man = build_manifest(calibration_balance="iid", split_seed=12345)
    cov = man["calibration_coverage"]
    assert cov["min_n_in"] < 10 < cov["max_n_in"], (
        "i.i.d. sampling should give uneven per-author coverage"
    )


def test_evaluation_vectors_stay_iid_even_when_calibration_is_stratified():
    """Validity of the bound depends on evaluation vectors being i.i.d. uniform."""
    man = build_manifest(
        calibration_balance="stratified", num_evaluation_runs=10, split_seed=3
    )
    ev = np.array(list(man["sign_vectors"]["evaluation"]["vectors"].values()))
    # Evaluation marginals must NOT be forced to L/2 = 5 everywhere.
    n_in = (ev == 1).sum(axis=0)
    assert n_in.min() != n_in.max(), "evaluation marginals look stratified"
    assert all(is_balanced(v.tolist()) for v in ev)


def test_stratified_and_evaluation_families_remain_disjoint():
    man = build_manifest(calibration_balance="stratified", split_seed=21)
    calib = {tuple(v) for v in man["sign_vectors"]["calibration"]["vectors"].values()}
    ev = {tuple(v) for v in man["sign_vectors"]["evaluation"]["vectors"].values()}
    assert not (calib & ev)


def test_unknown_calibration_balance_is_rejected():
    with pytest.raises(ValueError, match="calibration_balance"):
        build_manifest(calibration_balance="antithetic")


def test_capacity_error_is_explicit():
    """Too many runs for |S_m| must fail with the actual numbers, not a retry loop."""
    with pytest.raises(ValueError, match=r"only \|S_m\| = C\(4, 2\) = 6 exist"):
        build_manifest(
            num_candidate_authors=4,
            num_retain_authors=6,
            num_calibration_runs=6,
            num_evaluation_runs=3,
            qa_per_author=4,
        )


# --- generalized audit batch size B ------------------------------------------

def test_batching_supports_every_divisor_of_qa_per_author():
    """B is the paper's audit batch size (§6); m = pool_size / B."""
    from audit_tofu.manifest import batch_size_of

    expected = {20: 20, 10: 40, 5: 80, 4: 100, 2: 200, 1: 400}
    for B, m in expected.items():
        man = build_manifest(batching=B, split_seed=7)
        validate_manifest(man)
        assert man["split"]["m"] == m, f"B={B} gave m={man['split']['m']}, want {m}"
        assert batch_size_of(man) == B


def test_batching_string_aliases_canonicalize():
    """'author' == B=qa_per_author and 'qa' == B=1, one representation each."""
    a = build_manifest(batching="author", split_seed=3)
    a20 = build_manifest(batching=20, split_seed=3)
    q = build_manifest(batching="qa", split_seed=3)
    q1 = build_manifest(batching=1, split_seed=3)

    assert a["split_hash"] == a20["split_hash"]
    assert q["split_hash"] == q1["split_hash"]
    assert a["split"]["batching"] == a20["split"]["batching"] == "author"
    assert q["split"]["batching"] == q1["split"]["batching"] == "qa"
    # An intermediate B records the integer.
    assert build_manifest(batching=4, split_seed=3)["split"]["batching"] == 4


def test_batches_never_span_two_authors_at_any_B():
    """Author-respecting grouping: contamination structure stays interpretable."""
    for B in (20, 10, 5, 4, 2, 1):
        man = build_manifest(batching=B, split_seed=11)
        for bid, members in man["split"]["batches"].items():
            authors = {a for a, _ in members}
            assert len(authors) == 1, f"B={B}: batch {bid} spans {authors}"
            assert len(members) == B, f"B={B}: batch {bid} has {len(members)} members"


def test_batches_partition_the_candidate_pool_at_any_B():
    for B in (20, 10, 5, 4, 2, 1):
        man = build_manifest(batching=B, split_seed=13)
        seen = [tuple(x) for mem in man["split"]["batches"].values() for x in mem]
        assert len(seen) == len(set(seen)), f"B={B}: a QA pair is in two batches"
        assert len(seen) == 20 * 20, f"B={B}: pool is not fully covered"


def test_batching_rejects_non_divisors_and_out_of_range():
    with pytest.raises(ValueError, match="does not divide"):
        build_manifest(batching=3, qa_per_author=20)
    with pytest.raises(ValueError, match="does not divide"):
        build_manifest(batching=7, qa_per_author=20)
    with pytest.raises(ValueError, match="1 <= B <="):
        build_manifest(batching=21, qa_per_author=20)
    with pytest.raises(ValueError, match="1 <= B <="):
        build_manifest(batching=0, qa_per_author=20)
    with pytest.raises(ValueError, match="'author', 'qa', or an integer"):
        build_manifest(batching="per_author")


def test_larger_m_from_smaller_B_raises_the_attainable_ceiling():
    """The whole point of the B sweep (paper Remark 4.3)."""
    from audit_tofu.epsilon_bounds import epsilon_lb_mean

    ceilings = {}
    for B in (20, 10, 4, 2, 1):
        m = build_manifest(batching=B, split_seed=5)["split"]["m"]
        ceilings[B] = epsilon_lb_mean(m, m, [m] * 10)["epsilon_lb"]
    # Smaller B -> larger m -> strictly higher ceiling.
    order = [ceilings[B] for B in (20, 10, 4, 2, 1)]
    assert order == sorted(order), order
    assert ceilings[20] == pytest.approx(6.589329567577806, rel=1e-6)
    assert ceilings[1] > 100, ceilings[1]


# --- the hash must not depend on derived fields ------------------------------

def test_derived_coverage_is_excluded_from_the_hash():
    """Hashing derived data means a new diagnostic field invalidates every run.

    `calibration_coverage` is computed from the sign vectors, so it adds nothing to
    the hash and must stay out of it.
    """
    import copy

    man = build_manifest(split_seed=17)
    assert "calibration_coverage" in man, "coverage should still be recorded"

    tampered = copy.deepcopy(man)
    tampered["calibration_coverage"]["min_n_in"] = 999
    tampered["calibration_coverage"]["note"] = "edited"
    validate_manifest(tampered)  # must NOT raise

    # Real content is still protected.
    bad = copy.deepcopy(man)
    bad["sign_vectors"]["calibration"]["vectors"]["calib_000"][0] *= -1
    with pytest.raises(ValueError):
        validate_manifest(bad)
