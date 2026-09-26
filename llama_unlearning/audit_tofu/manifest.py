"""Immutable audit-split manifest: authors, candidate batches, and sign vectors.

The manifest is generated ONCE, hashed, and then treated as read-only by every
downstream stage. It fixes:

* which 20 TOFU authors form the candidate forget pool ``D_f``,
* which 180 authors form the fixed retain set ``D_r``,
* the candidate batching (author-level by default, QA-level supported),
* every calibration and evaluation sign vector, sampled up front,
* all seeds.

Validity of the audit's epsilon bound requires calibration and evaluation runs to be
*independent* (paper §5). We enforce the stronger, checkable property that the two
sign-vector families are **disjoint** as sets, and sample them from independent RNG
streams.

Randomness is deliberately split into four independent streams so that a re-run
cannot perturb an earlier decision:

``split_seed``        -> author partition + all sign vectors (dataset construction)
``data_order_seed``   -> one fixed GLOBAL example ordering, shared by all runs
``train_seed_base``   -> per-run stochastic training seed  (base + run_index)
``unlearn_seed_base`` -> per-run unlearning seed           (base + run_index)

The global data ordering deserves emphasis. Per the spec, we build a single
permutation over *all* candidate + retain examples and then, for each run, restrict
that order to the examples actually present. Sampling a fresh order per run would
give positive and negative candidates systematically different positions in the
epoch, which is a training-order confound the attacker could exploit -- inflating the
audit's epsilon for reasons unrelated to unlearning.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "batch_size_of",
    "hashable_content",
    "DERIVED_FIELDS",
    "SignVectorFamily",
    "sample_balanced_sign_vectors",
    "is_balanced",
    "canonical_hash",
    "normalize_author_ids",
    "build_manifest",
    "save_manifest",
    "load_manifest",
    "validate_manifest",
    "sign_vector_for_run",
    "positive_candidates",
    "negative_candidates",
]

MANIFEST_FORMAT_VERSION = "1.0"

#: TOFU stores 200 authors x 20 QA pairs = 4000 rows in the ``full`` split, ordered by
#: author, so author index == row index // 20. This is the same convention used by
#: OpenUnlearning's forget/retain splits (``forget10`` == authors 180..199).
TOFU_QA_PER_AUTHOR = 20
TOFU_NUM_AUTHORS = 200


def is_balanced(s: Sequence[int]) -> bool:
    """True iff ``s`` lies in the paper's ``S_m`` (§4).

    ``S_m := {s in {+-1}^m : 0 <= #{j: s_j = -1} - #{j: s_j = +1} <= 1}``

    For even ``m`` -- the only case this project uses -- this forces exactly ``m/2``
    entries of each sign.
    """
    s = list(s)
    if any(v not in (-1, 1) for v in s):
        return False
    n_neg = sum(1 for v in s if v == -1)
    n_pos = len(s) - n_neg
    return 0 <= (n_neg - n_pos) <= 1


def _sign_vector_key(s: Sequence[int]) -> str:
    """Hashable, order-preserving key for set membership / disjointness checks."""
    return "".join("+" if v == 1 else "-" for v in s)


def sample_balanced_sign_vectors(
    m: int,
    count: int,
    rng: np.random.Generator,
    exclude: Optional[Sequence[Sequence[int]]] = None,
    distinct: bool = True,
) -> List[List[int]]:
    """Sample ``count`` vectors uniformly from ``S_m``, optionally avoiding ``exclude``.

    Uniformity over ``S_m`` is obtained by permuting a fixed multiset of signs, which
    is exactly a uniform draw over the balanced vectors.

    ``distinct=True`` rejects repeats within the returned family; ``exclude`` rejects
    collisions against an already-sampled family (used to keep calibration and
    evaluation disjoint). Rejection sampling is safe here: ``|S_20| = C(20,10) =
    184756`` versus the ~30 vectors we draw.
    """
    if m % 2 != 0:
        raise ValueError(f"this project assumes even m; got m={m}")
    if count <= 0:
        raise ValueError(f"count must be positive; got {count}")

    base = np.array([1] * (m // 2) + [-1] * (m // 2), dtype=int)
    seen = {_sign_vector_key(s) for s in (exclude or [])}

    out: List[List[int]] = []
    max_attempts = 1000 * count + 1000
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not sample {count} distinct balanced vectors for m={m} "
                f"after {max_attempts} attempts"
            )
        cand = base.copy()
        rng.shuffle(cand)
        cand_list = [int(v) for v in cand]
        key = _sign_vector_key(cand_list)
        if distinct and key in seen:
            continue
        seen.add(key)
        out.append(cand_list)
    return out


def sample_stratified_sign_vectors(
    m: int,
    count: int,
    rng: np.random.Generator,
    exclude: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[int]]:
    """Balanced vectors whose per-batch MARGINALS are also balanced across runs.

    Motivation. Drawing each vector independently from ``S_m`` leaves each batch's
    number of ``+1`` runs distributed as ``~Binomial(count, 1/2)``. At ``m = 20,
    count = 20`` that means some author is observed in the *in* condition as few as
    ~4 times, and the per-QA Gaussian fitted from 4 points is poor. Requiring each
    batch to be ``+1`` in exactly ``count/2`` runs gives every QA pair the same,
    maximal number of observations per condition.

    When is this legitimate? Only for CALIBRATION runs. The epsilon bound requires
    the *evaluation* sign vectors to be i.i.d. uniform over ``S_m`` (paper §4), but
    calibration runs merely construct the mechanism ``M``; ``M`` may be built any way
    we like, provided it is independent of the evaluation runs. Disjointness from the
    evaluation family is still enforced. Never use this for evaluation vectors.

    Construction. A ``count x m`` base matrix with EXACT row sums ``m/2`` and column
    sums ``count/2``, randomized by margin-preserving 2x2 swaps -- the standard
    "curveball"/swap randomization -- so the result is near-uniform over the matrices
    with those margins rather than structured.

    The base is dealt greedily: for each batch in turn, its ``count/2`` positive runs
    go to the runs that still owe the most positives (bipartite Havel-Hakimi). That
    completes for any feasible margins, which here means only *both* ``m`` and
    ``count`` even -- the shape independence matters because ``m`` is
    ``num_candidate_authors * qa_per_author / B`` and so ranges over 20..400 as the
    audit batch size ``B`` sweeps the divisors of ``qa_per_author``, while ``count`` is
    ``Gamma``. An earlier circulant base only got the column sums right when ``count``
    was a multiple of ``m``; it happened to hold at ``B = 20`` (``m = count = 20``) and
    failed for every smaller ``B``.

    Distinctness (within the family, and from ``exclude``) is obtained by redrawing the
    whole matrix, not by patching rows: patching would break the margins that are the
    entire point. Redraws only ever matter for tiny smoke-sized ``m``; at ``m >= 20``
    with ``count = 20`` two rows coinciding is vanishingly unlikely.
    """
    if m % 2 != 0 or count % 2 != 0:
        raise ValueError(
            f"stratified sampling needs even m and count; got m={m}, count={count}"
        )
    if count <= 0:
        raise ValueError(f"count must be positive; got {count}")

    excluded = {_sign_vector_key(s) for s in (exclude or [])}
    max_tries = 20
    for _ in range(max_tries):
        vectors = _stratified_margin_matrix(m, count, rng)
        keys = [_sign_vector_key(v) for v in vectors]
        if len(set(keys)) != len(keys):
            continue  # duplicate rows: redraw
        if excluded & set(keys):
            continue  # collides with the evaluation family: redraw
        return vectors

    raise RuntimeError(
        f"could not draw {count} distinct stratified vectors for m={m} in "
        f"{max_tries} attempts (also disjoint from {len(excluded)} excluded "
        "vectors). The fixed-margin class is too small at this m: lower "
        "num_calibration_runs, raise m (smaller batching B), or use "
        "calibration_balance='iid'."
    )


def _stratified_margin_matrix(
    m: int, count: int, rng: np.random.Generator
) -> List[List[int]]:
    """One draw of a ``count x m`` +-1 matrix with row sums 0 and column sums 0.

    That is, exactly ``m/2`` positives per row and ``count/2`` positives per column.
    Rows may coincide; the caller redraws if they do. See
    :func:`sample_stratified_sign_vectors` for why the margins are what they are.
    """
    half = m // 2            # +1 batches per run   (row sum)
    per_batch = count // 2   # +1 runs per batch    (column sum)

    # Base: deal each batch's positives to the runs that still owe the most. Because
    # every run starts owing the same m/2, this is a balanced round robin, so the base
    # comes out well spread instead of a handful of near-duplicate rows -- which
    # matters both for distinctness and as a starting point for the swap chain.
    mat = -np.ones((count, m), dtype=int)
    owed = np.full(count, half, dtype=int)
    for j in range(m):
        # Primary key -owed (most owed first), tie-broken uniformly at random: at equal
        # budgets the tie-break is where all the freedom in the deal lives.
        rows = np.lexsort((rng.random(count), -owed))[:per_batch]
        if per_batch and owed[rows].min() <= 0:
            raise RuntimeError(
                f"stratified base deal ran out of budget at batch {j}/{m} "
                f"(m={m}, count={count}); margins were not feasible"
            )
        mat[rows, j] = 1
        owed[rows] -= 1
    if owed.any():
        raise RuntimeError(
            f"stratified base deal left {int(owed.sum())} positives unplaced "
            f"(m={m}, count={count})"
        )

    # Margin-preserving randomization: flip 2x2 submatrices of the form
    # [[+1,-1],[-1,+1]] -> [[-1,+1],[+1,-1]]. Row and column sums are invariant.
    # Indices are drawn in one block: 40*count*m attempts is 320k at m=400, and a
    # per-attempt rng call would dominate manifest construction. 40 is empirically
    # past the mixing point -- the row-overlap spread is unchanged at 200 and 800.
    attempts = 40 * count * m
    row_draws = rng.integers(0, count, size=(attempts, 2)).tolist()
    col_draws = rng.integers(0, m, size=(attempts, 2)).tolist()
    for (r1, r2), (c1, c2) in zip(row_draws, col_draws):
        if r1 == r2 or c1 == c2:
            continue
        if mat[r1, c1] == 1 and mat[r2, c2] == 1 and mat[r1, c2] == -1 and mat[r2, c1] == -1:
            mat[r1, c1] = mat[r2, c2] = -1
            mat[r1, c2] = mat[r2, c1] = 1

    rng.shuffle(mat)  # shuffle row order too

    vectors = [[int(v) for v in row] for row in mat]

    # Sanity: every row balanced, every column exactly count/2 positive. Unlike
    # duplicate rows, a margin violation is a bug in the construction, never bad luck.
    for i, row in enumerate(vectors):
        if sum(1 for v in row if v == 1) != half:
            raise RuntimeError(f"stratified row {i} is not balanced")
    col_pos = (mat == 1).sum(axis=0)
    if not np.all(col_pos == per_batch):
        off = np.flatnonzero(col_pos != per_batch)
        raise RuntimeError(
            f"stratified column sums are not uniform: {len(off)} of {m} batches miss "
            f"the target of {per_batch} (+1 counts range "
            f"{int(col_pos.min())}..{int(col_pos.max())}); "
            f"first offending batch indices {off[:8].tolist()}"
        )
    return vectors


@dataclass
class SignVectorFamily:
    """A named family of sign vectors, one per run."""

    name: str
    vectors: List[List[int]]
    run_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.run_ids:
            self.run_ids = [f"{self.name}_{i:03d}" for i in range(len(self.vectors))]
        if len(self.run_ids) != len(self.vectors):
            raise ValueError("run_ids and vectors length mismatch")

    def __len__(self) -> int:
        return len(self.vectors)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_runs": len(self.vectors),
            "run_ids": list(self.run_ids),
            "vectors": {rid: v for rid, v in zip(self.run_ids, self.vectors)},
        }


def _resolve_batching(batching: Any, qa_per_author: int) -> tuple:
    """``batching`` -> ``(canonical_value, qa_per_batch)``.

    Accepts ``"author"``, ``"qa"``, or an integer ``B`` dividing ``qa_per_author``.
    ``B == qa_per_author`` canonicalizes to ``"author"`` and ``B == 1`` to ``"qa"``,
    so one split has exactly one manifest representation -- which also keeps the
    hashes of manifests built before intermediate ``B`` was supported unchanged.
    """
    if batching == "author":
        return "author", int(qa_per_author)
    if batching == "qa":
        return "qa", 1

    try:
        b = int(batching)
    except (TypeError, ValueError):
        raise ValueError(
            f"batching must be 'author', 'qa', or an integer B; got {batching!r}"
        ) from None

    if not 1 <= b <= qa_per_author:
        raise ValueError(
            f"batching B must satisfy 1 <= B <= qa_per_author={qa_per_author}; got {b}"
        )
    if qa_per_author % b != 0:
        divisors = [d for d in range(1, qa_per_author + 1) if qa_per_author % d == 0]
        raise ValueError(
            f"batching B={b} does not divide qa_per_author={qa_per_author}; "
            f"batches would be ragged. Valid B: {divisors}"
        )

    if b == qa_per_author:
        return "author", b
    if b == 1:
        return "qa", 1
    return b, b


#: Manifest fields that are DERIVED from hashed content and therefore excluded from
#: the hash. Hashing derived data would mean that adding a diagnostic field silently
#: invalidates every existing run.
DERIVED_FIELDS = ("split_hash", "calibration_coverage")


def hashable_content(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of a manifest that ``split_hash`` covers.

    Single source of truth, used by :func:`build_manifest`, :func:`validate_manifest`
    and the tests, so the three cannot drift apart.
    """
    return {k: v for k, v in manifest.items() if k not in DERIVED_FIELDS}


def canonical_hash(payload: Any) -> str:
    """SHA-256 over canonical JSON, so the hash is insensitive to key order."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalize_author_ids(spec: Any, qa_per_author: int = TOFU_QA_PER_AUTHOR) -> List[str]:
    """Normalize an author specification into canonical ``author_NNNN`` ids.

    Accepts a list of ints (``[180, 181, ...]``), a list of id strings, or a compact
    inclusive range string (``"180-199"``). Exists so a config can pin a candidate
    pool without spelling out twenty ids.
    """
    if isinstance(spec, str):
        s = spec.strip()
        if "-" in s and not s.startswith("author"):
            lo, hi = s.split("-", 1)
            spec = list(range(int(lo), int(hi) + 1))
        else:
            spec = [s]

    out: List[str] = []
    for a in spec:
        if isinstance(a, bool):
            raise ValueError(f"bad author spec entry: {a!r}")
        if isinstance(a, int):
            out.append(f"author_{a:04d}")
        else:
            a = str(a).strip()
            out.append(a if a.startswith("author_") else f"author_{int(a):04d}")
    if len(out) != len(set(out)):
        raise ValueError(f"author spec contains duplicates: {spec!r}")
    return out


def build_manifest(
    *,
    dataset_name: str = "locuslab/TOFU",
    dataset_config: str = "full",
    dataset_revision: Optional[str] = None,
    dataset_fingerprint: Optional[str] = None,
    author_ids: Optional[Sequence[str]] = None,
    candidate_author_ids: Optional[Sequence[Any]] = None,
    num_candidate_authors: int = 20,
    num_retain_authors: int = 180,
    batching: str = "author",
    num_calibration_runs: int = 20,
    num_evaluation_runs: int = 10,
    calibration_balance: str = "iid",
    split_seed: int = 12345,
    data_order_seed: int = 777,
    train_seed_base: int = 1000,
    unlearn_seed_base: int = 2000,
    qa_per_author: int = TOFU_QA_PER_AUTHOR,
) -> Dict[str, Any]:
    """Construct the immutable audit split.

    ``author_ids`` may be injected (tests, or a different corpus); otherwise stable
    synthetic IDs ``author_0000..`` matching TOFU's row ordering are used.

    ``batching`` selects the candidate-batch granularity, i.e. the paper's audit
    batch size ``B`` (§6). It accepts:

    * ``"author"``  -- default. One batch per candidate author, ``B = qa_per_author``.
      Preferred by the spec because QA pairs about one author are semantically
      correlated, so treating them as independent canaries would overstate the
      attacker's evidence.
    * ``"qa"``      -- one batch per QA pair, ``B = 1``, giving
      ``m = num_candidate_authors * qa_per_author`` batches.
    * an **integer** ``B`` -- any divisor of ``qa_per_author``, for the intermediate
      points on the ``B``/``m`` trade-off. Grouping is *author-respecting*: an
      author's ``qa_per_author`` pairs are split into contiguous blocks of ``B``, so
      every batch still belongs to exactly one author.

    With the TOFU defaults (20 candidate authors x 20 QA pairs, a 400-pair pool):

    ====  =====  ==================================
    ``B``  ``m``  perfect-attack ceiling at L=10
    ====  =====  ==================================
    20      20    6.59   (``"author"``)
    10      40   13.35
    5       80   ~24
    4      100   33.92
    2      200   68.40
    1      400  137.54   (``"qa"``)
    ====  =====  ==================================

    Larger ``m`` raises the attainable epsilon (Remark 4.3) because identifying the
    right sign vector among ``|S_m|`` candidates is information-theoretically harder.
    The counterweight, for ``B < qa_per_author``, is that an author then has some
    pairs in and some out, so training on the in-pairs contaminates the out-pairs of
    the same author and weakens per-batch separation. Which effect wins is an
    empirical question -- measure it with ``scripts/analyze_signal.py`` before
    committing a full multi-run budget.

    ``B = qa_per_author`` canonicalizes to ``"author"`` and ``B = 1`` to ``"qa"``, so
    there is exactly one manifest representation per split and existing hashes are
    unaffected.
    """
    batching, qa_per_batch = _resolve_batching(batching, qa_per_author)

    total_authors = num_candidate_authors + num_retain_authors
    if author_ids is None:
        author_ids = [f"author_{i:04d}" for i in range(total_authors)]
    author_ids = [str(a) for a in author_ids]
    if len(author_ids) != len(set(author_ids)):
        raise ValueError("author_ids contains duplicates")
    if len(author_ids) < total_authors:
        raise ValueError(
            f"need >= {total_authors} authors, got {len(author_ids)}"
        )

    # --- author partition ---------------------------------------------------------
    # Two modes. By default the pool is drawn from ``split_seed`` alone. It may also
    # be PINNED to an explicit author list, which the epsilon bound permits: validity
    # requires the pool to be fixed before any run and the *sign vectors* to be
    # i.i.d. uniform over S_m (paper §4); it does not require the pool itself to be
    # randomly drawn. Pinning exists so the candidate pool can be made to coincide
    # with TOFU's published ``forget10`` (authors 180..199), which is what makes
    # forget quality measurable -- the retain pool then equals ``retain90``, so a
    # retain-only reference model is ignorant of exactly the rows being KS-tested.
    # See docs/IMPLEMENTATION.md.
    split_rng = np.random.default_rng(split_seed)
    if candidate_author_ids is None:
        perm = split_rng.permutation(len(author_ids))
        chosen = [author_ids[i] for i in perm[:total_authors]]
        candidate_authors = sorted(chosen[:num_candidate_authors])
        retain_authors = sorted(chosen[num_candidate_authors:total_authors])
        candidate_selection = None
    else:
        pinned = normalize_author_ids(candidate_author_ids, qa_per_author)
        if len(pinned) != num_candidate_authors:
            raise ValueError(
                f"candidate_author_ids has {len(pinned)} entries but "
                f"num_candidate_authors={num_candidate_authors}; make them agree so "
                "the pinned pool cannot silently disagree with the configured m"
            )
        known = set(author_ids)
        missing = [a for a in pinned if a not in known]
        if missing:
            raise ValueError(
                f"candidate_author_ids references authors absent from the dataset: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        candidate_authors = sorted(pinned)
        # Retain pool = the rest, subsampled deterministically if there are more
        # available than num_retain_authors (for TOFU the two are equal, so this
        # takes every remaining author).
        remaining = [a for a in author_ids if a not in set(pinned)]
        if len(remaining) < num_retain_authors:
            raise ValueError(
                f"only {len(remaining)} non-candidate authors remain but "
                f"num_retain_authors={num_retain_authors}"
            )
        rperm = split_rng.permutation(len(remaining))
        retain_authors = sorted(remaining[i] for i in rperm[:num_retain_authors])
        candidate_selection = "pinned"

    assert not (set(candidate_authors) & set(retain_authors)), "pool overlap"

    # --- candidate batches --------------------------------------------------------
    # batch_id -> ordered list of (author_id, qa_index) members.
    # Grouping is author-respecting at every B: a batch never spans two authors.
    batches: Dict[str, List[List[Any]]] = {}
    if batching == "author":
        for a in candidate_authors:
            batches[a] = [[a, q] for q in range(qa_per_author)]
    elif batching == "qa":
        for a in candidate_authors:
            for q in range(qa_per_author):
                batches[f"{a}__qa{q:02d}"] = [[a, q]]
    else:
        n_blocks = qa_per_author // qa_per_batch
        for a in candidate_authors:
            for k in range(n_blocks):
                lo = k * qa_per_batch
                batches[f"{a}__g{k:02d}"] = [
                    [a, q] for q in range(lo, lo + qa_per_batch)
                ]
    batch_ids = sorted(batches)
    m = len(batch_ids)

    if m % 2 != 0:
        raise ValueError(
            f"m = {m} is odd; the balanced sign vectors this project uses assume "
            "even m. Choose a different B or candidate-author count."
        )
    expected_m = num_candidate_authors * (qa_per_author // qa_per_batch)
    assert m == expected_m, f"batch construction gave m={m}, expected {expected_m}"

    # --- sign vectors, from independent RNG streams -------------------------------
    # Strict disjointness needs enough distinct balanced vectors to go around:
    # |S_m| = C(m, m/2). Trivial at m=20 (184756), but binding for tiny smoke
    # configurations, so fail early with the actual numbers rather than after
    # rejection sampling gives up.
    capacity = int(math.comb(m, m // 2))
    needed = num_calibration_runs + num_evaluation_runs
    if needed > capacity:
        raise ValueError(
            f"cannot draw {num_calibration_runs} calibration + "
            f"{num_evaluation_runs} evaluation disjoint balanced sign vectors: "
            f"only |S_m| = C({m}, {m // 2}) = {capacity} exist. "
            f"Reduce the run counts or increase m (e.g. batching='qa')."
        )

    # Independent streams (not sequential draws from one) so that changing the number
    # of calibration runs cannot shift the evaluation vectors.
    calib_rng = np.random.default_rng([split_seed, 0xC0FFEE])
    eval_rng = np.random.default_rng([split_seed, 0xE7A1])

    if calibration_balance == "stratified":
        calib_vectors = sample_stratified_sign_vectors(
            m, num_calibration_runs, calib_rng
        )
    elif calibration_balance == "iid":
        calib_vectors = sample_balanced_sign_vectors(m, num_calibration_runs, calib_rng)
    else:
        raise ValueError(
            f"calibration_balance must be 'iid' or 'stratified'; "
            f"got {calibration_balance!r}"
        )

    # Evaluation vectors are ALWAYS i.i.d. uniform over S_m -- the epsilon bound
    # depends on it -- and additionally exclude every calibration vector.
    eval_vectors = sample_balanced_sign_vectors(
        m, num_evaluation_runs, eval_rng, exclude=calib_vectors
    )

    calibration = SignVectorFamily("calib", calib_vectors)
    evaluation = SignVectorFamily("eval", eval_vectors)

    # Record the realized per-batch observation counts: these determine how many
    # samples each QA Gaussian is fitted from, and a small minimum is the main
    # small-sample risk in the attack.
    calib_arr = np.array(calib_vectors)
    n_in = (calib_arr == 1).sum(axis=0)
    n_out = (calib_arr == -1).sum(axis=0)

    content: Dict[str, Any] = {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "dataset": {
            "name": dataset_name,
            "config": dataset_config,
            "revision": dataset_revision,
            "fingerprint": dataset_fingerprint,
            "qa_per_author": int(qa_per_author),
        },
        "split": {
            "batching": batching,
            "num_candidate_authors": int(num_candidate_authors),
            "num_retain_authors": int(num_retain_authors),
            "candidate_authors": candidate_authors,
            "retain_authors": retain_authors,
            "m": int(m),
            "batch_ids": batch_ids,
            "batches": batches,
        },
        "seeds": {
            "split_seed": int(split_seed),
            "data_order_seed": int(data_order_seed),
            "train_seed_base": int(train_seed_base),
            "unlearn_seed_base": int(unlearn_seed_base),
        },
        "sign_vectors": {
            "calibration_balance": calibration_balance,
            "calibration": calibration.as_dict(),
            "evaluation": evaluation.as_dict(),
        },
        "calibration_coverage": {
            "min_n_in": int(n_in.min()),
            "max_n_in": int(n_in.max()),
            "min_n_out": int(n_out.min()),
            "max_n_out": int(n_out.max()),
            "note": (
                "Observations per condition per candidate batch, across calibration "
                "runs. Each of a batch's QA pairs gets this many samples for its "
                "Gaussian fit. A small minimum is why pooled variance is the default."
            ),
        },
    }

    # Recorded ONLY when pinning is used, so a default (randomly partitioned)
    # manifest hashes exactly as it did before pinning existed and pre-existing
    # audits stay valid.
    if candidate_selection is not None:
        content["split"]["candidate_selection"] = candidate_selection

    # `calibration_coverage` is derived from the sign vectors, so it stays in the
    # file but out of the hash -- see DERIVED_FIELDS.
    manifest = dict(content)
    manifest["split_hash"] = canonical_hash(hashable_content(content))
    return manifest


def save_manifest(manifest: Dict[str, Any], path: str | Path, overwrite: bool = False) -> Path:
    """Write the manifest. Refuses to clobber unless ``overwrite=True``.

    The manifest is meant to be immutable: silently regenerating it after runs exist
    would invalidate every stored result and, worse, could break calibration/eval
    independence without any visible error.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists. The audit manifest is immutable; pass overwrite=True "
            "only if no runs have been executed against it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """Re-check every invariant the audit's validity depends on. Raises on failure."""
    if "split_hash" not in manifest:
        raise ValueError("manifest has no split_hash")

    recomputed = canonical_hash(hashable_content(manifest))
    if recomputed != manifest["split_hash"]:
        raise ValueError(
            "manifest hash mismatch: the split was edited after generation "
            f"(stored {manifest['split_hash'][:12]}..., recomputed {recomputed[:12]}...)"
        )

    split = manifest["split"]
    m = split["m"]

    if set(split["candidate_authors"]) & set(split["retain_authors"]):
        raise ValueError("candidate and retain author pools overlap")
    if len(split["batch_ids"]) != m:
        raise ValueError("m disagrees with the number of batch_ids")

    # Every candidate batch must draw only on candidate authors.
    cand = set(split["candidate_authors"])
    for bid, members in split["batches"].items():
        for author, _qa in members:
            if author not in cand:
                raise ValueError(
                    f"batch {bid} references non-candidate author {author}"
                )

    calib = manifest["sign_vectors"]["calibration"]
    ev = manifest["sign_vectors"]["evaluation"]

    for fam_name, fam in (("calibration", calib), ("evaluation", ev)):
        for rid, vec in fam["vectors"].items():
            if len(vec) != m:
                raise ValueError(f"{fam_name}/{rid}: length {len(vec)} != m={m}")
            if not is_balanced(vec):
                n_pos = sum(1 for v in vec if v == 1)
                raise ValueError(
                    f"{fam_name}/{rid}: not balanced "
                    f"({n_pos} positive of {len(vec)}); required by S_m"
                )

    calib_keys = {_sign_vector_key(v) for v in calib["vectors"].values()}
    eval_keys = {_sign_vector_key(v) for v in ev["vectors"].values()}
    shared = calib_keys & eval_keys
    if shared:
        raise ValueError(
            f"{len(shared)} sign vector(s) shared between calibration and evaluation. "
            "The epsilon bound requires independent families."
        )


def load_manifest(path: str | Path, validate: bool = True) -> Dict[str, Any]:
    """Load and (by default) validate a manifest."""
    with open(path) as f:
        manifest = json.load(f)
    if validate:
        validate_manifest(manifest)
    return manifest


def batch_size_of(manifest: Dict[str, Any]) -> int:
    """The audit batch size ``B`` (QA pairs per candidate batch) for a manifest.

    Derived rather than stored, so that adding intermediate-``B`` support did not
    change the hashed content of manifests built before it existed.
    """
    split = manifest["split"]
    sizes = {len(v) for v in split["batches"].values()}
    if len(sizes) != 1:
        raise ValueError(f"ragged batches: sizes {sorted(sizes)}")
    return sizes.pop()


def sign_vector_for_run(manifest: Dict[str, Any], run_id: str) -> List[int]:
    """Look up a run's hidden sign vector.

    NOTE: this is ground truth. It is legitimately needed to *construct* a run's
    training set, and to score an evaluation run *after* predictions are finalized.
    It must never be reachable from the attack code -- see :mod:`audit_tofu.attack`.
    """
    for fam in ("calibration", "evaluation"):
        vectors = manifest["sign_vectors"][fam]["vectors"]
        if run_id in vectors:
            return list(vectors[run_id])
    raise KeyError(f"unknown run_id {run_id!r}")


def positive_candidates(manifest: Dict[str, Any], run_id: str) -> List[str]:
    """Batch ids with ``S_j = +1``: included in training, then handed to unlearning."""
    s = sign_vector_for_run(manifest, run_id)
    return [b for b, sj in zip(manifest["split"]["batch_ids"], s) if sj == 1]


def negative_candidates(manifest: Dict[str, Any], run_id: str) -> List[str]:
    """Batch ids with ``S_j = -1``: never included in training."""
    s = sign_vector_for_run(manifest, run_id)
    return [b for b, sj in zip(manifest["split"]["batch_ids"], s) if sj == -1]
