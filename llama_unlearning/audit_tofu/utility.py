"""Standard TOFU utility metrics, for the held-out evaluation runs only.

These are **complementary** to the audit. The audit score and epsilon lower bound
come from the hypothesis test in :mod:`audit_tofu.epsilon_bounds`; TOFU forget
quality is a behavioural/utility metric. They answer different questions and are
reported separately -- a method can score well on forget quality while still
admitting a large epsilon, and that gap is exactly what the audit is for.

Metrics, following Maini et al. (2024):

``probability``
    Length-normalized answer probability ``P(a|q)^(1/|a|)``. For the perturbed
    splits it is normalized across the true and perturbed answers.
``rouge``
    ROUGE-L recall between the greedily generated answer and the ground truth, via
    upstream's own ``rouge_score`` with ``use_stemmer=True``. Requires generation, so
    it is opt-in (``compute_rouge``).
``truth_ratio``
    ``mean_perturbed P(a_pert)^(1/|a_pert|) / P(a_para)^(1/|a_para|)``. Low means the
    model prefers the true answer over wrong ones.
``model_utility``
    Harmonic mean of the retain / real-authors / world-facts metrics, as in TOFU.
``forget_quality``
    KS-test p-value comparing the forget-set truth-ratio distribution against a
    reference (retain-only) model. Requires ``reference_truth_ratios``; reported as
    ``None`` when unavailable, never silently faked.

Per the spec, the full suite runs only on the ``L`` evaluation runs; calibration runs
need candidate losses alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "TOFU_UTILITY_SPLITS",
    "rouge_l_recall",
    "answer_logprob_normalized",
    "evaluate_split",
    "model_utility",
    "forget_quality",
    "run_utility_suite",
    "tofu_identity_map",
    "load_reference_truth_ratios",
    "forget_quality_slices",
    "ks_with_stats",
]

#: (metric group name, TOFU config) pairs used by the utility suite.
TOFU_UTILITY_SPLITS = {
    "retain": "retain_perturbed",
    "forget": "forget10_perturbed",
    "real_authors": "real_authors_perturbed",
    "world_facts": "world_facts_perturbed",
}

_IDENTITY_CACHE: Dict[tuple, Dict[tuple, tuple]] = {}


def tofu_identity_map(
    dataset_name: str = "locuslab/TOFU",
    cache_dir: Optional[str] = None,
    qa_per_author: int = 20,
) -> Dict[tuple, tuple]:
    """``{(question, answer) -> (author_id, qa_id)}`` for every row of ``full``.

    Why this exists. The per-row truth ratios are the raw material of forget quality,
    and to slice them by a run's sign vector we need to know which author each row
    belongs to. The perturbed eval configs do not carry an author field, so identity
    has to be recovered by joining against ``full``.

    Joining on text rather than on row position is deliberate: the eval configs are
    author-ordered *today*, so ``author = 180 + i // 20`` happens to work for
    ``forget10_perturbed``, but that is an undocumented coincidence of upstream's file
    layout and would silently mis-assign every row if it ever changed. Rows that do
    not join (``real_authors``, ``world_facts`` -- real trivia, not TOFU authors) are
    simply left without identity by the caller.

    The key is the ``(question, answer)`` PAIR, not the question alone. ``full``
    contains one duplicated question -- ``"What is the full name of the author?"``, at
    row 100 (author 5) and row 440 (author 22) -- so a question-keyed dict is
    last-write-wins and silently relabels row 100 as author 22, collapsing 4000 rows
    to 3999 keys. The pair is unique across all 4000 rows, and both author-bearing
    eval configs join 400/400 on it (verified against the cached dataset), so the
    stricter key costs nothing. Collision is the failure mode a text join has and a
    positional one does not, which is the other half of the tradeoff above.
    """
    key = (dataset_name, cache_dir, qa_per_author)
    if key in _IDENTITY_CACHE:
        return _IDENTITY_CACHE[key]

    from datasets import load_dataset

    ds = load_dataset(dataset_name, "full", cache_dir=cache_dir)
    split = "train" if "train" in ds else list(ds.keys())[0]
    out: Dict[tuple, tuple] = {}
    for i, row in enumerate(ds[split]):
        ident = (f"author_{i // qa_per_author:04d}", i % qa_per_author)
        out[(row["question"], row["answer"])] = ident
    if len(out) != len(ds[split]):
        # Not fatal -- the map is still usable and only the colliding rows are
        # affected -- but it means upstream added a duplicated (question, answer)
        # pair, so n_authors and the sign slices can no longer be trusted blindly.
        import warnings

        warnings.warn(
            f"{dataset_name}/full has {len(ds[split])} rows but only {len(out)} "
            "distinct (question, answer) pairs; identity is ambiguous for the "
            "duplicates and those rows will be labelled with the LAST occurrence.",
            RuntimeWarning,
            stacklevel=2,
        )
    _IDENTITY_CACHE[key] = out
    return out


_ROUGE_SCORER: Any = None


def rouge_l_recall(prediction: str, reference: str) -> float:
    """ROUGE-L recall, delegated to upstream TOFU's own scorer.

    Uses ``rouge_score`` with ``use_stemmer=True`` -- the exact configuration in
    ``locuslab/tofu``'s ``evaluate_util.py`` -- so values are directly comparable to
    published TOFU numbers. Recall, not F1, is the TOFU convention.

    This was previously a hand-rolled LCS over ``str.split()`` tokens, to avoid a
    dependency for one metric. That is NOT the same metric. ``rouge_score``
    lowercases, strips non-alphanumerics, and Porter-stems tokens longer than three
    characters, so ``"Paris."`` against ``"paris"`` scored 0 under the hand-rolled
    version and 1 here; likewise ``"novels"`` against ``"novel"``. The hand-rolled
    scores were biased low by an amount that varied with the model's punctuation and
    inflection habits rather than by a constant offset, so a reader could not have
    corrected for it. Nothing stored was affected: the metric is opt-in via
    ``compute_rouge``, which had never been enabled when this changed.

    Argument order: ``rouge_score``'s signature is ``score(target, prediction)`` and
    upstream calls ``scorer.score(gt, gen)``. ``reference`` is the target, so recall
    is ``|LCS| / |reference tokens|`` -- the same denominator as before.
    """
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        # Lazily imported and cached. Constructing the scorer builds a PorterStemmer,
        # and this is called once per row over a few thousand rows; the import stays
        # out of module scope so the epsilon path, which imports this module for
        # nothing else, does not need rouge_score present.
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(_ROUGE_SCORER.score(reference, prediction)["rougeL"].recall)


def answer_logprob_normalized(
    model: Any,
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    max_length: int = 512,
    append_eos: bool = True,
    system_prompt: Optional[str] = None,
) -> tuple[float, int]:
    """Return ``(mean answer logprob, num answer tokens)``.

    ``exp(mean logprob)`` is the length-normalized probability ``P(a|q)^(1/|a|)``.
    """
    import torch

    from .scoring import score_examples
    from .tofu_data import QAExample

    ex = QAExample(author_id="_", qa_id=0, question=question, answer=answer, row_index=0)
    rows = score_examples(
        model, tokenizer, [ex],
        max_length=max_length, append_eos=append_eos,
        system_prompt=system_prompt, batch_size=1,
    )
    # score_examples returns the NEGATIVE mean logprob.
    return -float(rows[0]["loss"]), int(rows[0]["num_answer_tokens"])


def evaluate_split(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    *,
    max_length: int = 512,
    append_eos: bool = True,
    system_prompt: Optional[str] = None,
    compute_rouge: bool = False,
    max_new_tokens: int = 200,
    limit: Optional[int] = None,
    identities: Optional[Dict[tuple, tuple]] = None,
) -> Dict[str, Any]:
    """Compute probability / truth-ratio / (optional) ROUGE over one TOFU split.

    ``identities`` is an optional ``{(question, answer) -> (author_id, qa_id)}`` map
    used to label each emitted row, so downstream code can slice the per-row truth
    ratios by a run's sign vector. Rows absent from the map are labelled with
    ``None``.
    """
    import torch

    if limit is not None:
        rows = list(rows)[:limit]

    probs: List[float] = []
    truth_ratios: List[float] = []
    rouges: List[float] = []
    per_row: List[Dict[str, Any]] = []

    for row in rows:
        question = row["question"]
        answer = row["answer"]

        gt_lp, _ = answer_logprob_normalized(
            model, tokenizer, question, answer,
            max_length=max_length, append_eos=append_eos, system_prompt=system_prompt,
        )

        perturbed = row.get("perturbed_answer") or []
        if isinstance(perturbed, str):
            perturbed = [perturbed]
        pert_lps = [
            answer_logprob_normalized(
                model, tokenizer, question, p,
                max_length=max_length, append_eos=append_eos,
                system_prompt=system_prompt,
            )[0]
            for p in perturbed
        ]

        # Normalized probability across {true} U perturbed, when perturbations exist.
        if pert_lps:
            all_lp = np.array([gt_lp] + pert_lps, dtype=np.float64)
            probs.append(float(np.exp(gt_lp) / np.exp(all_lp).sum()))
        else:
            probs.append(float(np.exp(gt_lp)))

        # Truth ratio uses the paraphrased answer as the denominator when present.
        para = row.get("paraphrased_answer")
        denom_lp = (
            answer_logprob_normalized(
                model, tokenizer, question, para,
                max_length=max_length, append_eos=append_eos,
                system_prompt=system_prompt,
            )[0]
            if para
            else gt_lp
        )
        if pert_lps:
            num = float(np.mean([np.exp(lp) for lp in pert_lps]))
            den = float(np.exp(denom_lp))
            truth_ratios.append(num / den if den > 0 else float("nan"))

        if compute_rouge:
            from .tofu_data import encode_example

            enc = encode_example(
                tokenizer, question, answer,
                max_length=max_length, append_eos=append_eos,
                system_prompt=system_prompt,
            )
            n_prompt = enc["num_prompt_tokens"]
            prompt_ids = torch.tensor(
                [enc["input_ids"][:n_prompt]], device=next(model.parameters()).device
            )
            was_cache = model.config.use_cache
            model.config.use_cache = True
            model.eval()
            with torch.no_grad():
                gen = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            model.config.use_cache = was_cache
            text = tokenizer.decode(
                gen[0][prompt_ids.shape[1]:], skip_special_tokens=True
            )
            rouges.append(rouge_l_recall(text, answer))

        # One identified record per row. ``truth_ratio`` is explicitly None when the
        # row had no perturbed answers, because the flat ``truth_ratios`` list below
        # skips those rows and so is NOT positionally aligned with ``rows``; per-row
        # slicing must not silently inherit that misalignment.
        ident = (identities or {}).get((question, answer))
        per_row.append({
            "index": len(per_row),
            "author_id": ident[0] if ident else None,
            "qa_id": ident[1] if ident else None,
            "probability": probs[-1],
            "truth_ratio": float(truth_ratios[-1]) if pert_lps else None,
            "rouge_l_recall": float(rouges[-1]) if compute_rouge else None,
        })

    def _mean(xs: Sequence[float]) -> Optional[float]:
        vals = [x for x in xs if np.isfinite(x)]
        return float(np.mean(vals)) if vals else None

    n_ident = sum(1 for r in per_row if r["author_id"] is not None)
    return {
        "n": len(rows),
        "probability": _mean(probs),
        "truth_ratio": _mean(truth_ratios),
        "truth_ratios": [float(t) for t in truth_ratios],
        "rouge_l_recall": _mean(rouges) if compute_rouge else None,
        "per_row": per_row,
        "n_identified": n_ident,
    }


def model_utility(split_results: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Harmonic mean over the non-forget metric groups, as in TOFU.

    Truth ratio is inverted (``max(0, 1 - tr)``) so that, like the other metrics,
    larger is better before the harmonic mean is taken.

    NOT numerically equal to TOFU's published Model Utility, by two deliberate
    choices -- see ``docs/METRICS.md`` 2.1 before changing either:

    * **Zero terms are dropped** (below), whereas ``scipy.stats.hmean`` returns
      ``0.0`` if any input is 0. This matters constantly rather than rarely: on
      Llama-3.2-1B every measured run has ``truth_ratio >= 1`` on ``world_facts``
      (480/480 files) and ``real_authors`` (479/480), so ``max(0, 1 - tr)`` clamps
      to 0 and the faithful hmean would be identically 0.0 for every run, ranking
      nothing. Upstream's published numbers come from Llama-2-7B / Phi-1.5, which
      know real-world trivia, so their terms never hit zero. Kept as-is so the
      column can still separate methods; the divergence is recorded, not silent.
    * **The term count varies.** Upstream's is a fixed 9 (3 groups x {ROUGE,
      Probability, Truth Ratio}); here ROUGE is absent unless ``compute_rouge``, so
      the default is a 6-term mean.
    """
    values: List[float] = []
    for group in ("retain", "real_authors", "world_facts"):
        res = split_results.get(group)
        if not res:
            continue
        for key in ("probability", "rouge_l_recall"):
            v = res.get(key)
            if v is not None and np.isfinite(v):
                values.append(float(v))
        tr = res.get("truth_ratio")
        if tr is not None and np.isfinite(tr):
            values.append(float(max(0.0, 1.0 - tr)))

    values = [v for v in values if v > 0]
    if not values:
        return None
    return float(len(values) / np.sum([1.0 / v for v in values]))


def _rows_by_identity(per_row: Sequence[Dict[str, Any]]) -> Dict[tuple, float]:
    """``{(author_id, qa_id) -> truth_ratio}`` for rows carrying both."""
    out: Dict[tuple, float] = {}
    for r in per_row or []:
        a, q, tr = r.get("author_id"), r.get("qa_id"), r.get("truth_ratio")
        if a is None or q is None or tr is None:
            continue
        tr = float(tr)
        if np.isfinite(tr):
            out[(str(a), int(q))] = tr
    return out


def _flat_finite(per_row: Sequence[Dict[str, Any]]) -> List[float]:
    return [float(r["truth_ratio"]) for r in (per_row or [])
            if r.get("truth_ratio") is not None
            and np.isfinite(float(r["truth_ratio"]))]


def ks_with_stats(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """KS test plus the descriptive stats needed to read its direction."""
    import warnings

    with warnings.catch_warnings():
        # At n = m = 200 scipy falls back from the exact to the asymptotic p-value
        # and says so once per test. The fallback is intended at this sample size, so
        # the notice is pure noise across dozens of tests.
        warnings.filterwarnings(
            "ignore", message=".*Exact calculation unsuccessful.*",
            category=RuntimeWarning,
        )
        res = forget_quality(list(a), list(b))
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    res["n_model"] = int(av.size)
    res["n_reference"] = int(bv.size)
    res["mean_tr_model"] = float(av.mean()) if av.size else None
    res["mean_tr_reference"] = float(bv.mean()) if bv.size else None
    res["median_tr_model"] = float(np.median(av)) if av.size else None
    res["median_tr_reference"] = float(np.median(bv)) if bv.size else None
    return res


def forget_quality_slices(
    model_per_row: Sequence[Dict[str, Any]],
    reference_per_row: Sequence[Dict[str, Any]],
    sign_of: Optional[Dict[tuple, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """KS of a model's truth ratios against a reference's, split by sign.

    THE single implementation of forget quality, shared by the inline path in
    ``scripts/run_single.py`` and the post-hoc ``scripts/forget_quality.py`` so the
    two can never disagree. Returns a dict of slices; slices with no usable rows are
    omitted rather than reported as null.

    ``all``     every row with a finite truth ratio on both sides. For the ``forget``
                group this is the conventional TOFU number, and it is biased toward
                "good forgetting": under a balanced sign vector half its rows were
                never trained on and so match the reference by construction.
    ``plus``    rows whose ``(author, qa)`` has ``S_j = +1`` -- trained then
                unlearned. The headline.
    ``minus``   rows with ``S_j = -1`` -- never trained by EITHER model. A negative
                control; under the null its p-value is Uniform(0,1), so judge its
                mean across runs (~0.5 = the reference is exchangeable with the
                runs), never a single value.

    ``plus``/``minus`` compare the SAME rows on both sides. Comparing a run's 200
    ``+1`` rows against all 400 reference rows would partly measure author
    composition rather than forgetting. They are omitted when ``sign_of`` is None or
    the group carries no author identity (``real_authors``, ``world_facts``).
    """
    out: Dict[str, Dict[str, Any]] = {}

    a_flat, b_flat = _flat_finite(model_per_row), _flat_finite(reference_per_row)
    if a_flat and b_flat:
        rec = ks_with_stats(a_flat, b_flat)
        rec["n_authors"] = len({a for a, _ in _rows_by_identity(model_per_row)}) or None
        out["all"] = rec

    if not sign_of:
        return out

    model_rows = _rows_by_identity(model_per_row)
    ref_rows = _rows_by_identity(reference_per_row)
    for slice_name, want in (("plus", 1), ("minus", -1)):
        # Sign is keyed by (author, qa), so this is exact at every batch size B --
        # including B < qa_per_author, where one author's pairs carry mixed signs.
        keys = [k for k in model_rows if k in ref_rows and sign_of.get(k) == want]
        if not keys:
            continue
        rec = ks_with_stats([model_rows[k] for k in keys],
                            [ref_rows[k] for k in keys])
        rec["n_authors"] = len({k[0] for k in keys})
        out[slice_name] = rec
    return out


def load_reference_truth_ratios(payload: Any, group: str = "forget") -> List[float]:
    """Coerce a reference payload into a flat list of truth ratios.

    Two shapes are accepted, because two exist in practice and confusing them used to
    fail deep inside numpy:

    * a **flat list of floats** -- what ``utility.reference_truth_ratios_path``
      originally expected;
    * the **nested dict** that ``scripts/train_reference.py`` writes to
      ``reference/truth_ratios.json`` (``{"groups": {<group>: {"per_row": [...]}}}``),
      which is the artifact this repo actually produces.

    Pointing the config at the nested file used to raise
    ``TypeError: ufunc 'isfinite' not supported`` -- numpy iterating a dict's string
    keys -- roughly 20 minutes into a run, with nothing to indicate the cause.
    """
    if payload is None:
        return []

    if isinstance(payload, dict):
        groups = payload.get("groups")
        if not isinstance(groups, dict) or group not in groups:
            raise ValueError(
                f"reference payload is a dict but has no groups[{group!r}]; "
                f"found keys {sorted(payload)[:8]}. Expected either a flat list of "
                "truth ratios or the output of scripts/train_reference.py."
            )
        rows = (groups[group] or {}).get("per_row")
        if rows is None:
            raise ValueError(
                f"reference groups[{group!r}] has no 'per_row'; retrain with "
                "scripts/train_reference.py"
            )
        out = [float(r["truth_ratio"]) for r in rows
               if isinstance(r, dict) and r.get("truth_ratio") is not None]
        if not out:
            raise ValueError(f"reference groups[{group!r}] has no usable truth ratios")
        return out

    if isinstance(payload, (list, tuple)):
        try:
            return [float(x) for x in payload]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reference list contains non-numeric entries: {exc}"
            ) from exc

    raise ValueError(
        f"unsupported reference payload type {type(payload).__name__}; expected a "
        "list of floats or the dict written by scripts/train_reference.py"
    )


def forget_quality(
    forget_truth_ratios: Sequence[float],
    reference_truth_ratios: Optional[Sequence[float]],
) -> Dict[str, Any]:
    """TOFU forget quality: KS-test p-value against a retain-only reference model.

    A high p-value means the forget-set truth-ratio distribution is statistically
    indistinguishable from the reference -- TOFU's notion of successful forgetting.
    Returns ``None`` when no reference is supplied rather than substituting a proxy.
    """
    if reference_truth_ratios is None or len(reference_truth_ratios) == 0:
        return {
            "forget_quality": None,
            "ks_statistic": None,
            "note": (
                "No reference (retain-only) model truth ratios supplied, so TOFU "
                "forget quality is undefined here. Train a retain-only reference to "
                "enable it. The audit's epsilon bound does not depend on this."
            ),
        }
    # A dict here means someone passed the nested reference file straight through;
    # numpy would otherwise iterate its string keys and raise an opaque
    # "ufunc 'isfinite' not supported" from inside the KS call.
    if isinstance(reference_truth_ratios, dict):
        raise TypeError(
            "reference_truth_ratios must be a flat sequence of floats, got a dict. "
            "Pass it through audit_tofu.utility.load_reference_truth_ratios() first, "
            "which accepts scripts/train_reference.py's output."
        )
    from scipy.stats import ks_2samp

    a = np.asarray([t for t in forget_truth_ratios if np.isfinite(t)], dtype=float)
    b = np.asarray([t for t in reference_truth_ratios if np.isfinite(t)], dtype=float)
    if a.size == 0 or b.size == 0:
        return {"forget_quality": None, "ks_statistic": None, "note": "empty sample"}
    res = ks_2samp(a, b)
    return {
        "forget_quality": float(res.pvalue),
        "ks_statistic": float(res.statistic),
        "note": "KS-test p-value vs reference; higher = closer to the reference model.",
    }


def run_utility_suite(
    model: Any,
    tokenizer: Any,
    *,
    dataset_name: str = "locuslab/TOFU",
    splits: Optional[Dict[str, str]] = None,
    reference_truth_ratios: Optional[Sequence[float]] = None,
    max_length: int = 512,
    append_eos: bool = True,
    system_prompt: Optional[str] = None,
    compute_rouge: bool = False,
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the utility suite. Missing splits are skipped with a recorded reason."""
    from datasets import load_dataset

    splits = splits or TOFU_UTILITY_SPLITS
    log = (logger.info if logger else print)

    # Author identity for every row we can join back to the `full` config. Failure is
    # non-fatal: the suite still produces its aggregate metrics, only the per-run
    # sign-vector slicing in scripts/forget_quality.py becomes unavailable.
    identities: Dict[tuple, tuple] = {}
    if dataset_name != "__synthetic__":
        try:
            identities = tofu_identity_map(dataset_name, cache_dir)
        except Exception as exc:
            log(f"[utility] no author identity map ({exc}); per-row slicing disabled")

    results: Dict[str, Any] = {}
    for group, config in splits.items():
        try:
            ds = load_dataset(dataset_name, config, cache_dir=cache_dir)
            key = "train" if "train" in ds else list(ds.keys())[0]
            rows = list(ds[key])
        except Exception as exc:
            results[group] = {"error": f"could not load {config}: {exc}"}
            log(f"[utility] skipping {group} ({config}): {exc}")
            continue

        log(f"[utility] {group} ({config}): {len(rows)} rows")
        results[group] = evaluate_split(
            model, tokenizer, rows,
            max_length=max_length, append_eos=append_eos,
            system_prompt=system_prompt, compute_rouge=compute_rouge, limit=limit,
            identities=identities,
        )
        n_id = results[group].get("n_identified")
        if n_id is not None:
            log(f"[utility] {group}: {n_id}/{len(rows)} rows carry author identity")

    out: Dict[str, Any] = {"splits": results}
    out["model_utility"] = model_utility(results)
    forget_res = results.get("forget") or {}
    fq = forget_quality(forget_res.get("truth_ratios", []), reference_truth_ratios)
    # Keep forget_quality's own explanation under its OWN key. It used to be merged
    # into `note` and then immediately overwritten by the generic note below, so a
    # null forget_quality was recorded with no indication of why -- which is exactly
    # the question every reader of these files asks first.
    out["forget_quality"] = fq.get("forget_quality")
    out["ks_statistic"] = fq.get("ks_statistic")
    out["forget_quality_note"] = fq.get("note")
    out["note"] = (
        "Utility/behaviour metrics only. The audit's epsilon lower bound is computed "
        "separately in audit_tofu.epsilon_bounds and is not a function of these. "
        "See forget_quality_note for the status of forget quality specifically."
    )
    return out
