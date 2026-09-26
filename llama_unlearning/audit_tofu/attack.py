"""The batchwise in/out likelihood-ratio attack (paper §5, Instantiation I).

Pipeline
--------
1. :func:`fit_calibration` consumes ONLY calibration-run losses and fits, per
   candidate QA pair ``z``, two Gaussians::

       l_z | S_j = +1  ~  N(mu_z_in,  sigma^2_z_in)     "included then unlearned"
       l_z | S_j = -1  ~  N(mu_z_out, sigma^2_z_out)    "never included"

2. :func:`predict` consumes an evaluation run's losses plus the frozen calibration,
   and returns a guess vector in ``{-1, 0, +1}^m`` with exactly ``r/2`` entries
   ``+1`` and ``r/2`` entries ``-1``.

3. :func:`overlap` is applied only afterwards, by the caller, once the sign vector
   is revealed.

Label hygiene
-------------
The spec requires that evaluation labels cannot be fed to the predictor. That is
enforced structurally rather than by convention:

* :func:`predict` takes ``(scores, calibration, r, ...)``. It has no parameter for a
  sign vector, labels, or a manifest, and it never imports
  :mod:`audit_tofu.manifest`. A caller cannot smuggle labels in without a
  ``TypeError``.
* :func:`overlap` is the only function that touches ground truth, and it cannot
  influence a prediction because it consumes one.
* ``tests/test_attack.py`` asserts the signature property by introspection, so an
  accidental future refactor that adds a label argument fails CI.

Small-sample handling
---------------------
With ``Gamma = 20`` calibration runs and balanced sign vectors, each condition gets
only ~10 observations, so the fits are fragile. Three mitigations, all configurable:

* a **variance floor** (``var_floor``), applied after fitting;
* **pooled variance** (``pool_variance="author"``), sharing one variance estimate
  across the 20 QA pairs of an author within each condition, which trades a little
  bias for a large variance reduction;
* explicit **degeneracy warnings** when a condition has too few observations or
  (near-)zero spread.

Log densities are evaluated in a numerically stable closed form rather than via
``scipy.stats.norm.logpdf`` so that a zero/negative variance can never silently
produce ``nan``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CalibrationWarning",
    "QAGaussian",
    "Calibration",
    "fit_calibration",
    "log_normal_pdf",
    "predict",
    "overlap",
    "LOG_2PI",
]

LOG_2PI = math.log(2.0 * math.pi)

#: Below this many observations in a condition, the Gaussian fit is flagged.
MIN_OBS_WARN = 3

DEFAULT_VAR_FLOOR = 1e-6


class CalibrationWarning(UserWarning):
    """Raised for degenerate or insufficient calibration distributions."""


def log_normal_pdf(x: float | np.ndarray, mean: float, var: float) -> float | np.ndarray:
    """``log N(x; mean, var)``, evaluated stably.

    ``var`` must already be floored; a non-positive variance is a programming error
    here rather than something to paper over, so it raises.
    """
    if not np.isfinite(var) or var <= 0.0:
        raise ValueError(f"variance must be finite and positive; got {var!r}")
    z = (np.asarray(x, dtype=np.float64) - mean)
    return -0.5 * (LOG_2PI + math.log(var)) - 0.5 * (z * z) / var


@dataclass
class QAGaussian:
    """Fitted in/out Gaussians for one candidate QA pair."""

    batch_id: str
    qa_id: int
    mu_in: float
    var_in: float
    n_in: int
    mu_out: float
    var_out: float
    n_out: int
    degenerate: bool = False
    reasons: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.n_in > 0 and self.n_out > 0

    def llr(self, loss: float) -> float:
        """``lambda_z = log p_in(l_z) - log p_out(l_z)`` (paper §4/§5)."""
        return float(
            log_normal_pdf(loss, self.mu_in, self.var_in)
            - log_normal_pdf(loss, self.mu_out, self.var_out)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "qa_id": int(self.qa_id),
            "mu_in": self.mu_in,
            "var_in": self.var_in,
            "n_in": self.n_in,
            "mu_out": self.mu_out,
            "var_out": self.var_out,
            "n_out": self.n_out,
            "degenerate": self.degenerate,
            "reasons": list(self.reasons),
        }


@dataclass
class Calibration:
    """Frozen calibration: the attack's only knowledge beyond the observed losses."""

    batch_ids: List[str]
    gaussians: Dict[Tuple[str, int], QAGaussian]
    config: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.batch_ids)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_ids": list(self.batch_ids),
            "config": dict(self.config),
            "diagnostics": dict(self.diagnostics),
            "gaussians": [g.as_dict() for g in self.gaussians.values()],
        }


def _fit_one(values: Sequence[float]) -> Tuple[float, float, int]:
    """Mean and *unbiased* variance. ``n < 2`` yields variance ``0`` for flooring."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan"), 0.0, 0
    if n == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.var(ddof=1)), n


def fit_calibration(
    calibration_losses: Dict[str, Dict[Tuple[str, int], float]],
    calibration_signs: Dict[str, Dict[str, int]],
    batch_ids: Sequence[str],
    *,
    var_floor: float = DEFAULT_VAR_FLOOR,
    pool_variance: Optional[str] = None,
    warn: bool = True,
) -> Calibration:
    """Fit per-QA in/out Gaussians from calibration runs only.

    Parameters
    ----------
    calibration_losses
        ``{run_id: {(batch_id, qa_id): loss}}`` for calibration runs.
    calibration_signs
        ``{run_id: {batch_id: +1 or -1}}``. Calibration labels ARE used -- that is
        what calibration means. Evaluation labels are never passed here.
    batch_ids
        Canonical ordering of the ``m`` candidate batches, from the manifest.
    var_floor
        Lower bound applied to every fitted variance.
    pool_variance
        ``None`` for per-QA variances; ``"author"`` to share one variance across all
        QA pairs of a batch, per condition.

    Notes
    -----
    A run contributes an observation to the *in* distribution for batch ``j`` when
    ``S_j = +1`` in that run, and to *out* when ``S_j = -1``.
    """
    if pool_variance not in (None, "author"):
        raise ValueError(f"pool_variance must be None or 'author'; got {pool_variance!r}")
    if var_floor <= 0:
        raise ValueError(f"var_floor must be positive; got {var_floor}")

    batch_ids = list(batch_ids)
    batch_set = set(batch_ids)

    # Bucket observations by (batch, qa) and condition.
    obs: Dict[Tuple[str, int], Dict[int, List[float]]] = {}
    for run_id, losses in calibration_losses.items():
        if run_id not in calibration_signs:
            raise KeyError(f"no sign vector for calibration run {run_id!r}")
        signs = calibration_signs[run_id]
        for (bid, qa_id), loss in losses.items():
            if bid not in batch_set:
                continue
            if bid not in signs:
                raise KeyError(f"run {run_id!r} has no sign for batch {bid!r}")
            sj = signs[bid]
            if sj not in (-1, 1):
                raise ValueError(f"sign must be +-1; got {sj!r}")
            obs.setdefault((bid, qa_id), {1: [], -1: []})[sj].append(float(loss))

    if not obs:
        raise ValueError("no calibration observations found")

    # Optional pooled variance per (batch, condition).
    pooled: Dict[Tuple[str, int], float] = {}
    if pool_variance == "author":
        grouped: Dict[Tuple[str, int], List[float]] = {}
        for (bid, _qa), by_cond in obs.items():
            for cond, vals in by_cond.items():
                if len(vals) >= 2:
                    # Centre each QA pair's observations, then pool the residuals:
                    # QA pairs differ in mean difficulty, so pooling raw values would
                    # inflate the variance with between-QA spread.
                    mu = float(np.mean(vals))
                    grouped.setdefault((bid, cond), []).extend(
                        [v - mu for v in vals]
                    )
        for key, residuals in grouped.items():
            if len(residuals) >= 2:
                pooled[key] = float(np.var(np.asarray(residuals), ddof=1))

    gaussians: Dict[Tuple[str, int], QAGaussian] = {}
    n_degenerate = 0
    messages: List[str] = []

    for key in sorted(obs):
        bid, qa_id = key
        by_cond = obs[key]
        mu_in, var_in, n_in = _fit_one(by_cond[1])
        mu_out, var_out, n_out = _fit_one(by_cond[-1])

        if pool_variance == "author":
            var_in = pooled.get((bid, 1), var_in)
            var_out = pooled.get((bid, -1), var_out)

        reasons: List[str] = []
        if n_in == 0 or n_out == 0:
            reasons.append(f"empty condition (n_in={n_in}, n_out={n_out})")
        else:
            if n_in < MIN_OBS_WARN or n_out < MIN_OBS_WARN:
                reasons.append(f"few observations (n_in={n_in}, n_out={n_out})")
            if var_in <= var_floor or var_out <= var_floor:
                reasons.append(
                    f"variance at/below floor (var_in={var_in:.3g}, var_out={var_out:.3g})"
                )

        # Floor AFTER inspection so the warning reflects the raw fit.
        var_in = max(float(var_in), var_floor)
        var_out = max(float(var_out), var_floor)

        degenerate = bool(reasons)
        if degenerate:
            n_degenerate += 1
            if len(messages) < 10:
                messages.append(f"{bid}/qa{qa_id}: " + "; ".join(reasons))

        gaussians[key] = QAGaussian(
            batch_id=bid,
            qa_id=int(qa_id),
            mu_in=mu_in,
            var_in=var_in,
            n_in=n_in,
            mu_out=mu_out,
            var_out=var_out,
            n_out=n_out,
            degenerate=degenerate,
            reasons=reasons,
        )

    missing = [b for b in batch_ids if not any(k[0] == b for k in gaussians)]
    if missing and warn:
        warnings.warn(
            f"{len(missing)} candidate batch(es) have no calibration data: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            CalibrationWarning,
            stacklevel=2,
        )
    if n_degenerate and warn:
        warnings.warn(
            f"{n_degenerate}/{len(gaussians)} QA Gaussians are degenerate or "
            f"under-observed. Consider pool_variance='author' or a larger Gamma. "
            f"Examples: " + " | ".join(messages),
            CalibrationWarning,
            stacklevel=2,
        )

    n_runs = len(calibration_losses)
    diagnostics = {
        "num_calibration_runs": n_runs,
        "num_qa_fitted": len(gaussians),
        "num_degenerate": n_degenerate,
        "degenerate_examples": messages,
        "batches_missing_data": missing,
        "min_n_in": min((g.n_in for g in gaussians.values()), default=0),
        "min_n_out": min((g.n_out for g in gaussians.values()), default=0),
    }
    config = {
        "var_floor": float(var_floor),
        "pool_variance": pool_variance,
        "min_obs_warn": MIN_OBS_WARN,
    }
    return Calibration(
        batch_ids=batch_ids, gaussians=gaussians, config=config, diagnostics=diagnostics
    )


def _batch_scores(
    scores: Dict[Tuple[str, int], float],
    calibration: Calibration,
    aggregate: str,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Aggregate per-QA log-likelihood ratios into per-batch ``Lambda_j``."""
    per_batch: Dict[str, List[float]] = {b: [] for b in calibration.batch_ids}
    for (bid, qa_id), loss in scores.items():
        g = calibration.gaussians.get((bid, qa_id))
        if g is None or not g.usable:
            continue
        if bid in per_batch:
            per_batch[bid].append(g.llr(float(loss)))

    lambdas: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for bid, vals in per_batch.items():
        counts[bid] = len(vals)
        if not vals:
            # No evidence: a neutral score, so the batch sinks to the abstain region
            # rather than being pushed to either extreme.
            lambdas[bid] = 0.0
        elif aggregate == "sum":
            lambdas[bid] = float(np.sum(vals))
        elif aggregate == "mean":
            lambdas[bid] = float(np.mean(vals))
        else:
            raise ValueError(f"aggregate must be 'sum' or 'mean'; got {aggregate!r}")
    return lambdas, counts


def predict(
    scores: Dict[Tuple[str, int], float],
    calibration: Calibration,
    r: int,
    *,
    aggregate: str = "sum",
) -> Dict[str, Any]:
    """Produce a guess vector from observed losses and frozen calibration.

    This function deliberately accepts NO ground-truth argument. See the module
    docstring on label hygiene.

    Parameters
    ----------
    scores
        ``{(batch_id, qa_id): loss}`` from ONE evaluation run's final unlearned model.
    calibration
        Output of :func:`fit_calibration`, fitted on calibration runs only.
    r
        Total guess budget, even. ``r/2`` guesses ``+1``, ``r/2`` guess ``-1``,
        ``m - r`` abstain.
    aggregate
        ``"sum"`` (default, per spec) or ``"mean"`` (ablation) over an author's QA
        pairs.

    Returns
    -------
    dict with ``guess`` (list of ``m`` values in ``{-1,0,+1}``, in manifest batch
    order), ``lambdas``, and the chosen positive/negative batch ids.

    Tie handling
    ------------
    Batches are ranked by ``(-Lambda_j, batch_id)`` for the positive side and
    ``(Lambda_j, batch_id)`` for the negative side. Ties therefore break on the
    lexicographic batch id -- deterministic, reproducible, and independent of labels
    and of dict iteration order.
    """
    m = calibration.m
    if r % 2 != 0:
        raise ValueError(f"r must be even; got {r}")
    if not 0 < r <= m:
        raise ValueError(f"need 0 < r <= m={m}; got r={r}")

    lambdas, counts = _batch_scores(scores, calibration, aggregate)

    half = r // 2
    by_desc = sorted(calibration.batch_ids, key=lambda b: (-lambdas[b], b))
    positives = by_desc[:half]
    # Take the negative side from the opposite end of the SAME ordering, so a batch
    # can never be selected twice even when many Lambda values coincide.
    negatives = list(reversed(by_desc))[:half]

    assert not (set(positives) & set(negatives)), "positive/negative guess overlap"

    guess = []
    pos_set, neg_set = set(positives), set(negatives)
    for b in calibration.batch_ids:
        guess.append(1 if b in pos_set else (-1 if b in neg_set else 0))

    n_pos = sum(1 for g in guess if g == 1)
    n_neg = sum(1 for g in guess if g == -1)
    assert n_pos == half and n_neg == half, f"budget violated: {n_pos}/{n_neg} vs {half}"

    return {
        "guess": guess,
        "r": int(r),
        "aggregate": aggregate,
        "lambdas": {b: float(lambdas[b]) for b in calibration.batch_ids},
        "qa_counts": {b: int(counts[b]) for b in calibration.batch_ids},
        "predicted_positive": sorted(positives),
        "predicted_negative": sorted(negatives),
        "num_abstain": int(m - r),
    }


def overlap(guess: Sequence[int], sign_vector: Sequence[int]) -> int:
    """The audit's overlap statistic ``V`` (paper §4).

    ``V := sum_j max{0, guess_j * S_j}`` -- the number of correct non-zero guesses,
    out of ``r``. Abstentions contribute ``0``.

    Call this ONLY after predictions are finalized.
    """
    if len(guess) != len(sign_vector):
        raise ValueError(
            f"length mismatch: guess {len(guess)} vs sign vector {len(sign_vector)}"
        )
    for v in sign_vector:
        if v not in (-1, 1):
            raise ValueError(f"sign vector entries must be +-1; got {v!r}")
    for g in guess:
        if g not in (-1, 0, 1):
            raise ValueError(f"guess entries must be in {{-1,0,1}}; got {g!r}")
    return int(sum(max(0, int(g) * int(s)) for g, s in zip(guess, sign_vector)))
