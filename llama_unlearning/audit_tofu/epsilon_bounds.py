"""Epsilon lower bounds for the unlearning audit.

This module is a *thin, documented wrapper* around :mod:`audit_tofu.cum_runs_eps_lab`,
which is vendored **byte-identically** from the tested implementation released with the
paper's reference code (the Shakespeare experiment).

We do not rederive or modify that math. Everything below is
either (a) a convention adapter or (b) reporting. Two adapters are required, because
the upstream *caller* (``evaluate_llr_predictions.py``) uses conventions that differ
from the paper's notation:

1. ``r`` is PER-SIDE upstream.
   Upstream ``--r 50`` means "top 50 and bottom 50", and the caller then passes
   ``epsilon_r = 2 * r_value`` into the bound. The paper (§4, "Fix an even guess
   budget r") and this project's spec define ``r`` as the TOTAL guess budget, with
   ``r/2`` positive and ``r/2`` negative guesses. We therefore pass ``r`` straight
   through, unhalved and undoubled. ``V`` ranges over ``{0, ..., r}``.

2. Upstream reports the LDP epsilon, NOT ``eps_LB``.
   Lemma 4.1 of the paper shows that if ``(A, U)`` is ``(eps, 0)``-certified
   unlearning then the auditor mechanism ``M`` is ``(2*eps, 0)``-locally
   differentially private. The paper's §4.2 states: "dividing by 2 to undo the
   reduction of Theorem 4.1 yields the reported eps_LB". The upstream caller omits
   this division. We apply it here and report BOTH quantities so the distinction is
   never silent:

       epsilon_ldp_lb : solution of the LDP bound   (== upstream's "epsilon_lb")
       epsilon_lb     : epsilon_ldp_lb / 2          (the unlearning bound, reported)

   The division is **conditional**, because a newer revision of
   ``cum_runs_eps_lab.py`` performs it internally (returning both ``epsilon_lb_ldp``
   and an already-halved ``epsilon_lb``). :func:`vendored_halves_internally` detects
   that revision and the wrapper then passes the value through untouched, so
   swapping the vendored file cannot halve the bound twice. Every report says which
   layer did it under ``halving_applied_by``. That revision also dropped the
   ``delta`` argument, so it is passed only when the signature accepts it.

The relevant statement, for reference (paper Lemma 4.2). With
``M' = C(m, floor(m/2))`` and ``r`` even::

    pi_eps(u) = [ sum_{a1,a2 in {0..r/2}, a1+a2=u}
                    C(m-r, ceil((m-r)/2) - (a1-a2)) * C(r/2,a1) * C(r/2,a2) ]
                * e^eps / (e^eps + M' - 1)

    mean:   Pr[ (1/L) sum_l V_l >= v ] <= inf_{lam>=0}
                exp( L*log( sum_u e^{lam u} pi_eps(u) ) - lam*L*v )

    median: Pr[ Median{V_l} >= v ] <= C(L, ceil(L/2)) * P_eps(v)^{ceil(L/2)}

Setting the right-hand side to ``zeta``, fixing ``delta = 0`` and solving for ``eps``
gives the LDP bound; halving it gives ``eps_LB``. Both bounds are monotone
non-decreasing in ``eps`` (paper Thm. B.2), so evaluating at ``eps = eps_LB``
suffices to reject ``H0 : eps <= eps_LB`` at level ``zeta``.

In the vendored module, ``log_f_values(m, r)`` is the bracketed combinatorial sum
``f(v)`` and ``log_Z_closed_form(m)`` is ``log M'``; the ``e^eps/(e^eps + M'-1)``
factor enters through ``epsilon_lb_from_logM``.
"""

from __future__ import annotations

import inspect
from typing import Iterable, Sequence

from . import cum_runs_eps_lab as _vendored
from .cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb,
    compute_median_v_test_epsilon_lb,
)

__all__ = [
    "LDP_REDUCTION_FACTOR",
    "vendored_halves_internally",
    "epsilon_lb_mean",
    "epsilon_lb_median",
    "epsilon_lb_report",
]

#: Lemma 4.1: certified ``(eps, 0)`` unlearning implies a ``(2*eps, 0)``-LDP auditor.
#: The solved LDP epsilon is divided by this to obtain the reported ``eps_LB``.
LDP_REDUCTION_FACTOR = 2.0


def vendored_halves_internally(raw: dict | None = None) -> bool:
    """Does the vendored module already apply the Lemma 4.1 ``/2`` itself?

    **This is the guard against halving twice.** The vendored file in this repo is
    the upstream one, which reports the LDP epsilon and leaves the division to the
    caller. A newer revision of ``cum_runs_eps_lab.py`` does the division inside,
    returning *both* ``epsilon_lb_ldp`` (the LDP solution) and ``epsilon_lb`` (already
    the unlearning epsilon). Dropping that file in without this check would halve the
    reported bound a second time and silently understate every audit by 2x.

    Two independent signals, either of which is sufficient:

    * ``_unlearning_eps_from_ldp`` exists in the module -- the helper that performs
      the division. Module-level, so it covers every entry point.
    * the result dict carries an ``epsilon_lb_ldp`` key -- i.e. it is reporting the
      un-halved value separately, which only the halving revision does.

    ``r`` never enters this decision, so the two convention adapters stay independent.
    """
    if hasattr(_vendored, "_unlearning_eps_from_ldp"):
        return True
    return bool(raw is not None and "epsilon_lb_ldp" in raw)


def _vendored_accepts_delta(fn) -> bool:
    """Does this vendored entry point still take a ``delta`` argument?

    The upstream revision does (and ignores it for anything but ``delta = 0``); the
    halving revision dropped it, because the Lemma 4.1 reduction does not extend to
    ``delta > 0``. Probing the signature keeps this wrapper working against either.
    """
    return "delta" in inspect.signature(fn).parameters


def _delta_kwargs(fn, delta: float) -> dict:
    if _vendored_accepts_delta(fn):
        return {"delta": float(delta)}
    if float(delta) != 0.0:
        raise ValueError(
            f"the vendored {fn.__name__} takes no delta argument (pure-eps revision), "
            f"so delta={delta} cannot be honoured; the audit fixes delta = 0"
        )
    return {}

#: The paper fixes ``delta = 0`` for the main test (§4.1); the LDP reduction does not
#: extend cleanly to ``delta > 0`` (paper footnote 3).
DEFAULT_DELTA = 0.0

#: Confidence level ``zeta`` used throughout the paper (§7).
DEFAULT_ZETA = 0.05


def _validate(m: int, r: int, v_list: Sequence[int]) -> None:
    if r % 2 != 0:
        raise ValueError(f"r must be even (r/2 guesses per side); got r={r}")
    if not 0 < r <= m:
        raise ValueError(f"need 0 < r <= m; got r={r}, m={m}")
    if len(v_list) == 0:
        raise ValueError("v_list is empty; need at least one evaluation run")
    for i, v in enumerate(v_list):
        if not (0 <= v <= r):
            raise ValueError(
                f"overlap v_list[{i}]={v} outside [0, r]=[0, {r}]. "
                "V counts correct non-zero guesses out of r."
            )


def _finalize(raw: dict, m: int, r: int, v_list: Sequence[int], zeta: float,
              delta: float, statistic: str) -> dict:
    """Attach the Lemma 4.1 halving and audit metadata to a raw upstream result.

    Halves **only if the vendored module did not already do it** -- see
    :func:`vendored_halves_internally`. Either way the report carries the same two
    numbers (``epsilon_ldp_lb`` and ``epsilon_lb = epsilon_ldp_lb / 2``) plus
    ``halving_applied_by``, so which layer performed the division is on the record
    instead of being inferred from the magnitude of the answer.
    """
    already_halved = vendored_halves_internally(raw)

    if already_halved:
        # The module returned the unlearning epsilon. Keep it verbatim, and recover
        # the LDP value for the report from the module's own field when it has one,
        # else by undoing the division (exact: scaling by 2 is exact in binary FP).
        module_eps = raw.get("epsilon_lb")
        ldp = raw.get("epsilon_lb_ldp")
        if ldp is None and module_eps is not None:
            ldp = module_eps * LDP_REDUCTION_FACTOR
    else:
        module_eps = None          # nothing halved yet; this wrapper does it below
        ldp = raw.get("epsilon_lb")

    if ldp is None:
        # Upstream returns None when the bound is infeasible even at eps=0, i.e. the
        # observed overlap is not distinguishable from chance at this confidence.
        eps_lb = None
        note = (
            "Bound infeasible at eps=0: observed overlap is consistent with random "
            "guessing at this confidence level. No positive lower bound is certified."
        )
    elif ldp < 0.0:
        # A NEGATIVE solution certifies nothing: eps >= 0 by definition, so
        # "eps > negative" is vacuously true for every mechanism. The two upstream
        # entry points differ here -- the mean test checks feasibility at eps=0 and
        # returns None (cum_runs_eps_lab.py:193), but the median test solves
        # `log_numerator - log_denominator` unclamped (:958) and so can go negative
        # when the observed median sits at or below chance. We normalise the two to
        # the same contract rather than reporting a meaningless negative bound.
        eps_lb = None
        note = (
            f"Solved epsilon was negative ({float(ldp) / LDP_REDUCTION_FACTOR:.4f}), "
            "which certifies nothing since eps >= 0 by definition. Reported as None: "
            "the observed statistic is at or below chance. The raw negative value is "
            "kept under 'upstream' for inspection."
        )
    elif ldp == float("inf"):
        eps_lb = float("inf")
        note = "Upstream reported an unbounded/degenerate solution; inspect diagnostics."
    elif module_eps is not None:
        # Already halved inside the vendored module: pass it through untouched.
        eps_lb = float(module_eps)
        note = (
            "epsilon_lb came already halved from the vendored module (Lemma 4.1); the "
            "wrapper did NOT halve it again. Reject H0: eps <= epsilon_lb at level zeta."
        )
    else:
        eps_lb = float(ldp) / LDP_REDUCTION_FACTOR
        note = (
            "epsilon_lb = epsilon_ldp_lb / 2 per Lemma 4.1 (the auditor mechanism is "
            "(2*eps,0)-LDP). Reject H0: eps <= epsilon_lb at level zeta."
        )

    n = len(v_list)
    return {
        "statistic": statistic,
        "epsilon_lb": eps_lb,
        "epsilon_ldp_lb": None if (ldp is None or ldp < 0.0) else float(ldp),
        # Preserved verbatim so a negative/vacuous solve is still inspectable.
        "epsilon_ldp_raw": None if ldp is None else float(ldp),
        "certified": eps_lb is not None and eps_lb > 0.0,
        "ldp_reduction_factor": LDP_REDUCTION_FACTOR,
        # Which layer performed the /2. Guards against halving twice if the vendored
        # file is ever swapped for a revision that does it internally.
        "halving_applied_by": "vendored_module" if already_halved else "wrapper",
        "m": int(m),
        "r": int(r),
        "L": n,
        "zeta": float(zeta),
        "delta": float(delta),
        "v_list": [int(v) for v in v_list],
        "v_mean": sum(v_list) / n,
        "random_guess_baseline": r / 2.0,
        "max_attainable_v": int(r),
        "upstream": raw,
        "note": note,
    }


def epsilon_lb_mean(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    delta: float = DEFAULT_DELTA,
    theta_max: float = 50.0,
) -> dict:
    """Mean-based epsilon lower bound (the statistic the spec asks us to report).

    Parameters
    ----------
    m : int
        Number of candidate forget batches (candidate authors). ``m = 20`` here.
    r : int
        TOTAL guess budget, even. The auditor makes ``r/2`` positive and ``r/2``
        negative guesses and abstains on the remaining ``m - r`` candidates.
    v_list : iterable of int
        One overlap score ``V`` per independent evaluation run, each in ``[0, r]``.
    zeta : float
        Confidence level; the test's Type I error is at most ``zeta``.
    delta : float
        Held at ``0`` for the paper's main test.

    Returns
    -------
    dict with ``epsilon_lb`` (the reported unlearning bound) and
    ``epsilon_ldp_lb`` (before the Lemma 4.1 halving), plus diagnostics.
    """
    v_list = [int(v) for v in v_list]
    _validate(m, r, v_list)
    raw = compute_avg_v_test_epsilon_lb(
        m=int(m),
        r=int(r),
        T=len(v_list),
        v_list=v_list,
        ci_delta=float(zeta),
        direction="ge",  # we test avg(V) >= observed mean
        theta_max=float(theta_max),
        **_delta_kwargs(compute_avg_v_test_epsilon_lb, delta),
    )
    return _finalize(raw, m, r, v_list, zeta, delta, statistic="mean")


def epsilon_lb_median(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """Median-based epsilon lower bound (reported alongside the mean as a check)."""
    v_list = [int(v) for v in v_list]
    _validate(m, r, v_list)
    raw = compute_median_v_test_epsilon_lb(
        m=int(m),
        r=int(r),
        T=len(v_list),
        v_list=v_list,
        ci_delta=float(zeta),
        **_delta_kwargs(compute_median_v_test_epsilon_lb, delta),
    )
    return _finalize(raw, m, r, v_list, zeta, delta, statistic="median")


def epsilon_lb_report(
    m: int,
    r: int,
    v_list: Iterable[int],
    zeta: float = DEFAULT_ZETA,
    delta: float = DEFAULT_DELTA,
    theta_max: float = 50.0,
) -> dict:
    """Both statistics at one ``r``. ``mean`` is the headline number."""
    v_list = [int(v) for v in v_list]
    return {
        "mean": epsilon_lb_mean(m, r, v_list, zeta, delta, theta_max),
        "median": epsilon_lb_median(m, r, v_list, zeta, delta),
    }
