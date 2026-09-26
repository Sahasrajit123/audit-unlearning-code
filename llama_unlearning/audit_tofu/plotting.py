"""Diagnostic plots for the audit.

Matplotlib only, no seaborn, one chart per figure, default colours -- these are
diagnostics meant to be read quickly, not presentation graphics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "plot_calibration_distributions",
    "plot_lambda_distribution",
    "plot_epsilon_vs_r",
    "plot_overlap_per_run",
]


def _save(fig: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_calibration_distributions(
    calibration: Any, out_path: str | Path, method: str = ""
) -> Path:
    """Histograms of the fitted in/out calibration means, plus per-QA separation.

    The right panel is the diagnostic that matters for the ``noop`` control: if
    ``mu_out - mu_in`` is not clearly positive for most QA pairs, the attack has no
    signal to work with and the pipeline should be debugged before running NPO.
    """
    import matplotlib.pyplot as plt

    gs = [g for g in calibration.gaussians.values() if g.usable]
    mu_in = np.array([g.mu_in for g in gs])
    mu_out = np.array([g.mu_out for g in gs])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(mu_in, bins=30, alpha=0.6, label="in (trained, then unlearned)")
    axes[0].hist(mu_out, bins=30, alpha=0.6, label="out (never trained)")
    axes[0].set_xlabel("mean answer loss")
    axes[0].set_ylabel("number of QA pairs")
    axes[0].set_title(f"Calibration loss distributions {method}".strip())
    axes[0].legend()

    sep = mu_out - mu_in
    axes[1].hist(sep, bins=30, color="tab:purple", alpha=0.75)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"$\mu_{out} - \mu_{in}$  (positive = leakage)")
    axes[1].set_ylabel("number of QA pairs")
    axes[1].set_title(
        f"Per-QA separation (median {np.median(sep):.4f})" if sep.size else "no data"
    )

    return _save(fig, out_path)


def plot_lambda_distribution(
    lambdas_pos: Sequence[float],
    lambdas_neg: Sequence[float],
    out_path: str | Path,
    method: str = "",
) -> Path:
    """Author-level ``Lambda_j`` for true positive vs true negative candidates.

    Revealed only after predictions are finalized; a diagnostic, not an input.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = 30
    if len(lambdas_pos) or len(lambdas_neg):
        allv = np.concatenate(
            [np.asarray(lambdas_pos, float), np.asarray(lambdas_neg, float)]
        )
        finite = allv[np.isfinite(allv)]
        if finite.size:
            bins = np.linspace(finite.min(), finite.max(), 31)

    ax.hist(lambdas_pos, bins=bins, alpha=0.6, label="true $S_j=+1$")
    ax.hist(lambdas_neg, bins=bins, alpha=0.6, label="true $S_j=-1$")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(r"author log-likelihood ratio $\Lambda_j$")
    ax.set_ylabel("count")
    ax.set_title(f"Author likelihood ratios {method}".strip())
    ax.legend()
    return _save(fig, out_path)


def plot_epsilon_vs_r(
    per_r: Dict[int, Dict[str, Any]], out_path: str | Path, method: str = ""
) -> Path:
    """``epsilon_LB`` against the support size ``r``.

    The paper (Remark 4.3) predicts an inverted U: the attainable bound grows with
    ``r``, then falls once the auditor must commit to low-confidence batches.
    """
    import matplotlib.pyplot as plt

    rs = sorted(per_r)
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for stat, style in (("mean", "-o"), ("median", "--s")):
        ys, xs = [], []
        for r in rs:
            v = (per_r[r].get(stat) or {}).get("epsilon_lb")
            if v is not None and np.isfinite(v):
                xs.append(r)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, style, label=f"{stat}-based")

    ax.set_xlabel("support size $r$ (total guesses)")
    ax.set_ylabel(r"$\varepsilon_{LB}$")
    ax.set_title(f"Epsilon lower bound vs r {method}".strip())
    ax.set_xticks(rs)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_overlap_per_run(
    v_list: Sequence[int], r: int, out_path: str | Path, method: str = ""
) -> Path:
    """Per-run overlap ``V`` against the random-guessing baseline ``r/2``."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    idx = np.arange(len(v_list))
    ax.bar(idx, list(v_list), alpha=0.8)
    ax.axhline(r / 2.0, color="tab:red", linestyle="--", label=f"chance = r/2 = {r / 2:g}")
    ax.axhline(float(r), color="tab:green", linestyle=":", label=f"max = r = {r}")
    if len(v_list):
        ax.axhline(
            float(np.mean(v_list)), color="k", linewidth=1,
            label=f"mean = {np.mean(v_list):.2f}",
        )
    ax.set_xlabel("evaluation run")
    ax.set_ylabel("overlap $V$")
    ax.set_title(f"Per-run overlap (r={r}) {method}".strip())
    ax.set_xticks(idx)
    ax.legend()
    return _save(fig, out_path)
