"""
Membership-inference-style auditing for a run_sweep.py sweep.

Adapted from the PyTorch pointwise-stats path of an earlier audit codebase: `create_model`,
`find_model_file`, `compute_phi_and_loss_per_point`,
`_summarize_pointwise_values`, `_resolve_or_compute_pointwise_stats_path`,
`_log_pdf_gaussian`, `compute_point_llr_pointwise`, and `evaluate_batch_llr_predictions`.
That file's JAX/orbax batch-loading machinery (checkpoint restore, JAX batch loaders) is
NOT ported here -- it's specific to that repo's data/checkpoint format, not this project's.

Also ported: the epsilon-lower-bound "guessing game" audit (`build_v_s_vectors`,
`compute_overlap_score`, `kl_bernoulli`, `epsilon_lower_bound_from_vs`,
`compute_eps_bounds_for_all_runs`, adapted from `compute_eps_lower_bound` /
`_compute_v_for_single_run` / `compute_eps_bounds_for_all_runs`), plus its combinatorial
math dependency `cum_runs_eps_lab.py` (ported verbatim -- pure numpy/scipy, no
JAX/torch/data-format dependency). See that section's docstring below for how it works.

The same overlap scores feed three parallel audits, one per privacy notion, all sharing
the single expensive attack pass (`_collect_v_list_for_runs`):
  - eps-DP    (`compute_eps_bounds_for_all_runs`) -- pure (eps, 0) only; the reported
    epsilon_lb is HALF the rejected LDP parameter of the audit mechanism, because
    "(eps,0)-certified unlearning => M is (2 eps,0)-LDP". There is no delta > 0 version.
  - rho-zCDP  (`compute_rho_bounds_for_all_runs`) -- no halving (the factor sits inside
    the local RDP bound).
  - mu-GDP    (`compute_mu_bounds_for_all_runs`) -- no halving (the factor of 2 is inside
    mu_loc(mu) = 2 mu, the exact f-DP group operation).
`compute_all_bounds_for_all_runs` runs all three off one attack pass. The zCDP/GDP audits
additionally report "eps_estimate", an (eps, conv_delta) conversion of rho_lb/mu_lb that
is comparable with epsilon_lb on the same axis but is NOT itself a lower bound on eps.

Core idea (LiRA-style membership inference, Carlini et al. 2022): run_sweep.py's runs each
randomly include/exclude a different subset of the forget set (see run_sweep.py's
--forget-prob and forget_indices.npy), so every forget point ends up "in" (its index
appears in that run's forget_indices.npy -- it was actually part of that run's training
set) for roughly half the runs and "out" for the other half. For each point, computing a
per-run statistic from that run's saved model -- phi = log-odds of the true class,
log(p / (1-p)), or the cross-entropy loss -- and bucketing the observations by in/out
gives two empirical distributions per point. Fitting a Gaussian to each (mean, variance) --
plus the median, since a heavy-tailed metric like loss can have mean != median -- lets you
later score a NEW observation via the log-likelihood ratio between the two Gaussians
(`compute_point_llr_pointwise`): a positive LLR says "this observation looks more like the
point was in than out." For an unlearned model, points from the forget set should score
close to the "out" distribution -- if they still score like "in", unlearning left a
detectable trace.

Typical usage:
    generate_path = compute_pointwise_forget_stats(
        run_dir="runs_cifar100_bs_1", dataset="cifar100",
        data_dir="data/cifar100/data_split/cifar100_bs_1", use_trained_model=False,
    )
    stats = json.loads(Path(generate_path).read_text())
    llr = compute_point_llr_pointwise(stats["points"]["point_5"], metric="phi", obs_value=1.23)
"""
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audit.cum_runs_eps_lab import (
    compute_avg_v_test_epsilon_lb, compute_median_v_test_epsilon_lb,
    compute_avg_v_test_rho_lb, compute_median_v_test_rho_lb,
    compute_avg_v_test_mu_lb, compute_median_v_test_mu_lb,
)
from core.data_utils import load_full_forget
from core.model import build_model, dataset_defaults


def _model_filename(use_trained_model: bool = False, unlearn_epoch: Optional[int] = None) -> str:
    """
    Decide which checkpoint filename to use for a run:
      - unlearn_epoch given: `unlearned_model_epoch_{unlearn_epoch}.pth` -- an intermediate
        checkpoint, only present for a run_sweep.py run launched with --unlearn-epochs > 1
        (see unlearn.py's on_epoch_end hook / run_sweep.py's _save_intermediate_unlearn_checkpoint).
        Takes precedence over use_trained_model.
      - use_trained_model=True: `trained_model.pth` (pre-unlearning).
      - otherwise (default): `unlearned_model.pth` (the final, post-unlearning checkpoint).
    """
    if unlearn_epoch is not None:
        return f"unlearned_model_epoch_{unlearn_epoch}.pth"
    return "trained_model.pth" if use_trained_model else "unlearned_model.pth"


def _model_source_label(use_trained_model: bool = False, unlearn_epoch: Optional[int] = None) -> str:
    """The `model_source` string stored in a pointwise-stats JSON and read back by attack_run
    to know which checkpoint file to load for the run being attacked."""
    if unlearn_epoch is not None:
        return f"unlearn_epoch_{unlearn_epoch}"
    return "trained" if use_trained_model else "unlearned"


def _model_filename_from_source(model_source: str) -> str:
    """Inverse of _model_source_label: turn a stored model_source string back into a filename."""
    if model_source == "trained":
        return "trained_model.pth"
    if model_source.startswith("unlearn_epoch_"):
        epoch = model_source[len("unlearn_epoch_"):]
        return f"unlearned_model_epoch_{epoch}.pth"
    return "unlearned_model.pth"


def find_model_file(run_dir: Union[str, Path], use_trained: bool = False,
                     unlearn_epoch: Optional[int] = None) -> Path:
    """
    Locate a run_sweep.py run's saved model. Simpler than the reference repo's version
    (no keyword search / models-subdir fallback) since run_sweep.py always writes exactly
    `trained_model.pth` / `unlearned_model.pth` (and, for --unlearn-epochs > 1,
    `unlearned_model_epoch_{N}.pth`) directly under each run_XX/ folder.
    """
    run_dir = Path(run_dir)
    filename = _model_filename(use_trained, unlearn_epoch)
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found in {run_dir}")
    return path


@torch.no_grad()
def compute_phi_and_loss_per_point(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                                    device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-point phi (log-odds of the true class) and cross-entropy loss for one batch.
    phi = log(p / (1-p)) where p = softmax(logits)[true class] -- the standard LiRA
    statistic (Carlini et al. 2022), clamped away from 0/1 to avoid +-inf.
    Returns (phi, loss), each shape (N,).
    """
    model.eval()
    x, y = x.to(device), y.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    idx = torch.arange(len(y), device=y.device)
    p = probs[idx, y].double().clamp(min=1e-9, max=1 - 1e-9)
    phi = torch.log(p / (1.0 - p)).cpu().numpy()
    loss = F.nll_loss(F.log_softmax(logits, dim=1), y, reduction="none").cpu().numpy()
    return phi, loss


def score_forget_set(model: torch.nn.Module, full_forget_set, device: torch.device,
                      batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (phi, loss) for every point in the full forget set, in fixed index order."""
    loader = DataLoader(full_forget_set, batch_size=batch_size, shuffle=False)
    all_phi, all_loss = [], []
    for x, y in loader:
        phi, loss = compute_phi_and_loss_per_point(model, x, y, device)
        all_phi.append(phi)
        all_loss.append(loss)
    return np.concatenate(all_phi), np.concatenate(all_loss)


def _summarize_pointwise_values(vals: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean, var, median) for a list of values; (None, None, None) if empty."""
    if len(vals) == 0:
        return None, None, None
    arr = np.array(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    var = float(np.var(arr)) if len(vals) > 1 else 0.0
    median = float(np.median(arr))
    return mean, var, median


def compute_pointwise_forget_stats(
    run_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    use_trained_model: bool = False,
    unlearn_epoch: Optional[int] = None,
    device: str = "cpu",
    batch_size: int = 256,
    output_path: Optional[Union[str, Path]] = None,
    print_progress: bool = True,
) -> Path:
    """
    Stage 1 of the audit: generate the in/out pointwise-stats JSON that the rest of this
    module (`compute_point_llr_pointwise`, `evaluate_point_llr_predictions`) consumes.

    For every forget point, across every run_XX/ folder under `run_dir`:
      - load that run's model and its forget_indices.npy,
      - compute (phi, loss) for every point in the full forget set under that model,
      - bucket each point's observation into "in" (point_idx in that run's
        forget_indices.npy) or "out" (not in it).
    Then summarize each point's in-list and out-list with (mean, var, median) for both
    metrics, and write everything to JSON.

    Args:
        run_dir: a run_sweep.py sweep directory containing run_01/, run_02/, ...
        dataset, data_dir: passed to data_utils.load_full_forget / model.dataset_defaults.
        model_name, num_classes: default to dataset_defaults(dataset) if not given.
        use_trained_model: score with the pre-unlearning model (trained_model.pth) instead
            of the post-unlearning model (unlearned_model.pth, the default). Ignored if
            unlearn_epoch is given. Comparing use_trained_model=True/False audits is a
            natural way to check whether unlearning actually reduced the "in" signal on
            forget points.
        unlearn_epoch: instead of the final unlearned_model.pth, score with the
            intermediate checkpoint from unlearning epoch `unlearn_epoch`
            (unlearned_model_epoch_{unlearn_epoch}.pth -- only present for runs launched
            with --unlearn-epochs > 1, see run_sweep.py). Takes precedence over
            use_trained_model. Lets you watch how the "in" signal decays epoch by epoch
            through unlearning, not just before-vs-after.
        output_path: defaults to run_dir/forget_pointwise_stats_{trained,unlearned,
            unlearn_epoch_N}.json.

    Returns:
        Path to the written JSON file.
    """
    run_dir = Path(run_dir)
    run_subdirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
    if not run_subdirs:
        raise FileNotFoundError(f"No run_* subfolders found under {run_dir}")

    model_source = _model_source_label(use_trained_model, unlearn_epoch)
    model_filename = _model_filename(use_trained_model, unlearn_epoch)

    default_input_size, default_num_classes, default_model_name = dataset_defaults(dataset)
    input_size = default_input_size
    num_classes = num_classes if num_classes is not None else default_num_classes
    model_name = model_name if model_name is not None else default_model_name

    device_obj = torch.device(device)
    model = build_model(model_name, num_classes, input_size=input_size, filters_percentage=filters).to(device_obj)

    full_forget_set = load_full_forget(dataset, data_dir)
    num_points = len(full_forget_set)
    loader = DataLoader(full_forget_set, batch_size=batch_size, shuffle=False)

    in_phi: List[List[float]] = [[] for _ in range(num_points)]
    in_loss: List[List[float]] = [[] for _ in range(num_points)]
    out_phi: List[List[float]] = [[] for _ in range(num_points)]
    out_loss: List[List[float]] = [[] for _ in range(num_points)]

    valid_run_count = 0
    for run_subdir in run_subdirs:
        indices_path = run_subdir / "forget_indices.npy"
        model_path = run_subdir / model_filename
        if not indices_path.exists() or not model_path.exists():
            if print_progress:
                print(f"[audit] skipping {run_subdir.name}: missing forget_indices.npy or model checkpoint")
            continue

        in_set: Set[int] = set(int(i) for i in np.load(indices_path).tolist())
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        valid_run_count += 1

        point_offset = 0
        for x, y in loader:
            phi, loss = compute_phi_and_loss_per_point(model, x, y, device_obj)
            for i in range(len(phi)):
                point_idx = point_offset + i
                if point_idx in in_set:
                    in_phi[point_idx].append(float(phi[i]))
                    in_loss[point_idx].append(float(loss[i]))
                else:
                    out_phi[point_idx].append(float(phi[i]))
                    out_loss[point_idx].append(float(loss[i]))
            point_offset += len(phi)

        if print_progress:
            print(f"[audit] {run_subdir.name}: {len(in_set)}/{num_points} points in ({model_source} model)")

    if valid_run_count == 0:
        raise ValueError(f"No usable runs found under {run_dir}")

    points: Dict[str, Dict] = {}
    for point_idx in range(num_points):
        in_mean_phi, in_var_phi, in_med_phi = _summarize_pointwise_values(in_phi[point_idx])
        in_mean_loss, in_var_loss, in_med_loss = _summarize_pointwise_values(in_loss[point_idx])
        out_mean_phi, out_var_phi, out_med_phi = _summarize_pointwise_values(out_phi[point_idx])
        out_mean_loss, out_var_loss, out_med_loss = _summarize_pointwise_values(out_loss[point_idx])
        points[f"point_{point_idx}"] = {
            "point_idx": point_idx,
            "in": {
                "count": len(in_phi[point_idx]),
                "mean_phi": in_mean_phi, "var_phi": in_var_phi, "median_phi": in_med_phi,
                "mean_loss": in_mean_loss, "var_loss": in_var_loss, "median_loss": in_med_loss,
            },
            "out": {
                "count": len(out_phi[point_idx]),
                "mean_phi": out_mean_phi, "var_phi": out_var_phi, "median_phi": out_med_phi,
                "mean_loss": out_mean_loss, "var_loss": out_var_loss, "median_loss": out_med_loss,
            },
        }

    payload = {
        "generated_by": "compute_pointwise_forget_stats",
        "run_dir": str(run_dir),
        "model_source": model_source,
        "dataset": dataset, "data_dir": str(data_dir),
        "model": model_name, "num_classes": num_classes,
        "num_runs": valid_run_count, "num_points": num_points,
        "points": points,
    }

    if output_path is None:
        output_path = run_dir / f"forget_pointwise_stats_{model_source}.json"
    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload, indent=2))
    if print_progress:
        print(f"[audit] saved pointwise in/out stats ({num_points} points, {valid_run_count} runs) to {output_path}")
    return output_path


def _log_pdf_gaussian(x: float, mu: float, var: float, eps: float = 1e-10) -> float:
    """Log probability density of x under Gaussian(mu, var)."""
    var = max(var, eps)
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mu) ** 2) / var


def compute_point_llr_pointwise(
    point_stats: Dict,
    metric: str,
    obs_value: float,
    *,
    in_key: str = "in",
    out_key: str = "out",
    eps: float = 1e-10,
) -> float:
    """
    LLR = log p_in(obs) - log p_out(obs) for one point, under Gaussians fit by
    compute_pointwise_forget_stats. Positive LLR => obs looks more like the point was
    included in training than excluded.

    Args:
        point_stats: one entry from stats["points"][...], with "in"/"out" sub-dicts
            holding mean_phi/var_phi/mean_loss/var_loss.
        metric: "phi" or "loss".
        obs_value: the observed phi or loss for this point under the model being audited.

    Raises:
        ValueError: if mean/var is missing for either side (e.g. a point that was never
            "in" or never "out" across the sweep -- widen --num-runs or --forget-prob).
    """
    if metric not in ("phi", "loss"):
        raise ValueError("metric must be 'phi' or 'loss'")
    in_stats = point_stats.get(in_key, {})
    out_stats = point_stats.get(out_key, {})
    mu_in, var_in = in_stats.get(f"mean_{metric}"), in_stats.get(f"var_{metric}")
    mu_out, var_out = out_stats.get(f"mean_{metric}"), out_stats.get(f"var_{metric}")
    if mu_in is None or var_in is None or mu_out is None or var_out is None:
        raise ValueError(
            f"Point stats missing mean/var for metric={metric!r}: "
            f"in=({mu_in}, {var_in}) out=({mu_out}, {var_out})"
        )
    return _log_pdf_gaussian(obs_value, mu_in, var_in, eps) - _log_pdf_gaussian(obs_value, mu_out, var_out, eps)


def attack_run(
    stats: Dict,
    run_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
) -> Tuple[Dict[int, float], Set[int]]:
    """
    Run the LiRA-style membership-inference attack against ONE run: score every forget
    point under that run's saved model (trained or unlearned, matching
    stats["model_source"]) and compute each point's LLR against the Gaussians in `stats`.

    IMPORTANT: `stats` should be built (via compute_pointwise_forget_stats) from DIFFERENT
    runs than `run_dir` -- e.g. stats from run_01..run_50 attacking held-out run_51..run_60
    -- otherwise the attack partly "cheats" by having the very observation it's trying to
    classify baked into its own reference distribution.

    Args:
        stats: a loaded pointwise-stats payload (i.e. json.loads of a
            compute_pointwise_forget_stats output), or just its "points" dict.
        run_dir: the run_XX/ folder being attacked -- must have forget_indices.npy (used
            only as ground truth for scoring the attack, never as attack input) and the
            model file matching stats["model_source"].
        metric: "phi" or "loss" -- which statistic's Gaussians to score against.

    Returns:
        (point_llrs, ground_truth_in): point_llrs maps point_idx -> LLR; ground_truth_in
        is the set of point indices actually in that run's forget_indices.npy (the answer
        the attack is trying to guess -- pass to evaluate_point_llr_predictions).
    """
    run_dir = Path(run_dir)
    points = stats.get("points", stats)
    model_path = run_dir / _model_filename_from_source(stats.get("model_source", "unlearned"))
    indices_path = run_dir / "forget_indices.npy"
    if not model_path.exists() or not indices_path.exists():
        raise FileNotFoundError(f"Missing {model_path.name} or forget_indices.npy in {run_dir}")

    default_input_size, default_num_classes, default_model_name = dataset_defaults(dataset)
    input_size = default_input_size
    num_classes = num_classes if num_classes is not None else default_num_classes
    model_name = model_name if model_name is not None else default_model_name

    device_obj = torch.device(device)
    model = build_model(model_name, num_classes, input_size=input_size, filters_percentage=filters).to(device_obj)
    model.load_state_dict(torch.load(model_path, map_location=device_obj, weights_only=False))
    model.eval()

    full_forget_set = load_full_forget(dataset, data_dir)
    phi, loss = score_forget_set(model, full_forget_set, device_obj, batch_size)
    obs = phi if metric == "phi" else loss

    ground_truth_in: Set[int] = set(int(i) for i in np.load(indices_path).tolist())

    point_llrs: Dict[int, float] = {}
    for point_idx in range(len(obs)):
        point_stats = points.get(f"point_{point_idx}")
        if point_stats is None:
            continue
        try:
            point_llrs[point_idx] = compute_point_llr_pointwise(point_stats, metric, float(obs[point_idx]))
        except ValueError:
            continue  # this point was never observed as both in and out across the stats-building sweep

    return point_llrs, ground_truth_in


def evaluate_point_llr_predictions(
    point_llrs: Dict[Union[int, str], float],
    ground_truth_in: Set[int],
    k: int,
    *,
    threshold: float = 0.0,
) -> Dict:
    """
    Evaluate point-level LLR predictions against ground truth: top-k / bottom-k accuracy
    plus a confusion matrix at `threshold`. Adapted from the reference repo's
    evaluate_batch_llr_predictions (renamed point-level since this project scores
    individual forget points rather than whole training batches).

    Args:
        point_llrs: point_idx -> LLR (e.g. from compute_point_llr_pointwise for a run
            under audit).
        ground_truth_in: set of point indices actually "in" for the run being audited
            (i.e. that run's forget_indices.npy).
        k: top-k / bottom-k size.
        threshold: predict "in" when llr > threshold.
    """
    gt = set(int(x) for x in ground_truth_in)
    llrs = {int(p): float(v) for p, v in point_llrs.items()}
    if not llrs:
        return {
            "top_k_accuracy": 0.0, "bottom_k_accuracy": 0.0, "combined_accuracy": 0.0,
            "precision": 0.0, "recall": 0.0,
            "true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0,
            "num_points": 0, "num_in_points": len(gt),
        }
    sorted_items = sorted(llrs.items(), key=lambda t: t[1], reverse=True)
    point_ids = [p for p, _ in sorted_items]
    llr_arr = np.array([v for _, v in sorted_items], dtype=float)
    k_actual = min(k, len(point_ids))
    top_k_ids, bottom_k_ids = point_ids[:k_actual], point_ids[-k_actual:]

    gt_arr = np.array([p in gt for p in point_ids], dtype=bool)
    top_k_correct = sum(1 for p in top_k_ids if p in gt)
    bottom_k_correct = sum(1 for p in bottom_k_ids if p not in gt)

    pred_in = llr_arr > threshold
    tp = int(((pred_in) & gt_arr).sum())
    fp = int(((pred_in) & (~gt_arr)).sum())
    tn = int(((~pred_in) & (~gt_arr)).sum())
    fn = int(((~pred_in) & gt_arr).sum())

    return {
        "top_k_accuracy": top_k_correct / k_actual if k_actual else 0.0,
        "bottom_k_accuracy": bottom_k_correct / k_actual if k_actual else 0.0,
        "combined_accuracy": (top_k_correct + bottom_k_correct) / (2 * k_actual) if k_actual else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
        "num_points": len(llrs), "num_in_points": len(gt),
    }


# ---------------------------------------------------------------------
# Epsilon lower bound ("guessing game" DP audit)
# ---------------------------------------------------------------------
# Adapted from the reference repo's compute_eps_lower_bound / _compute_v_for_single_run /
# compute_eps_bounds_for_all_runs. The underlying scheme (Steinke, Nasr & Jagielski,
# "Privacy Auditing with One (1) Training Run", 2023): for a run whose true "in" set is
# known (forget_indices.npy), pick the k points the attack is MOST confident are "in" (top-k
# LLR) and the k points it's most confident are "out" (bottom-k LLR) -- 2k guesses total --
# then count v = how many of those 2k guesses were actually correct. Under a
# perfectly-forgetting model, an attacker's guesses on the forget set should be no better
# than chance, so v should look like a random draw from a known null (hypergeometric-ish)
# distribution; if v is implausibly high under that null, that's evidence of an epsilon
# lower bound -- a certificate that the (unlearning) mechanism leaks at least this much
# privacy. `cum_runs_eps_lab.py` (ported verbatim, pure numpy/scipy) turns v (or v's
# collected across several independent runs, T of them) into that epsilon lower bound via
# a Chernoff-style concentration bound on the null distribution.
#
# What the audit certifies, and the factor of 2. cum_runs_eps_lab's solver bounds the LDP
# parameter of the audit mechanism M ("which forget set was unlearned?"). The bridge to a
# certified-unlearning statement is the lemma
#     (eps, 0)-certified unlearning  =>  M is (2 eps, 0)-LDP,
# so rejecting every LDP parameter <= eps_ldp rejects every unlearning epsilon <= eps_ldp/2.
# That halving is applied inside cum_runs_eps_lab (see _unlearning_eps_from_ldp), so the
# "epsilon_lb" reported here is already a certified-unlearning epsilon; the un-halved LDP
# value travels alongside it as "epsilon_lb_ldp" in the detail dicts.
#
# The lemma rests on the triangle inequality for indistinguishability, which does NOT
# survive delta > 0 (X ~_{eps,delta} Y ~_{eps,delta} Z only gives X ~_{2 eps, (1+e^eps)
# delta} Z), so this audit is pure-eps: `delta` must be 0. For an approximate-DP-flavoured
# guarantee, use the zCDP (rho) or GDP (mu) audits below instead -- their reductions carry
# the factor of 2 internally (eps_gamma^loc for zCDP, mu_loc(mu) = 2 mu for GDP), so their
# bounds are NOT halved.
#
# This project's point-level LLR machinery (attack_run, compute_point_llr_pointwise) plays
# the role of the reference repo's batch-level select_batches_by_likelihood -- "select the
# top/bottom k FORGET POINTS by LLR" instead of "top/bottom k forget BATCHES".

def build_v_s_vectors(selected_idx, non_selected_idx, chosen_idx, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the attack's guess vector `v` and the ground-truth vector `s`, both length N.
      v[i] = +1 if point i was guessed "in" (top-k LLR), -1 if guessed "out" (bottom-k
             LLR), 0 if not guessed (not in either top-k or bottom-k).
      s[i] = +1 if point i was actually in the audited run's training set
             (i in chosen_idx, i.e. forget_indices.npy), else -1.
    """
    v = np.zeros(N, dtype=int)
    v[selected_idx] = +1
    v[non_selected_idx] = -1

    s = np.full(N, -1, dtype=int)
    s[chosen_idx] = +1
    return v, s


def compute_overlap_score(v: np.ndarray, s: np.ndarray) -> int:
    """v = number of correct guesses out of the 2k made: sum over i of max(0, v_i * s_i)
    (a guess only contributes +1 when it was actually made -- v_i != 0 -- AND correct)."""
    return int(np.maximum(v * s, 0).sum())


def kl_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    """KL divergence between Bernoulli(p) and Bernoulli(q). Ported for completeness from
    the reference repo; not used by the Chernoff-based bounds below, which use
    cum_runs_eps_lab.py's tighter combinatorial null distribution instead."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def epsilon_lower_bound_from_vs(
    result: Union[int, List[int]],
    k: int,
    N: int,
    confidence_level: float = 0.95,
    delta: float = 0.0,
    use_median: bool = False,
) -> Union[float, Dict]:
    """
    Turn one v (single run) or a list of v's (T independent runs) into an epsilon lower
    bound, via cum_runs_eps_lab.py's average-based or median-based Chernoff test.

    This is the (eps, 0) audit only -- see the section header on why the LDP reduction it
    relies on does not extend to delta > 0. The returned epsilon is a lower bound on the
    certified-unlearning epsilon, i.e. HALF the rejected LDP parameter of the audit
    mechanism M (cum_runs_eps_lab applies that factor of 2; the un-halved value is
    available as "epsilon_lb_ldp" in the result dict).

    Args:
        result: overlap score (from compute_overlap_score) for one run, or a list of them
            for T runs.
        k: top/bottom-k size used to build v (so r = 2k guesses were made per run).
        N: total number of forget points (m in cum_runs_eps_lab.py's notation).
        confidence_level: mapped to ci_delta = 1 - confidence_level.
        delta: kept only so existing call sites keep parsing; must be 0.0.
        use_median: use the median-based test instead of the average-based test.

    Returns:
        For a single run (T=1): the epsilon lower bound (float), None if the observation is
        consistent with eps=0, or inf if it is impossible under any finite eps. For
        multiple runs: the full result dict from cum_runs_eps_lab.py (has "epsilon_lb",
        "epsilon_lb_ldp", plus diagnostic fields).
    """
    if delta != 0.0:
        raise ValueError(
            f"delta={delta}: the overlap audit only supports delta = 0. The LDP reduction "
            "it relies on ((eps,0)-certified unlearning => M is (2 eps,0)-LDP) does not "
            "extend to delta > 0. Use rho_lower_bound_from_vs (zCDP) or "
            "mu_lower_bound_from_vs (GDP) if an approximate-DP-style guarantee is wanted; "
            "both report an (eps, conv_delta) estimate alongside their bound."
        )

    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level

    v_list = list(result) if isinstance(result, (list, np.ndarray)) else [result]
    T = len(v_list)

    if use_median:
        result_dict = compute_median_v_test_epsilon_lb(m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta)
    else:
        result_dict = compute_avg_v_test_epsilon_lb(
            m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta, direction="ge", theta_max=50.0,
        )

    if T == 1:
        return result_dict.get("epsilon_lb")
    return result_dict


def rho_lower_bound_from_vs(
    result: Union[int, List[int]],
    k: int,
    N: int,
    confidence_level: float = 0.95,
    use_median: bool = False,
    gamma_max: float = 1e4,
    conv_delta: float = 1e-3,
) -> Union[float, Dict]:
    """
    zCDP analogue of epsilon_lower_bound_from_vs: turn one v (single run) or a list of
    v's (T independent runs) into a rho lower bound, via cum_runs_eps_lab.py's
    average-based or median-based zCDP test (compute_avg_v_test_rho_lb /
    compute_median_v_test_rho_lb). Unlike the eps-DP test, this has no audit-noise
    `delta` parameter -- the zCDP pointwise bound pi_eps(u) is delta-free.

    Unlike the eps-DP bound, rho_lb is NOT halved: the factor from the reduction
    through the reference law already sits inside the local RDP bound
    eps_gamma^loc(rho) = 4 rho gamma, so rho_lb bounds the certified-unlearning rho
    directly.

    Args:
        result: overlap score (from compute_overlap_score) for one run, or a list of them
            for T runs.
        k: top/bottom-k size used to build v (so r = 2k guesses were made per run).
        N: total number of forget points (m in cum_runs_eps_lab.py's notation).
        confidence_level: mapped to ci_delta = 1 - confidence_level.
        use_median: use the median-based test instead of the average-based test.
        gamma_max: upper bound for the Renyi-order search in the zCDP conversion.
        conv_delta: delta at which the result dict's "eps_estimate" reports the
            (eps, delta) conversion of rho_lb. That conversion runs in the forward
            direction (a rho guarantee implies an (eps, delta) guarantee), so applied to
            an audited lower bound it is an estimate, NOT a lower bound on eps.

    Returns:
        For a single run (T=1): the rho lower bound (float), or None if infeasible at
        rho=0. For multiple runs: the full result dict from cum_runs_eps_lab.py (has
        "rho_lb", "eps_estimate", "conv_delta", plus diagnostic fields).
    """
    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level

    v_list = list(result) if isinstance(result, (list, np.ndarray)) else [result]
    T = len(v_list)

    if use_median:
        result_dict = compute_median_v_test_rho_lb(m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta,
                                                   gamma_max=gamma_max, conv_delta=conv_delta)
    else:
        result_dict = compute_avg_v_test_rho_lb(m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta,
                                                gamma_max=gamma_max, conv_delta=conv_delta)

    if T == 1:
        return result_dict.get("rho_lb")
    return result_dict


def mu_lower_bound_from_vs(
    result: Union[int, List[int]],
    k: int,
    N: int,
    confidence_level: float = 0.95,
    use_median: bool = False,
    theta_max: float = 50.0,
    conv_delta: float = 1e-3,
) -> Union[float, Dict]:
    """
    Gaussian-DP (mu-GDP) analogue of epsilon_lower_bound_from_vs / rho_lower_bound_from_vs:
    turn one v (single run) or a list of v's (T independent runs) into a mu lower bound,
    via cum_runs_eps_lab.py's compute_avg_v_test_mu_lb / compute_median_v_test_mu_lb.

    Those audit the GDP parameter straight from its hypothesis-testing semantics: the
    chance-overlap null tail bound q at the observed statistic is turned into
        mu_lb = sup{mu : Phi(Phi^{-1}(q) + 2 sqrt(T) mu) <= ci_delta} = tau/2,
        tau  = (Phi^{-1}(ci_delta) - Phi^{-1}(q)) / sqrt(T),
    using the exact f-DP group operation mu_loc(mu) = 2 mu (Dong, Roth & Su 2022, Thm 3
    with k=2) for the reduction through the common reference law. No Renyi detour, and no
    assumption that the model laws are Gaussian.

    Like rho_lb and unlike the eps-DP bound, mu_lb is NOT halved -- the factor of 2 is
    already inside mu_loc -- so it bounds the certified-unlearning GDP parameter directly.

    Args:
        result: overlap score (from compute_overlap_score) for one run, or a list of them
            for T runs.
        k: top/bottom-k size used to build v (so r = 2k guesses were made per run).
        N: total number of forget points (m in cum_runs_eps_lab.py's notation).
        confidence_level: mapped to ci_delta = 1 - confidence_level.
        use_median: use the median-based test instead of the average-based test.
        theta_max: cap for the Chernoff lambda search in the mean test's null tail bound.
        conv_delta: delta at which the result dict's "eps_estimate" reports the
            (eps, delta) conversion of mu_lb, obtained by inverting delta = theta_eps(mu).
            That inversion is exact for GDP, but applied to an audited lower bound the
            result is an estimate, NOT a lower bound on eps.

    Returns:
        For a single run (T=1): the mu lower bound (float), 0.0 when the observation is
        not significant at this confidence level. For multiple runs: the full result dict
        from cum_runs_eps_lab.py (has "mu_lb", "eps_estimate", "conv_delta", plus
        diagnostic fields).
    """
    r = 2 * k
    m = N
    ci_delta = 1 - confidence_level

    v_list = list(result) if isinstance(result, (list, np.ndarray)) else [result]
    T = len(v_list)

    if use_median:
        result_dict = compute_median_v_test_mu_lb(m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta,
                                                  conv_delta=conv_delta)
    else:
        result_dict = compute_avg_v_test_mu_lb(m=m, r=r, T=T, v_list=v_list, ci_delta=ci_delta,
                                               theta_max=theta_max, conv_delta=conv_delta)

    if T == 1:
        return result_dict.get("mu_lb")
    return result_dict


def _compute_v_for_single_run(
    stats: Dict,
    run_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    k: int,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
) -> Tuple[int, int]:
    """
    Compute the overlap score v for one run: attack it (see attack_run), take the top-k
    and bottom-k forget points by LLR as the attack's 2k guesses, and score them against
    that run's real forget_indices.npy. Returns (v, m) where m = total forget points.
    """
    point_llrs, ground_truth_in = attack_run(
        stats, run_dir, dataset, data_dir, model_name=model_name, num_classes=num_classes,
        filters=filters, device=device, batch_size=batch_size, metric=metric,
    )
    m = stats.get("num_points", len(point_llrs))
    if len(point_llrs) < 2 * k:
        raise ValueError(
            f"Need at least 2*k={2 * k} points with valid LLRs to pick top-k/bottom-k; "
            f"got {len(point_llrs)}. Use a smaller k or a sweep with more runs "
            f"(so more points were observed as both in and out)."
        )

    sorted_items = sorted(point_llrs.items(), key=lambda t: t[1], reverse=True)
    selected_idx = [p for p, _ in sorted_items[:k]]       # guessed "in"
    non_selected_idx = [p for p, _ in sorted_items[-k:]]  # guessed "out"

    v, s = build_v_s_vectors(selected_idx, non_selected_idx, sorted(ground_truth_in), N=m)
    return compute_overlap_score(v, s), m


def _collect_v_list_for_runs(
    stats: Dict,
    test_runs_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    k: int,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
    print_progress: bool = True,
    tag: str = "audit",
) -> Tuple[List[int], List[str], List[Tuple[str, str]], int]:
    """
    Attack every held-out run under `test_runs_dir` and collect one overlap score v per run
    (see _compute_v_for_single_run). This is the only expensive part of the audit -- one
    model-inference pass per run -- and it is identical whichever privacy notion the v's
    are later turned into, so the eps / rho / mu drivers below all share it, and
    compute_all_bounds_for_all_runs gets all three bounds out of a single pass.

    Runs that fail are collected in failed_runs rather than aborting the sweep.

    Returns (v_list, run_ids, failed_runs, m) where m = total forget points.
    """
    test_runs_dir = Path(test_runs_dir)
    run_dirs = sorted(d for d in test_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* subfolders found under {test_runs_dir}")

    v_list, run_ids, failed_runs = [], [], []
    m = None
    for run_dir in run_dirs:
        try:
            v, m_run = _compute_v_for_single_run(
                stats, run_dir, dataset, data_dir, k, model_name=model_name, num_classes=num_classes,
                filters=filters, device=device, batch_size=batch_size, metric=metric,
            )
            v_list.append(v)
            run_ids.append(run_dir.name)
            m = m_run if m is None else m
            if print_progress:
                print(f"[{tag}] {run_dir.name}: v={v} (of r={2 * k} guesses)")
        except Exception as e:
            failed_runs.append((run_dir.name, str(e)))
            if print_progress:
                print(f"[{tag}] {run_dir.name}: FAILED - {e}")

    if not v_list:
        raise ValueError(f"No usable runs found under {test_runs_dir}")

    return v_list, run_ids, failed_runs, m


def _eps_bounds_from_v_list(
    v_list: List[int],
    k: int,
    m: int,
    confidence_level: float = 0.95,
    delta: float = 0.0,
    print_progress: bool = True,
) -> Dict:
    """
    Turn a collected v_list into average- and median-based epsilon lower bounds
    (see epsilon_lower_bound_from_vs). The eps_lb_* values are certified-unlearning
    epsilons -- already halved; the un-halved LDP parameters are reported alongside as
    eps_lb_ldp_*.
    """
    T, r = len(v_list), 2 * k
    result_avg = epsilon_lower_bound_from_vs(v_list, k=k, N=m, confidence_level=confidence_level,
                                              delta=delta, use_median=False)
    result_median = epsilon_lower_bound_from_vs(v_list, k=k, N=m, confidence_level=confidence_level,
                                                 delta=delta, use_median=True)
    eps_lb_avg = result_avg.get("epsilon_lb") if isinstance(result_avg, dict) else result_avg
    eps_lb_median = result_median.get("epsilon_lb") if isinstance(result_median, dict) else result_median
    eps_lb_ldp_avg = result_avg.get("epsilon_lb_ldp") if isinstance(result_avg, dict) else None
    eps_lb_ldp_median = result_median.get("epsilon_lb_ldp") if isinstance(result_median, dict) else None

    if print_progress:
        print(f"\n[eps-audit] T={T} runs, m={m} forget points, r={r} (2*k, k={k})")
        print(f"[eps-audit] v_list={v_list}")
        print(f"[eps-audit] epsilon_lb (average-based): {eps_lb_avg}")
        print(f"[eps-audit] epsilon_lb (median-based):  {eps_lb_median}")
        print(f"[eps-audit] (un-halved LDP parameters of M -- average: {eps_lb_ldp_avg}, "
              f"median: {eps_lb_ldp_median})")

    return {
        "eps_lb_avg": eps_lb_avg, "eps_lb_median": eps_lb_median,
        "eps_lb_ldp_avg": eps_lb_ldp_avg, "eps_lb_ldp_median": eps_lb_ldp_median,
        "result_dict_avg": result_avg, "result_dict_median": result_median,
    }


def _rho_bounds_from_v_list(
    v_list: List[int],
    k: int,
    m: int,
    confidence_level: float = 0.95,
    gamma_max: float = 1e4,
    conv_delta: float = 1e-3,
    print_progress: bool = True,
) -> Dict:
    """
    Turn a collected v_list into average- and median-based rho (zCDP) lower bounds
    (see rho_lower_bound_from_vs), plus their (eps, conv_delta) estimates -- which are
    NOT lower bounds on eps.
    """
    T, r = len(v_list), 2 * k
    result_avg = rho_lower_bound_from_vs(v_list, k=k, N=m, confidence_level=confidence_level,
                                          use_median=False, gamma_max=gamma_max, conv_delta=conv_delta)
    result_median = rho_lower_bound_from_vs(v_list, k=k, N=m, confidence_level=confidence_level,
                                             use_median=True, gamma_max=gamma_max, conv_delta=conv_delta)
    rho_lb_avg = result_avg.get("rho_lb") if isinstance(result_avg, dict) else result_avg
    rho_lb_median = result_median.get("rho_lb") if isinstance(result_median, dict) else result_median
    eps_est_avg = result_avg.get("eps_estimate") if isinstance(result_avg, dict) else None
    eps_est_median = result_median.get("eps_estimate") if isinstance(result_median, dict) else None

    if print_progress:
        print(f"\n[rho-audit] T={T} runs, m={m} forget points, r={r} (2*k, k={k})")
        print(f"[rho-audit] v_list={v_list}")
        print(f"[rho-audit] rho_lb (average-based): {rho_lb_avg}")
        print(f"[rho-audit] rho_lb (median-based):  {rho_lb_median}")
        print(f"[rho-audit] (eps, delta={conv_delta}) estimates from rho -- NOT lower bounds "
              f"-- average: {eps_est_avg}, median: {eps_est_median}")

    return {
        "rho_lb_avg": rho_lb_avg, "rho_lb_median": rho_lb_median,
        "eps_estimate_avg": eps_est_avg, "eps_estimate_median": eps_est_median,
        "conv_delta": conv_delta,
        "result_dict_avg": result_avg, "result_dict_median": result_median,
    }


def _mu_bounds_from_v_list(
    v_list: List[int],
    k: int,
    m: int,
    confidence_level: float = 0.95,
    theta_max: float = 50.0,
    conv_delta: float = 1e-3,
    print_progress: bool = True,
) -> Dict:
    """
    Turn a collected v_list into average- and median-based mu (GDP) lower bounds
    (see mu_lower_bound_from_vs), plus their (eps, conv_delta) estimates -- which are
    NOT lower bounds on eps.
    """
    T, r = len(v_list), 2 * k
    result_avg = mu_lower_bound_from_vs(v_list, k=k, N=m, confidence_level=confidence_level,
                                         use_median=False, theta_max=theta_max, conv_delta=conv_delta)
    result_median = mu_lower_bound_from_vs(v_list, k=k, N=m, confidence_level=confidence_level,
                                            use_median=True, conv_delta=conv_delta)
    mu_lb_avg = result_avg.get("mu_lb") if isinstance(result_avg, dict) else result_avg
    mu_lb_median = result_median.get("mu_lb") if isinstance(result_median, dict) else result_median
    eps_est_avg = result_avg.get("eps_estimate") if isinstance(result_avg, dict) else None
    eps_est_median = result_median.get("eps_estimate") if isinstance(result_median, dict) else None

    if print_progress:
        print(f"\n[mu-audit] T={T} runs, m={m} forget points, r={r} (2*k, k={k})")
        print(f"[mu-audit] v_list={v_list}")
        print(f"[mu-audit] mu_lb (average-based): {mu_lb_avg}")
        print(f"[mu-audit] mu_lb (median-based):  {mu_lb_median}")
        print(f"[mu-audit] (eps, delta={conv_delta}) estimates from mu -- NOT lower bounds "
              f"-- average: {eps_est_avg}, median: {eps_est_median}")

    return {
        "mu_lb_avg": mu_lb_avg, "mu_lb_median": mu_lb_median,
        "eps_estimate_avg": eps_est_avg, "eps_estimate_median": eps_est_median,
        "conv_delta": conv_delta,
        "result_dict_avg": result_avg, "result_dict_median": result_median,
    }


def compute_eps_bounds_for_all_runs(
    stats: Dict,
    test_runs_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    k: int,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
    delta: float = 0.0,
    confidence_level: float = 0.95,
    print_progress: bool = True,
) -> Dict:
    """
    Run the full "guessing game" epsilon-lower-bound audit across every held-out run under
    `test_runs_dir` (e.g. runs_cifar100_bs_1/test_run/), and combine them into both an
    average-based and a median-based epsilon lower bound (see epsilon_lower_bound_from_vs).

    `stats` must come from runs DIFFERENT than `test_runs_dir` (see attack_run's docstring)
    -- e.g. stats built from run_01..run_50 auditing held-out run_51..run_60.

    This is the (eps, 0) audit only: `delta` must be 0.0 (see the section header). The
    reported eps_lb_* are certified-unlearning epsilons -- half the rejected LDP parameter
    of the audit mechanism -- with the un-halved values in eps_lb_ldp_*.

    Returns a dict with eps_lb_avg, eps_lb_median, eps_lb_ldp_avg, eps_lb_ldp_median,
    v_list (one overlap score per run), run_ids (which runs succeeded), failed_runs,
    T (run count), m (forget point count), r (2*k), and the full result dicts from both
    tests.
    """
    v_list, run_ids, failed_runs, m = _collect_v_list_for_runs(
        stats, test_runs_dir, dataset, data_dir, k, model_name=model_name,
        num_classes=num_classes, filters=filters, device=device, batch_size=batch_size,
        metric=metric, print_progress=print_progress, tag="eps-audit",
    )

    bounds = _eps_bounds_from_v_list(v_list, k=k, m=m, confidence_level=confidence_level,
                                     delta=delta, print_progress=print_progress)

    if print_progress and failed_runs:
        print(f"[eps-audit] {len(failed_runs)} run(s) failed: {failed_runs}")

    return {
        **bounds,
        "v_list": v_list, "run_ids": run_ids, "failed_runs": failed_runs,
        "T": len(v_list), "m": m, "r": 2 * k,
    }


def compute_rho_bounds_for_all_runs(
    stats: Dict,
    test_runs_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    k: int,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
    confidence_level: float = 0.95,
    gamma_max: float = 1e4,
    conv_delta: float = 1e-3,
    print_progress: bool = True,
) -> Dict:
    """
    zCDP analogue of compute_eps_bounds_for_all_runs: same "guessing game" audit
    (reuses _collect_v_list_for_runs to collect one overlap score v per held-out run),
    but combines the v's into average-based and median-based rho lower bounds via
    rho_lower_bound_from_vs instead of epsilon lower bounds. Unlike the eps-DP version,
    there's no `delta` audit-noise parameter -- the zCDP bound is delta-free -- and
    `gamma_max` (upper bound for the Renyi-order search in the zCDP conversion) takes
    its place. rho_lb is not halved (see rho_lower_bound_from_vs).

    `stats` must come from runs DIFFERENT than `test_runs_dir` (see attack_run's
    docstring) -- e.g. stats built from run_01..run_50 auditing held-out run_51..run_60.

    Returns a dict with rho_lb_avg, rho_lb_median, their (eps, conv_delta) estimates
    (eps_estimate_avg / eps_estimate_median -- comparable with epsilon_lb on the same
    axis, but NOT lower bounds on eps), conv_delta, v_list (one overlap score per run),
    run_ids (which runs succeeded), failed_runs, T (run count), m (forget point count),
    r (2*k), and the full result dicts from both tests.
    """
    v_list, run_ids, failed_runs, m = _collect_v_list_for_runs(
        stats, test_runs_dir, dataset, data_dir, k, model_name=model_name,
        num_classes=num_classes, filters=filters, device=device, batch_size=batch_size,
        metric=metric, print_progress=print_progress, tag="rho-audit",
    )

    bounds = _rho_bounds_from_v_list(v_list, k=k, m=m, confidence_level=confidence_level,
                                     gamma_max=gamma_max, conv_delta=conv_delta,
                                     print_progress=print_progress)

    if print_progress and failed_runs:
        print(f"[rho-audit] {len(failed_runs)} run(s) failed: {failed_runs}")

    return {
        **bounds,
        "v_list": v_list, "run_ids": run_ids, "failed_runs": failed_runs,
        "T": len(v_list), "m": m, "r": 2 * k,
    }


def compute_mu_bounds_for_all_runs(
    stats: Dict,
    test_runs_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    k: int,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
    confidence_level: float = 0.95,
    theta_max: float = 50.0,
    conv_delta: float = 1e-3,
    print_progress: bool = True,
) -> Dict:
    """
    Gaussian-DP analogue of compute_eps_bounds_for_all_runs / compute_rho_bounds_for_all_runs:
    same "guessing game" audit (reuses _collect_v_list_for_runs to collect one overlap score
    v per held-out run), but combines the v's into average-based and median-based mu lower
    bounds via mu_lower_bound_from_vs. Like the zCDP version there's no audit-noise `delta`;
    mu_lb is not halved (the factor of 2 is inside mu_loc(mu) = 2 mu).

    `stats` must come from runs DIFFERENT than `test_runs_dir` (see attack_run's
    docstring) -- e.g. stats built from run_01..run_50 auditing held-out run_51..run_60.

    Returns a dict with mu_lb_avg, mu_lb_median, their (eps, conv_delta) estimates
    (eps_estimate_avg / eps_estimate_median -- NOT lower bounds on eps), conv_delta,
    v_list (one overlap score per run), run_ids (which runs succeeded), failed_runs,
    T (run count), m (forget point count), r (2*k), and the full result dicts from both
    tests.
    """
    v_list, run_ids, failed_runs, m = _collect_v_list_for_runs(
        stats, test_runs_dir, dataset, data_dir, k, model_name=model_name,
        num_classes=num_classes, filters=filters, device=device, batch_size=batch_size,
        metric=metric, print_progress=print_progress, tag="mu-audit",
    )

    bounds = _mu_bounds_from_v_list(v_list, k=k, m=m, confidence_level=confidence_level,
                                    theta_max=theta_max, conv_delta=conv_delta,
                                    print_progress=print_progress)

    if print_progress and failed_runs:
        print(f"[mu-audit] {len(failed_runs)} run(s) failed: {failed_runs}")

    return {
        **bounds,
        "v_list": v_list, "run_ids": run_ids, "failed_runs": failed_runs,
        "T": len(v_list), "m": m, "r": 2 * k,
    }


def compute_all_bounds_for_all_runs(
    stats: Dict,
    test_runs_dir: Union[str, Path],
    dataset: str,
    data_dir: str,
    k: int,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    filters: float = 1.0,
    device: str = "cpu",
    batch_size: int = 256,
    metric: str = "phi",
    delta: float = 0.0,
    confidence_level: float = 0.95,
    gamma_max: float = 1e4,
    theta_max: float = 50.0,
    conv_delta: float = 1e-3,
    print_progress: bool = True,
) -> Dict:
    """
    Run all three audits -- eps-DP, rho-zCDP and mu-GDP -- off a SINGLE attack pass over
    the held-out runs. The expensive step (_collect_v_list_for_runs: one model-inference
    pass per run) is shared, so this costs roughly a third of calling
    compute_eps_bounds_for_all_runs, compute_rho_bounds_for_all_runs and
    compute_mu_bounds_for_all_runs separately, and is guaranteed to report all three
    notions off the very same overlap scores.

    `stats` must come from runs DIFFERENT than `test_runs_dir` (see attack_run's
    docstring) -- e.g. stats built from run_01..run_50 auditing held-out run_51..run_60.

    Returns a dict with:
      - "eps", "rho", "mu": the three per-notion sub-dicts, shaped exactly like the
        bound fields of the individual drivers (eps_lb_avg / rho_lb_avg / mu_lb_avg etc.).
      - the shared v_list, run_ids, failed_runs, T, m, r.
    Remember the asymmetry: eps_lb_* are halved (certified-unlearning epsilons, with the
    LDP values in eps_lb_ldp_*), while rho_lb_* and mu_lb_* are not, and the zCDP/GDP
    "eps_estimate_*" fields are conversions at conv_delta, not lower bounds on eps.
    """
    v_list, run_ids, failed_runs, m = _collect_v_list_for_runs(
        stats, test_runs_dir, dataset, data_dir, k, model_name=model_name,
        num_classes=num_classes, filters=filters, device=device, batch_size=batch_size,
        metric=metric, print_progress=print_progress, tag="audit",
    )

    eps_bounds = _eps_bounds_from_v_list(v_list, k=k, m=m, confidence_level=confidence_level,
                                         delta=delta, print_progress=print_progress)
    rho_bounds = _rho_bounds_from_v_list(v_list, k=k, m=m, confidence_level=confidence_level,
                                         gamma_max=gamma_max, conv_delta=conv_delta,
                                         print_progress=print_progress)
    mu_bounds = _mu_bounds_from_v_list(v_list, k=k, m=m, confidence_level=confidence_level,
                                       theta_max=theta_max, conv_delta=conv_delta,
                                       print_progress=print_progress)

    if print_progress and failed_runs:
        print(f"[audit] {len(failed_runs)} run(s) failed: {failed_runs}")

    return {
        "eps": eps_bounds, "rho": rho_bounds, "mu": mu_bounds,
        "v_list": v_list, "run_ids": run_ids, "failed_runs": failed_runs,
        "T": len(v_list), "m": m, "r": 2 * k,
    }


def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="A run_sweep.py sweep directory")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default=None, help="Default: chosen per dataset")
    parser.add_argument("--num-classes", type=int, default=None, help="Default: chosen per dataset")
    parser.add_argument("--filters", type=float, default=1.0)
    parser.add_argument("--use-trained-model", action="store_true", default=False,
                         help="Score with trained_model.pth (pre-unlearning) instead of unlearned_model.pth. "
                              "Ignored if --both or --unlearn-epoch is set.")
    parser.add_argument("--unlearn-epoch", type=int, default=None,
                         help="Score with the intermediate checkpoint unlearned_model_epoch_{N}.pth instead "
                              "of the final unlearned_model.pth (only present for runs launched with "
                              "--unlearn-epochs > 1 in run_sweep.py). Takes precedence over --use-trained-model; "
                              "ignored if --both is set. Pass -1 to generate stats for EVERY intermediate "
                              "unlearning epoch found (one output file per epoch) in a single invocation.")
    parser.add_argument("--both", action="store_true", default=False,
                         help="Generate stats for BOTH trained_model.pth and unlearned_model.pth "
                              "(two output files) in one run, instead of picking one via --use-trained-model.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default=None,
                         help="Default: <run-dir>/forget_pointwise_stats_*.json. Ignored if --both is set "
                              "(each model source gets its own default-named file).")
    return parser


def _discover_unlearn_epochs(run_dir: Union[str, Path]) -> List[int]:
    """
    Find which intermediate unlearn-epoch checkpoints exist, by scanning the first run_*
    subfolder (every run in a sweep shares the same --unlearn-epochs setting, so they all
    have the same set of unlearned_model_epoch_{N}.pth files).
    """
    run_dir = Path(run_dir)
    run_subdirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
    if not run_subdirs:
        raise FileNotFoundError(f"No run_* subfolders found under {run_dir}")
    epochs = []
    for f in run_subdirs[0].glob("unlearned_model_epoch_*.pth"):
        m = re.match(r"unlearned_model_epoch_(\d+)\.pth$", f.name)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise FileNotFoundError(
            f"--unlearn-epoch -1 requested but no unlearned_model_epoch_*.pth files found under "
            f"{run_subdirs[0]} (the sweep may have been launched with --unlearn-epochs 1, which "
            f"only ever writes the final unlearned_model.pth)."
        )
    return sorted(epochs)


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.unlearn_epoch == -1:
        epochs = _discover_unlearn_epochs(args.run_dir)
        print(f"[audit] --unlearn-epoch -1: found {len(epochs)} intermediate unlearning "
              f"epoch(s) {epochs}; generating one stats file per epoch (trained-model stats, "
              f"if requested via --use-trained-model/--both, are generated once, not per epoch)")
        for epoch in epochs:
            compute_pointwise_forget_stats(
                run_dir=args.run_dir, dataset=args.dataset, data_dir=args.data_dir,
                model_name=args.model, num_classes=args.num_classes, filters=args.filters,
                unlearn_epoch=epoch, device=args.device, batch_size=args.batch_size, output_path=None,
            )
        if args.use_trained_model or args.both:
            compute_pointwise_forget_stats(
                run_dir=args.run_dir, dataset=args.dataset, data_dir=args.data_dir,
                model_name=args.model, num_classes=args.num_classes, filters=args.filters,
                use_trained_model=True, device=args.device, batch_size=args.batch_size, output_path=None,
            )
    else:
        use_trained_options = [True, False] if args.both else [args.use_trained_model]
        for use_trained in use_trained_options:
            compute_pointwise_forget_stats(
                run_dir=args.run_dir, dataset=args.dataset, data_dir=args.data_dir,
                model_name=args.model, num_classes=args.num_classes, filters=args.filters,
                use_trained_model=use_trained, unlearn_epoch=None if args.both else args.unlearn_epoch,
                device=args.device, batch_size=args.batch_size, output_path=None if args.both else args.output,
            )
    
