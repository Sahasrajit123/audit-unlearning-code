#!/usr/bin/env python
"""
Cumulative privacy audit across the run_sweep.py sweeps: one driver, all three bounds.

For every sweep directory, and every pointwise-stats file in it (each stats file pins the
checkpoint that gets attacked via its "model_source" field -- trained_model.pth,
unlearned_model.pth, or an intermediate unlearned_model_epoch_N.pth), this:

  1. attacks each held-out run under <sweep>/test_run/ ONCE (audit_utils.attack_run) and
     keeps that run's LLR ranking of the forget points;
  2. slices the rankings into the attack's 2k guesses and scores them into one overlap
     score v per run -- for every requested k, off the same rankings;
  3. feeds each k's single v_list to ALL THREE lower bounds -- pure-eps DP, rho-zCDP and
     mu-GDP -- by the mean-v and median-v tests both.

Why not call audit_utils.compute_{eps,rho,mu}_bounds_for_all_runs? Each of those re-scores
every held-out model from scratch, so asking for all three would do the expensive work
three times over for identical v's. Neither the checkpoint restore, the per-point scoring
nor the ranking depends on k or on which privacy parameter you bound afterwards, so this
collects the rankings once per (sweep, model source) and feeds the same v_list to the three
cheap closed-form/Chernoff steps. A k-sweep costs one scoring pass, not one per k.

There are no per-run output files. The bounds are cross-run by construction -- the mean-v
and median-v tests combine the whole v_list over all T runs -- so no per-run bound exists,
and the per-run v's are already saved in full inside the files below.

Output:
  <sweep>/audit_bounds_<model_source>_<metric>.json
      The cumulative audit for that sweep and checkpoint, next to the
      forget_pointwise_stats_*.json it came from: config, the whole v_list, the run ids,
      the failed runs, every bound and the full per-test detail dicts. Each k lives under
      "by_k", and a later invocation with a different k merges into the same file (refused
      if the saved file used a different confidence_level / conv_delta / gamma_max /
      theta_max, since those are not in the file name -- pass --replace or --output-name).
      Written as soon as that sweep/source finishes, so an audit that dies partway through
      still leaves everything it completed on disk.
  --summary-out <path>
      Optional combined summary across all sweeps: one flat row per
      (sweep, model_source, k), loadable straight into a DataFrame, and enough to re-derive
      every bound later via --from-json.

Reading the numbers:
  - eps_lb is a CERTIFIED-UNLEARNING epsilon at delta = 0. The test rejects an LDP
    parameter of the audit mechanism M, and (eps,0)-certified unlearning => M is
    (2 eps,0)-LDP, so the certified epsilon is half of it. cum_runs_eps_lab already applies
    that halving; the un-halved value is reported alongside as eps_lb_ldp. Pure epsilon
    only: the LDP reduction degrades to (2 eps, (1+e^eps) delta) once delta > 0, so this
    audit passes delta = 0.0 and audit_utils rejects anything else.
  - rho_lb and mu_lb are NOT halved -- their factor of 2 already sits inside
    eps_gamma^loc(rho) and mu_loc(mu) = 2 mu.
  - eps_estimate_from_rho_* / eps_estimate_from_mu_* are FORWARD (eps, conv_delta)
    conversions of rho_lb / mu_lb. They are comparable with eps_lb on the same axis but
    are ESTIMATES, not lower bounds on eps. conv_delta is stored next to every one of them.
  - null means the observation is consistent with a perfectly private mechanism; Infinity
    means it is impossible under any finite parameter. Neither is coerced to 0.

Typical usage:
    # every sweep that has stats + test_run/, final checkpoints only, on a GPU
    python -m audit.run_cumulative_audit --device cuda:0 --summary-out audit_summary.json

    # one sweep, sweep k off a single scoring pass, every intermediate unlearn epoch
    python -m audit.run_cumulative_audit runs_cifar100_bs_1 --k 100 250 500 1000 \
        --all-epochs --device cuda:0

    # re-derive the bounds from saved v_lists after changing the math or conv_delta
    # (instant: no checkpoints are touched)
    python -m audit.run_cumulative_audit --from-json audit_summary.json --conv-delta 1e-5
"""
import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from audit.audit_utils import (
    _eps_bounds_from_v_list,
    _mu_bounds_from_v_list,
    _rho_bounds_from_v_list,
    attack_run,
    build_v_s_vectors,
    compute_overlap_score,
)

# Settings that every k entry in one output file must share. They are not part of the file
# name (unlike model_source / metric), so merging into a file computed under different ones
# would leave its saved config describing only the newest entries; that is refused instead.
GUARDED_SETTINGS = ["confidence_level", "conv_delta", "gamma_max", "theta_max"]

BOUNDS_NOTE = (
    "eps_lb is a certified-unlearning epsilon at delta=0, already halved from eps_lb_ldp "
    "((eps,0)-certified unlearning => the audit mechanism is (2 eps,0)-LDP). rho_lb and "
    "mu_lb are NOT halved -- their factor of 2 is inside eps_gamma^loc(rho) and "
    "mu_loc(mu)=2mu. eps_estimate_from_rho_* / eps_estimate_from_mu_* are forward "
    "(eps, conv_delta) conversions of rho_lb / mu_lb: estimates, NOT lower bounds on eps. "
    "null = the observation is consistent with a perfectly private mechanism; Infinity = "
    "impossible under any finite parameter."
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _source_sort_key(model_source: str):
    """Order model sources the way the run progresses: trained, then each intermediate
    unlearn epoch in numeric order, then the final unlearned model."""
    if model_source == "trained":
        return (0, 0)
    m = re.fullmatch(r"unlearn_epoch_(\d+)", model_source)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


def discover_sweeps(patterns: List[str], cwd: Path = Path(".")) -> List[Path]:
    """Resolve the sweep directories to audit. A pattern may be a directory or a glob;
    directories without a test_run/ or without any stats file are reported and dropped,
    so `--sweeps 'runs_*'` does the right thing on a repo with half-finished sweeps."""
    candidates: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            candidates.append(path)
            continue
        matched = sorted(p for p in cwd.glob(pattern) if p.is_dir())
        if not matched:
            print(f"[audit] {pattern}: no such directory (or glob matched nothing), skipping")
        candidates.extend(matched)

    sweeps = []
    for path in candidates:
        if not (path / "test_run").is_dir():
            print(f"[audit] {path}: no test_run/ subdirectory, skipping")
            continue
        if not list(path.glob("forget_pointwise_stats_*.json")):
            print(f"[audit] {path}: no forget_pointwise_stats_*.json, skipping "
                  f"(run audit_utils.py on it first)")
            continue
        sweeps.append(path)
    return sweeps


def discover_stats_files(sweep_dir: Path, include_epochs: bool = False,
                         only: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    """The pointwise-stats files to audit in one sweep, as (model_source, path) pairs in run
    order. Intermediate unlearn-epoch stats are skipped unless asked for."""
    found = []
    for path in sorted(sweep_dir.glob("forget_pointwise_stats_*.json")):
        try:
            model_source = json.loads(path.read_text()).get("model_source", "unlearned")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[audit] {sweep_dir.name}/{path.name}: unreadable, skipping - {e}")
            continue
        if only is not None:
            # An explicitly named source wins over the include_epochs default: asking for
            # --stats unlearn_epoch_1 should not also require --all-epochs.
            if model_source not in only:
                continue
        elif model_source.startswith("unlearn_epoch_") and not include_epochs:
            continue
        found.append((model_source, path))
    return sorted(found, key=lambda t: _source_sort_key(t[0]))


# ---------------------------------------------------------------------------
# The expensive pass: LLR rankings, once per (sweep, model source)
# ---------------------------------------------------------------------------

def collect_rankings(stats: Dict, test_runs_dir: Path, dataset: str, data_dir: str,
                     device: str, batch_size: int, metric: str, model_name=None,
                     num_classes=None, filters: float = 1.0, label: str = ""):
    """
    Attack every held-out run under `test_runs_dir` once and keep its ranking of the forget
    points by LLR (most-confidently-"in" first), together with that run's ground truth.

    This is the only expensive step of the audit -- restore the checkpoint, score all m
    forget points, sort them -- and it depends on neither k nor which privacy parameter is
    bounded afterwards, so it is hoisted out of both loops. audit_utils._compute_v_for_single_run
    does this and the k-dependent slice together, which is the right shape for a single k
    but would re-score every checkpoint per k in a sweep.

    Returns (rankings, failed_runs, m) where rankings is a list of
    (run_id, ranked_point_ids, ground_truth_in) and m is the total number of forget points.
    """
    run_dirs = sorted(d for d in test_runs_dir.iterdir()
                      if d.is_dir() and d.name.startswith("run_"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* subfolders found under {test_runs_dir}")

    rankings, failed_runs = [], []
    m = None
    for run_dir in run_dirs:
        try:
            point_llrs, ground_truth_in = attack_run(
                stats, run_dir, dataset, data_dir, model_name=model_name,
                num_classes=num_classes, filters=filters, device=device,
                batch_size=batch_size, metric=metric,
            )
            # Same m as _compute_v_for_single_run: the stats' own point count, so a run
            # where some points had no usable LLR still scores against the full forget set.
            m_run = stats.get("num_points", len(point_llrs))
            m = m_run if m is None else m
            ranked = [p for p, _ in sorted(point_llrs.items(), key=lambda t: t[1], reverse=True)]
            rankings.append((run_dir.name, ranked, sorted(ground_truth_in)))
            print(f"[audit] {label} {run_dir.name}: ranked {len(ranked)} points with a usable LLR")
        except Exception as e:
            failed_runs.append((run_dir.name, str(e)))
            print(f"[audit] {label} {run_dir.name}: FAILED - {e}")

    if not rankings:
        raise ValueError(f"No usable runs found under {test_runs_dir}: {failed_runs}")
    if m is None:
        raise ValueError(f"Could not determine the forget-set size m for {test_runs_dir}")
    return rankings, failed_runs, m


def v_list_from_rankings(rankings, k: int, m: int, label: str = ""):
    """
    Slice the already-computed rankings into each run's 2k guesses -- top-k guessed "in",
    bottom-k guessed "out" -- and score them against that run's ground truth.

    Identical to the tail of audit_utils._compute_v_for_single_run (same slices, same
    build_v_s_vectors / compute_overlap_score), just applied to a ranking that was computed
    once and is reused for every k. Pure bookkeeping: no model is touched.

    Returns (v_list, run_ids, failed_runs).
    """
    v_list, run_ids, failed_runs = [], [], []
    for run_id, ranked, ground_truth_in in rankings:
        if len(ranked) < 2 * k:
            reason = (f"Need at least 2*k={2 * k} points with valid LLRs to pick "
                      f"top-k/bottom-k; got {len(ranked)}")
            failed_runs.append((run_id, reason))
            print(f"[audit] {label} {run_id}: SKIPPED at k={k} - {reason}")
            continue
        selected_idx = ranked[:k]        # guessed "in"
        non_selected_idx = ranked[-k:]   # guessed "out"
        v, s = build_v_s_vectors(selected_idx, non_selected_idx, ground_truth_in, N=m)
        v_list.append(int(compute_overlap_score(v, s)))
        run_ids.append(run_id)
    return v_list, run_ids, failed_runs


# ---------------------------------------------------------------------------
# The cheap part: one v_list -> all three bounds
# ---------------------------------------------------------------------------

def _try(fn, what: str):
    """Run one bound and swallow its failure: a sweep where (say) the zCDP order search
    blows up should still report the other two bounds rather than abort the audit."""
    try:
        return fn()
    except Exception as e:
        print(f"[audit]   {what} bound failed: {e}")
        return None


def bounds_from_v_list(v_list: List[int], k: int, m: int, confidence_level: float,
                       conv_delta: float, gamma_max: float = 1e4, theta_max: float = 50.0,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Turn one already-collected v_list into the eps / rho / mu lower bounds, mean-v and
    median-v tests both. Pure math on the v's -- no model scoring, milliseconds.

    delta is pinned to 0.0 for the epsilon audit: the LDP reduction it rests on does not
    survive delta > 0, and audit_utils.epsilon_lower_bound_from_vs raises on anything else.
    """
    eps = _try(lambda: _eps_bounds_from_v_list(
        v_list, k=k, m=m, confidence_level=confidence_level, delta=0.0,
        print_progress=verbose), "eps")
    rho = _try(lambda: _rho_bounds_from_v_list(
        v_list, k=k, m=m, confidence_level=confidence_level, gamma_max=gamma_max,
        conv_delta=conv_delta, print_progress=verbose), "rho")
    mu = _try(lambda: _mu_bounds_from_v_list(
        v_list, k=k, m=m, confidence_level=confidence_level, theta_max=theta_max,
        conv_delta=conv_delta, print_progress=verbose), "mu")

    def get(d: Optional[Dict], key: str):
        return None if d is None else d.get(key)

    return {
        # audited lower bounds on the certified-unlearning parameters
        "eps_lb_avg": get(eps, "eps_lb_avg"),
        "eps_lb_median": get(eps, "eps_lb_median"),
        "eps_lb_ldp_avg": get(eps, "eps_lb_ldp_avg"),
        "eps_lb_ldp_median": get(eps, "eps_lb_ldp_median"),
        "eps_delta": 0.0,
        "rho_lb_avg": get(rho, "rho_lb_avg"),
        "rho_lb_median": get(rho, "rho_lb_median"),
        "mu_lb_avg": get(mu, "mu_lb_avg"),
        "mu_lb_median": get(mu, "mu_lb_median"),
        # forward conversions -- estimates, NOT lower bounds on eps; conv_delta travels
        # alongside so the numbers can never be read without their delta
        "eps_estimate_from_rho_avg": get(rho, "eps_estimate_avg"),
        "eps_estimate_from_rho_median": get(rho, "eps_estimate_median"),
        "eps_estimate_from_mu_avg": get(mu, "eps_estimate_avg"),
        "eps_estimate_from_mu_median": get(mu, "eps_estimate_median"),
        "conv_delta": conv_delta,
        "details": {
            "eps_mean": get(eps, "result_dict_avg"),
            "eps_median": get(eps, "result_dict_median"),
            "rho_mean": get(rho, "result_dict_avg"),
            "rho_median": get(rho, "result_dict_median"),
            "mu_mean": get(mu, "result_dict_avg"),
            "mu_median": get(mu, "result_dict_median"),
        },
    }


# ---------------------------------------------------------------------------
# Saving (per sweep/source, with merge-on-rerun across k)
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """numpy scalars/arrays and tuples -> plain Python, so json.dump doesn't choke. inf
    survives as Infinity and None as null, both of which json.load reads back fine (and
    both are meaningful here -- see BOUNDS_NOTE -- so neither is flattened away)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


class SettingsMismatch(Exception):
    """An existing output file was computed under different GUARDED_SETTINGS."""


def output_path(sweep: Path, model_source: str, metric: str,
                output_name: Optional[str] = None) -> Path:
    """<sweep>/audit_bounds_<model_source>_<metric>.json -- next to the stats file it came
    from. model_source and metric are in the name so phi/loss and different checkpoints
    never overwrite each other; k is not, because every k lives under "by_k" in one file."""
    if output_name:
        return sweep / output_name
    return sweep / f"audit_bounds_{model_source}_{metric}.json"


def _k_sort_key(key):
    try:
        return (0, int(key), "")
    except (TypeError, ValueError):
        return (1, 0, str(key))


def read_existing(path: Path) -> Dict[str, Any]:
    """Load an output file to merge into; {} if absent or unreadable."""
    if not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[audit]   could not read {path} to merge into ({e}); it will be replaced")
        return {}
    return previous if isinstance(previous, dict) else {}


def assert_merge_compatible(previous: Dict[str, Any], config: Dict[str, Any], path: Path):
    """Refuse to merge into a file computed under different GUARDED_SETTINGS, so one file
    never mixes k entries that are not comparable. Checked before the scoring pass (so a
    mismatch costs no GPU time) and again at save time."""
    old_config = previous.get("config", {})
    differing = {
        key: (old_config.get(key), config.get(key))
        for key in GUARDED_SETTINGS
        if key in old_config and old_config.get(key) != config.get(key)
    }
    if differing:
        details = ", ".join(f"{key}: saved={old!r} now={new!r}"
                            for key, (old, new) in differing.items())
        raise SettingsMismatch(
            f"{path} was computed with different settings ({details}). Merging would leave "
            f"its saved config describing only the new entries. Pass --replace to overwrite "
            f"the file, or --output-name to write a separate one."
        )


def merge_by_k(result: Dict[str, Any], previous: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Keep the by_k entries of an existing file that this run did not recompute."""
    assert_merge_compatible(previous, result["config"], path)
    old_by_k = previous.get("by_k", {})
    if not isinstance(old_by_k, dict):
        return result
    kept = {key: value for key, value in old_by_k.items() if key not in result["by_k"]}
    if kept:
        print(f"[audit]   keeping {len(kept)} previously saved k entr(y/ies): "
              f"{sorted(kept, key=_k_sort_key)}")
    merged = dict(kept)
    merged.update(result["by_k"])
    result["by_k"] = dict(sorted(merged.items(), key=lambda item: _k_sort_key(item[0])))
    return result


def build_result(group: Dict[str, Any], config: Dict[str, Any], generated_at: str,
                 verbose: bool = False) -> Dict[str, Any]:
    """Compute the bounds for every k of one (sweep, model source) and assemble the payload
    that gets saved. group["per_k"] maps k -> {v_list, run_ids, failed_runs}."""
    by_k: Dict[str, Any] = {}
    for k in sorted(group["per_k"]):
        entry_in = group["per_k"][k]
        v_list = entry_in["v_list"]
        entry: Dict[str, Any] = {
            "k": k,
            "r": 2 * k,
            "T": len(v_list),
            "v_list": v_list,
            "run_ids": entry_in.get("run_ids"),
            "failed_runs": entry_in.get("failed_runs"),
            "generated_at": generated_at,
        }
        if not v_list:
            entry["error"] = f"No usable runs at k={k} (each run needs 2*k scored points)"
            print(f"[audit]   k={k}: {entry['error']}")
            by_k[str(k)] = entry
            continue

        entry["v_mean"] = float(np.mean(v_list))
        entry["v_median"] = float(np.median(v_list))
        print(f"[audit]   k={k}: T={len(v_list)} runs, r={2 * k}, "
              f"v_mean={entry['v_mean']:.1f}, v_median={entry['v_median']:.1f}")
        entry["bounds"] = bounds_from_v_list(
            v_list, k=k, m=group["m"],
            confidence_level=config["confidence_level"], conv_delta=config["conv_delta"],
            gamma_max=config["gamma_max"], theta_max=config["theta_max"], verbose=verbose,
        )
        by_k[str(k)] = entry

    return {
        "config": config,
        "sweep": str(group["sweep"]),
        "model_source": group["model_source"],
        "stats_file": group.get("stats_file"),
        "dataset": group.get("dataset"),
        "data_dir": group.get("data_dir"),
        "metric": group["metric"],
        "m": group["m"],
        "run_ids": group.get("run_ids"),
        "scoring_failed_runs": group.get("scoring_failed_runs"),
        "by_k": by_k,
        "generated_at": generated_at,
        "note": BOUNDS_NOTE,
    }


def save_result(result: Dict[str, Any], path: Path, replace: bool) -> Path:
    """Write one (sweep, model source)'s audit, merging k entries into any existing file."""
    previous = {} if replace else read_existing(path)
    if previous:
        result = merge_by_k(result, previous, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(result), indent=2))
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one result into one row per k: flat enough for a DataFrame, and complete
    enough (v_list, m, k) to re-derive every bound later via --from-json."""
    rows = []
    for entry in result["by_k"].values():
        row = {
            "sweep": result["sweep"],
            "model_source": result["model_source"],
            "metric": result["metric"],
            "k": entry["k"],
            "r": entry["r"],
            "T": entry["T"],
            "m": result["m"],
            "v_mean": entry.get("v_mean"),
            "v_median": entry.get("v_median"),
            "v_list": entry["v_list"],
            "run_ids": entry.get("run_ids"),
            "stats_file": result.get("stats_file"),
            "dataset": result.get("dataset"),
            "data_dir": result.get("data_dir"),
            "generated_at": entry.get("generated_at"),
        }
        if "error" in entry:
            row["error"] = entry["error"]
        bounds = entry.get("bounds") or {}
        for key, value in bounds.items():
            if key != "details":
                row[key] = value
        rows.append(row)
    return rows


def _fmt(x, width: int = 11) -> str:
    if x is None:
        return "-".rjust(width)
    if isinstance(x, str):
        return x[:width].rjust(width)
    if not np.isfinite(x):
        return ("inf" if x > 0 else "-inf").rjust(width)
    return f"{x:{width}.4f}"


def print_summary(rows: List[Dict[str, Any]], confidence_level: float, conv_delta: float):
    """One two-line block (mean-v test, median-v test) per (sweep, model source, k)."""
    if not rows:
        print("\n[audit] nothing audited.")
        return

    width = 152
    print("\n" + "=" * width)
    print(f"CUMULATIVE AUDIT SUMMARY  (confidence={confidence_level}, conv_delta={conv_delta})")
    print("=" * width)
    print("eps_lb : certified-unlearning epsilon at delta=0, already halved from eps_ldp.")
    print("rho_lb / mu_lb : NOT halved. eps~rho / eps~mu : forward (eps, conv_delta) "
          "conversions -- ESTIMATES, not lower bounds.")
    print("'-' = consistent with a perfectly private mechanism, or the test did not apply; "
          "'inf' = impossible under any finite parameter.\n")

    head = (f"{'sweep':<34} {'model source':<16} {'k':>5} {'T':>3} {'v med':>7} "
            f"{'eps_lb':>11} {'eps_ldp':>11} {'rho_lb':>11} {'mu_lb':>11} "
            f"{'eps~rho':>11} {'eps~mu':>11}  test")
    print(head)
    print("-" * width)

    for row in rows:
        if "error" in row:
            print(f"{row['sweep'][:34]:<34} {row['model_source'][:16]:<16} "
                  f"{row['k']:>5} {row['T']:>3} {'-':>7}  {row['error']}")
            print("-" * width)
            continue
        v_median = row.get("v_median")
        for which, label in (("avg", "mean"), ("median", "median")):
            # The left-hand identifying columns are printed once, on the mean-v line.
            first = which == "avg"
            sweep_cell = row["sweep"][:34] if first else ""
            source_cell = row["model_source"][:16] if first else ""
            k_cell = str(row["k"]) if first else ""
            t_cell = str(row["T"]) if first else ""
            v_cell = f"{v_median:.1f}" if first and v_median is not None else ""
            print(f"{sweep_cell:<34} {source_cell:<16} {k_cell:>5} {t_cell:>3} {v_cell:>7} "
                  f"{_fmt(row.get(f'eps_lb_{which}'))} {_fmt(row.get(f'eps_lb_ldp_{which}'))} "
                  f"{_fmt(row.get(f'rho_lb_{which}'))} {_fmt(row.get(f'mu_lb_{which}'))} "
                  f"{_fmt(row.get(f'eps_estimate_from_rho_{which}'))} "
                  f"{_fmt(row.get(f'eps_estimate_from_mu_{which}'))}  {label}")
        print("-" * width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="*", default=None,
                   help="Sweep directories (or globs) to audit. Default: every runs_* "
                        "directory that has both a test_run/ and a stats file")
    p.add_argument("--k", type=int, nargs="+", default=[500],
                   help="Top/bottom-k guesses per run, so r = 2*k. Several values sweep k "
                        "off ONE scoring pass per sweep (default: 500, of m=4500)")
    p.add_argument("--metric", default="phi", choices=["phi", "loss"],
                   help="Per-point statistic the LLR is computed on (default: phi)")
    p.add_argument("--device", default="cpu", help='e.g. "cuda:0" (default: cpu)')
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dataset", default=None,
                   help="Default: the dataset recorded in each stats file")
    p.add_argument("--data-dir", default=None,
                   help="Default: the data_dir recorded in each stats file")
    p.add_argument("--model", default=None, help="Default: chosen per dataset")
    p.add_argument("--num-classes", type=int, default=None, help="Default: chosen per dataset")
    p.add_argument("--filters", type=float, default=1.0)
    p.add_argument("--confidence-level", type=float, default=0.95,
                   help="Maps to ci_delta = 1 - confidence_level (default: 0.95)")
    p.add_argument("--conv-delta", type=float, default=1e-3,
                   help="Delta for the (eps, delta) conversions of rho_lb / mu_lb, which "
                        "are estimates and not lower bounds (default: 1e-3)")
    p.add_argument("--gamma-max", type=float, default=1e4,
                   help="Cap for the Renyi-order search in the zCDP conversion")
    p.add_argument("--theta-max", type=float, default=50.0,
                   help="Cap for the Chernoff lambda search in the mean-v tests")
    p.add_argument("--all-epochs", action="store_true", default=False,
                   help="Also audit the intermediate unlearn-epoch stats files, not just "
                        "trained_model.pth and the final unlearned_model.pth")
    p.add_argument("--stats", nargs="+", default=None,
                   help="Audit only these model sources (e.g. --stats unlearned trained)")
    p.add_argument("--summary-out", default=None,
                   help="Also write a combined flat summary of every (sweep, source, k) "
                        "to this JSON path")
    p.add_argument("--output-name", default=None,
                   help="Override the per-sweep output file name (default: "
                        "audit_bounds_<model_source>_<metric>.json)")
    p.add_argument("--no-per-sweep", dest="per_sweep", action="store_false", default=True,
                   help="Do not write the <sweep>/audit_bounds_*.json files")
    p.add_argument("--replace", action="store_true",
                   help="Rewrite each output file with only this run's k entries, dropping "
                        f"previously saved ones. The default merges, and refuses if the "
                        f"saved {GUARDED_SETTINGS} differ")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a (sweep, model source) whose output file already exists")
    p.add_argument("--from-json", nargs="+", default=None,
                   help="Skip all model scoring: re-derive the bounds from the v_lists saved "
                        "in previous --summary-out files and/or per-sweep audit_bounds files. "
                        "Use after changing the bound math, the confidence level or "
                        "conv_delta -- the bounds are instant, the scoring is not")
    p.add_argument("--verbose", action="store_true",
                   help="Print each bound's own progress lines as well")
    p.add_argument("--dry-run", action="store_true",
                   help="List the planned (sweep, model source) pairs and output paths, "
                        "then exit without touching a checkpoint")
    return p


def groups_from_json(paths: List[str], metric_default: str) -> List[Dict[str, Any]]:
    """
    Rebuild the audit groups from saved v_lists, so the bounds can be recomputed without
    re-scoring anything. Accepts both output shapes: a --summary-out file (a "rows" list)
    and a per-sweep audit_bounds file (a "by_k" map).
    """
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def add(sweep, model_source, metric, m, k, v_list, run_ids, failed_runs, extra):
        key = (str(sweep), str(model_source), str(metric))
        group = groups.setdefault(key, {
            "sweep": Path(sweep), "model_source": model_source, "metric": metric,
            "m": m, "per_k": {}, **extra,
        })
        group["per_k"][int(k)] = {
            "v_list": [int(v) for v in v_list],
            "run_ids": run_ids,
            "failed_runs": failed_runs,
        }

    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            for row in payload["rows"]:
                if not row.get("v_list"):
                    continue
                add(row["sweep"], row["model_source"], row.get("metric", metric_default),
                    row["m"], row["k"], row["v_list"], row.get("run_ids"),
                    row.get("failed_runs"),
                    {"stats_file": row.get("stats_file"), "dataset": row.get("dataset"),
                     "data_dir": row.get("data_dir"), "run_ids": row.get("run_ids"),
                     "scoring_failed_runs": None})
        elif isinstance(payload, dict) and isinstance(payload.get("by_k"), dict):
            for entry in payload["by_k"].values():
                if not entry.get("v_list"):
                    continue
                add(payload["sweep"], payload["model_source"],
                    payload.get("metric", metric_default), payload["m"], entry["k"],
                    entry["v_list"], entry.get("run_ids"), entry.get("failed_runs"),
                    {"stats_file": payload.get("stats_file"),
                     "dataset": payload.get("dataset"), "data_dir": payload.get("data_dir"),
                     "run_ids": payload.get("run_ids"),
                     "scoring_failed_runs": payload.get("scoring_failed_runs")})
        else:
            raise ValueError(
                f"{path}: not a recognised audit file (expected a 'rows' list from "
                f"--summary-out, or a 'by_k' map from a per-sweep audit_bounds file)")

    if not groups:
        raise ValueError(f"No saved v_lists found in {paths}")
    return list(groups.values())


def plan_groups(args) -> List[Dict[str, Any]]:
    """The (sweep, model source) pairs to audit, from the CLI, without scoring anything."""
    patterns = args.sweeps if args.sweeps else ["runs_*"]
    sweeps = discover_sweeps(patterns)
    if not sweeps:
        raise ValueError(f"No auditable sweep directories matched {patterns} (each needs a "
                         f"test_run/ and at least one forget_pointwise_stats_*.json)")

    planned = []
    for sweep in sweeps:
        stats_files = discover_stats_files(sweep, include_epochs=args.all_epochs,
                                           only=args.stats)
        if not stats_files:
            print(f"[audit] {sweep}: no matching stats files "
                  f"({'--stats ' + ' '.join(args.stats) if args.stats else 'final checkpoints'}"
                  f"{'' if args.all_epochs else '; pass --all-epochs for intermediate epochs'})"
                  f", skipping")
            continue
        for model_source, stats_path in stats_files:
            planned.append({"sweep": sweep, "model_source": model_source,
                            "stats_file": stats_path, "metric": args.metric})
    return planned


def main():
    args = build_arg_parser().parse_args()

    config = {
        "metric": args.metric,
        "confidence_level": args.confidence_level,
        "ci_delta": 1 - args.confidence_level,
        "conv_delta": args.conv_delta,
        "gamma_max": args.gamma_max,
        "theta_max": args.theta_max,
        "delta": 0.0,           # the epsilon audit is (eps, 0) only
        "k": list(args.k),
        "device": args.device,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": BOUNDS_NOTE,
    }

    all_rows: List[Dict[str, Any]] = []
    saved_paths: List[str] = []
    failures: List[Dict[str, str]] = []

    # ---- re-derivation from saved v_lists: no scoring, no discovery ----
    if args.from_json:
        groups = groups_from_json(args.from_json, metric_default=args.metric)
        print(f"[audit] re-deriving bounds for {len(groups)} (sweep, model source) "
              f"combination(s) from saved v_lists -- no checkpoints will be read")
        for group in groups:
            label = f"{group['sweep']}/{group['model_source']}"
            print(f"\n[audit] === {label} (k={sorted(group['per_k'])}) ===")
            generated_at = datetime.now().isoformat(timespec="seconds")
            result = build_result(group, config, generated_at, verbose=args.verbose)
            if args.per_sweep:
                out_path = output_path(group["sweep"], group["model_source"],
                                       group["metric"], args.output_name)
                try:
                    saved = save_result(result, out_path, replace=args.replace)
                    print(f"[audit]   wrote {saved}")
                    saved_paths.append(str(saved))
                except SettingsMismatch as e:
                    print(f"[audit]   NOT SAVED: {e}")
                    failures.append({"target": label, "error": str(e)})
            all_rows.extend(summary_rows(result))

    # ---- the real thing: score the checkpoints ----
    else:
        planned = plan_groups(args)

        if args.dry_run:
            print(f"\n{len(planned)} (sweep, model source) pair(s) planned, k={args.k}:")
            for item in planned:
                print(f"  {item['sweep']}/{item['model_source']:<16} "
                      f"stats={item['stats_file'].name} -> "
                      f"{output_path(item['sweep'], item['model_source'], item['metric'], args.output_name)}")
            return 0

        for index, item in enumerate(planned, start=1):
            sweep, model_source = item["sweep"], item["model_source"]
            label = f"{sweep}/{model_source}"
            out_path = output_path(sweep, model_source, item["metric"], args.output_name)
            print(f"\n{'#' * 78}\n# [{index}/{len(planned)}] {label} "
                  f"(stats: {item['stats_file'].name})\n{'#' * 78}")

            if args.skip_existing and out_path.exists():
                print(f"[audit]   output already exists, skipping: {out_path}")
                continue

            # Check the merge target up front, so an incompatible file is reported before
            # a scoring pass is spent on it.
            previous = {} if (args.replace or not args.per_sweep) else read_existing(out_path)
            if previous:
                try:
                    assert_merge_compatible(previous, config, out_path)
                except SettingsMismatch as e:
                    print(f"[audit]   FAILED: {e}")
                    failures.append({"target": label, "error": str(e)})
                    continue

            try:
                stats = json.loads(item["stats_file"].read_text())
                dataset = args.dataset or stats.get("dataset")
                data_dir = args.data_dir or stats.get("data_dir")
                if not dataset or not data_dir:
                    raise ValueError(
                        f"{item['stats_file']} records no dataset/data_dir; pass --dataset "
                        f"and --data-dir explicitly")

                rankings, scoring_failed, m = collect_rankings(
                    stats, sweep / "test_run", dataset, data_dir, args.device,
                    args.batch_size, item["metric"], model_name=args.model,
                    num_classes=args.num_classes, filters=args.filters, label=label,
                )

                # One scoring pass above; every k below is a re-slice of the same rankings.
                per_k = {}
                for k in args.k:
                    v_list, run_ids, failed_k = v_list_from_rankings(rankings, k, m, label=label)
                    per_k[int(k)] = {
                        "v_list": v_list,
                        "run_ids": run_ids,
                        "failed_runs": [list(t) for t in (scoring_failed + failed_k)],
                    }

                group = {
                    "sweep": sweep, "model_source": model_source,
                    "stats_file": str(item["stats_file"]), "metric": item["metric"],
                    "dataset": dataset, "data_dir": data_dir, "m": m,
                    "run_ids": [run_id for run_id, _, _ in rankings],
                    "scoring_failed_runs": [list(t) for t in scoring_failed],
                    "per_k": per_k,
                }
            except Exception as e:
                print(f"[audit]   FAILED: {e}")
                if args.verbose:
                    traceback.print_exc()
                failures.append({"target": label, "error": str(e)})
                continue

            generated_at = datetime.now().isoformat(timespec="seconds")
            result = build_result(group, config, generated_at, verbose=args.verbose)

            # Saved per sweep/source as soon as it is done, so a crash later keeps this.
            if args.per_sweep:
                try:
                    saved = save_result(result, out_path, replace=args.replace)
                    print(f"[audit]   wrote {saved}")
                    saved_paths.append(str(saved))
                except SettingsMismatch as e:
                    print(f"[audit]   NOT SAVED: {e}")
                    failures.append({"target": label, "error": str(e)})

            all_rows.extend(summary_rows(result))

    print_summary(all_rows, confidence_level=args.confidence_level,
                  conv_delta=args.conv_delta)

    if failures:
        print(f"\n[audit] {len(failures)} target(s) failed:")
        for failure in failures:
            print(f"  {failure['target']}: {failure['error']}")

    if args.summary_out:
        payload = {
            "config": config,
            "rows": all_rows,
            "saved_files": saved_paths,
            "failures": failures,
            "note": BOUNDS_NOTE,
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_jsonable(payload), indent=2))
        print(f"\n[audit] wrote combined summary -> {summary_path} "
              f"({len(all_rows)} row(s) over {len(set((r['sweep'], r['model_source']) for r in all_rows))} "
              f"(sweep, model source) combination(s))")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
