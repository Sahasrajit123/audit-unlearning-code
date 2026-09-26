#!/usr/bin/env python3
"""Aggregate runs into the final epsilon lower bound, plus diagnostics and plots.

    python scripts/aggregate_audit.py --config configs/base.yaml --method noop

What it does, in the order that keeps the audit valid:

1. Load the CALIBRATION runs' candidate losses and fit the in/out Gaussians.
   Evaluation observations are never touched here.
2. For each EVALUATION run, call ``attack.predict`` with only the observed losses
   and the frozen calibration -- no sign vector in scope.
3. Only then reveal each run's sign vector and compute the overlap ``V``.
4. Feed ``{V}`` into the vendored mean-based bound and halve it (Lemma 4.1). The same
   ``{V}`` also goes into the zCDP and GDP audits of ``audit_tofu.rho_mu_bounds``,
   which are *not* halved -- their reductions carry the factor internally.
5. Report every ``r``, plus per-author/per-QA accuracy as diagnostics, plots, and
   the measured resource roll-up.

Selecting ``r``: use ``--select_r_from calibration`` to pick ``r`` by held-out
cross-validation *within the calibration runs only*, freeze it, and report it
separately from the full sweep. Picking ``r`` by maximising epsilon on the
evaluation runs would invalidate the confidence level.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.attack import CalibrationWarning, fit_calibration, overlap, predict
from audit_tofu.config import load_config
from audit_tofu.epsilon_bounds import epsilon_lb_report
from audit_tofu.manifest import load_manifest
from audit_tofu.rho_mu_bounds import DEFAULT_CONV_DELTA, mu_lb_report, rho_lb_report
from audit_tofu.run_manager import load_json, resolve_run_paths, save_json


def _dp_variant_reports(
    m: int, r: int, v_list: List[int], eps_cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The zCDP and GDP audits of the same overlaps, at the same confidence level.

    Same ``(m, r, v_list, zeta)`` as the epsilon bound -- these are three readings of
    one observation, not three experiments. Neither is halved: the Lemma 4.1 factor
    lives inside ``eps_gamma^loc(rho)`` and ``mu_loc(mu) = 2 mu``.

    ``conv_delta``/``gamma_max`` are optional config keys, defaulted here so an older
    config still aggregates.
    """
    conv_delta = float(eps_cfg.get("conv_delta", DEFAULT_CONV_DELTA))
    gamma_max = float(eps_cfg.get("gamma_max", 1e4))
    zeta = float(eps_cfg["zeta"])
    rho = rho_lb_report(
        m, r, v_list, zeta=zeta, conv_delta=conv_delta,
        theta_max=float(eps_cfg["theta_max"]), gamma_max=gamma_max,
    )
    mu = mu_lb_report(
        m, r, v_list, zeta=zeta, conv_delta=conv_delta,
        theta_max=float(eps_cfg["theta_max"]),
    )
    return rho, mu


def _load_family_losses(
    cfg: Dict[str, Any], manifest: Dict[str, Any], family: str, method: str
) -> Dict[str, Dict[Tuple[str, int], float]]:
    """``{run_id: {(batch_id, qa_id): loss}}`` for every completed run in a family."""
    out: Dict[str, Dict[Tuple[str, int], float]] = {}
    for run_id in manifest["sign_vectors"][family]["run_ids"]:
        paths = resolve_run_paths(cfg["experiment"]["output_root"], run_id)
        p = paths.method_dir(method) / "losses.json"
        if not p.exists():
            continue
        payload = load_json(p)
        if payload.get("split_hash") != manifest["split_hash"]:
            raise RuntimeError(
                f"{run_id}/{method}: losses were produced against a different split"
            )
        scores: Dict[Tuple[str, int], float] = {}
        for rec in payload["records"]:
            if rec.get("batch_id") is None:
                continue
            scores[(rec["batch_id"], int(rec["qa_id"]))] = float(rec["loss"])
        out[run_id] = scores
    return out


def _signs_for(manifest: Dict[str, Any], family: str) -> Dict[str, Dict[str, int]]:
    """``{run_id: {batch_id: +-1}}``."""
    batch_ids = manifest["split"]["batch_ids"]
    vecs = manifest["sign_vectors"][family]["vectors"]
    return {rid: dict(zip(batch_ids, v)) for rid, v in vecs.items()}


def _select_r_on_calibration(
    calib_losses, calib_signs, batch_ids, r_values, attack_cfg, eps_cfg, n_eval: int
) -> Dict[str, Any]:
    """Leave-one-out over CALIBRATION runs to choose ``r``, using no eval data.

    For each held-out calibration run we refit on the remaining ones, predict, and
    score the overlap. We then pick the ``r`` that maximises the **epsilon lower
    bound** those calibration overlaps would produce.

    Optimising epsilon directly -- rather than, say, the overlap margin over chance
    -- matters. The normalized margin ``(V - r/2)/(r/2)`` equals 1.0 for *every* r
    under a perfect attack, so it cannot distinguish them and ties resolve
    arbitrarily; but epsilon still grows with r (a perfect attack at r=20 certifies
    6.59 versus 1.18 at r=4). Since epsilon is the reported quantity, it is the right
    objective, and it is what makes the inverted-U of Remark 4.3 visible.

    The LOO overlaps are evaluated at ``L = n_eval`` -- the number of evaluation runs
    the frozen r will actually face -- so the comparison across r reflects the real
    operating point rather than the larger calibration sample.
    """
    run_ids = sorted(calib_losses)
    overlaps: Dict[int, List[int]] = {r: [] for r in r_values}

    for held in run_ids:
        rest = {k: v for k, v in calib_losses.items() if k != held}
        if len(rest) < 4:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CalibrationWarning)
            cal = fit_calibration(
                rest, calib_signs, batch_ids,
                var_floor=float(attack_cfg["var_floor"]),
                pool_variance=attack_cfg["pool_variance"],
                warn=False,
            )
        for r in r_values:
            if r > len(batch_ids):
                continue
            pred = predict(
                calib_losses[held], cal, r, aggregate=attack_cfg["aggregate"]
            )
            v = overlap(pred["guess"], [calib_signs[held][b] for b in batch_ids])
            overlaps[r].append(int(v))

    m = len(batch_ids)
    margin_by_r: Dict[int, Any] = {}
    epsilon_by_r: Dict[int, Any] = {}

    for r, vs in overlaps.items():
        if not vs:
            margin_by_r[r] = epsilon_by_r[r] = None
            continue
        margin_by_r[r] = float(np.mean([(v - r / 2.0) / (r / 2.0) for v in vs]))
        # Project the observed mean overlap onto L = n_eval runs. Rounding down keeps
        # the projection conservative rather than optimistic.
        v_mean = float(np.mean(vs))
        projected = [int(np.floor(v_mean))] * max(1, n_eval)
        try:
            rep = epsilon_lb_report(
                m, r, projected,
                zeta=float(eps_cfg["zeta"]),
                delta=float(eps_cfg["delta"]),
                theta_max=float(eps_cfg["theta_max"]),
            )
            epsilon_by_r[r] = rep["mean"]["epsilon_lb"]
        except ValueError:
            epsilon_by_r[r] = None

    scored = {r: e for r, e in epsilon_by_r.items() if e is not None}
    if scored:
        # Ties break toward the SMALLER r: fewer forced low-confidence guesses.
        best = max(sorted(scored), key=lambda r: (scored[r], -r))
    else:
        # No r certified anything on calibration; fall back to the largest r, which
        # has the highest attainable ceiling.
        best = max(r_values) if r_values else None

    return {
        "method": "leave-one-out over calibration runs; argmax of projected epsilon_lb",
        "objective": "epsilon_lb",
        "mean_overlap_by_r": {r: (float(np.mean(v)) if v else None)
                              for r, v in overlaps.items()},
        "normalized_margin_by_r": margin_by_r,
        "projected_epsilon_lb_by_r": epsilon_by_r,
        "projected_at_L": n_eval,
        "selected_r": best,
        "note": (
            "r chosen using calibration runs only, then frozen. Evaluation runs were "
            "not consulted, so the reported confidence level remains valid. The "
            "projected epsilon values are a selection statistic, not the audit's "
            "result; the reported bound comes from the evaluation runs."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--method", required=True, help="noop | npo | retain_ft")
    ap.add_argument("--out_dir", default=None, help="defaults to <output_root>/audit")
    ap.add_argument(
        "--select_r_from",
        choices=["none", "calibration"],
        default="calibration",
        help="how to freeze r before looking at evaluation runs",
    )
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    manifest = load_manifest(cfg["experiment"]["manifest_path"])
    method = args.method
    batch_ids = manifest["split"]["batch_ids"]
    m = manifest["split"]["m"]

    out_dir = Path(
        args.out_dir or (Path(cfg["experiment"]["output_root"]) / "audit" / method)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_cfg = cfg["attack"]
    eps_cfg = cfg["epsilon"]

    # ---- 1. calibration only ----------------------------------------------------
    calib_losses = _load_family_losses(cfg, manifest, "calibration", method)
    eval_losses = _load_family_losses(cfg, manifest, "evaluation", method)
    calib_signs = _signs_for(manifest, "calibration")
    eval_signs = _signs_for(manifest, "evaluation")

    print(f"[aggregate] method={method}  m={m}")
    print(f"[aggregate] calibration runs found: {len(calib_losses)}")
    print(f"[aggregate] evaluation  runs found: {len(eval_losses)}")
    if not calib_losses:
        raise SystemExit("no calibration losses found; run the calibration runs first")
    if not eval_losses:
        raise SystemExit("no evaluation losses found; run the evaluation runs first")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", CalibrationWarning)
        calibration = fit_calibration(
            calib_losses, calib_signs, batch_ids,
            var_floor=float(attack_cfg["var_floor"]),
            pool_variance=attack_cfg["pool_variance"],
        )
    for w in caught:
        print(f"[aggregate] WARNING: {w.message}")

    diag = calibration.diagnostics
    print(f"[aggregate] fitted {diag['num_qa_fitted']} QA Gaussians, "
          f"{diag['num_degenerate']} degenerate, "
          f"min n_in={diag['min_n_in']} min n_out={diag['min_n_out']}")
    save_json(out_dir / "calibration.json", calibration.as_dict())

    # ---- 2. freeze r using calibration only -------------------------------------
    r_values = [int(r) for r in attack_cfg["r_values"] if int(r) <= m]
    r_selection = None
    if args.select_r_from == "calibration":
        r_selection = _select_r_on_calibration(
            calib_losses, calib_signs, batch_ids, r_values, attack_cfg,
            eps_cfg, len(eval_losses),
        )
        print(f"[aggregate] r selected on calibration: {r_selection['selected_r']} "
              f"(projected eps by r: "
              f"{ {k: None if v is None else round(v, 3) for k, v in r_selection['projected_epsilon_lb_by_r'].items()} })")
    frozen_r = attack_cfg.get("r_frozen") or (
        r_selection["selected_r"] if r_selection else None
    )

    # ---- 3. predict per evaluation run, THEN reveal labels ----------------------
    per_r: Dict[int, Dict[str, Any]] = {}
    per_run_records: List[Dict[str, Any]] = []

    for r in r_values:
        v_list: List[int] = []
        runs: List[Dict[str, Any]] = []

        for run_id in sorted(eval_losses):
            # Prediction: no labels in scope.
            pred = predict(
                eval_losses[run_id], calibration, r,
                aggregate=attack_cfg["aggregate"],
            )
            # Reveal only now.
            true_vec = [eval_signs[run_id][b] for b in batch_ids]
            v = overlap(pred["guess"], true_vec)
            v_list.append(v)

            correct_pos = sum(
                1 for g, s in zip(pred["guess"], true_vec) if g == 1 and s == 1
            )
            correct_neg = sum(
                1 for g, s in zip(pred["guess"], true_vec) if g == -1 and s == -1
            )
            runs.append({
                "run_id": run_id,
                "overlap_v": int(v),
                "correct_positive": int(correct_pos),
                "correct_negative": int(correct_neg),
                "guesses_per_side": r // 2,
                "author_accuracy": v / r,
                "predicted_positive": pred["predicted_positive"],
                "predicted_negative": pred["predicted_negative"],
                "lambdas": pred["lambdas"],
                "true_sign_vector": true_vec,
            })

        report = epsilon_lb_report(
            m, r, v_list,
            zeta=float(eps_cfg["zeta"]),
            delta=float(eps_cfg["delta"]),
            theta_max=float(eps_cfg["theta_max"]),
        )
        rho_report, mu_report = _dp_variant_reports(m, r, v_list, eps_cfg)

        per_r[r] = report
        per_r[r]["runs"] = runs
        per_r[r]["v_list"] = v_list
        per_r[r]["rho"] = rho_report
        per_r[r]["mu"] = mu_report

        eps = report["mean"]["epsilon_lb"]
        rho_v = rho_report["mean"]["rho_lb"]
        mu_v = mu_report["mean"]["mu_lb"]
        print(f"[aggregate] r={r:3d}  mean V={np.mean(v_list):5.2f}/{r}  "
              f"(chance {r/2:g})  eps_LB={eps if eps is None else round(eps, 4)}"
              f"  rho_LB={rho_v if rho_v is None else round(rho_v, 4)}"
              f"  mu_LB={mu_v if mu_v is None else round(mu_v, 4)}")
        if r == frozen_r:
            per_run_records = runs

    # ---- 4. per-QA diagnostic accuracy ------------------------------------------
    qa_correct, qa_total = 0, 0
    for run_id in sorted(eval_losses):
        for (bid, qa_id), loss in eval_losses[run_id].items():
            g = calibration.gaussians.get((bid, qa_id))
            if g is None or not g.usable:
                continue
            pred_sign = 1 if g.llr(loss) > 0 else -1
            qa_total += 1
            qa_correct += int(pred_sign == eval_signs[run_id][bid])
    qa_accuracy = qa_correct / qa_total if qa_total else None

    headline = per_r.get(frozen_r) or per_r[max(per_r)]
    summary = {
        "method": method,
        "m": m,
        "L": len(eval_losses),
        "Gamma": len(calib_losses),
        "zeta": float(eps_cfg["zeta"]),
        "delta": float(eps_cfg["delta"]),
        "aggregate": attack_cfg["aggregate"],
        "frozen_r": frozen_r,
        "r_selection": r_selection,
        "headline": {
            "r": headline["mean"]["r"],
            "epsilon_lb_mean": headline["mean"]["epsilon_lb"],
            "epsilon_ldp_lb_mean": headline["mean"]["epsilon_ldp_lb"],
            "epsilon_lb_median": headline["median"]["epsilon_lb"],
            "halving_applied_by": headline["mean"]["halving_applied_by"],
            # zCDP / GDP readings of the SAME v_list, not halved.
            "rho_lb_mean": headline["rho"]["mean"]["rho_lb"],
            "rho_lb_median": headline["rho"]["median"]["rho_lb"],
            "mu_lb_mean": headline["mu"]["mean"]["mu_lb"],
            "mu_lb_median": headline["mu"]["median"]["mu_lb"],
            "eps_estimate_from_rho_mean": headline["rho"]["mean"]["eps_estimate"],
            "eps_estimate_from_mu_mean": headline["mu"]["mean"]["eps_estimate"],
            "conv_delta": headline["rho"]["mean"]["conv_delta"],
            "mean_overlap": headline["mean"]["v_mean"],
            "v_list": headline["v_list"],
            "random_guess_baseline": headline["mean"]["random_guess_baseline"],
        },
        "per_r": {
            str(r): {
                "mean_overlap": float(np.mean(per_r[r]["v_list"])),
                "median_overlap": float(np.median(per_r[r]["v_list"])),
                "v_list": per_r[r]["v_list"],
                "epsilon_lb_mean": per_r[r]["mean"]["epsilon_lb"],
                "epsilon_lb_median": per_r[r]["median"]["epsilon_lb"],
                "epsilon_ldp_lb_mean": per_r[r]["mean"]["epsilon_ldp_lb"],
                "rho_lb_mean": per_r[r]["rho"]["mean"]["rho_lb"],
                "rho_lb_median": per_r[r]["rho"]["median"]["rho_lb"],
                "mu_lb_mean": per_r[r]["mu"]["mean"]["mu_lb"],
                "mu_lb_median": per_r[r]["mu"]["median"]["mu_lb"],
                "eps_estimate_from_rho_mean": per_r[r]["rho"]["mean"]["eps_estimate"],
                "eps_estimate_from_mu_mean": per_r[r]["mu"]["mean"]["eps_estimate"],
            }
            for r in sorted(per_r)
        },
        "diagnostics": {
            "per_qa_llr_accuracy": qa_accuracy,
            "per_qa_n": qa_total,
            "calibration": diag,
        },
        "note": (
            "epsilon_lb is the unlearning bound (LDP solution halved per Lemma 4.1; "
            "'halving_applied_by' records which layer divided, so the halving cannot "
            "happen twice). rho_lb (zCDP) and mu_lb (GDP) audit the SAME v_list at the "
            "same zeta and are NOT halved -- the reference-law factor is already inside "
            "eps_gamma^loc(rho) and mu_loc(mu)=2mu. eps_estimate_from_* are (eps, "
            "conv_delta) conversions for readability, NOT lower bounds on epsilon. "
            "Per-author and per-QA accuracies are diagnostics only and are not "
            "inputs to the bound. TOFU forget quality, if computed, lives in each "
            "run's utility.json and is a separate utility metric."
        ),
    }

    save_json(out_dir / "audit_summary.json", summary)
    save_json(out_dir / "per_r_detail.json", {str(k): v for k, v in per_r.items()})

    # ---- 5. plots ---------------------------------------------------------------
    if not args.no_plots:
        from audit_tofu.plotting import (
            plot_calibration_distributions,
            plot_epsilon_vs_r,
            plot_lambda_distribution,
            plot_overlap_per_run,
        )

        plot_calibration_distributions(
            calibration, out_dir / "calibration_losses.png", method=f"({method})"
        )
        plot_epsilon_vs_r(per_r, out_dir / "epsilon_vs_r.png", method=f"({method})")

        recs = per_run_records or per_r[max(per_r)]["runs"]
        pos_l, neg_l = [], []
        for rec in recs:
            for b, s in zip(batch_ids, rec["true_sign_vector"]):
                (pos_l if s == 1 else neg_l).append(rec["lambdas"][b])
        plot_lambda_distribution(
            pos_l, neg_l, out_dir / "lambda_distribution.png", method=f"({method})"
        )
        r_show = headline["mean"]["r"]
        plot_overlap_per_run(
            headline["v_list"], r_show, out_dir / "overlap_per_run.png",
            method=f"({method})",
        )
        print(f"[aggregate] plots -> {out_dir}")

    print("\n" + "=" * 70)
    print(f"AUDIT SUMMARY  method={method}  m={m}  Gamma={len(calib_losses)}  L={len(eval_losses)}")
    print("=" * 70)
    h = summary["headline"]
    print(f"  frozen r            : {frozen_r}")
    print(f"  mean overlap        : {h['mean_overlap']:.3f} / {h['r']}  "
          f"(chance {h['random_guess_baseline']:g})")
    print(f"  per-run overlaps    : {h['v_list']}")
    print(f"  epsilon_LB (mean)   : {h['epsilon_lb_mean']}")
    print(f"  epsilon_LB (median) : {h['epsilon_lb_median']}")
    print(f"  rho_LB    zCDP mean : {h['rho_lb_mean']}   (median {h['rho_lb_median']})")
    print(f"  mu_LB      GDP mean : {h['mu_lb_mean']}   (median {h['mu_lb_median']})")
    print(f"  eps est. from rho/mu: {h['eps_estimate_from_rho_mean']} / "
          f"{h['eps_estimate_from_mu_mean']}  (delta={h['conv_delta']}, not a bound)")
    print(f"  per-QA LLR accuracy : {qa_accuracy}")
    print(f"  results             : {out_dir}")
    if method == "noop" and (h["epsilon_lb_mean"] in (None, 0.0)):
        print("\n  !! noop is the positive leakage control and produced NO bound.")
        print("     Diagnose training exposure, loss masking, or aggregation")
        print("     before spending compute on NPO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
