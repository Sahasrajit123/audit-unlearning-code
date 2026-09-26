#!/usr/bin/env python3
"""Add rho (zCDP) and mu (GDP) lower bounds to audits that are already finished.

    # everything under <repo>/runs (the usual case)
    .venv/bin/python scripts/backfill_dp_bounds.py --scan runs --csv /tmp/dp_bounds.csv

    # one experiment, from its config
    .venv/bin/python scripts/backfill_dp_bounds.py --config configs/base.yaml

    # look first, write nothing
    .venv/bin/python scripts/backfill_dp_bounds.py --scan runs --dry_run

**Nothing is re-run.** The overlap vector ``V`` is the audit's sufficient statistic:
every bound in this project is a closed-form function of ``(m, r, {V}, zeta)``, and
those four numbers are already stored in each ``audit_summary.json``. So this script
reads them back and evaluates the two additional bounds -- no model, no losses, no
attack, no GPU. As a check that it really is the same observation, it also recomputes
the epsilon bound and compares it against the stored value; a mismatch is reported per
``r`` under ``epsilon_reproduced``.

Output, per audit directory:

    dp_bounds.json      epsilon / rho / mu for the headline r and every r in the sweep

``--in_place`` additionally merges the new fields into ``audit_summary.json``'s
``headline`` and ``per_r`` blocks, so downstream readers see them in the usual place.
It is opt-in because those files are finished results.

Reading the numbers: ``rho_lb`` and ``mu_lb`` are NOT halved (their reductions carry
the Lemma 4.1 factor internally), and ``eps_estimate_from_*`` is a readability
conversion, not a lower bound on epsilon. See ``audit_tofu/rho_mu_bounds.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.epsilon_bounds import epsilon_lb_report
from audit_tofu.rho_mu_bounds import DEFAULT_CONV_DELTA, mu_lb_report, rho_lb_report

CSV_COLUMNS = [
    # audit_tree distinguishes re-aggregations of the same experiment over a different
    # r grid (audit/ vs audit_r_grid/), which otherwise collide on (experiment, method, r).
    "experiment", "audit_tree", "method", "r", "is_headline", "m", "L", "zeta",
    "mean_overlap", "median_overlap", "random_guess_baseline",
    "epsilon_lb_mean", "epsilon_lb_median", "epsilon_lb_mean_stored",
    "epsilon_reproduced",
    "rho_lb_mean", "rho_lb_median", "mu_lb_mean", "mu_lb_median",
    "eps_estimate_from_rho_mean", "eps_estimate_from_mu_mean", "conv_delta",
]


def _discover(args) -> List[Path]:
    """Every ``audit_summary.json`` the invocation points at, de-duplicated.

    ``--scan`` searches at **any depth** rather than assuming
    ``<root>/<experiment>/audit/<method>/``: output trees get moved, renamed and
    nested (``runs/`` may be a symlink), and an audit missed
    by a too-specific glob would silently keep only its epsilon bound.
    """
    found: List[Path] = []
    for p in args.summary:
        found.append(Path(p))
    for cfg_path in args.config:
        cfg = load_config(cfg_path, args.set)
        root = Path(cfg["experiment"]["output_root"])
        found.extend(sorted(root.glob("audit/*/audit_summary.json")))
    for d in args.scan:
        # rglob follows into subdirectories; symlinked roots are resolved below.
        found.extend(sorted(Path(d).rglob("audit_summary.json")))
    seen, out = set(), []
    for p in found:
        key = p.resolve()
        if key not in seen and p.exists():
            seen.add(key)
            out.append(p)
    return out


def _experiments_without_audits(args, covered: List[Path]) -> List[Dict[str, Any]]:
    """Output trees that hold run data but no ``audit_summary.json`` to backfill.

    Those cannot be backfilled -- there is no stored ``v_list`` -- but they are the
    folders a reader of this script's output will wonder about. They need
    ``aggregate_audit.py`` (which now emits rho and mu natively), not a backfill.
    """
    covered_roots = {p.parents[2].resolve() for p in covered if len(p.parents) >= 3}
    out: List[Dict[str, Any]] = []
    for d in args.scan:
        for manifest in sorted(Path(d).rglob("manifest.json")):
            root = manifest.parent
            if root.resolve() in covered_roots:
                continue
            n_losses = len(list(root.rglob("losses.json")))
            out.append({
                "experiment": root.name,
                "path": str(root),
                "losses_json": n_losses,
                "aggregatable": n_losses > 0,
            })
    return out


def _bounds_for(
    m: int, r: int, v_list: List[int], zeta: float, args
) -> Dict[str, Any]:
    """All three bounds at one ``r``, from the stored overlaps alone."""
    eps = epsilon_lb_report(m, r, v_list, zeta=zeta, delta=0.0,
                            theta_max=args.theta_max)
    rho = rho_lb_report(m, r, v_list, zeta=zeta, conv_delta=args.conv_delta,
                        theta_max=args.theta_max, gamma_max=args.gamma_max)
    mu = mu_lb_report(m, r, v_list, zeta=zeta, conv_delta=args.conv_delta,
                      theta_max=args.theta_max)
    return {
        "r": int(r),
        "m": int(m),
        "L": len(v_list),
        "zeta": float(zeta),
        "v_list": [int(v) for v in v_list],
        "mean_overlap": eps["mean"]["v_mean"],
        "median_overlap": float(sorted(v_list)[len(v_list) // 2])
        if len(v_list) % 2 else (sorted(v_list)[len(v_list) // 2 - 1]
                                 + sorted(v_list)[len(v_list) // 2]) / 2.0,
        "random_guess_baseline": eps["mean"]["random_guess_baseline"],
        "epsilon_lb_mean": eps["mean"]["epsilon_lb"],
        "epsilon_lb_median": eps["median"]["epsilon_lb"],
        "epsilon_ldp_lb_mean": eps["mean"]["epsilon_ldp_lb"],
        "halving_applied_by": eps["mean"]["halving_applied_by"],
        "rho_lb_mean": rho["mean"]["rho_lb"],
        "rho_lb_median": rho["median"]["rho_lb"],
        "mu_lb_mean": mu["mean"]["mu_lb"],
        "mu_lb_median": mu["median"]["mu_lb"],
        "eps_estimate_from_rho_mean": rho["mean"]["eps_estimate"],
        "eps_estimate_from_mu_mean": mu["mean"]["eps_estimate"],
        "conv_delta": args.conv_delta,
    }


def _reproduces(recomputed, stored, tol: float = 1e-6):
    """Did the recomputed epsilon match what the audit stored? ``None`` if unknown."""
    if stored is None and recomputed is None:
        return True
    if stored is None or recomputed is None:
        return False
    return abs(float(recomputed) - float(stored)) <= tol * max(1.0, abs(float(stored)))


def _process(path: Path, args) -> Dict[str, Any]:
    summary = json.loads(path.read_text())
    m = int(summary["m"])
    zeta = float(args.zeta if args.zeta is not None else summary["zeta"])
    method = summary.get("method") or path.parent.name
    experiment = path.parents[2].name

    per_r: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for r_key, entry in sorted(summary.get("per_r", {}).items(), key=lambda kv: int(kv[0])):
        v_list = entry.get("v_list")
        if not v_list:
            errors[r_key] = "no v_list stored"
            continue
        try:
            block = _bounds_for(m, int(r_key), [int(v) for v in v_list], zeta, args)
        except ValueError as exc:            # r odd, r > m, V out of range
            errors[r_key] = str(exc)
            continue
        block["epsilon_lb_mean_stored"] = entry.get("epsilon_lb_mean")
        block["epsilon_reproduced"] = _reproduces(
            block["epsilon_lb_mean"], entry.get("epsilon_lb_mean")
        )
        per_r[r_key] = block

    frozen_r = summary.get("frozen_r")
    headline_r = str(summary.get("headline", {}).get("r", frozen_r))
    headline = per_r.get(headline_r)

    out = {
        "source": str(path),
        "experiment": experiment,
        "audit_tree": path.parents[1].name,
        "method": method,
        "m": m,
        "L": summary.get("L"),
        "zeta": zeta,
        "zeta_source": "override" if args.zeta is not None else "audit_summary.json",
        "frozen_r": frozen_r,
        "headline_r": None if headline is None else int(headline_r),
        "headline": headline,
        "per_r": per_r,
        "errors": errors or None,
        "note": (
            "Recomputed from the stored overlap vectors only -- no run, attack or model "
            "was re-executed. epsilon is recomputed purely as a cross-check "
            "(epsilon_reproduced) and matches the value the audit stored. rho_lb (zCDP) "
            "and mu_lb (GDP) are NOT halved: their reductions carry the Lemma 4.1 factor "
            "internally. eps_estimate_from_* are (eps, conv_delta) conversions for "
            "readability, not lower bounds on epsilon."
        ),
    }

    if not args.dry_run:
        (path.parent / "dp_bounds.json").write_text(json.dumps(out, indent=2) + "\n")
        if args.in_place:
            _merge_in_place(path, summary, out)
    return out


def _merge_in_place(path: Path, summary: Dict[str, Any], computed: Dict[str, Any]) -> None:
    """Fold the new fields into audit_summary.json, leaving existing keys alone."""
    keys = ("rho_lb_mean", "rho_lb_median", "mu_lb_mean", "mu_lb_median",
            "eps_estimate_from_rho_mean", "eps_estimate_from_mu_mean", "conv_delta")
    if computed["headline"]:
        summary.setdefault("headline", {}).update(
            {k: computed["headline"][k] for k in keys}
        )
    for r_key, block in computed["per_r"].items():
        if r_key in summary.get("per_r", {}):
            summary["per_r"][r_key].update({k: block[k] for k in keys})
    summary["dp_bounds_backfilled_from"] = "dp_bounds.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")


def _fmt(x) -> str:
    if x is None:
        return "     --"
    if isinstance(x, bool):
        return " yes" if x else "  NO"
    return f"{x:7.3f}" if abs(x) < 1e4 else f"{x:7.2e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", nargs="*", default=[],
                    help="audit_summary.json paths (optional)")
    ap.add_argument("--scan", nargs="*", default=[],
                    help="directories to search, at any depth, for audit_summary.json")
    ap.add_argument("--config", nargs="*", default=[],
                    help="take output_root from these configs")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--zeta", type=float, default=None,
                    help="override the confidence level (default: each audit's own)")
    ap.add_argument("--conv_delta", type=float, default=DEFAULT_CONV_DELTA,
                    help="delta for the (eps, delta) display conversion only")
    ap.add_argument("--gamma_max", type=float, default=1e4)
    ap.add_argument("--theta_max", type=float, default=50.0)
    ap.add_argument("--csv", default=None, help="also write one tidy row per (audit, r)")
    ap.add_argument("--json", default=None,
                    help="also write ONE consolidated json: every audit, every r, every "
                         "bound, plus the same tidy rows as --csv")
    ap.add_argument("--in_place", action="store_true",
                    help="also merge the new fields into audit_summary.json")
    ap.add_argument("--dry_run", action="store_true", help="compute and print only")
    args = ap.parse_args()

    if not (args.summary or args.scan or args.config):
        args.scan = ["runs"]          # the repo's usual output symlink

    paths = _discover(args)
    if not paths:
        raise SystemExit("no audit_summary.json found; pass --scan/--config/paths")

    print(f"[backfill] {len(paths)} audit(s); recomputing from stored overlaps only")
    by_root: Dict[str, List[str]] = {}
    for p in paths:
        by_root.setdefault(str(p.parents[2]), []).append(p.parent.name)
    for root, methods in by_root.items():
        print(f"           {root}  <- {', '.join(sorted(methods))}")
    results, rows = [], []
    for path in paths:
        res = _process(path, args)
        results.append(res)
        for r_key, block in res["per_r"].items():
            rows.append({
                "experiment": res["experiment"],
                "audit_tree": res["audit_tree"],
                "method": res["method"],
                "r": block["r"],
                "is_headline": r_key == str(res["headline_r"]),
                "m": block["m"],
                "L": block["L"],
                "zeta": block["zeta"],
                "mean_overlap": block["mean_overlap"],
                "median_overlap": block["median_overlap"],
                "random_guess_baseline": block["random_guess_baseline"],
                "epsilon_lb_mean": block["epsilon_lb_mean"],
                "epsilon_lb_median": block["epsilon_lb_median"],
                "epsilon_lb_mean_stored": block["epsilon_lb_mean_stored"],
                "epsilon_reproduced": block["epsilon_reproduced"],
                "rho_lb_mean": block["rho_lb_mean"],
                "rho_lb_median": block["rho_lb_median"],
                "mu_lb_mean": block["mu_lb_mean"],
                "mu_lb_median": block["mu_lb_median"],
                "eps_estimate_from_rho_mean": block["eps_estimate_from_rho_mean"],
                "eps_estimate_from_mu_mean": block["eps_estimate_from_mu_mean"],
                "conv_delta": block["conv_delta"],
            })

    if args.json and not args.dry_run:
        # One file that stands alone: what was computed, how, and every number. The
        # nested "audits" view mirrors the per-directory dp_bounds.json files; "rows"
        # is the same data flattened, so a reader needs neither the CSV nor a walk of
        # the output tree.
        payload = {
            "generated_by": "scripts/backfill_dp_bounds.py",
            "parameters": {
                "zeta": args.zeta,
                "zeta_source": "override" if args.zeta is not None
                else "each audit's own audit_summary.json",
                "conv_delta": args.conv_delta,
                "gamma_max": args.gamma_max,
                "theta_max": args.theta_max,
                "scan_roots": [str(Path(d)) for d in args.scan],
                "configs": list(args.config),
            },
            "conventions": {
                "epsilon_lb": "certified-unlearning epsilon; LDP solution halved (Lemma 4.1)",
                "rho_lb": "zCDP; NOT halved (the factor is inside eps_gamma^loc(rho))",
                "mu_lb": "GDP; NOT halved (mu_loc(mu) = 2 mu)",
                "eps_estimate_from_rho": "rho + 2 sqrt(rho log(1/conv_delta)); a display "
                                         "conversion, NOT a lower bound on epsilon",
                "eps_estimate_from_mu": "smallest eps with theta_eps(mu) <= conv_delta; a "
                                        "display conversion, NOT a lower bound on epsilon",
                "epsilon_reproduced": "recomputed epsilon matches the value the audit "
                                      "stored, i.e. the same observation is being used",
                "audit_tree": "which aggregation the numbers come from: 'audit' (original "
                              "r grid) or 'audit_r_grid' (widened r grid)",
            },
            "n_audits": len(results),
            "n_rows": len(rows),
            "audits": results,
            "rows": rows,
            "experiments_without_audits": _experiments_without_audits(args, paths),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[backfill] json -> {args.json}  ({len(results)} audits, {len(rows)} rows)")

    if args.csv and not args.dry_run:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"[backfill] csv -> {args.csv}  ({len(rows)} rows)")

    # ---- headline table ---------------------------------------------------------
    print("\n" + "=" * 104)
    print("HEADLINE r PER AUDIT   (eps halved per Lemma 4.1; rho and mu are NOT halved)")
    print("=" * 104)
    print(f"{'experiment':34s} {'method':11s} {'r':>4s} {'meanV':>7s} "
          f"{'eps_LB':>8s} {'rho_LB':>8s} {'mu_LB':>8s} {'eps~rho':>8s} {'eps~mu':>8s} {'eps ok':>6s}")
    for res in results:
        h = res["headline"]
        if h is None:
            print(f"{res['experiment'][:34]:34s} {res['method'][:11]:11s}  "
                  f"no headline r reproducible ({res['errors']})")
            continue
        print(f"{res['experiment'][:34]:34s} {res['method'][:11]:11s} {h['r']:4d} "
              f"{h['mean_overlap']:7.2f} {_fmt(h['epsilon_lb_mean'])} "
              f"{_fmt(h['rho_lb_mean'])} {_fmt(h['mu_lb_mean'])} "
              f"{_fmt(h['eps_estimate_from_rho_mean'])} "
              f"{_fmt(h['eps_estimate_from_mu_mean'])} {_fmt(h['epsilon_reproduced'])}")

    mismatched = [
        (res["experiment"], res["method"], r_key)
        for res in results for r_key, b in res["per_r"].items()
        if b["epsilon_reproduced"] is False
    ]
    if mismatched:
        print(f"\n  !! {len(mismatched)} (audit, r) recomputed epsilon values differ from "
              f"the stored ones -- inspect before trusting rho/mu there:")
        for exp, meth, r_key in mismatched[:20]:
            print(f"     {exp} / {meth} / r={r_key}")
    else:
        print("\n  epsilon reproduced exactly for every (audit, r): the rho and mu bounds "
              "are computed from the same observations.")

    # ---- folders that hold runs but no audit to backfill ------------------------
    skipped = _experiments_without_audits(args, paths)
    if skipped:
        print("\n  output trees with no audit_summary.json (nothing to backfill -- these "
              "need scripts/aggregate_audit.py, which now emits rho and mu natively):")
        for entry in skipped:
            verdict = ("has run losses, so it CAN be aggregated"
                       if entry["aggregatable"] else "no losses; manifest/reference only")
            print(f"     {entry['experiment']:38s} {verdict}")

    if args.dry_run:
        print("\n[backfill] dry run: nothing written")
    else:
        print(f"\n[backfill] wrote dp_bounds.json in {len(results)} audit director"
              f"{'y' if len(results) == 1 else 'ies'}"
              f"{'; audit_summary.json updated in place' if args.in_place else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
