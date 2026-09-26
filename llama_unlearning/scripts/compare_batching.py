#!/usr/bin/env python3
"""Compare realized epsilon across audit batch sizes B, by leave-one-out.

    python scripts/compare_batching.py \
        --config configs/base.yaml --config configs/qa_level.yaml --method noop

For each config it fits calibration on all-but-one completed run, predicts the
held-out one, reveals the labels, and computes the epsilon lower bound. Sweeping r
gives the realized bound at that B, which is what should decide the batching -- not
the perfect-attack ceiling, and not a simulation.

This is a MODEL-SELECTION diagnostic, not an audit: it reuses calibration runs as
pseudo-evaluation, violating the calibration/evaluation independence the bound needs.
Use it to pick B, then run the real audit with the disjoint eval_* family.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.attack import CalibrationWarning, fit_calibration, overlap, predict
from audit_tofu.config import load_config
from audit_tofu.epsilon_bounds import epsilon_lb_mean
from audit_tofu.manifest import batch_size_of, load_manifest
from audit_tofu.run_manager import load_json, resolve_run_paths


def leave_one_out(cfg: dict, method: str, r_values: Optional[List[int]] = None) -> dict:
    man = load_manifest(cfg["experiment"]["manifest_path"])
    bids = man["split"]["batch_ids"]
    m = man["split"]["m"]
    B = batch_size_of(man)
    fam = man["sign_vectors"]["calibration"]

    losses, signs, runs = {}, {}, []
    for rid in fam["run_ids"]:
        p = resolve_run_paths(cfg["experiment"]["output_root"], rid).method_dir(method)
        f = p / "losses.json"
        if not f.exists():
            continue
        runs.append(rid)
        losses[rid] = {
            (r["batch_id"], int(r["qa_id"])): float(r["loss"])
            for r in load_json(f)["records"] if r.get("batch_id")
        }
        signs[rid] = dict(zip(bids, fam["vectors"][rid]))

    if len(runs) < 3:
        raise SystemExit(f"need >= 3 completed runs; found {len(runs)} for {method}")

    r_values = [r for r in (r_values or cfg["attack"]["r_values"]) if int(r) <= m]
    pool = cfg["attack"]["pool_variance"]
    print(f"=== B={B}, m={m}, {len(runs)} runs, pool_variance={pool!r} ===")
    print(f"  {'r':>5} {'mean V':>10} {'V/r':>7} {'chance':>8} {'eps_LB':>10}")

    best = {"r": None, "epsilon_lb": None}
    per_r = {}
    for rv in r_values:
        rv = int(rv)
        vs = []
        for held in runs:
            rest = {k: v for k, v in losses.items() if k != held}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", CalibrationWarning)
                cal = fit_calibration(
                    rest, signs, bids,
                    var_floor=float(cfg["attack"]["var_floor"]),
                    pool_variance=pool, warn=False,
                )
            pred = predict(losses[held], cal, rv, aggregate=cfg["attack"]["aggregate"])
            vs.append(overlap(pred["guess"], [signs[held][b] for b in bids]))
        eps = epsilon_lb_mean(
            m, rv, vs,
            zeta=float(cfg["epsilon"]["zeta"]), delta=float(cfg["epsilon"]["delta"]),
        )["epsilon_lb"]
        per_r[rv] = {"v_list": vs, "mean_v": float(np.mean(vs)), "epsilon_lb": eps}
        print(f"  {rv:>5} {np.mean(vs):>10.2f} {np.mean(vs)/rv:>7.1%} {rv/2:>8.1f} "
              f"{('None' if eps is None else format(eps, '.3f')):>10}")
        if eps is not None and (best["epsilon_lb"] is None or eps > best["epsilon_lb"]):
            best = {"r": rv, "epsilon_lb": eps}

    ceiling = epsilon_lb_mean(m, m, [m] * len(runs),
                              zeta=float(cfg["epsilon"]["zeta"]),
                              delta=float(cfg["epsilon"]["delta"]))["epsilon_lb"]
    frac = (best["epsilon_lb"] / ceiling) if best["epsilon_lb"] else 0.0
    print(f"  -> best eps_LB = {best['epsilon_lb']} at r={best['r']}  "
          f"({frac:.0%} of the {ceiling:.2f} ceiling)\n")
    return {"B": B, "m": m, "runs": runs, "per_r": per_r,
            "best": best, "ceiling": ceiling}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True,
                    help="repeat to compare several batchings")
    ap.add_argument("--method", default="noop")
    ap.add_argument("--r_values", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    results = []
    for c in args.config:
        cfg = load_config(c)
        res = leave_one_out(cfg, args.method, args.r_values)
        res["config"] = c
        results.append(res)

    if len(results) > 1:
        print("=" * 68)
        print(f"{'config':<28}{'B':>4}{'m':>6}{'best r':>8}{'eps_LB':>11}{'ceiling':>10}")
        for r in results:
            print(f"{Path(r['config']).name:<28}{r['B']:>4}{r['m']:>6}"
                  f"{str(r['best']['r']):>8}"
                  f"{('None' if r['best']['epsilon_lb'] is None else format(r['best']['epsilon_lb'], '.3f')):>11}"
                  f"{r['ceiling']:>10.2f}")
        vals = [r["best"]["epsilon_lb"] for r in results if r["best"]["epsilon_lb"]]
        if len(vals) > 1:
            print(f"\nbest/worst realized ratio: {max(vals)/min(vals):.1f}x")
        print("\nPick B on the REALIZED bound, not the ceiling. Then run the real audit")
        print("against the disjoint eval_* family -- this LOO is selection only.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
