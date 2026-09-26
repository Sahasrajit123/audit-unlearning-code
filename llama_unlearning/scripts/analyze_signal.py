#!/usr/bin/env python3
"""Does the attack have signal? Decompose the variance in candidate losses.

    python scripts/analyze_signal.py --config configs/proxy_pilot.yaml --method noop

A single run cannot answer this. The raw in/out gap in one run is swamped by
*between-QA* difficulty variance -- some questions are simply harder than others --
and removing exactly that nuisance is what the per-QA calibration does. What
determines the attack's power is the **within-QA, run-to-run** spread, which needs
several runs with differing sign vectors to estimate.

This script separates the two and reports the implied effect sizes:

  sigma_between   SD of per-QA mean loss across QA pairs      (nuisance, calibrated away)
  sigma_within    SD of one QA pair's loss across runs        (the real noise floor)
  gap_z           mean over QA pairs of (mu_out - mu_in)      (the signal)
  d_qa            gap_z / sigma_within                        (per-QA-pair effect size)
  d_batch         d_qa * sqrt(B)                              (after summing Lambda_j)

`d_batch` is the quantity that decides everything: it sets the probability that a
true positive batch outranks a true negative one, hence the expected overlap V, hence
epsilon. The script converts it into a projected overlap and epsilon so the go/no-go
on the full run is quantitative rather than a guess.

Note `B` is the *batch* size, read from the manifest -- not `qa_per_author`. The two
coincide only for author-level batching. At `B = 1` a candidate batch IS a single QA
pair, so `Lambda_j = lambda_z` and there is no aggregation gain: `d_batch == d_qa`.

Pooling note: with only a handful of runs, each (QA pair, condition) cell holds 2-3
observations, far too few individually. But pooling the centred residuals over all
400 QA pairs gives hundreds of degrees of freedom, so sigma_within is well estimated
even when no single cell is.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.epsilon_bounds import epsilon_lb_mean
from audit_tofu.manifest import batch_size_of, load_manifest
from audit_tofu.run_manager import load_json, resolve_run_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default="noop")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--L", type=int, default=10, help="evaluation runs the audit will use")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    manifest = load_manifest(cfg["experiment"]["manifest_path"])
    batch_ids = manifest["split"]["batch_ids"]
    m = manifest["split"]["m"]
    qa_per_author = manifest["dataset"]["qa_per_author"]
    # Lambda_j aggregates over the QA pairs IN A BATCH, which is B -- not
    # qa_per_author. They coincide only for author-level batching; at B=1 a batch is
    # a single QA pair and there is no aggregation gain at all.
    B = batch_size_of(manifest)

    # Collect every completed run's losses, with that run's signs.
    obs: Dict[tuple, Dict[int, List[float]]] = {}
    runs_used = []
    for family in ("calibration", "evaluation"):
        fam = manifest["sign_vectors"][family]
        for run_id in fam["run_ids"]:
            p = resolve_run_paths(cfg["experiment"]["output_root"], run_id)
            f = p.method_dir(args.method) / "losses.json"
            if not f.exists():
                continue
            signs = dict(zip(batch_ids, fam["vectors"][run_id]))
            runs_used.append(run_id)
            for rec in load_json(f)["records"]:
                b, q = rec["batch_id"], int(rec["qa_id"])
                if b is None:
                    continue
                obs.setdefault((b, q), {1: [], -1: []})[signs[b]].append(float(rec["loss"]))

    n_runs = len(runs_used)
    print(f"method       : {args.method}")
    print(f"model        : {cfg['model']['id']}")
    print(f"runs found   : {n_runs}  {runs_used}")
    print(f"m            : {m} candidate batches, B = {B} QA pair(s) each "
          f"({qa_per_author} per author)")
    if n_runs < 2:
        raise SystemExit(
            f"\nOnly {n_runs} run(s). Estimating within-QA run-to-run variance needs "
            "at least 2, and 3-5 for a usable number. Run more before deciding."
        )

    # --- variance decomposition ----------------------------------------------
    qa_means, gaps = [], []
    resid_in, resid_out = [], []
    n_in_cells, n_out_cells = [], []

    for key, by_cond in sorted(obs.items()):
        ins, outs = by_cond[1], by_cond[-1]
        n_in_cells.append(len(ins))
        n_out_cells.append(len(outs))
        if ins and outs:
            gaps.append(np.mean(outs) - np.mean(ins))
        allv = ins + outs
        if allv:
            qa_means.append(np.mean(allv))
        # Centre within each (QA, condition) cell, so what remains is run-to-run noise.
        if len(ins) >= 2:
            resid_in.extend(np.asarray(ins) - np.mean(ins))
        if len(outs) >= 2:
            resid_out.extend(np.asarray(outs) - np.mean(outs))

    resid = np.asarray(resid_in + resid_out, dtype=float)
    dof_in = sum(max(0, c - 1) for c in n_in_cells)
    dof_out = sum(max(0, c - 1) for c in n_out_cells)
    dof = dof_in + dof_out
    if dof < 10:
        raise SystemExit(
            f"only {dof} residual degrees of freedom; add more runs before trusting this"
        )

    sigma_within = float(np.sqrt(np.sum(resid ** 2) / dof))
    sigma_between = float(np.std(np.asarray(qa_means), ddof=1))
    gap = float(np.mean(gaps))
    gap_se = float(np.std(gaps, ddof=1) / math.sqrt(len(gaps)))

    print(f"\nobservations per (QA, condition) cell: in {min(n_in_cells)}-{max(n_in_cells)}, "
          f"out {min(n_out_cells)}-{max(n_out_cells)}   (residual dof = {dof})")

    print("\n--- variance decomposition -----------------------------------------")
    print(f"  sigma_between (QA difficulty spread) : {sigma_between:.4f}   <- nuisance")
    print(f"  sigma_within  (run-to-run, pooled)   : {sigma_within:.4f}   <- noise floor")
    print(f"  ratio between/within                 : {sigma_between / sigma_within:.2f}x")
    print(f"  gap_z = mean(mu_out - mu_in)         : {gap:.4f} +/- {gap_se:.4f}")

    d_qa = gap / sigma_within if sigma_within > 0 else float("inf")
    # Lambda_j sums B per-QA terms. Independent noise across them => sqrt(B).
    # At B=1 this is 1.0: a batch IS a QA pair, so there is nothing to aggregate.
    d_batch = d_qa * math.sqrt(B)
    d_uncalibrated = gap / math.sqrt(sigma_within ** 2 + sigma_between ** 2)

    print("\n--- effect sizes ---------------------------------------------------")
    print(f"  d_qa      (per QA pair, calibrated)  : {d_qa:.3f}")
    print(f"  d_batch   (per batch, Lambda_j)      : {d_batch:.3f}")
    print(f"  d_uncalibrated (single run, raw loss): {d_uncalibrated:.3f}"
          f"   <- what a naive attack would see")
    print(f"  calibration gain                     : {d_batch / d_uncalibrated:.1f}x")

    # --- project the attack's overlap ----------------------------------------
    # Rank m/2 positives against m/2 negatives on Lambda_j ~ N(+-d_batch/2, 1).
    # Simulate rather than approximate: the top-r/2 selection couples the ranks.
    rng = np.random.default_rng(0)
    n_sim = 20000
    half = m // 2
    r_values = [int(r) for r in cfg["attack"]["r_values"] if int(r) <= m]

    print("\n--- projected audit outcome ---------------------------------------")
    print(f"  (simulating Lambda_j ~ N(+-d_batch/2, 1), {n_sim} draws, L={args.L})")
    print(f"\n  {'r':>4} {'E[V]':>8} {'chance':>8} {'eps_LB':>10}")
    best = None
    for r in r_values:
        k = r // 2
        vs = []
        for _ in range(n_sim):
            pos = rng.normal(+d_batch / 2, 1.0, half)
            neg = rng.normal(-d_batch / 2, 1.0, half)
            scores = np.concatenate([pos, neg])
            is_pos = np.concatenate([np.ones(half, bool), np.zeros(half, bool)])
            order = np.argsort(-scores)
            v = int(is_pos[order[:k]].sum() + (~is_pos[order[-k:]]).sum())
            vs.append(v)
        ev = float(np.mean(vs))
        # Floor the mean overlap: epsilon_lb_mean takes integer overlaps.
        proj = [int(math.floor(ev))] * args.L
        try:
            eps = epsilon_lb_mean(m, r, proj, zeta=float(cfg["epsilon"]["zeta"]),
                                 delta=float(cfg["epsilon"]["delta"]))["epsilon_lb"]
        except ValueError:
            eps = None
        print(f"  {r:>4} {ev:>8.2f} {r / 2:>8.1f} "
              f"{'None' if eps is None else format(eps, '.3f'):>10}")
        if eps is not None and (best is None or eps > best[1]):
            best = (r, eps)

    ceiling = epsilon_lb_mean(m, m, [m] * args.L,
                              zeta=float(cfg["epsilon"]["zeta"]),
                              delta=float(cfg["epsilon"]["delta"]))["epsilon_lb"]

    print("\n--- verdict --------------------------------------------------------")
    print(f"  perfect-attack ceiling at m={m}, L={args.L} : {ceiling:.3f}")
    if best is None:
        print("  PROJECTED epsilon_LB: none at any r.")
        print("  The attack is too weak at this m to certify anything. Options:")
        print("    - raise the in/out gap (more epochs / higher LR -> more memorization)")
        print("    - use split.batching=qa (m=400) to raise the ceiling")
        print("    - increase L (more evaluation runs tightens the bound)")
        print("  DO NOT spend the full 30-run budget as configured.")
    else:
        r_b, eps_b = best
        frac = eps_b / ceiling
        print(f"  PROJECTED best epsilon_LB: {eps_b:.3f} at r={r_b} "
              f"({100 * frac:.0f}% of ceiling)")
        if frac > 0.9:
            print("  Attack is near-perfect. The full run should produce a clean bound.")
        elif frac > 0.4:
            print("  Attack has real signal but is not saturated. Worth running.")
        else:
            print("  Attack is weak. Consider raising the gap or m before committing.")

    print("\nCaveats: d_batch assumes per-QA noise is independent within a batch;")
    print("correlation would reduce the sqrt(B) gain (at B=1 there is no gain to lose).")
    print("assumes calibration Gaussians are estimated well, which needs Gamma >> these")
    print("few runs. Treat this as an order-of-magnitude go/no-go, not a prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
