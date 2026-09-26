#!/usr/bin/env python3
"""Per-method leakage and damage diagnostics, side by side.

    python scripts/compare_methods.py --config configs/base.yaml

For every method with results, reports the raw in/out separation on the candidate
pool alongside how badly the model was damaged getting there. Both numbers are
needed, because a method can score well on the first for the wrong reason.

Columns
-------
``mean_in``      mean answer loss on candidates that were trained on then unlearned
``mean_out``     mean answer loss on candidates never trained on
``gap``          ``mean_out - mean_in``. Positive = trained-on candidates are still
                 easier, i.e. leakage. NEGATIVE = the method *over*-forgot, pushing
                 them above never-seen ones -- equally detectable by the attack,
                 which fits per-QA Gaussians and does not care about the sign.
``d``            Cohen's d for the gap. Single-run RAW effect size, so it badly
                 understates the calibrated attack (see docs/RESOURCE_ESTIMATE.md
                 §0b: calibration bought 107x on this data). Use it to compare
                 methods against each other, not to predict epsilon.
``%uniform``     ``mean loss / ln(vocab_size)``. A model emitting a uniform
                 distribution over the vocabulary scores exactly ``ln(V)``, so 100%
                 means the model has collapsed to noise and predicts nothing.
``fgt_nll``      final forget-set NLL during unlearning, from metrics.json
``retain``       final retain loss during unlearning
``div``          ``YES`` if the run recorded ``diverged``; ``yes*`` if collapse is
                 inferred from ``%uniform`` but the stored flag is False (runs made
                 before the threshold became relative to ``ln(vocab)``)

Why ``%uniform`` matters
------------------------
A near-zero ``d`` can mean two opposite things: the method genuinely removed the
leakage, or it destroyed the model so there is no signal left anywhere. Those look
identical in ``gap`` and ``d`` but are trivially distinguished by ``%uniform``.
Measured example: ``grad_ascent`` reaches d = 0.05 -- and 101% of uniform, on BOTH
conditions. It did not selectively forget; it stopped predicting.

This is a diagnostic, not the audit. The epsilon lower bound comes from
``scripts/aggregate_audit.py``, which uses per-QA calibrated likelihood ratios over
independent evaluation runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.manifest import load_manifest
from audit_tofu.run_manager import load_json, resolve_run_paths


def _vocab_size(model_id: str) -> Optional[int]:
    """Vocabulary size for the uniform-loss reference. Config only, no weights."""
    try:
        from transformers import AutoConfig

        return int(AutoConfig.from_pretrained(model_id).vocab_size)
    except Exception as exc:
        print(f"[compare] could not read vocab_size from {model_id}: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run_id", default=None,
                    help="restrict to one run; default aggregates over all completed")
    ap.add_argument("--vocab_size", type=int, default=None,
                    help="override; otherwise read from the model config")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    man = load_manifest(cfg["experiment"]["manifest_path"])
    root = cfg["experiment"]["output_root"]
    batch_ids = man["split"]["batch_ids"]

    V = args.vocab_size or _vocab_size(cfg["model"]["id"])
    uniform = math.log(V) if V else None

    # run_id -> {batch_id: sign}
    signs: Dict[str, Dict[str, int]] = {}
    for fam in ("calibration", "evaluation"):
        f = man["sign_vectors"][fam]
        for rid in f["run_ids"]:
            signs[rid] = dict(zip(batch_ids, f["vectors"][rid]))

    wanted = [args.run_id] if args.run_id else list(signs)
    per_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rid in wanted:
        if rid not in signs:
            raise SystemExit(f"unknown run_id {rid!r}")
        paths = resolve_run_paths(root, rid)
        if not paths.methods_dir.exists():
            continue
        for mdir in sorted(paths.methods_dir.iterdir()):
            lf = mdir / "losses.json"
            if not mdir.is_dir() or not lf.exists():
                continue
            recs = load_json(lf)["records"]
            ins = [r["loss"] for r in recs if signs[rid].get(r["batch_id"]) == 1]
            outs = [r["loss"] for r in recs if signs[rid].get(r["batch_id"]) == -1]
            if len(ins) < 2 or len(outs) < 2:
                continue
            mi, mo = float(np.mean(ins)), float(np.mean(outs))
            sd = math.sqrt((np.var(ins, ddof=1) + np.var(outs, ddof=1)) / 2)

            entry = {
                "run_id": rid,
                "mean_in": mi,
                "mean_out": mo,
                "gap": mo - mi,
                "cohens_d": (mo - mi) / sd if sd > 0 else float("nan"),
                "mean_all": float(np.mean(ins + outs)),
            }
            mf = mdir / "metrics.json"
            if mf.exists():
                try:
                    unl = (load_json(mf).get("unlearning") or {})
                    hist = unl.get("history") or []
                    entry["final_forget_nll"] = unl.get("final_forget_nll")
                    entry["final_retain_loss"] = (
                        hist[-1].get("retain_loss") if hist else None
                    )
                    entry["diverged"] = unl.get("diverged")
                except Exception:
                    pass
            per_method[mdir.name].append(entry)

    if not per_method:
        raise SystemExit(f"no results found under {root}/runs")

    print(f"model      : {cfg['model']['id']}")
    if uniform:
        print(f"vocab      : {V:,}  ->  uniform-output loss = ln(V) = {uniform:.3f}")
    print(f"runs       : {len(wanted)} requested, "
          f"{len({e['run_id'] for v in per_method.values() for e in v})} with results")
    print(f"m          : {man['split']['m']}  batching={man['split']['batching']}")

    # Order: control first, then by |gap| descending so the informative ones lead.
    order = sorted(per_method, key=lambda m: (m != "noop",
                                              -abs(np.mean([e["gap"] for e in per_method[m]]))))
    hdr = (f"\n{'method':<13}{'n':>3}{'mean_in':>9}{'mean_out':>10}{'gap':>9}"
           f"{'d':>7}{'%unif':>7}{'fgt_nll':>9}{'retain':>8}{'div':>5}")
    print(hdr)
    print("-" * len(hdr.strip("\n")))

    rows_out = []
    for m in order:
        es = per_method[m]
        gaps = [e["gap"] for e in es]
        ds = [e["cohens_d"] for e in es]
        mi = np.mean([e["mean_in"] for e in es])
        mo = np.mean([e["mean_out"] for e in es])
        allm = np.mean([e["mean_all"] for e in es])
        pu = f"{100 * allm / uniform:.0f}%" if uniform else "-"
        fn = [e.get("final_forget_nll") for e in es if e.get("final_forget_nll") is not None]
        rl = [e.get("final_retain_loss") for e in es if e.get("final_retain_loss") is not None]
        # The stored `diverged` flag is whatever the run recorded. Runs made before
        # the threshold was made relative to ln(vocab) can carry a false negative, so
        # fall back to the %uniform criterion, which needs no stored state.
        dv_stored = any(e.get("diverged") for e in es)
        dv_derived = uniform is not None and (100 * allm / uniform) > 80
        dv = dv_stored or dv_derived
        dv_mark = "YES" if dv_stored else ("yes*" if dv_derived else "-")
        print(f"{m:<13}{len(es):>3}{mi:>9.4f}{mo:>10.4f}{np.mean(gaps):>+9.4f}"
              f"{np.mean(ds):>7.2f}{pu:>7}"
              f"{(f'{np.mean(fn):.2f}' if fn else '-'):>9}"
              f"{(f'{np.mean(rl):.2f}' if rl else '-'):>8}"
              f"{dv_mark:>5}")
        rows_out.append({
            "method": m, "n_runs": len(es),
            "mean_in": float(mi), "mean_out": float(mo),
            "gap": float(np.mean(gaps)), "gap_sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None,
            "cohens_d": float(np.mean(ds)),
            "pct_of_uniform": float(100 * allm / uniform) if uniform else None,
            "final_forget_nll": float(np.mean(fn)) if fn else None,
            "final_retain_loss": float(np.mean(rl)) if rl else None,
            "diverged": dv,
            "diverged_stored": dv_stored,
            "diverged_from_pct_uniform": dv_derived,
        })

    # --- interpretation flags ------------------------------------------------
    print("\nflags:")
    any_flag = False
    for r in rows_out:
        m, pu, d = r["method"], r["pct_of_uniform"], r["cohens_d"]
        if pu is not None and pu > 80:
            any_flag = True
            print(f"  {m}: COLLAPSED — {pu:.0f}% of the uniform-output loss. A small "
                  f"epsilon here\n      means there is no model left, not that "
                  f"unlearning succeeded.")
        elif r["diverged"]:
            any_flag = True
            print(f"  {m}: diverged flag set (forget NLL "
                  f"{r['final_forget_nll']:.1f}). Read epsilon with the utility metrics.")
        if r["gap"] < -0.05:
            any_flag = True
            print(f"  {m}: OVER-forgets (gap {r['gap']:+.3f}) — trained-on candidates "
                  f"now score\n      ABOVE never-seen ones. Still fully detectable: the "
                  f"attack is sign-agnostic.")
    if not any_flag:
        print("  none")

    noop = next((r for r in rows_out if r["method"] == "noop"), None)
    if noop:
        print(f"\nrelative to the noop control (gap {noop['gap']:+.4f}, d {noop['cohens_d']:.2f}):")
        for r in rows_out:
            if r["method"] == "noop":
                continue
            frac = abs(r["gap"]) / abs(noop["gap"]) if noop["gap"] else float("nan")
            print(f"  {r['method']:<13} |gap| is {frac:5.0%} of control's")
        print("\n  ~100% means the method left the leakage essentially untouched.")

    if any(r["diverged_from_pct_uniform"] and not r["diverged_stored"] for r in rows_out):
        print("\n  'yes*' = collapse inferred from %uniform, but the run's stored")
        print("  `diverged` flag is False. Those runs predate the fix that made the")
        print("  threshold relative to ln(vocab); see docs/METHOD_DIAGNOSTICS.md.")

    print("\nNOTE: raw single-run effect sizes; the calibrated attack is far stronger.")
    print("      The audit's epsilon comes from scripts/aggregate_audit.py.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": cfg["model"]["id"], "vocab_size": V,
             "uniform_loss": uniform, "methods": rows_out,
             "per_run": {k: v for k, v in per_method.items()}},
            indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
