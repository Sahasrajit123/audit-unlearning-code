#!/usr/bin/env python3
"""Consolidate every run's losses into one tidy table for later analysis.

    python scripts/collect_results.py --config configs/base.yaml

Walks the run tree and emits, under ``<output_root>/collected/``:

``losses.csv``       one row per (run_id, method, candidate_author, qa_id)
``losses.parquet``   same, if pyarrow is available (much faster to reload)
``runs.csv``         one row per (run_id, method): timings, memory, unlearning
                     metrics, in/out separation, and every utility metric
``truth_ratios.csv`` one row per scored QA pair per group, with author identity and
                     sign -- the grain forget quality is computed on
``manifest_flat.csv`` one row per (run_id, batch_id): the ground-truth sign vector
``summary.json``     counts, coverage, and the split hash

The loss table carries the ground-truth ``sign`` column, because at this point the
predictions are already made and stored -- this file is for *analysis*, not for the
attack. The attack itself never sees labels; see ``audit_tofu/attack.py``.

Everything is keyed by ``split_hash`` and the script refuses to mix runs from
different splits, so a regenerated manifest cannot silently contaminate a table.
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
from audit_tofu.manifest import batch_size_of, load_manifest
from audit_tofu.run_manager import load_json, resolve_run_paths


def _utility_seconds(run_metrics: Dict[str, Any], method: str) -> Any:
    """Utility-suite duration for one method, from the RUN-level metrics.

    Not from the method's own ``metrics.json``: that file is written before the
    utility suite runs, so it never carries utility timing.
    """
    methods = run_metrics.get("methods") or {}
    if not isinstance(methods, dict):
        return None
    resources = (methods.get(method) or {}).get("resources") or {}
    return (resources.get("utility") or {}).get("duration_seconds")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out_dir", default=None, help="defaults to <output_root>/collected")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    man = load_manifest(cfg["experiment"]["manifest_path"])
    root = Path(cfg["experiment"]["output_root"])
    out = Path(args.out_dir or (root / "collected"))
    out.mkdir(parents=True, exist_ok=True)

    batch_ids = man["split"]["batch_ids"]
    split_hash = man["split_hash"]

    # run_id -> (family, {batch_id: sign})
    signs: Dict[str, Any] = {}
    for fam in ("calibration", "evaluation"):
        f = man["sign_vectors"][fam]
        for rid in f["run_ids"]:
            signs[rid] = (fam, dict(zip(batch_ids, f["vectors"][rid])))

    owner = {}
    for bid, members in man["split"]["batches"].items():
        for a, q in members:
            owner[(str(a), int(q))] = bid

    loss_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    tr_rows: List[Dict[str, Any]] = []
    seen_methods, seen_runs = set(), set()

    for rid, (family, sign_map) in signs.items():
        paths = resolve_run_paths(root, rid)
        if not paths.run_dir.exists():
            continue

        run_metrics = {}
        rm = paths.run_dir / "metrics.json"
        if rm.exists():
            try:
                run_metrics = load_json(rm)
            except Exception as exc:
                print(f"[collect] {rid}: unreadable metrics.json ({exc})")

        if not paths.methods_dir.exists():
            continue
        for mdir in sorted(paths.methods_dir.iterdir()):
            if not mdir.is_dir():
                continue
            method = mdir.name
            lf = mdir / "losses.json"
            if not lf.exists():
                continue

            payload = load_json(lf)
            if payload.get("split_hash") != split_hash:
                raise SystemExit(
                    f"{rid}/{method}: split_hash {payload.get('split_hash','?')[:12]}... "
                    f"!= manifest {split_hash[:12]}.... Refusing to mix splits."
                )
            seen_methods.add(method)
            seen_runs.add(rid)

            for rec in payload["records"]:
                bid = rec.get("batch_id") or owner.get(
                    (rec["candidate_author"], int(rec["qa_id"]))
                )
                loss_rows.append({
                    "run_id": rid,
                    "family": family,
                    "method": method,
                    "batch_id": bid,
                    "candidate_author": rec["candidate_author"],
                    "qa_id": int(rec["qa_id"]),
                    # Ground truth. Present because this table is for analysis of
                    # already-finalized predictions, never an input to the attack.
                    "sign": sign_map.get(bid),
                    "loss": float(rec["loss"]),
                    "num_answer_tokens": int(rec["num_answer_tokens"]),
                })

            # per (run, method) metadata
            mm = {}
            mf = mdir / "metrics.json"
            if mf.exists():
                try:
                    mm = load_json(mf)
                except Exception:
                    mm = {}
            unl = (mm.get("unlearning") or {})
            res = (mm.get("resources") or {})
            hist = unl.get("history") or []
            row = {
                "run_id": rid,
                "family": family,
                "method": method,
                "split_hash": split_hash,
                "n_losses": sum(1 for r in payload["records"]),
                "unlearn_seconds": (res.get("unlearn") or {}).get("duration_seconds"),
                "unlearn_peak_gib": (res.get("unlearn") or {}).get("peak_allocated_gib"),
                "score_seconds": (res.get("scoring") or {}).get("duration_seconds"),
                "variant": unl.get("variant"),
                "epochs": unl.get("epochs"),
                "retain_weight": unl.get("retain_weight"),
                "forget_weight": unl.get("forget_weight"),
                "used_reference_model": unl.get("used_reference_model"),
                "diverged": unl.get("diverged"),
                "final_forget_nll": unl.get("final_forget_nll"),
                "final_retain_loss": hist[-1].get("retain_loss") if hist else None,
                "train_seconds": (run_metrics.get("train") or {}).get("duration_seconds"),
                "train_final_loss": (run_metrics.get("train") or {}).get("final_loss"),
                "train_seed": (run_metrics.get("seeds") or {}).get("train_seed"),
                "unlearn_seed": (run_metrics.get("seeds") or {}).get("unlearn_seed"),
            }
            # Method telemetry that used to be computed and then dropped here.
            # divergence_threshold in particular: without it, `diverged: False`
            # cannot be distinguished from "the check did not run", which is exactly
            # the ambiguity older runs exhibit.
            row.update({
                "divergence_threshold": unl.get("divergence_threshold"),
                "optimizer_steps": unl.get("optimizer_steps"),
                "unlearn_peak_reserved_gib": (res.get("unlearn") or {}).get(
                    "peak_reserved_gib"),
                "beta": unl.get("beta"),
                "sequence_reduction": unl.get("sequence_reduction"),
                "num_forget_examples": unl.get("num_forget_examples"),
                "num_retain_examples": unl.get("num_retain_examples"),
                "utility_seconds": _utility_seconds(run_metrics, method),
                "config_hash": run_metrics.get("config_hash"),
                "num_parameters": (run_metrics.get("model_info") or {}).get(
                    "num_parameters"),
                "train_optimizer_steps": (run_metrics.get("train") or {}).get(
                    "optimizer_steps"),
            })

            # in/out separation (written by run_single.py; recomputed below for runs
            # that predate it, since it is cheap and the inputs are all here)
            sf = mdir / "separation.json"
            sep = {}
            if sf.exists():
                try:
                    sep = load_json(sf)
                except Exception:
                    sep = {}
            if not sep:
                ins = [float(r["loss"]) for r in payload["records"]
                       if sign_map.get(r.get("batch_id") or owner.get(
                           (r["candidate_author"], int(r["qa_id"])))) == 1]
                outs = [float(r["loss"]) for r in payload["records"]
                        if sign_map.get(r.get("batch_id") or owner.get(
                            (r["candidate_author"], int(r["qa_id"])))) == -1]
                if ins and outs:
                    import statistics as _st

                    mi, mo = _st.fmean(ins), _st.fmean(outs)
                    sd = ((_st.variance(ins) + _st.variance(outs)) / 2) ** 0.5
                    sep = {"mean_in": mi, "mean_out": mo, "gap": mo - mi,
                           "cohens_d": (mo - mi) / sd if sd > 0 else None,
                           "n_in": len(ins), "n_out": len(outs)}
            for k in ("mean_in", "mean_out", "gap", "cohens_d", "n_in", "n_out"):
                row[f"sep_{k}"] = sep.get(k)

            # utility, if the suite ran
            uf = mdir / "utility.json"
            if uf.exists():
                try:
                    u = load_json(uf)
                    row["model_utility"] = u.get("model_utility")
                    row["forget_quality"] = u.get("forget_quality")
                    row["ks_statistic"] = u.get("ks_statistic")
                    for grp, r in (u.get("splits") or {}).items():
                        if isinstance(r, dict):
                            row[f"util_{grp}_n"] = r.get("n")
                            row[f"util_{grp}_probability"] = r.get("probability")
                            row[f"util_{grp}_truth_ratio"] = r.get("truth_ratio")
                            row[f"util_{grp}_rouge_l_recall"] = r.get("rouge_l_recall")
                            # Per-row truth ratios go to truth_ratios.csv, not here.
                            for pr in (r.get("per_row") or []):
                                tr_rows.append({
                                    "run_id": rid, "family": family,
                                    "method": method, "group": grp,
                                    "index": pr.get("index"),
                                    "author_id": pr.get("author_id"),
                                    "qa_id": pr.get("qa_id"),
                                    "sign": sign_map.get(
                                        owner.get((str(pr.get("author_id")),
                                                   int(pr["qa_id"])))
                                    ) if pr.get("qa_id") is not None else None,
                                    "probability": pr.get("probability"),
                                    "truth_ratio": pr.get("truth_ratio"),
                                    "rouge_l_recall": pr.get("rouge_l_recall"),
                                })
                except Exception as exc:
                    print(f"[collect] {rid}/{method}: unreadable utility.json ({exc})")
            run_rows.append(row)

    if not loss_rows:
        raise SystemExit(f"no losses found under {root}/runs — nothing to collect")

    # --- write ---------------------------------------------------------------
    loss_fields = ["run_id", "family", "method", "batch_id", "candidate_author",
                   "qa_id", "sign", "loss", "num_answer_tokens"]
    _write_csv(out / "losses.csv", loss_rows, loss_fields)

    run_fields: List[str] = []
    for r in run_rows:
        for k in r:
            if k not in run_fields:
                run_fields.append(k)
    _write_csv(out / "runs.csv", run_rows, run_fields)

    flat = [
        {"run_id": rid, "family": fam, "batch_id": b, "sign": s,
         "split_hash": split_hash}
        for rid, (fam, sm) in signs.items() for b, s in sm.items()
    ]
    _write_csv(out / "manifest_flat.csv", flat,
               ["run_id", "family", "batch_id", "sign", "split_hash"])

    # Per-row utility detail. Separate from runs.csv because it is a different grain
    # (one row per scored QA pair per group), and it is what makes forget quality
    # recomputable on any slice without re-scoring a model.
    if tr_rows:
        _write_csv(out / "truth_ratios.csv", tr_rows,
                   ["run_id", "family", "method", "group", "index", "author_id",
                    "qa_id", "sign", "probability", "truth_ratio", "rouge_l_recall"])

    parquet = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(loss_rows), out / "losses.parquet")
        parquet = str(out / "losses.parquet")
    except ImportError:
        pass

    n_calib = sum(1 for r in seen_runs if signs[r][0] == "calibration")
    n_eval = len(seen_runs) - n_calib
    summary = {
        "split_hash": split_hash,
        "model": cfg["model"]["id"],
        "output_root": str(root),
        "m": man["split"]["m"],
        "batching": man["split"]["batching"],
        "B_qa_per_batch": batch_size_of(man),
        "methods": sorted(seen_methods),
        "runs_with_results": len(seen_runs),
        "calibration_runs": n_calib,
        "evaluation_runs": n_eval,
        "calibration_expected": len(man["sign_vectors"]["calibration"]["run_ids"]),
        "evaluation_expected": len(man["sign_vectors"]["evaluation"]["run_ids"]),
        "loss_rows": len(loss_rows),
        "run_method_rows": len(run_rows),
        "truth_ratio_rows": len(tr_rows),
        "files": {
            "losses_csv": str(out / "losses.csv"),
            "losses_parquet": parquet,
            "runs_csv": str(out / "runs.csv"),
            "manifest_flat_csv": str(out / "manifest_flat.csv"),
            "truth_ratios_csv": str(out / "truth_ratios.csv") if tr_rows else None,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"collected -> {out}")
    print(f"  losses.csv        {len(loss_rows):>8,} rows")
    if parquet:
        print(f"  losses.parquet    {len(loss_rows):>8,} rows")
    else:
        print("  losses.parquet    (skipped: pip install pyarrow)")
    print(f"  runs.csv          {len(run_rows):>8,} rows")
    print(f"  manifest_flat.csv {len(flat):>8,} rows")
    if tr_rows:
        print(f"  truth_ratios.csv  {len(tr_rows):>8,} rows")
    print(f"\n  m={summary['m']} (B={summary['B_qa_per_batch']}), methods={summary['methods']}")
    print(f"  calibration {n_calib}/{summary['calibration_expected']}   "
          f"evaluation {n_eval}/{summary['evaluation_expected']}")
    if n_eval < summary["evaluation_expected"]:
        print("\n  NOTE: the epsilon bound needs the evaluation family; it is incomplete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
