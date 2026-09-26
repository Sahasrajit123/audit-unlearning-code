#!/usr/bin/env python3
"""Compute missing evaluations on ALREADY-TRAINED runs, from saved checkpoints.

    # what is missing, and which checkpoints survive? (no GPU)
    python scripts/backfill_eval.py --config configs/base.yaml --dry_run

    # add ROUGE to every evaluation run, reusing the saved unlearned models
    python scripts/backfill_eval.py --config configs/base.yaml --what utility \
        --gpu 3 --set utility.compute_rouge=true

    # prove a saved checkpoint is the one that produced the stored losses
    python scripts/backfill_eval.py --config configs/base.yaml --what verify --gpu 3

Why this exists. Adding a metric used to mean re-running the whole audit, because
``retention: delete_after_scoring`` threw the weights away -- which is exactly how
we ended up unable to compute forget quality on 30 finished runs. With
``retention: keep_all`` and ``save_unlearned_checkpoint: true`` the post-unlearning
model for every (run, method) is on disk, so a new metric costs a forward pass
instead of a fine-tune. This script is the thing that spends that forward pass.

It never re-trains and never re-unlearns. If a checkpoint is missing it says so and
skips, rather than silently reconstructing a *different* model.

Modes (``--what``, repeatable):

``utility``  Run the TOFU utility suite and MERGE the result into the existing
             ``utility.json``. Merge rather than replace, so a backfill that only
             adds ROUGE cannot drop fields an older run recorded. Also fills in
             per-row truth-ratio identity for runs that predate it, which is what
             ``scripts/forget_quality.py`` needs for the +1/-1 slicing.
``losses``   Re-score the candidate pool and rewrite ``losses.json``. Use only if
             the scoring config genuinely changed; prefer ``verify`` first.
``verify``   Re-score and COMPARE against the stored losses without writing. The
             cheap integrity check: a saved checkpoint that reproduces the stored
             losses to ~1e-5 is provably the model that produced them.

Which model is loaded: ``runs/<id>/methods/<m>/model`` -- the post-unlearning
checkpoint, i.e. the exact model the audit scored. ``trained/`` is NOT a substitute;
it is the pre-unlearning model and would give every method identical numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.manifest import load_manifest
from audit_tofu.run_manager import (
    ResourceTracker,
    load_json,
    resolve_run_paths,
    save_json,
    setup_run_logger,
)

WHAT_CHOICES = ("utility", "losses", "verify")

#: Absolute loss difference above which a re-score is treated as a mismatch rather
#: than bf16/kernel nondeterminism. Scoring is FP32 over a fixed batch order, so
#: agreement is normally far tighter than this.
LOSS_TOLERANCE = 1e-4


def _row_count_conflicts(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Groups whose row count changed between the stored and the fresh suite.

    A changed ``n`` means the two suites scored DIFFERENT row sets -- almost always
    because ``utility.limit`` was set for the backfill, sometimes because the dataset
    revision moved. Merging across different row sets would overwrite the stored
    400-row ``per_row`` with a truncated one and quietly break the +1/-1 slicing in
    scripts/forget_quality.py, so this is a hard stop rather than a warning.
    """
    bad: List[str] = []
    for group, o in (old.get("splits") or {}).items():
        n = (new.get("splits") or {}).get(group)
        if not isinstance(o, dict) or not isinstance(n, dict):
            continue
        if o.get("n") is not None and n.get("n") is not None and o["n"] != n["n"]:
            bad.append(f"{group}: stored n={o['n']} vs fresh n={n['n']}")
    return bad


def _merge_utility(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a freshly computed suite into an existing one, preferring new values.

    Per-split dicts are merged key-by-key so that a partial backfill (say, ROUGE
    only) keeps whatever the original run recorded. A new value of ``None`` never
    overwrites a non-None old value -- that would let a suite run with
    ``compute_rouge=false`` erase a ROUGE number a previous backfill computed.
    """
    out = dict(old)
    for k, v in new.items():
        if k == "splits":
            continue
        if v is not None or k not in out:
            out[k] = v

    old_splits = dict(old.get("splits") or {})
    new_splits = dict(new.get("splits") or {})
    merged: Dict[str, Any] = {}
    for group in set(old_splits) | set(new_splits):
        o, n = old_splits.get(group), new_splits.get(group)
        if not isinstance(o, dict) or not isinstance(n, dict):
            merged[group] = n if n is not None else o
            continue
        g = dict(o)
        for k, v in n.items():
            if v is not None or k not in g:
                g[k] = v
        merged[group] = g
    out["splits"] = merged
    return out


def _find_targets(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
    families: Tuple[str, ...],
    only_runs: Optional[List[str]],
    only_methods: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Inventory every (run, method) with results, and whether its model survives."""
    root = Path(cfg["experiment"]["output_root"])
    out: List[Dict[str, Any]] = []
    for fam in families:
        for run_id in manifest["sign_vectors"][fam]["run_ids"]:
            if only_runs and run_id not in only_runs:
                continue
            paths = resolve_run_paths(root, run_id)
            if not paths.methods_dir.exists():
                continue
            for mdir in sorted(paths.methods_dir.iterdir()):
                if not mdir.is_dir():
                    continue
                method = mdir.name
                if only_methods and method not in only_methods:
                    continue
                if not (mdir / "losses.json").exists():
                    continue
                ckpt = mdir / "model"
                uf = mdir / "utility.json"
                util = None
                if uf.exists():
                    try:
                        util = load_json(uf)
                    except Exception:
                        util = None
                splits = (util or {}).get("splits") or {}
                has_rouge = any(
                    isinstance(v, dict) and v.get("rouge_l_recall") is not None
                    for v in splits.values()
                )
                has_perrow = any(
                    isinstance(v, dict) and v.get("per_row") is not None
                    for v in splits.values()
                )
                out.append({
                    "run_id": run_id,
                    "family": fam,
                    "method": method,
                    "mdir": mdir,
                    "ckpt": ckpt,
                    "has_ckpt": (ckpt / "config.json").exists(),
                    "has_utility": util is not None,
                    "has_rouge": has_rouge,
                    "has_per_row": has_perrow,
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--what", nargs="*", default=["utility"], choices=WHAT_CHOICES)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--runs", nargs="*", default=None, help="restrict to these run ids")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=["evaluation"],
                    choices=["calibration", "evaluation"],
                    help="default evaluation only, matching utility.calibration_runs")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the metric is already present")
    ap.add_argument("--dry_run", action="store_true",
                    help="report what is missing and skip all GPU work")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    manifest = load_manifest(cfg["experiment"]["manifest_path"])
    root = Path(cfg["experiment"]["output_root"])

    targets = _find_targets(cfg, manifest, tuple(args.families),
                            args.runs, args.methods)
    if not targets:
        raise SystemExit(
            f"no (run, method) results found under {root}/runs for families "
            f"{args.families}. Has the audit been run?"
        )

    n_ck = sum(1 for t in targets if t["has_ckpt"])
    want_rouge = bool(cfg["utility"]["compute_rouge"])

    print(f"[backfill] config      {args.config}")
    print(f"[backfill] output_root {root}")
    print(f"[backfill] split_hash  {manifest['split_hash'][:16]}...")
    print(f"[backfill] what        {args.what}   families={args.families}")
    print(f"[backfill] targets     {len(targets)} (run, method) pairs; "
          f"{n_ck} have a saved model, {len(targets) - n_ck} do not")
    print(f"[backfill] compute_rouge={want_rouge}  "
          f"(from utility.compute_rouge; override with --set)")
    print()

    # ---- inventory --------------------------------------------------------------
    missing_rouge = [t for t in targets if not t["has_rouge"]]
    missing_perrow = [t for t in targets if not t["has_per_row"]]
    missing_util = [t for t in targets if not t["has_utility"]]
    print(f"  without utility.json      : {len(missing_util)}")
    print(f"  without per-row identity  : {len(missing_perrow)}  "
          f"(needed by forget_quality.py for +1/-1 slicing)")
    print(f"  without ROUGE             : {len(missing_rouge)}")
    no_ck = [t for t in targets if not t["has_ckpt"]]
    if no_ck:
        print()
        print(f"  {len(no_ck)} pair(s) have NO saved model and cannot be backfilled:")
        for t in no_ck[:6]:
            print(f"    {t['run_id']}/{t['method']}  (expected {t['ckpt']})")
        if len(no_ck) > 6:
            print(f"    ... and {len(no_ck) - 6} more")
        print("  Those runs were made under retention=delete_after_scoring, or before")
        print("  storage.save_unlearned_checkpoint existed. Only a re-run recovers them.")

    if args.dry_run:
        print()
        print("[backfill] --dry_run: nothing computed.")
        return 0

    todo = [t for t in targets if t["has_ckpt"]]
    if not todo:
        raise SystemExit(
            "every target is missing its saved model; nothing can be backfilled. "
            "Re-run with storage.retention=keep_all and "
            "storage.save_unlearned_checkpoint=true."
        )

    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import torch

    from audit_tofu.modeling import load_model_and_tokenizer
    from audit_tofu.scoring import losses_to_records, score_examples
    from audit_tofu.tofu_data import candidate_examples, load_examples
    from audit_tofu.utility import run_utility_suite

    logger = setup_run_logger(root / "backfill", "backfill")
    log = logger.info
    log("=" * 78)
    log(f"BACKFILL  what={args.what}  targets={len(todo)}")
    log("=" * 78)

    n_authors = (int(cfg["split"]["num_candidate_authors"])
                 + int(cfg["split"]["num_retain_authors"]))
    examples = load_examples(cfg["dataset"], num_authors=n_authors)
    score_ex = candidate_examples(manifest, examples)

    tracker = ResourceTracker()
    summary: List[Dict[str, Any]] = []

    for i, t in enumerate(todo, 1):
        run_id, method, mdir = t["run_id"], t["method"], t["mdir"]
        rec: Dict[str, Any] = {"run_id": run_id, "method": method, "actions": []}

        need_util = "utility" in args.what and (
            args.force or not t["has_utility"]
            or (want_rouge and not t["has_rouge"])
            or not t["has_per_row"]
        )
        need_losses = "losses" in args.what
        need_verify = "verify" in args.what
        if not (need_util or need_losses or need_verify):
            rec["actions"].append("skipped (nothing missing; use --force)")
            summary.append(rec)
            continue

        log("-" * 78)
        log(f"[{i}/{len(todo)}] {run_id}/{method}  <- {t['ckpt']}")
        tracker.start(f"load:{run_id}:{method}")
        model, tokenizer, _ = load_model_and_tokenizer(
            str(t["ckpt"]),
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            gradient_checkpointing=False,     # inference only
            trust_remote_code=cfg["model"]["trust_remote_code"],
            for_training=False,
        )
        if torch.cuda.is_available():
            model.to("cuda")
        lres = tracker.stop()
        log(f"  loaded in {lres['duration_seconds']:.1f}s")

        try:
            # ---- verify / losses -------------------------------------------------
            if need_verify or need_losses:
                tracker.start(f"score:{run_id}:{method}")
                rows = score_examples(
                    model, tokenizer, score_ex,
                    max_length=int(cfg["scoring"]["max_seq_length"]),
                    append_eos=bool(cfg["scoring"]["append_eos"]),
                    system_prompt=cfg["training"].get("system_prompt"),
                    batch_size=int(cfg["scoring"]["batch_size"]),
                )
                records = losses_to_records(rows, run_id, method, manifest)
                sres = tracker.stop()

                if need_verify:
                    stored = load_json(mdir / "losses.json")["records"]
                    a = {(r["candidate_author"], int(r["qa_id"])): float(r["loss"])
                         for r in stored}
                    b = {(r["candidate_author"], int(r["qa_id"])): float(r["loss"])
                         for r in records}
                    shared = sorted(set(a) & set(b))
                    if not shared:
                        rec["verify"] = {"status": "no overlap"}
                    else:
                        d = np.array([abs(a[k] - b[k]) for k in shared])
                        ok = bool(d.max() <= LOSS_TOLERANCE)
                        rec["verify"] = {
                            "n": len(shared),
                            "max_abs_diff": float(d.max()),
                            "mean_abs_diff": float(d.mean()),
                            "within_tolerance": ok,
                            "tolerance": LOSS_TOLERANCE,
                        }
                        log(f"  verify: n={len(shared)} max|d|={d.max():.2e} "
                            f"{'OK' if ok else 'MISMATCH'}")
                        if not ok:
                            log("  MISMATCH: the saved checkpoint does not reproduce "
                                "the stored losses. Do not backfill metrics from it "
                                "until this is explained.")
                    rec["actions"].append("verify")

                if need_losses:
                    save_json(mdir / "losses.json", {
                        "run_id": run_id, "method": method,
                        "split_hash": manifest["split_hash"],
                        "records": records,
                        "backfilled": True,
                    })
                    log(f"  rewrote losses.json ({len(records)} records, "
                        f"{sres['duration_seconds']:.1f}s)")
                    rec["actions"].append("losses")

            # ---- utility ---------------------------------------------------------
            if need_util:
                tracker.start(f"utility:{run_id}:{method}")
                util = run_utility_suite(
                    model, tokenizer,
                    dataset_name=cfg["dataset"]["name"],
                    reference_truth_ratios=None,   # forget quality is post-hoc
                    max_length=int(cfg["scoring"]["max_seq_length"]),
                    append_eos=bool(cfg["scoring"]["append_eos"]),
                    system_prompt=cfg["training"].get("system_prompt"),
                    compute_rouge=want_rouge,
                    limit=cfg["utility"].get("limit"),
                    cache_dir=cfg["dataset"].get("cache_dir"),
                    logger=logger,
                )
                ures = tracker.stop()
                uf = mdir / "utility.json"
                if uf.exists():
                    try:
                        stored_util = load_json(uf)
                    except Exception as exc:
                        log(f"  unreadable utility.json ({exc}); writing the fresh "
                            "suite")
                        stored_util = None
                    if stored_util is not None:
                        conflicts = _row_count_conflicts(stored_util, util)
                        if conflicts and not args.force:
                            raise SystemExit(
                                f"{run_id}/{method}: refusing to merge -- the fresh "
                                "suite scored a different number of rows than the "
                                "stored one:\n    "
                                + "\n    ".join(conflicts)
                                + "\n  This usually means utility.limit is set. Merging "
                                  "would overwrite the stored per-row truth ratios with "
                                  "a truncated set and break the +1/-1 slicing in "
                                  "scripts/forget_quality.py.\n"
                                  "  Drop utility.limit, or pass --force to overwrite "
                                  "deliberately."
                            )
                        if conflicts:
                            log(f"  --force: merging despite row-count change "
                                f"({'; '.join(conflicts)})")
                        util = _merge_utility(stored_util, util)
                util["backfilled"] = True
                save_json(uf, util)
                log(f"  utility done in {ures['duration_seconds']:.1f}s  "
                    f"model_utility={util.get('model_utility')}")
                rec["actions"].append("utility" + ("+rouge" if want_rouge else ""))
                rec["model_utility"] = util.get("model_utility")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary.append(rec)

    # ---- report -----------------------------------------------------------------
    out = root / "backfill"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "backfill_summary.json", {
        "config": args.config,
        "split_hash": manifest["split_hash"],
        "what": args.what,
        "families": args.families,
        "compute_rouge": want_rouge,
        "n_targets": len(targets),
        "n_processed": len(todo),
        "resources": tracker.as_dict(),
        "results": summary,
    })

    print()
    print(f"[backfill] processed {len(todo)} pair(s) -> {out}/backfill_summary.json")
    ver = [r for r in summary if "verify" in r]
    if ver:
        bad = [r for r in ver if not r["verify"].get("within_tolerance")]
        worst = max((r["verify"].get("max_abs_diff", 0.0) for r in ver), default=0.0)
        print(f"[backfill] verify: {len(ver) - len(bad)}/{len(ver)} reproduce the "
              f"stored losses (worst max|d| = {worst:.2e})")
        for r in bad:
            print(f"    MISMATCH {r['run_id']}/{r['method']}  "
                  f"max|d|={r['verify']['max_abs_diff']:.3e}")
    r = tracker.as_dict()
    print(f"[backfill] {r['total_duration_seconds'] / 60:.1f} min total, "
          f"peak {r['overall_peak_allocated_gib']:.2f} GiB")
    if "utility" in args.what:
        print("[backfill] next: python scripts/forget_quality.py --config "
              f"{args.config}")
        print("[backfill]       python scripts/collect_results.py --config "
              f"{args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
