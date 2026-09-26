#!/usr/bin/env python3
"""Build the immutable audit-split manifest. Run ONCE, before any experiment.

    python scripts/build_manifest.py --config configs/base.yaml

Samples the author partition and every calibration/evaluation sign vector up front,
then hashes the result. Every later stage reads this file and verifies the hash, so
the split cannot drift underneath the results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.manifest import (
    build_manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override key.path=value")
    ap.add_argument("--out", default=None, help="defaults to experiment.manifest_path")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing manifest (INVALIDATES every existing run)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out = Path(args.out or cfg["experiment"]["manifest_path"])

    if out.exists() and not args.overwrite:
        manifest = load_manifest(out)
        print(f"[build_manifest] manifest already exists: {out}")
        print(f"[build_manifest] split_hash = {manifest['split_hash']}")
        print("[build_manifest] validated OK; pass --overwrite to regenerate")
        return 0

    ds = cfg["dataset"]
    split = cfg["split"]
    seeds = cfg["seeds"]

    # Fingerprint the real dataset so upstream drift is detectable. Skipped for the
    # synthetic smoke dataset, which is generated deterministically in-process.
    fingerprint = None
    if ds["name"] != "__synthetic__":
        from audit_tofu.tofu_data import dataset_fingerprint, load_examples

        print(f"[build_manifest] loading {ds['name']}/{ds['config']} to fingerprint...")
        examples = load_examples(ds)
        fingerprint = dataset_fingerprint(examples)
        n_authors = len({e.author_id for e in examples})
        print(f"[build_manifest] {len(examples)} examples, {n_authors} authors")
        print(f"[build_manifest] fingerprint = {fingerprint[:16]}...")
        need = split["num_candidate_authors"] + split["num_retain_authors"]
        if n_authors < need:
            raise SystemExit(
                f"dataset has {n_authors} authors but the split needs {need}"
            )

    manifest = build_manifest(
        dataset_name=ds["name"],
        dataset_config=ds["config"],
        dataset_revision=ds.get("revision"),
        dataset_fingerprint=fingerprint,
        candidate_author_ids=split.get("candidate_author_ids"),
        num_candidate_authors=int(split["num_candidate_authors"]),
        num_retain_authors=int(split["num_retain_authors"]),
        batching=split["batching"],
        num_calibration_runs=int(split["num_calibration_runs"]),
        num_evaluation_runs=int(split["num_evaluation_runs"]),
        calibration_balance=split.get("calibration_balance", "iid"),
        split_seed=int(seeds["split_seed"]),
        data_order_seed=int(seeds["data_order_seed"]),
        train_seed_base=int(seeds["train_seed_base"]),
        unlearn_seed_base=int(seeds["unlearn_seed_base"]),
        qa_per_author=int(ds["qa_per_author"]),
    )

    validate_manifest(manifest)
    save_manifest(manifest, out, overwrite=args.overwrite)

    s = manifest["split"]
    print(f"\n[build_manifest] wrote {out}")
    print(f"  split_hash           : {manifest['split_hash']}")
    print(f"  m (candidate batches): {s['m']}  (batching={s['batching']})")
    sel = s.get("candidate_selection") or "random (from split_seed)"
    print(f"  candidate selection  : {sel}")
    print(f"  candidate authors    : {len(s['candidate_authors'])}"
          f"  [{s['candidate_authors'][0]}..{s['candidate_authors'][-1]}]")
    print(f"  retain authors       : {len(s['retain_authors'])}"
          f"  [{s['retain_authors'][0]}..{s['retain_authors'][-1]}]")
    print(f"  calibration runs     : {len(manifest['sign_vectors']['calibration']['run_ids'])}")
    print(f"  evaluation runs      : {len(manifest['sign_vectors']['evaluation']['run_ids'])}")
    print(f"  calibration_balance  : {manifest['sign_vectors']['calibration_balance']}")
    cov = manifest["calibration_coverage"]
    print(f"  calib obs / batch    : n_in [{cov['min_n_in']},{cov['max_n_in']}]  "
          f"n_out [{cov['min_n_out']},{cov['max_n_out']}]")
    if cov["min_n_in"] < 6 or cov["min_n_out"] < 6:
        print("    NOTE: some candidate batches have few observations in one "
              "condition.\n"
              "          Keep attack.pool_variance='author', or rebuild with "
              "split.calibration_balance=stratified.")
    print("  calibration/evaluation sign vectors verified disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
