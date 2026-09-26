#!/usr/bin/env python3
"""TOFU forget quality, computed post-hoc from stored per-row truth ratios.

    python scripts/forget_quality.py --config configs/forgetq5_pinned180.yaml

Runs entirely on CPU from JSON already on disk: every run's ``utility.json`` plus the
reference model's ``truth_ratios.json``. No GPU, no model loading. That is deliberate
-- it means a botched or missing reference is a 20-minute fix rather than a re-run of
the whole experiment, and new KS variants can be added without re-scoring anything.

Three slices per metric group, and the second is the one that makes the first
readable:

``plus``     the run's ``S_j = +1`` candidate authors -- trained, then unlearned.
             THE headline. Compared against the same rows from the reference, which
             never trained on them.
``minus``    the run's ``S_j = -1`` candidates -- never trained by EITHER model.
             A negative control, and necessary rather than decorative: the reference
             differs from each run's model by training seed and by ~63 optimizer
             steps (3600 vs 3800 examples at fixed epochs), both of which move truth
             ratios for reasons unrelated to unlearning. ``minus`` measures that
             nuisance floor so ``plus`` can be read relative to it.

             How to read it: under the null (no systematic difference) the p-value is
             Uniform(0,1), so a SINGLE minus p-value near 0.1 means nothing. Judge it
             across runs and methods -- the mean should sit near 0.5 and the values
             should look uniform. A mean well below 0.5, or a consistently large
             ``ks_statistic``, says the reference is not exchangeable with the runs on
             data neither model ever saw, and then ``plus`` cannot be read at face
             value. Expecting "p near 1" would be wrong.
``all``      every row in the group. For the ``forget`` group this is the
             conventional TOFU number, reported for comparability -- but it mixes the
             two conditions above and so is biased toward "good forgetting": half the
             rows match the reference by construction.

For groups whose rows are not candidate authors (``retain``, ``real_authors``,
``world_facts``) only ``all`` is defined. Those KS tests are still informative --
they measure collateral damage and general capability drift against the same
reference -- so they are computed too.

Both samples are always the SAME rows on both sides. Comparing a run's 200 ``+1``
rows against all 400 reference rows would partly measure "these ten authors have
longer answers than those ten" rather than "this model forgot".

Outputs, under ``<output_root>/forget_quality/``::

    forget_quality.csv    one row per (run_id, method, group, slice)
    forget_quality.json   the same, plus provenance and the reference summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.manifest import load_manifest, sign_vector_for_run
from audit_tofu.run_manager import load_json, resolve_run_paths
from audit_tofu.utility import forget_quality, forget_quality_slices, ks_with_stats

REFERENCE_DIRNAME = "reference"

#: Slices that only make sense for groups made of candidate authors.
SIGN_SLICES = ("plus", "minus")


def _rows_by_identity(per_row: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], float]:
    """``{(author_id, qa_id) -> truth_ratio}`` for rows that have both."""
    out: Dict[Tuple[str, int], float] = {}
    for r in per_row:
        a, q, tr = r.get("author_id"), r.get("qa_id"), r.get("truth_ratio")
        if a is None or q is None or tr is None:
            continue
        if not np.isfinite(float(tr)):
            continue
        out[(str(a), int(q))] = float(tr)
    return out


def _qa_sign_map(manifest: Dict[str, Any], run_id: str) -> Dict[Tuple[str, int], int]:
    """``{(author_id, qa_id) -> +1/-1}`` for every candidate QA pair.

    Resolved at QA granularity rather than per author, because that is the grain the
    sign vector actually operates on. At ``B = qa_per_author`` ("author" batching) an
    author owns one batch and all 20 of its pairs share a sign, so this reduces to a
    per-author map. But at ``B < qa_per_author`` -- ``batching: 5``, ``10``, or
    ``qa`` -- an author owns several batches that are signed independently, so the
    same author can have some pairs trained-then-unlearned and others never trained.
    An author-level map would have to drop every such author as "mixed", which at
    ``batching: qa`` is essentially all of them, silently emptying the plus/minus
    slices. Keying by QA pair is exact at every ``B``.
    """
    signs = sign_vector_for_run(manifest, run_id)
    batches = manifest["split"]["batches"]
    out: Dict[Tuple[str, int], int] = {}
    for bid, sj in zip(manifest["split"]["batch_ids"], signs):
        for author, qa in batches[bid]:
            out[(str(author), int(qa))] = int(sj)
    return out




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--reference", default=None,
                    help="path to reference truth_ratios.json; defaults to "
                         "<output_root>/reference/truth_ratios.json")
    ap.add_argument("--out_dir", default=None,
                    help="defaults to <output_root>/forget_quality")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    manifest = load_manifest(cfg["experiment"]["manifest_path"])
    root = Path(cfg["experiment"]["output_root"])
    out = Path(args.out_dir or (root / "forget_quality"))
    out.mkdir(parents=True, exist_ok=True)

    # Same resolution order as train_reference.py, so a shared reference is found
    # without having to pass --reference for every config.
    ref_dir = Path(cfg["utility"].get("reference_dir") or (root / REFERENCE_DIRNAME))
    ref_path = Path(args.reference) if args.reference else ref_dir / "truth_ratios.json"
    if not ref_path.exists():
        raise SystemExit(
            f"no reference truth ratios at {ref_path}\n"
            f"  build one first:  python scripts/train_reference.py --config "
            f"{args.config} --gpu <id>\n"
            f"  (utility.reference_dir = {cfg['utility'].get('reference_dir')})"
        )
    ref = load_json(ref_path)

    # Validate on the RETAIN SET, not on split_hash. The reference model is a function
    # of D_r alone, so it is legitimately shared across configs that differ only in
    # candidate batching B (all four audit configs pin the same 180 retain authors but
    # have different split_hashes because m differs). What must match exactly is the
    # data the reference trained on.
    ref_retain = ref.get("retain_authors")
    man_retain = manifest["split"]["retain_authors"]
    if ref_retain is None:
        print("[fq] WARNING: reference predates retain_authors recording; falling back "
              "to a split_hash check")
        if ref.get("split_hash") != manifest["split_hash"]:
            raise SystemExit(
                f"reference split_hash {str(ref.get('split_hash'))[:12]}... != "
                f"manifest {manifest['split_hash'][:12]}..., and the reference does "
                "not record its retain authors. Retrain it with "
                "scripts/train_reference.py."
            )
    elif set(ref_retain) != set(man_retain):
        only_ref = sorted(set(ref_retain) - set(man_retain))[:5]
        only_man = sorted(set(man_retain) - set(ref_retain))[:5]
        raise SystemExit(
            f"reference retain set does not match this manifest's D_r "
            f"({len(ref_retain)} vs {len(man_retain)} authors).\n"
            f"  only in reference: {only_ref}\n"
            f"  only in manifest : {only_man}\n"
            "The reference must be trained on exactly this D_r, or it is not a valid "
            "'never saw the forget set' baseline."
        )
    elif ref.get("split_hash") != manifest["split_hash"]:
        print(f"[fq] note: reference split_hash {str(ref.get('split_hash'))[:12]}... "
              f"!= manifest {manifest['split_hash'][:12]}..., but D_r is identical "
              "({} authors) -- reusing it is correct (batching does not affect the "
              "reference model)".format(len(man_retain)))

    # Raw per-row records per group. forget_quality_slices() does the identity join
    # and the sign slicing itself, so nothing needs pre-flattening here.
    ref_per_row = {
        g: (v.get("per_row") or []) for g, v in (ref.get("groups") or {}).items()
    }
    ref_groups = {g: _rows_by_identity(rows) for g, rows in ref_per_row.items()}
    ref_flat = {
        g: [float(r["truth_ratio"]) for r in rows
            if r.get("truth_ratio") is not None
            and np.isfinite(float(r["truth_ratio"]))]
        for g, rows in ref_per_row.items()
    }

    cand_authors = set(manifest["split"]["candidate_authors"])
    print(f"[fq] reference : {ref_path}")
    print(f"[fq] split_hash: {manifest['split_hash'][:16]}...  "
          f"selection={manifest['split'].get('candidate_selection') or 'random'}")
    for g, rows in ref_groups.items():
        n_cand = sum(1 for (a, _q) in rows if a in cand_authors)
        print(f"[fq]   reference/{g:<13} {len(ref_flat[g]):>4} usable rows, "
              f"{n_cand:>4} on candidate authors")

    rows_out: List[Dict[str, Any]] = []
    families = ("calibration", "evaluation")

    for fam in families:
        for run_id in manifest["sign_vectors"][fam]["run_ids"]:
            paths = resolve_run_paths(root, run_id)
            if not paths.methods_dir.exists():
                continue
            sign_of = _qa_sign_map(manifest, run_id)

            for mdir in sorted(paths.methods_dir.iterdir()):
                if not mdir.is_dir():
                    continue
                uf = mdir / "utility.json"
                if not uf.exists():
                    continue
                method = mdir.name
                try:
                    util = load_json(uf)
                except Exception as exc:
                    print(f"[fq] {run_id}/{method}: unreadable utility.json ({exc})")
                    continue

                for group, res in (util.get("splits") or {}).items():
                    if not isinstance(res, dict) or "error" in res:
                        continue
                    per_row = res.get("per_row")
                    if per_row is None:
                        # Produced before per-row identity existed. The `all` slice is
                        # still computable from the flat list; sign slices are not.
                        flat = [t for t in (res.get("truth_ratios") or [])
                                if np.isfinite(float(t))]
                        if flat and ref_flat.get(group):
                            rec = ks_with_stats(flat, ref_flat[group])
                            rows_out.append({
                                "run_id": run_id, "family": fam, "method": method,
                                "group": group, "slice": "all",
                                "n_authors": None, "legacy_no_identity": True,
                                **rec,
                            })
                        continue

                    # ONE implementation of the slicing, shared with the inline path
                    # in scripts/run_single.py, so the two can never disagree.
                    slices = forget_quality_slices(
                        per_row, ref_per_row.get(group) or [], sign_of
                    )
                    for slice_name, rec in slices.items():
                        rows_out.append({
                            "run_id": run_id, "family": fam, "method": method,
                            "group": group, "slice": slice_name,
                            "legacy_no_identity": False,
                            **rec,
                        })

    if not rows_out:
        raise SystemExit(
            f"no utility.json files found under {root}/runs -- run the experiment "
            "first, or check that utility.enabled was true"
        )

    fields = ["run_id", "family", "method", "group", "slice", "n_authors",
              "n_model", "n_reference", "forget_quality", "ks_statistic",
              "mean_tr_model", "mean_tr_reference",
              "median_tr_model", "median_tr_reference", "legacy_no_identity", "note"]
    with open(out / "forget_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    payload = {
        "split_hash": manifest["split_hash"],
        "candidate_selection": manifest["split"].get("candidate_selection"),
        "candidate_authors": sorted(cand_authors),
        "reference": {
            "path": str(ref_path),
            "seed": ref.get("seed"),
            "compute_rouge": ref.get("compute_rouge"),
            "groups": {g: {"n": v.get("n"), "n_identified": v.get("n_identified"),
                           "truth_ratio": v.get("truth_ratio"),
                           "probability": v.get("probability"),
                           "rouge_l_recall": v.get("rouge_l_recall")}
                       for g, v in (ref.get("groups") or {}).items()},
        },
        "slices": {
            "plus": "S_j=+1 candidates: trained then unlearned. The headline.",
            "minus": "S_j=-1 candidates: never trained by either model. Negative "
                     "control. Under the null the p-value is Uniform(0,1), so judge "
                     "the MEAN over runs/methods (~0.5 = reference exchangeable with "
                     "the runs), never a single value.",
            "all": "Every row. For `forget` this is the conventional TOFU number, "
                   "but it mixes plus and minus and is biased toward high p.",
        },
        "rows": rows_out,
    }
    (out / "forget_quality.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- console summary: the forget group, evaluation runs ---------------------
    print(f"\nforget_quality -> {out}  ({len(rows_out)} rows)")
    sel = [r for r in rows_out
           if r["group"] == "forget" and r["family"] == "evaluation"]
    if sel:
        methods = sorted({r["method"] for r in sel})
        print("\nforget group, evaluation runs (mean over runs):")
        print(f"  {'method':<13}{'p(+1)':>9}{'KS(+1)':>9}{'p(-1)':>9}"
              f"{'KS(-1)':>9}{'p(all)':>9}{'n_runs':>8}")
        for m in methods:
            cells = []
            for sl in ("plus", "minus", "all"):
                vals = [r for r in sel if r["method"] == m and r["slice"] == sl]
                p = [v["forget_quality"] for v in vals
                     if v["forget_quality"] is not None]
                k = [v["ks_statistic"] for v in vals
                     if v["ks_statistic"] is not None]
                cells.append((float(np.mean(p)) if p else None,
                              float(np.mean(k)) if k else None))
            n_runs = len({r["run_id"] for r in sel if r["method"] == m
                          and r["slice"] == "plus"})

            def f(x: Optional[float]) -> str:
                return f"{x:>9.4f}" if x is not None else f"{'-':>9}"

            print(f"  {m:<13}{f(cells[0][0])}{f(cells[0][1])}{f(cells[1][0])}"
                  f"{f(cells[1][1])}{f(cells[2][0])}{n_runs:>8}")
        mp = [r["forget_quality"] for r in sel
              if r["slice"] == "minus" and r["forget_quality"] is not None]
        print("\n  p(+1) high = the forget set looks like a model that never trained "
              "on it (good forgetting).")
        print("  p(-1) is the NEGATIVE CONTROL on rows neither model ever saw. Under "
              "the null it is")
        print("  Uniform(0,1), so judge the mean over runs/methods, not a single "
              "value: ~0.5 means the")
        print("  reference is exchangeable with the runs and p(+1) is readable. Well "
              "below 0.5 means a")
        print("  systematic seed/step-count difference is contaminating p(+1).")
        if mp:
            mean_minus = float(np.mean(mp))
            # Below ~5 tests the mean of a Uniform(0,1) sample is too noisy to call,
            # and a spurious "SUSPECT" would be worse than no verdict at all.
            if len(mp) < 5:
                verdict = f"too few tests ({len(mp)}) to judge"
            elif mean_minus > 0.3:
                verdict = "looks exchangeable"
            else:
                verdict = "SUSPECT -- see above"
            print(f"    observed mean p(-1) = {mean_minus:.3f} over {len(mp)} "
                  f"tests: {verdict}")
        print("  p(all) mixes +1 and -1 rows, so it is biased high; kept only for "
              "comparability with")
        print("  the conventional TOFU number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
