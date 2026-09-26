#!/usr/bin/env python3
"""Tiny end-to-end smoke test: full pipeline, small LM, 4 candidate batches.

    python scripts/run_smoke.py --config configs/smoke.yaml

Exercises manifest -> train -> unlearn -> score -> attack -> epsilon in one process.
It is a PLUMBING test: with m=4 and a randomly-initialised tiny model the epsilon it
prints is meaningless. What it checks is that every stage hands the next one the
shape it expects, and that the invariants hold end to end.

Requires torch + transformers. Everything else in the test suite runs without them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.attack import CalibrationWarning, fit_calibration, overlap, predict
from audit_tofu.config import load_config
from audit_tofu.epsilon_bounds import epsilon_lb_mean
from audit_tofu.manifest import build_manifest, save_manifest, validate_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--keep", action="store_true", help="keep the smoke output dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["experiment"]["output_root"])
    if root.exists():
        shutil.rmtree(root)

    import torch

    from audit_tofu.modeling import load_model_and_tokenizer
    from audit_tofu.scoring import losses_to_records, records_to_score_dict, score_examples
    from audit_tofu.tofu_data import (
        candidate_examples,
        forget_examples_for_run,
        load_examples,
        retain_examples,
        training_examples_for_run,
    )
    from audit_tofu.train import train_model
    from audit_tofu.unlearn import apply_unlearning

    split, seeds, ds = cfg["split"], cfg["seeds"], cfg["dataset"]
    n_authors = split["num_candidate_authors"] + split["num_retain_authors"]

    print("[smoke] building manifest")
    manifest = build_manifest(
        dataset_name=ds["name"],
        dataset_config=ds["config"],
        num_candidate_authors=int(split["num_candidate_authors"]),
        num_retain_authors=int(split["num_retain_authors"]),
        batching=split["batching"],
        num_calibration_runs=int(split["num_calibration_runs"]),
        num_evaluation_runs=int(split["num_evaluation_runs"]),
        split_seed=int(seeds["split_seed"]),
        data_order_seed=int(seeds["data_order_seed"]),
        train_seed_base=int(seeds["train_seed_base"]),
        unlearn_seed_base=int(seeds["unlearn_seed_base"]),
        qa_per_author=int(ds["qa_per_author"]),
    )
    validate_manifest(manifest)
    save_manifest(manifest, root / "manifest.json", overwrite=True)
    m = manifest["split"]["m"]
    print(f"[smoke] m={m}, Gamma={split['num_calibration_runs']}, L={split['num_evaluation_runs']}")

    examples = load_examples(ds, num_authors=n_authors)
    print(f"[smoke] {len(examples)} synthetic examples")

    method = "noop"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_run_ids = (
        manifest["sign_vectors"]["calibration"]["run_ids"]
        + manifest["sign_vectors"]["evaluation"]["run_ids"]
    )

    losses_by_run = {}
    for run_id in all_run_ids:
        train_ex = training_examples_for_run(manifest, examples, run_id)
        forget_ex = forget_examples_for_run(manifest, examples, run_id)
        retain_ex = retain_examples(manifest, examples)
        score_ex = candidate_examples(manifest, examples)

        model, tokenizer, _ = load_model_and_tokenizer(
            cfg["model"]["id"],
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            gradient_checkpointing=False,
            for_training=True,
        )
        model.to(device)

        idx = all_run_ids.index(run_id)
        train_model(
            model, tokenizer, train_ex, cfg["training"],
            seed=int(seeds["train_seed_base"]) + idx, label=f"train:{run_id}",
        )
        apply_unlearning(
            method, model, tokenizer, forget_ex, retain_ex,
            cfg["unlearning"].get(method, {}),
            seed=int(seeds["unlearn_seed_base"]) + idx,
        )
        rows = score_examples(
            model, tokenizer, score_ex,
            max_length=int(cfg["scoring"]["max_seq_length"]),
            append_eos=bool(cfg["scoring"]["append_eos"]),
            batch_size=int(cfg["scoring"]["batch_size"]),
        )
        recs = losses_to_records(rows, run_id, method, manifest)
        losses_by_run[run_id] = records_to_score_dict(recs)
        print(f"[smoke] {run_id}: scored {len(recs)} candidate QA pairs")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch_ids = manifest["split"]["batch_ids"]
    calib_ids = manifest["sign_vectors"]["calibration"]["run_ids"]
    eval_ids = manifest["sign_vectors"]["evaluation"]["run_ids"]

    def signs(family):
        return {
            rid: dict(zip(batch_ids, v))
            for rid, v in manifest["sign_vectors"][family]["vectors"].items()
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        cal = fit_calibration(
            {k: losses_by_run[k] for k in calib_ids},
            signs("calibration"),
            batch_ids,
            var_floor=float(cfg["attack"]["var_floor"]),
            pool_variance=cfg["attack"]["pool_variance"],
            warn=False,
        )
    print(f"[smoke] calibration: {cal.diagnostics['num_qa_fitted']} QA Gaussians")

    ev_signs = signs("evaluation")
    for r in [int(x) for x in cfg["attack"]["r_values"] if int(x) <= m]:
        v_list = []
        for rid in eval_ids:
            pred = predict(losses_by_run[rid], cal, r, aggregate=cfg["attack"]["aggregate"])
            assert sum(1 for g in pred["guess"] if g == 1) == r // 2
            assert sum(1 for g in pred["guess"] if g == -1) == r // 2
            v_list.append(overlap(pred["guess"], [ev_signs[rid][b] for b in batch_ids]))
        out = epsilon_lb_mean(m, r, v_list, zeta=float(cfg["epsilon"]["zeta"]),
                              delta=float(cfg["epsilon"]["delta"]))
        print(f"[smoke] r={r}: V={v_list} eps_lb={out['epsilon_lb']}")

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)

    print("\n[smoke] PASS -- full pipeline ran end to end.")
    print("[smoke] (epsilon values here are meaningless: m=4, tiny random model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
