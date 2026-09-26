#!/usr/bin/env python3
"""Train the retain-only REFERENCE model and record its truth ratios.

    python scripts/train_reference.py --config configs/forgetq5_pinned180.yaml --gpu 3

What this is. TOFU forget quality is a KS test of an unlearned model's forget-set
truth-ratio distribution against a model that never trained on the forget set. This
script builds that second model: a fine-tune of the BASE checkpoint on ``D_r`` alone.

Why it cannot be an existing artifact:

* ``retain_ft`` is not a reference. It loads ``trained/`` -- the checkpoint that
  already trained on the forget data -- and fine-tunes it further on the retain set
  (OpenUnlearning's ``finetune`` heuristic). Both of its distributions are
  post-exposure, so KS between them measures nothing.
* Every run's weights are deleted by ``retention: delete_after_scoring``, so truth
  ratios on any new data cannot be recovered after the fact.

One model serves every run, because ``D_r`` is fixed across the whole audit by the
manifest. Under a pinned split it is exactly ``retain90``, so the reference is
ignorant of precisely the rows the KS test scores.

It is deliberately NOT written under ``runs/``: it has no sign vector and no
unlearning stage, and putting it there would let ``collect_results.py`` and
``run_audit.sh`` mistake it for an audit run.

Outputs, under ``<output_root>/reference/``::

    trained/            the D_r-only checkpoint (kept -- re-deriving costs 19 min)
    utility.json        full suite; its own forget_quality is null by construction
    truth_ratios.json   per-row, identified: what forget_quality.py consumes
    candidate_losses.json   answer-only NLL on all m candidate batches
    metrics.json        training + resource telemetry
    run.log             structured log (same convention as an audit run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config, config_hash
from audit_tofu.manifest import load_manifest
from audit_tofu.run_manager import (
    ResourceTracker,
    save_config_provenance,
    save_json,
    setup_run_logger,
)

#: Directory name under output_root. Not "runs/" -- see the module docstring.
REFERENCE_DIRNAME = "reference"
#: wandb identifiers for the reference, so it is visible alongside the audit runs
#: but cannot be confused with one.
REFERENCE_RUN_ID = "reference"
REFERENCE_METHOD = "retain_only"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed; defaults to seeds.train_seed_base - 1 so it "
                         "cannot collide with any audit run's seed")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a reference checkpoint already exists")
    ap.add_argument("--skip_candidate_scoring", action="store_true")
    ap.add_argument("--skip_utility", action="store_true",
                    help="train only, no utility suite. Produces NO truth ratios, so "
                         "forget_quality.py will have nothing to compare against; "
                         "intended for smoke-testing the training path.")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    manifest = load_manifest(cfg["experiment"]["manifest_path"])

    # utility.reference_dir lets several configs share one reference model, which is
    # correct because the reference depends only on D_r. Falls back to a
    # per-experiment directory when unset.
    out_dir = Path(cfg["utility"].get("reference_dir")
                   or (Path(cfg["experiment"]["output_root"]) / REFERENCE_DIRNAME))
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_dir = out_dir / "trained"

    logger = setup_run_logger(out_dir, REFERENCE_RUN_ID)
    log = logger.info

    # The reference seed must differ from every run's train seed, otherwise a run and
    # the reference would share stochasticity and the -1 control below would
    # understate the seed-driven nuisance floor it exists to measure.
    seed = args.seed if args.seed is not None else int(
        manifest["seeds"]["train_seed_base"]) - 1

    log("=" * 78)
    log("RETAIN-ONLY REFERENCE MODEL")
    log("=" * 78)
    log(f"config       : {args.config}")
    log(f"split_hash   : {manifest['split_hash']}")
    log(f"selection    : {manifest['split'].get('candidate_selection') or 'random'}")
    log(f"output       : {out_dir}")
    log(f"seed         : {seed}")

    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        log(f"gpu          : {args.gpu} (CUDA_VISIBLE_DEVICES)")

    import torch

    from audit_tofu.modeling import load_model_and_tokenizer, maybe_wrap_lora
    from audit_tofu.tofu_data import dataset_fingerprint, load_examples, retain_examples
    from audit_tofu.train import train_model
    from audit_tofu.utility import run_utility_suite
    from audit_tofu.wandb_logger import WandbRun, preflight

    save_config_provenance(out_dir, cfg, config_path=args.config,
                           overrides=args.set, argv=sys.argv)

    # ---- data ------------------------------------------------------------------
    n_authors = (int(cfg["split"]["num_candidate_authors"])
                 + int(cfg["split"]["num_retain_authors"]))
    examples = load_examples(cfg["dataset"], num_authors=n_authors)
    fp = dataset_fingerprint(examples)
    expected = manifest["dataset"].get("fingerprint")
    if expected and fp != expected:
        raise SystemExit(
            f"dataset fingerprint {fp[:16]}... != manifest {expected[:16]}.... The "
            "reference must be trained on the same corpus as the audit runs."
        )

    retain_ex = retain_examples(manifest, examples)
    cand_authors = set(manifest["split"]["candidate_authors"])
    leaked = [e for e in retain_ex if e.author_id in cand_authors]
    if leaked:
        # Would silently invalidate every KS test downstream.
        raise SystemExit(
            f"{len(leaked)} retain examples belong to candidate authors; the "
            "reference would not be ignorant of the forget set"
        )
    log(f"retain examples: {len(retain_ex)} "
        f"({len(manifest['split']['retain_authors'])} authors, candidate-free)")

    wb_ok, why = preflight(cfg)
    log(f"wandb        : {'ON' if wb_ok else 'off'} -- {why}")

    tracker = ResourceTracker()
    metrics = {
        "kind": "reference_retain_only",
        "split_hash": manifest["split_hash"],
        "seed": seed,
        "config_hash": config_hash(cfg),
        "dataset_fingerprint": fp,
        "counts": {
            "retain_examples": len(retain_ex),
            "retain_authors": len(manifest["split"]["retain_authors"]),
            "candidate_authors": len(manifest["split"]["candidate_authors"]),
        },
    }

    # Grouped by the RETAIN SET, not by any one audit. This model is shared by every
    # config that pins the same D_r (all four audit configs do), so filing it under
    # one audit's "<name>-<split_hash>" group would imply it belongs to that audit
    # alone. The group name is derived from a hash of D_r so two references trained
    # on different retain sets can never collide in the same group.
    import hashlib as _hashlib

    retain_hash = _hashlib.sha256(
        "|".join(manifest["split"]["retain_authors"]).encode()
    ).hexdigest()[:8]
    wb = WandbRun.create(
        cfg, REFERENCE_RUN_ID, REFERENCE_METHOD,
        split_hash=manifest["split_hash"], family="reference", gpu=args.gpu,
        logger=logger,
        group=f"reference-retain{len(manifest['split']['retain_authors'])}-{retain_hash}",
        extra_config={"stage": "reference", "seed": seed,
                      "retain_hash": retain_hash, **metrics["counts"]},
    ) if wb_ok else None

    # ---- train (or reuse) ------------------------------------------------------
    have = (trained_dir / "config.json").exists()
    if have and not args.force:
        log(f"reusing existing reference checkpoint at {trained_dir}")
        model, tokenizer, model_info = load_model_and_tokenizer(
            str(trained_dir),
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
            trust_remote_code=cfg["model"]["trust_remote_code"],
            cache_dir=cfg["dataset"].get("cache_dir"),
            for_training=False,
        )
        model_info["reused_checkpoint"] = True
        if torch.cuda.is_available():
            model.to("cuda")
    else:
        log("-" * 78)
        log("fine-tuning on D_r only")
        log("-" * 78)
        tracker.start("train")
        model, tokenizer, model_info = load_model_and_tokenizer(
            cfg["model"]["id"],
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
            trust_remote_code=cfg["model"]["trust_remote_code"],
            cache_dir=cfg["dataset"].get("cache_dir"),
            for_training=True,
        )
        model, lora_info = maybe_wrap_lora(model, cfg["model"].get("lora"))
        model_info.update(lora_info)
        log(f"model: {model_info}")
        if torch.cuda.is_available():
            model.to("cuda")

        train_metrics = train_model(
            model, tokenizer, retain_ex, cfg["training"],
            seed=seed, logger=log, label="reference", metrics_sink=wb,
        )
        metrics["train"] = dict(train_metrics)
        res = tracker.stop()
        log(f"train done in {res['duration_seconds']:.1f}s "
            f"peak_alloc={res['peak_allocated_gib']:.2f}GiB")
        if wb is not None:
            wb.summary({
                "reference/train_final_loss": train_metrics.get("final_loss"),
                "reference/train_duration_seconds": train_metrics.get("duration_seconds"),
                "reference/optimizer_steps": train_metrics.get("optimizer_steps"),
            })

        log(f"saving reference checkpoint -> {trained_dir}")
        trained_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(trained_dir))
        tokenizer.save_pretrained(str(trained_dir))

    metrics["model_info"] = model_info

    # ---- candidate losses ------------------------------------------------------
    # The never-trained loss floor on the candidate pool. Directly comparable to each
    # run's -1 candidates, and the ideal "out" distribution for reading in/out gaps.
    if not args.skip_candidate_scoring:
        from audit_tofu.scoring import score_examples
        from audit_tofu.tofu_data import candidate_examples

        log("-" * 78)
        log("scoring the candidate pool (never-trained loss floor)")
        tracker.start("candidate_scoring")
        cand_ex = candidate_examples(manifest, examples)
        records = score_examples(
            model, tokenizer, cand_ex,
            max_length=int(cfg["scoring"]["max_seq_length"]),
            append_eos=bool(cfg["scoring"]["append_eos"]),
            system_prompt=cfg["training"].get("system_prompt"),
            batch_size=int(cfg["scoring"]["batch_size"]),
        )
        cs = tracker.stop()
        save_json(out_dir / "candidate_losses.json", {
            "kind": "reference_candidate_losses",
            "split_hash": manifest["split_hash"],
            "note": "Reference model never trained on ANY candidate author, so these "
                    "are the ideal out-condition losses for the whole pool.",
            "records": records,
        })
        import numpy as _np

        mean_loss = float(_np.mean([r["loss"] for r in records]))
        metrics["candidate_losses"] = {
            "n": len(records), "mean_loss": mean_loss, "resources": cs,
        }
        log(f"candidate pool: {len(records)} QA pairs, mean answer NLL {mean_loss:.4f} "
            f"in {cs['duration_seconds']:.1f}s")
        if wb is not None:
            wb.summary({"reference/candidate_mean_loss": mean_loss,
                        "reference/candidate_n": len(records)})

    # ---- utility suite ---------------------------------------------------------
    # Deliberately NOT gated on cfg["utility"]["enabled"]. That flag controls whether
    # the *audit runs* pay for the suite; for the reference the suite is the entire
    # deliverable, since it is what produces the truth ratios. Use --skip_utility to
    # bypass it explicitly.
    if args.skip_utility:
        log("--skip_utility: no utility suite, so NO truth_ratios.json is written. "
            "scripts/forget_quality.py will have no reference to compare against.")
        metrics["resources"] = tracker.as_dict()
        save_json(out_dir / "metrics.json", metrics)
        if wb is not None:
            wb.finish()
        log(f"reference (training only) -> {out_dir}")
        return 0

    log("-" * 78)
    if not cfg["utility"]["enabled"]:
        log("note: utility.enabled=false applies to audit runs; the reference runs "
            "the suite regardless because its truth ratios are the deliverable")
    log("utility suite")
    tracker.start("utility")
    util = run_utility_suite(
        model, tokenizer,
        dataset_name=cfg["dataset"]["name"],
        reference_truth_ratios=None,   # a reference has no reference; stays null
        max_length=int(cfg["scoring"]["max_seq_length"]),
        append_eos=bool(cfg["scoring"]["append_eos"]),
        system_prompt=cfg["training"].get("system_prompt"),
        compute_rouge=bool(cfg["utility"]["compute_rouge"]),
        limit=cfg["utility"].get("limit"),
        cache_dir=cfg["dataset"].get("cache_dir"),
        logger=logger,
    )
    ures = tracker.stop()
    util["reference_note"] = (
        "This IS the reference model, so forget_quality is null by construction: a "
        "KS test of a distribution against itself is meaningless."
    )
    save_json(out_dir / "utility.json", util)
    metrics["utility_resources"] = ures

    # The artifact forget_quality.py consumes: per-row, identified truth ratios.
    tr_payload = {
        "kind": "reference_truth_ratios",
        "split_hash": manifest["split_hash"],
        "seed": seed,
        "compute_rouge": bool(cfg["utility"]["compute_rouge"]),
        "candidate_authors": manifest["split"]["candidate_authors"],
        # What this reference is actually a function of. The model depends ONLY on the
        # retain author set, not on the candidate batching, so a single reference is
        # valid for every config that shares D_r -- e.g. all four audit configs, which
        # pin the same 180 retain authors but differ in B and therefore have different
        # split_hashes. forget_quality.py validates on this, not on split_hash.
        "retain_authors": manifest["split"]["retain_authors"],
        "groups": {},
    }
    for group, res in (util.get("splits") or {}).items():
        if not isinstance(res, dict) or "per_row" not in res:
            continue
        tr_payload["groups"][group] = {
            "n": res.get("n"),
            "n_identified": res.get("n_identified"),
            "probability": res.get("probability"),
            "truth_ratio": res.get("truth_ratio"),
            "rouge_l_recall": res.get("rouge_l_recall"),
            "per_row": res["per_row"],
        }
    save_json(out_dir / "truth_ratios.json", tr_payload)

    metrics["resources"] = tracker.as_dict()
    save_json(out_dir / "metrics.json", metrics)

    if wb is not None:
        summ = {"reference/model_utility": util.get("model_utility")}
        for group, res in (util.get("splits") or {}).items():
            if isinstance(res, dict) and "error" not in res:
                summ[f"reference/{group}/probability"] = res.get("probability")
                summ[f"reference/{group}/truth_ratio"] = res.get("truth_ratio")
                summ[f"reference/{group}/rouge_l_recall"] = res.get("rouge_l_recall")
        wb.summary(summ)
        wb.finish()

    log("=" * 78)
    log(f"model_utility = {util.get('model_utility')}")
    for group, res in (util.get("splits") or {}).items():
        if isinstance(res, dict) and "error" not in res:
            log(f"  {group:<13} prob={res.get('probability')} "
                f"tr={res.get('truth_ratio')} rouge={res.get('rouge_l_recall')} "
                f"identified={res.get('n_identified')}/{res.get('n')}")
    log(f"reference -> {out_dir}")
    log("next: python scripts/forget_quality.py --config " + args.config)
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
