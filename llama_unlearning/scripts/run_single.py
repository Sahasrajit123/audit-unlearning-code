#!/usr/bin/env python3
"""Execute ONE audit run, identified by ``run_id``, so runs distribute across GPUs.

    python scripts/run_single.py --config configs/base.yaml --run_id calib_000 --gpu 0
    python scripts/run_single.py --config configs/base.yaml --run_id calib_000 --dry_run

Stages, in order:

1. fine-tune on ``D(S) = D_r U D_f(S)``   (once per run)
2. save the trained checkpoint            (branch point)
3. for each configured method: reload the checkpoint, unlearn, score candidates
4. for evaluation runs: optionally run the TOFU utility suite
5. write compact JSON results and apply the checkpoint retention policy

``--dry_run`` performs NO training: it resolves paths, builds the datasets, reports
sizes and the exact set of positive/negative candidates, and verifies the invariants.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import config_hash, load_config
from audit_tofu.manifest import (
    load_manifest,
    negative_candidates,
    positive_candidates,
)
from audit_tofu.run_manager import (
    ResourceTracker,
    save_config_provenance,
    load_or_create_run_state,
    prune_checkpoints,
    resolve_run_paths,
    save_json,
    setup_run_logger,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", required=True, help="e.g. calib_000 or eval_003")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="override experiment.methods")
    ap.add_argument(
        "--dry_run", action="store_true", help="no training; validate and report only"
    )
    ap.add_argument(
        "--force", action="store_true", help="recompute methods that already have results"
    )
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config, args.set)
    run_id = args.run_id

    manifest = load_manifest(cfg["experiment"]["manifest_path"])
    paths = resolve_run_paths(cfg["experiment"]["output_root"], run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    state = load_or_create_run_state(manifest, run_id, paths)
    logger = setup_run_logger(paths.run_dir, run_id)
    log = logger.info

    # Snapshot the configuration up front, so it survives even if the run crashes.
    prov = save_config_provenance(
        paths.run_dir, cfg,
        config_path=args.config, overrides=args.set, argv=sys.argv,
    )

    methods = args.methods or cfg["experiment"]["methods"]
    pos = positive_candidates(manifest, run_id)
    neg = negative_candidates(manifest, run_id)

    import socket

    gpu_str = "cpu" if args.gpu is None else f"cuda:{args.gpu}"
    log("=" * 78)
    log(f"RUN {run_id}  family={state.family}  index={state.run_index}")
    log(f"GPU {gpu_str}   host={socket.gethostname()}   pid={os.getpid()}")
    log(f"run log       : {paths.run_dir / 'run.log'}")
    log(f"config        : {args.config}"
        + (f"  --set {' '.join(args.set)}" if args.set else ""))
    log(f"config saved  : {paths.run_dir / 'config.effective.yaml'}"
        f"  (+ config_sources/, invocation.json)")
    log("=" * 78)
    log(f"split_hash    : {manifest['split_hash'][:16]}...")
    log(f"config_hash   : {config_hash(cfg)}")
    log(f"train_seed    : {state.train_seed}   unlearn_seed: {state.unlearn_seed}")
    log(f"methods       : {methods}")
    log(f"positive (+1) : {len(pos)} candidates {pos}")
    log(f"negative (-1) : {len(neg)} candidates {neg}")

    # ---- invariants that must hold before we spend any compute ------------------
    assert len(pos) + len(neg) == manifest["split"]["m"], "sign vector length"
    assert not (set(pos) & set(neg)), "a candidate is both positive and negative"
    if manifest["split"]["m"] % 2 == 0:
        assert len(pos) == len(neg), (
            f"unbalanced sign vector: {len(pos)} positive vs {len(neg)} negative"
        )

    from audit_tofu.tofu_data import (
        forget_examples_for_run,
        load_examples,
        retain_examples,
        training_examples_for_run,
    )

    n_authors = (
        cfg["split"]["num_candidate_authors"] + cfg["split"]["num_retain_authors"]
    )
    log("loading dataset...")
    examples = load_examples(cfg["dataset"], num_authors=n_authors)

    train_ex = training_examples_for_run(manifest, examples, run_id)
    forget_ex = forget_examples_for_run(manifest, examples, run_id)
    retain_ex = retain_examples(manifest, examples)
    from audit_tofu.tofu_data import candidate_examples

    score_ex = candidate_examples(manifest, examples)

    log(f"train  : {len(train_ex)} examples (retain {len(retain_ex)} + forget {len(forget_ex)})")
    log(f"forget : {len(forget_ex)} examples handed to the unlearner")
    log(f"score  : {len(score_ex)} candidate examples (both signs)")

    # The forget set must never contain a negative candidate.
    neg_set = set(neg)
    owner = {}
    for bid, members in manifest["split"]["batches"].items():
        for a, q in members:
            owner[(str(a), int(q))] = bid
    leaked = {owner.get(e.key) for e in forget_ex} & neg_set
    assert not leaked, f"forget set leaked negative candidates: {leaked}"

    # Negative candidates must not appear in the training set.
    train_batches = {owner.get(e.key) for e in train_ex} - {None}
    assert not (train_batches & neg_set), (
        f"negative candidates present in training data: {train_batches & neg_set}"
    )

    if args.dry_run:
        report = {
            "run_id": run_id,
            "dry_run": True,
            "family": state.family,
            "split_hash": manifest["split_hash"],
            "config_hash": config_hash(cfg),
            "methods": list(methods),
            "seeds": {"train": state.train_seed, "unlearn": state.unlearn_seed},
            "counts": {
                "train_examples": len(train_ex),
                "retain_examples": len(retain_ex),
                "forget_examples": len(forget_ex),
                "candidate_examples": len(score_ex),
                "positive_candidates": len(pos),
                "negative_candidates": len(neg),
            },
            "positive_candidates": pos,
            "negative_candidates": neg,
            "output_dir": str(paths.run_dir),
            "invariants": "all passed (balance, disjointness, no negative leakage)",
        }
        save_json(paths.run_dir / "dry_run.json", report)
        log("DRY RUN OK -- no training performed")
        log(f"wrote {paths.run_dir / 'dry_run.json'}")
        return 0

    # ---- real execution ---------------------------------------------------------
    import torch

    from audit_tofu.modeling import load_model_and_tokenizer, maybe_wrap_lora
    from audit_tofu.wandb_logger import WandbRun, preflight
    from audit_tofu.scoring import losses_to_records, score_examples
    from audit_tofu.train import train_model
    from audit_tofu.unlearn import apply_unlearning

    wb_ok, wb_reason = preflight(cfg)
    log(f"wandb         : {'ON' if wb_ok else 'off'} -- {wb_reason}")

    tracker = ResourceTracker()
    run_metrics = {
        "run_id": run_id,
        "family": state.family,
        "run_index": state.run_index,
        "split_hash": manifest["split_hash"],
        "config_hash": config_hash(cfg),
        "effective_config": cfg,
        "seeds": {
            "split_seed": manifest["seeds"]["split_seed"],
            "data_order_seed": manifest["seeds"]["data_order_seed"],
            "train_seed": state.train_seed,
            "unlearn_seed": state.unlearn_seed,
        },
        "counts": {
            "train_examples": len(train_ex),
            "forget_examples": len(forget_ex),
            "retain_examples": len(retain_ex),
            "candidate_examples": len(score_ex),
        },
        "methods": {},
    }

    trained_ckpt = paths.trained_dir
    have_ckpt = (trained_ckpt / "config.json").exists()

    if have_ckpt and not args.force:
        log(f"reusing existing trained checkpoint at {trained_ckpt}")
        model_info = {"model_id": cfg["model"]["id"], "reused_checkpoint": True}
    else:
        log("-" * 78)
        log("STAGE 1: fine-tuning")
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

        wb_train = WandbRun.create(
            cfg, run_id, "train", split_hash=manifest["split_hash"],
            family=state.family, gpu=args.gpu, logger=logger,
            extra_config={"stage": "fine-tune", **run_metrics["counts"]},
        ) if wb_ok else None
        train_metrics = train_model(
            model, tokenizer, train_ex, cfg["training"],
            seed=state.train_seed, logger=log, label="train",
            metrics_sink=wb_train,
        )
        run_metrics["train"] = dict(train_metrics)
        if wb_train is not None:
            wb_train.summary({
                "train/final_loss": train_metrics.get("final_loss"),
                "train/duration_seconds": train_metrics.get("duration_seconds"),
                "train/peak_allocated_gib": train_metrics.get("peak_allocated_gib"),
            })
            wb_train.finish()
        res = tracker.stop()
        log(f"train done in {res['duration_seconds']:.1f}s "
            f"peak_alloc={res['peak_allocated_gib']:.2f}GiB "
            f"peak_reserved={res['peak_reserved_gib']:.2f}GiB")

        if cfg["storage"]["save_trained_checkpoint"]:
            log(f"saving trained checkpoint -> {trained_ckpt}")
            trained_ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(trained_ckpt))
            tokenizer.save_pretrained(str(trained_ckpt))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_metrics["model_info"] = model_info

    # ---- branch into each method from the SAME trained checkpoint ---------------
    for method in methods:
        mdir = paths.method_dir(method)
        losses_path = mdir / "losses.json"
        if losses_path.exists() and not args.force:
            log(f"[{method}] results exist, skipping (use --force to recompute)")
            run_metrics["methods"][method] = {"skipped": True}
            continue

        log("-" * 78)
        log(f"STAGE 2/3: method = {method}")
        log("-" * 78)

        wb = WandbRun.create(
            cfg, run_id, method, split_hash=manifest["split_hash"],
            family=state.family, gpu=args.gpu, logger=logger,
        ) if wb_ok else None

        tracker.start(f"unlearn:{method}")
        model, tokenizer, _ = load_model_and_tokenizer(
            str(trained_ckpt) if trained_ckpt.exists() else cfg["model"]["id"],
            dtype=cfg["model"]["dtype"],
            attn_implementation=cfg["model"]["attn_implementation"],
            gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
            trust_remote_code=cfg["model"]["trust_remote_code"],
            cache_dir=cfg["dataset"].get("cache_dir"),
            for_training=(method != "noop"),
        )
        if torch.cuda.is_available():
            model.to("cuda")

        unlearn_metrics = apply_unlearning(
            method, model, tokenizer, forget_ex, retain_ex,
            cfg["unlearning"].get(method, {}),
            seed=state.unlearn_seed, logger=log, metrics_sink=wb,
        )
        unl_res = tracker.stop()
        log(f"[{method}] unlearn {unl_res['duration_seconds']:.1f}s "
            f"peak_alloc={unl_res['peak_allocated_gib']:.2f}GiB")

        # ---- loss extraction on all m candidate batches (both signs) ------------
        tracker.start(f"score:{method}")
        rows = score_examples(
            model, tokenizer, score_ex,
            max_length=int(cfg["scoring"]["max_seq_length"]),
            append_eos=bool(cfg["scoring"]["append_eos"]),
            system_prompt=cfg["training"].get("system_prompt"),
            batch_size=int(cfg["scoring"]["batch_size"]),
        )
        records = losses_to_records(rows, run_id, method, manifest)
        score_res = tracker.stop()
        log(f"[{method}] scored {len(records)} candidate QA pairs "
            f"in {score_res['duration_seconds']:.1f}s")

        save_json(losses_path, {
            "run_id": run_id,
            "method": method,
            "split_hash": manifest["split_hash"],
            "records": records,
        })
        save_json(mdir / "metrics.json", {
            "run_id": run_id,
            "method": method,
            "unlearning": unlearn_metrics,
            "resources": {
                "unlearn": unl_res,
                "scoring": score_res,
            },
            "effective_method_config": cfg["unlearning"].get(method, {}),
        })
        run_metrics["methods"][method] = {
            "unlearning": unlearn_metrics,
            "resources": {"unlearn": unl_res, "scoring": score_res},
        }

        # ---- persist the unlearned model ----------------------------------------
        # Saved to method_dir/"model", which is exactly the path prune_checkpoints
        # already targets, so the retention policy governs it with no extra wiring:
        # keep_all keeps it, keep_trained and delete_after_scoring remove it.
        #
        # Without this, `keep_all` and `keep_trained` were indistinguishable -- the
        # only save_pretrained in this script was for trained/, so the unlearned
        # weights were never written and those prune targets never existed. Any
        # metric invented later then required re-running the unlearning.
        if cfg["storage"].get("save_unlearned_checkpoint"):
            mckpt = mdir / "model"
            log(f"[{method}] saving unlearned checkpoint -> {mckpt}")
            tracker.start(f"save:{method}")
            mckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(mckpt))
            tokenizer.save_pretrained(str(mckpt))
            save_res = tracker.stop()
            run_metrics["methods"][method]["resources"]["save"] = save_res
            log(f"[{method}] checkpoint saved in {save_res['duration_seconds']:.1f}s")

        # ---- in/out separation --------------------------------------------------
        # Computed unconditionally and PERSISTED, not only pushed to wandb: this is
        # the first-look diagnostic for whether a run has any signal at all, and it
        # should not be lost when wandb is off or unreachable. Ground truth is
        # legitimate here -- the losses are already written and the attack never
        # reads this file (see audit_tofu/attack.py).
        import numpy as _np

        sign_of = dict(zip(manifest["split"]["batch_ids"], state.sign_vector))
        ins = [r["loss"] for r in records if sign_of.get(r["batch_id"]) == 1]
        outs = [r["loss"] for r in records if sign_of.get(r["batch_id"]) == -1]
        separation = {
            "run_id": run_id,
            "method": method,
            "split_hash": manifest["split_hash"],
            "n_in": len(ins),
            "n_out": len(outs),
            "mean_in": None,
            "mean_out": None,
            "gap": None,
            "cohens_d": None,
            "note": (
                "mean_in/mean_out are answer-only NLLs on candidate QA pairs that "
                "were trained-then-unlearned (+1) vs never trained (-1). A positive "
                "gap means the unlearned examples are still easier for the model, "
                "i.e. residual memorization the attacker can exploit."
            ),
        }
        if ins and outs:
            mi, mo = float(_np.mean(ins)), float(_np.mean(outs))
            sd = float(_np.sqrt((_np.var(ins, ddof=1) + _np.var(outs, ddof=1)) / 2))
            separation.update({
                "mean_in": mi,
                "mean_out": mo,
                "gap": mo - mi,
                "cohens_d": (mo - mi) / sd if sd > 0 else None,
            })
        save_json(mdir / "separation.json", separation)
        run_metrics["methods"][method]["separation"] = {
            k: v for k, v in separation.items() if k != "note"
        }

        if wb is not None:
            summ = {
                "unlearn/duration_seconds": unl_res["duration_seconds"],
                "unlearn/peak_allocated_gib": unl_res["peak_allocated_gib"],
                "score/duration_seconds": score_res["duration_seconds"],
                "score/n_candidates": len(records),
                "unlearn/diverged": bool(unlearn_metrics.get("diverged")),
                "unlearn/final_forget_nll": unlearn_metrics.get("final_forget_nll"),
            }
            if separation["mean_in"] is not None:
                summ.update({
                    "candidates/mean_in": separation["mean_in"],
                    "candidates/mean_out": separation["mean_out"],
                    "candidates/gap": separation["gap"],
                    "candidates/cohens_d": separation["cohens_d"],
                })
            wb.summary(summ)

        # ---- utility suite: evaluation runs only --------------------------------
        want_utility = cfg["utility"]["enabled"] and (
            state.family == "evaluation" or cfg["utility"]["calibration_runs"]
        )
        if want_utility:
            log(f"[{method}] running TOFU utility suite")
            tracker.start(f"utility:{method}")
            from audit_tofu.utility import run_utility_suite

            # Resolve the reference. Two sources, in priority order:
            #   1. utility.reference_truth_ratios_path -- an explicit file, kept for
            #      backward compatibility. Feeds only the flat/`all` comparison.
            #   2. utility.reference_dir -- AUTO-DISCOVERED. The retain-only model's
            #      truth_ratios.json, which carries per-row (author, qa) identity and
            #      so supports the proper +1/-1 slicing.
            # A configured-but-missing path used to fall through to ref=None
            # silently, so a typo'd path was indistinguishable from "no reference
            # configured" -- both just produced forget_quality: null.
            import json as _json

            from audit_tofu.utility import load_reference_truth_ratios

            ref = None                 # flat list -> the `all` comparison
            ref_payload = None         # full dict -> enables the sign slices
            ref_path = cfg["utility"].get("reference_truth_ratios_path")
            if ref_path:
                p = Path(ref_path)
                if not p.exists():
                    log(f"[{method}] WARNING: reference_truth_ratios_path={ref_path} "
                        "does not exist; forget_quality will be null. Fix the path or "
                        "set it to null to auto-discover utility.reference_dir.")
                else:
                    ref_payload = _json.loads(p.read_text())
                    ref = load_reference_truth_ratios(ref_payload)
                    log(f"[{method}] reference: {len(ref)} truth ratios from {ref_path}")
            else:
                ref_dir = cfg["utility"].get("reference_dir")
                cand = (Path(ref_dir) / "truth_ratios.json") if ref_dir else None
                if cand is not None and cand.exists():
                    ref_payload = _json.loads(cand.read_text())
                    stored_retain = ref_payload.get("retain_authors")
                    # The reference is only a valid baseline if it trained on exactly
                    # this D_r. Mismatch is a warning, not a failure: the run's own
                    # results are unaffected, and forget_quality is recomputable by
                    # scripts/forget_quality.py once a correct reference exists.
                    if (stored_retain is not None
                            and set(stored_retain) != set(
                                manifest["split"]["retain_authors"])):
                        log(f"[{method}] WARNING: reference at {cand} trained on a "
                            f"different retain set ({len(stored_retain)} authors vs "
                            f"{len(manifest['split']['retain_authors'])}); ignoring it")
                        ref_payload = None
                    else:
                        ref = load_reference_truth_ratios(ref_payload)
                        log(f"[{method}] reference auto-discovered: {len(ref)} truth "
                            f"ratios from {cand}")
                elif cand is not None:
                    log(f"[{method}] no reference at {cand}; forget_quality will be "
                        "null until scripts/train_reference.py has run")

            util = run_utility_suite(
                model, tokenizer,
                dataset_name=cfg["dataset"]["name"],
                reference_truth_ratios=ref,
                max_length=int(cfg["scoring"]["max_seq_length"]),
                append_eos=bool(cfg["scoring"]["append_eos"]),
                system_prompt=cfg["training"].get("system_prompt"),
                compute_rouge=bool(cfg["utility"]["compute_rouge"]),
                limit=cfg["utility"].get("limit"),
                cache_dir=cfg["dataset"].get("cache_dir"),
                logger=logger,
            )
            u_res = tracker.stop()

            # ---- forget quality, properly sliced --------------------------------
            # run_utility_suite can only compare the FLAT truth-ratio lists, which
            # for the forget group is the diluted `all` number: under a balanced sign
            # vector half those rows were never trained on and match the reference by
            # construction. Here we have the manifest and this run's sign vector, so
            # we compute the same plus/minus/all slices scripts/forget_quality.py
            # reports -- via the SAME helper, so the two cannot disagree.
            if ref_payload is not None:
                from audit_tofu.utility import forget_quality_slices

                sign_of = {}
                for bid, sj in zip(manifest["split"]["batch_ids"], state.sign_vector):
                    for author, qa in manifest["split"]["batches"][bid]:
                        sign_of[(str(author), int(qa))] = int(sj)

                ref_groups = ref_payload.get("groups") or {}
                fq_slices = {}
                for group, gres in (util.get("splits") or {}).items():
                    if not isinstance(gres, dict) or "per_row" not in gres:
                        continue
                    ref_rows = (ref_groups.get(group) or {}).get("per_row") or []
                    if not ref_rows:
                        continue
                    s = forget_quality_slices(gres["per_row"], ref_rows, sign_of)
                    if s:
                        fq_slices[group] = s
                util["forget_quality_slices"] = fq_slices
                util["forget_quality_reference"] = {
                    "seed": ref_payload.get("seed"),
                    "retain_authors": len(ref_payload.get("retain_authors") or []),
                    "split_hash": ref_payload.get("split_hash"),
                }
                # Promote the headline so it is readable without digging: the +1
                # slice of the forget group is the number that means "did it forget".
                head = (fq_slices.get("forget") or {}).get("plus") or {}
                if head:
                    util["forget_quality_plus"] = head.get("forget_quality")
                    util["ks_statistic_plus"] = head.get("ks_statistic")
                    log(f"[{method}] forget_quality(+1)={head.get('forget_quality')} "
                        f"KS={head.get('ks_statistic')} n={head.get('n_model')}  "
                        f"(control -1: "
                        f"{((fq_slices.get('forget') or {}).get('minus') or {}).get('forget_quality')})")

            save_json(mdir / "utility.json", util)
            run_metrics["methods"][method]["utility"] = util
            run_metrics["methods"][method]["resources"]["utility"] = u_res
            log(f"[{method}] utility done in {u_res['duration_seconds']:.1f}s "
                f"model_utility={util.get('model_utility')}")

            # The utility summary used to be omitted from wandb entirely, because the
            # only wb.summary() call happened before this block ran.
            if wb is not None:
                usumm = {
                    "utility/model_utility": util.get("model_utility"),
                    "utility/forget_quality": util.get("forget_quality"),
                    "utility/ks_statistic": util.get("ks_statistic"),
                    "utility/duration_seconds": u_res["duration_seconds"],
                }
                for group, res in (util.get("splits") or {}).items():
                    if not isinstance(res, dict) or "error" in res:
                        continue
                    for key in ("probability", "truth_ratio", "rouge_l_recall"):
                        usumm[f"utility/{group}/{key}"] = res.get(key)
                wb.summary(usumm)

        if wb is not None:
            wb.finish()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_metrics["resources"] = tracker.as_dict()
    save_json(paths.run_dir / "metrics.json", run_metrics)
    # config.json / config.effective.yaml / config_sources/ were written up front by
    # save_config_provenance, so there is nothing to re-save here.

    removed = prune_checkpoints(paths, cfg["storage"]["retention"], logger)
    run_metrics["checkpoints_removed"] = removed
    save_json(paths.run_dir / "metrics.json", run_metrics)

    r = tracker.as_dict()
    log("=" * 78)
    log(f"RUN {run_id} COMPLETE in {r['total_duration_seconds']:.1f}s")
    log(f"peak allocated {r['overall_peak_allocated_gib']:.2f} GiB | "
        f"peak reserved {r['overall_peak_reserved_gib']:.2f} GiB")
    log(f"results -> {paths.run_dir}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
