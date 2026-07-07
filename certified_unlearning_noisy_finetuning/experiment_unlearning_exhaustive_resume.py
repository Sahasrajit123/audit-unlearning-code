# -*- coding: utf-8 -*-
"""
Resume incomplete exhaustive-combo runs: reuse existing ckpt/ (training) and
run only the ckpt_trial_{j} that are not yet complete.

Use when a run was interrupted: _run_{combo_idx}.dir still exists. This script
  - finds run_dir via _run_{combo_idx}.dir
  - skips training if ckpt/ has a valid checkpoint
  - for each j in [0, num_unlearn_per_combo): skips if ckpt_trial_j/ has all
    required unlearn_epoch_* from config (see _expected_unlearn_epoch_numbers);
    else deletes partial ckpt_trial_j/, loads from ckpt/, runs unlearn,
    saves to ckpt_trial_j/, runs test
  - on full completion, removes _run_{combo_idx}.dir

Requires: run_vars.json, config.yaml, and (if training skipped) ckpt/ in run_dir.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

logging.getLogger("absl").setLevel(logging.ERROR)

n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
_xla_flags = os.environ.get("XLA_FLAGS", "")
for _flag in [
    "--xla_gpu_exclude_nondeterministic_ops",
    "--xla_gpu_strict_conv_algorithm_picker=false",
]:
    if _flag not in _xla_flags:
        _xla_flags = f"{_xla_flags} {_flag}".strip() if _xla_flags else _flag
os.environ["XLA_FLAGS"] = _xla_flags
os.environ.setdefault("JAX_DISABLE_MOST_FASTER_PATHS", "1")

import jax
import numpy as np
import wandb

from src.models.model import ModelFactory
from src.training.trainer import Trainer
from src.utils.utils import load_config, logger
from src.utils.data_cache import load_cifar_splits_with_batch_subset


def _get_data_split_base(
    results_dir: str,
    data_subfolder: str,
    dataset_name: str,
) -> Path:
    """Return base dir for data_split: prefer RESULTS_DIR/data_split, else data/{dataset_name}/data_split.

    The fallback must be dataset-aware so CIFAR-10 runs never incorrectly read
    from CIFAR-100 cache roots.
    """
    candidate = Path(results_dir) / "data_split"
    # sub = (
    #     data_subfolder
    #     if data_subfolder is not None
    #     else (f"cifar_{file_batch_size}" if file_batch_size != 128 else "cifar")
    # )
    if data_subfolder is not None:
        sub = data_subfolder
    else:
        raise ValueError("data_subfolder must be provided in run_vars for resume; remove fallback to file_batch_size-based subfolder")
    ##sub = data_subfolder if data_subfolder is not None else raise ValueError("data_subfolder must be provided in run_vars for resume; remove fallback to file_batch_size-based subfolder")
    forget_dir = candidate / sub / "forget"
    if forget_dir.is_dir() and list(forget_dir.glob("batch_*.pkl")):
        return candidate
    if dataset_name is not None:
        dataset_root = dataset_name
    else:
        raise ValueError("Dataset name must be provided in config if data_subfolder not set")
    return Path("data") / dataset_root / "data_split"



def _trial_complete(run_dir: Path, j: int, config) -> bool:
    """True iff ckpt_trial_{j} exists and contains the final unlearn epoch checkpoint."""
    d = run_dir / "ckpt_trial_{}".format(j)
    if not d.is_dir():
        return False
    ul = config.get("unlearning", {})
    last_epoch = int(ul.get("epochs", 50))
    return (d / "unlearn_epoch_{}".format(last_epoch)).exists()


def _training_complete(run_dir: Path, train_epochs: int) -> bool:
    """True if ckpt/ exists and has the final training checkpoint (or any checkpoint_*)."""
    ckpt = run_dir / "ckpt"
    if not ckpt.is_dir():
        return False
    # Prefer final epoch; fallback: any checkpoint_
    if (ckpt / f"checkpoint_{train_epochs}").exists():
        return True
    return bool(list(ckpt.glob("checkpoint_*")))


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Resume exhaustive-combo: reuse ckpt/, run only missing ckpt_trial_{j}.")
    p.add_argument("--results_dir", type=str, required=True, help="Base dir containing _run_{combo_idx}.dir")
    p.add_argument("--combo_idx", type=int, required=True, help="Combo to resume")
    p.add_argument("--num_trial_workers", type=int, default=1, help="Total number of parallel workers splitting trials")
    p.add_argument("--trial_worker_id", type=int, default=0, help="Index of this worker (0-based)")
    p.add_argument("--debug", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR = args.results_dir
    combo_idx = args.combo_idx
    num_trial_workers = args.num_trial_workers
    trial_worker_id = args.trial_worker_id

    run_dir_file = Path(RESULTS_DIR) / f"_run_{combo_idx}.dir"
    if not run_dir_file.is_file():
        raise FileNotFoundError(
            f"_run_{combo_idx}.dir not found in {RESULTS_DIR}. "
            "Resume only applies to incomplete runs (the .dir is removed on success)."
        )

    run_dir = Path(run_dir_file.read_text().strip())
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir from _run_{combo_idx}.dir does not exist: {run_dir}")

    run_vars_path = run_dir / "run_vars.json"
    if not run_vars_path.is_file():
        raise FileNotFoundError(f"run_vars.json not found in {run_dir}")
    with open(run_vars_path) as f:
        run_vars = json.load(f)

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    config = load_config(str(config_path))

    chosen_idx = np.array(run_vars["chosen_idx"], dtype=np.int64)
    NUM_UNLEARN_PER_COMBO = int(run_vars["num_unlearn_per_combo"])
    UNLEARN_SEED_OFFSET = int(run_vars["unlearn_seed_offset"])
    DATA_SUBFOLDER = run_vars["data_subfolder"]
    FORGET_FRACTION = run_vars["forget_fraction"]
    # FILE_BATCH_SIZE removed; use data_subfolder or dataset_type instead
    # Shuffle options are stored by the exhaustive launcher; default to False if absent
    SHUFFLE_TRAIN_SAMPLES = bool(run_vars.get("shuffle_train_samples", False))
    SHUFFLE_TRAIN_BATCHES_EACH_EPOCH = bool(run_vars.get("shuffle_train_batches_each_epoch", False))
    # Look for data_split under results_dir first, then fall back to shared
    # data/<dataset_name>/data_split.
    DATA_SPLIT_ROOT = str(
        _get_data_split_base(
            RESULTS_DIR,
            DATA_SUBFOLDER,
            str(config.get("dataset", {}).get("name", None)),
        )
    )
    train_epochs = int(config["training"]["epochs"])
    ckpt_dir = str(run_dir / "ckpt")
    unlearn_seeds = [UNLEARN_SEED_OFFSET + combo_idx * 1000 + j for j in range(NUM_UNLEARN_PER_COMBO)]

    if args.debug >= 1:
        logger.setLevel(logging.DEBUG)

    # Append resume info to the main training log used by the launcher
    train_log_path = run_dir / "run_train.log"
    train_log_file = open(train_log_path, "a", buffering=1)
    train_log_file.write("\n=== RESUME combo_idx={} ===\n".format(combo_idx))
    train_log_file.flush()
    sys.stdout = train_log_file
    sys.stderr = train_log_file
    os.dup2(train_log_file.fileno(), 1)
    os.dup2(train_log_file.fileno(), 2)

    if not _training_complete(run_dir, train_epochs):
        raise RuntimeError(
            f"Trained checkpoint missing or incomplete in {ckpt_dir}. "
            "Run the full experiment first before resuming."
        )
    logger.info("Resume: skipping training (ckpt/ already exists)")

    all_incomplete = [j for j in range(NUM_UNLEARN_PER_COMBO) if not _trial_complete(run_dir, j, config)]
    import math
    chunk_size = math.ceil(len(all_incomplete) / num_trial_workers) if num_trial_workers > 1 else len(all_incomplete)
    start = trial_worker_id * chunk_size
    to_run = all_incomplete[start:start + chunk_size]

    if to_run:
        logger.info("Resume worker %d/%d: running %d trials: %s", trial_worker_id, num_trial_workers, len(to_run), to_run)
    else:
        logger.info("Resume worker %d/%d: no trials to run", trial_worker_id, num_trial_workers)

    if to_run:
        # Trainer/unlearn_setup use wandb.log(); init so it doesn't raise.
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            id=run_dir.name,
            resume="allow",
            config=config,
            dir=config["wandb"].get("log_dir", "."),
        )

        (
            train_loader,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
            _,
        ) = load_cifar_splits_with_batch_subset(
            run_dir=DATA_SPLIT_ROOT,
            batch_size=config["dataset"]["batch_size"],
            forget_fraction=FORGET_FRACTION,
            rng_seed=0,
            data_subfolder=DATA_SUBFOLDER,
            chosen_idx_override=chosen_idx,
            shuffle_train_samples=SHUFFLE_TRAIN_SAMPLES,
            shuffle_train_batches_each_epoch=SHUFFLE_TRAIN_BATCHES_EACH_EPOCH,
        )

        for j in to_run:
            ckpt_trial_j_path = run_dir / f"ckpt_trial_{j}"
            if ckpt_trial_j_path.exists():
                shutil.rmtree(ckpt_trial_j_path)
                logger.info("Resume: removed partial ckpt_trial_%s for clean restart from trained ckpt", j)
            ckpt_trial_j = str(ckpt_trial_j_path)
            os.makedirs(ckpt_trial_j, exist_ok=True)
            config_trial = {**config, "checkpoint_path": ckpt_trial_j}

            # Per-trial log matches the launcher: run_unlearn_trial_{j}.log
            trial_log_path = run_dir / f"run_unlearn_trial_{j}.log"
            trial_log_file = open(trial_log_path, "a", buffering=1)
            prev_stdout, prev_stderr = sys.stdout, sys.stderr
            sys.stdout = trial_log_file
            sys.stderr = trial_log_file
            os.dup2(trial_log_file.fileno(), 1)
            os.dup2(trial_log_file.fileno(), 2)

            try:
                model_j = ModelFactory.create_model(
                    model_name=config["model"]["name"],
                    num_classes=config["model"]["n_classes"],
                )
                unlearn_key = jax.random.PRNGKey(unlearn_seeds[j])
                trainer_j = Trainer(
                    config=config_trial,
                    model=model_j,
                    key=unlearn_key,
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    test_dataloader=test_loader,
                    forget_dataloader=forget_loader,
                    retain_dataloader=retain_loader,
                    use_pretrained=False,
                    run_dir=str(run_dir),
                )

                state_template = trainer_j.train_setup()
                state_loaded = Trainer.load_checkpoint(ckpt_dir, state_template, train_epochs)

                if config["unlearning"]["algorithm"] in ("ascent_descent", "retain_finetune"):
                    from src.training.unlearn_ascent_descent import unlearn_ascent_descent
                    from src.training.unlearn_retain_finetune import unlearn_retain_finetune
                    if config["unlearning"]["algorithm"] == "ascent_descent":
                        state_loaded = unlearn_ascent_descent(
                            state=state_loaded,
                            forget_dataloader=forget_loader,
                            retain_dataloader=retain_loader,
                            config=config_trial,
                            val_dataloader=val_loader,
                            validate_fn=trainer_j.validate,
                            save_checkpoint_fn=lambda s, n, **kw: trainer_j.save_checkpoint(s, n, unlearn=True, **kw),
                            run_dir=str(run_dir),
                        )
                    else:
                        state_loaded = unlearn_retain_finetune(
                            state=state_loaded,
                            forget_dataloader=forget_loader,
                            retain_dataloader=retain_loader,
                            config=config_trial,
                            val_dataloader=val_loader,
                            validate_fn=trainer_j.validate,
                            save_checkpoint_fn=lambda s, n, **kw: trainer_j.save_checkpoint(s, n, unlearn=True, **kw),
                            run_dir=str(run_dir),
                        )
                else:
                    logger.info("Resume trial %s: unlearn_seed=%s", j, unlearn_seeds[j])
                    state_loaded = trainer_j.unlearn(state_loaded)

                trainer_j.test(state=state_loaded)
            finally:
                trial_log_file.flush()
                trial_log_file.close()
                # Restore training log as default stdout/stderr
                sys.stdout = train_log_file
                sys.stderr = train_log_file
                os.dup2(train_log_file.fileno(), 1)
                os.dup2(train_log_file.fileno(), 2)

    # Remove .dir only when all trials across all workers are complete
    all_done = all(_trial_complete(run_dir, j, config) for j in range(NUM_UNLEARN_PER_COMBO))
    if all_done:
        try:
            os.remove(run_dir_file)
            logger.info("Resume: removed _run_%s.dir (all trials complete)", combo_idx)
        except OSError as e:
            logger.warning("Resume: could not remove _run_%s.dir: %s", combo_idx, e)
    else:
        logger.info("Resume worker %d/%d: done, waiting for other workers to finish remaining trials", trial_worker_id, num_trial_workers)


if __name__ == "__main__":
    main()
