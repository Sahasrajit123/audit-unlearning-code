# -*- coding: utf-8 -*-
# main.py
"""
Entry point that
  - builds / re-uses cached CIFAR splits under <RESULTS_DIR>/data_split/
  - runs training/unlearning/testing 50 times with independent forget sampling
"""

import logging
import os
import shutil
import json
import sys
from pathlib import Path

# Suppress absl/Tensorflow warnings that happen at import time (before we can redirect)
logging.getLogger("absl").setLevel(logging.ERROR)

import jax
import numpy as np
import wandb

from src.models.model import ModelFactory
from src.training.trainer import Trainer
from src.utils.utils import load_config, logger
from src.utils.data_cache import load_cifar_splits_with_batch_subset

# ---------- Default values --------------------------------------------
# Default values for parameters
DEFAULT_RESULTS_DIR = "logs_multiple_runs_alternate_batch_1"
DEFAULT_FORGET_FRACTION = 0.50
DATA_SPLIT_FALLBACK_BASE = "data/cifar100"

# Thread settings
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"]            = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"]       = n_threads_str
os.environ["MKL_NUM_THREADS"]            = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"]     = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"]        = n_threads_str
os.environ["CUDA_DEVICE_ORDER"]          = "PCI_BUS_ID"
# ----------------------------------------------------------------------





def _resolve_data_split_root(results_dir: str, data_subfolder: str) -> str:
    """Use <results_dir>/data_split first; fallback to data/cifar100/data_split when missing."""
    candidate = Path(results_dir) / "data_split"
    forget_dir = candidate / data_subfolder / "forget"
    if forget_dir.is_dir() and list(forget_dir.glob("batch_*.pkl")):
        return str(candidate)
    return str(Path(DATA_SPLIT_FALLBACK_BASE) / "data_split")


def parse_args():
    """Parse command line arguments for the experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unlearning Experiment with Multiple Runs")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the config file."
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Debug level (default: 0)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Base directory for all runs (default: {})".format(DEFAULT_RESULTS_DIR)
    )
    parser.add_argument(
        "--run_index",
        type=int,
        required=True,
        help="Which run index (required)"
    )
    parser.add_argument(
        "--forget_fraction",
        type=float,
        default=DEFAULT_FORGET_FRACTION,
        help="Fraction of data to forget (default: {})".format(DEFAULT_FORGET_FRACTION)
    )
    parser.add_argument(
        "--num_forget_subsets",
        type=int,
        default=None,
        help="Number of forget subsets for multi-subset experiments (default: None, uses single forget directory)"
    )
    parser.add_argument(
        "--primary_subset_weight",
        type=float,
        default=None,
        help="Weight for primary forget subset when using multiple subsets. If not specified, uses uniform selection (default: None for uniform)"
    )
    parser.add_argument(
        "--data_subfolder",
        type=str,
        required=True,
        help="Data subfolder name (must be provided)"
    )
    parser.add_argument(
        "--single_forget_subset",
        action="store_true",
        help="If set, randomly picks ONE forget subset uniformly and uses only that subset (default: False, uses weighted combination)"
    )
    parser.add_argument(
        "--unlearn_seed_offset",
        type=int,
        default=10000,
        help="Offset to add to run_index for unlearn_seed (default: 10000). Unlearn seed = run_index + unlearn_seed_offset"
    )
    parser.add_argument(
        "--shuffle_train_samples",
        action="store_true",
        help="Shuffle at sample level once (reproducible via rng_seed); batch order same every epoch.",
    )
    parser.add_argument(
            "--shuffle_train_batches",
            action="store_true",
            help="Shuffle the order of the loaded training batches once without shuffling samples.",
    )
    parser.add_argument(
        "--shuffle_train_batches_each_epoch",
        action="store_true",
        help="Shuffle batch order at the start of each epoch (different order every epoch; uses rng_seed).",
    )
    
    return parser.parse_args()


def main():
    # 1) Parse arguments
    args = parse_args()
    
    # 2) Extract parameters from arguments
    RESULTS_DIR = args.results_dir
    RUN_INDEX = args.run_index
    FORGET_FRACTION = args.forget_fraction
    NUM_FORGET_SUBSETS = args.num_forget_subsets
    PRIMARY_SUBSET_WEIGHT = args.primary_subset_weight
    DATA_SUBFOLDER = args.data_subfolder
    SINGLE_FORGET_SUBSET = args.single_forget_subset
    UNLEARN_SEED_OFFSET = args.unlearn_seed_offset
    SHUFFLE_TRAIN_SAMPLES = args.shuffle_train_samples
    SHUFFLE_TRAIN_BATCHES_EACH_EPOCH = args.shuffle_train_batches_each_epoch
    SHUFFLE_TRAIN_BATCHES = args.shuffle_train_batches
    
    # Calculate unlearn seed: use run_index + offset for independent randomness in unlearning
    UNLEARN_SEED = RUN_INDEX + UNLEARN_SEED_OFFSET

    # Redirect stdout/stderr immediately so load_config, wandb.init, etc. don't print to terminal.
    # We don't have run_dir yet (it needs wandb.run.id), so write to a pre-log, then merge into run_dir/run.log.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pre_path = os.path.join(RESULTS_DIR, f"_run_{RUN_INDEX}_pre.log")
    pre_file = open(pre_path, "w", buffering=1)
    sys.stdout = pre_file
    sys.stderr = pre_file
    os.dup2(pre_file.fileno(), 1)
    os.dup2(pre_file.fileno(), 2)

    # 3) Load config
    config = load_config(args.config)

    if DATA_SUBFOLDER is None:
        raise ValueError("data_subfolder argument is required (e.g. 'cifar10_bs_128')")

    # 4) Initialize W&B (its "Tracking run" etc. now go to pre_file, not terminal)
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config,
        dir=config["wandb"]["log_dir"],
    )

    # 5) Bookkeeping: per‐run directory, merge pre-log into run.log, keep redirect
    run_dir = os.path.join(RESULTS_DIR, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    pre_file.flush()
    with open(pre_path, "r") as f:
        pre_content = f.read()
    try:
        os.remove(pre_path)
    except OSError:
        pass

    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, "w", buffering=1)
    log_file.write(pre_content)
    log_file.flush()
    sys.stdout = log_file
    sys.stderr = log_file
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)

    # Mapping so parallel runner can find this run's log on failure
    with open(os.path.join(RESULTS_DIR, f"_run_{RUN_INDEX}.dir"), "w") as f:
        f.write(run_dir)
    print("\n=== STARTING RUN {} ===\n".format(RUN_INDEX))

    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    
    # Also save config in the main results directory for easy debugging
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main_config_path = os.path.join(RESULTS_DIR, "config.yaml")
    if not os.path.exists(main_config_path):
        shutil.copy(args.config, main_config_path)

    checkpoint_path = os.path.join(run_dir, "ckpt")
    os.makedirs(checkpoint_path, exist_ok=True)
    config["checkpoint_path"] = checkpoint_path

    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)

    # 7) Pick shared data_split root: local under results_dir first, else fallback location
    DATA_SPLIT_ROOT = _resolve_data_split_root(RESULTS_DIR, DATA_SUBFOLDER)
    print(f"Using data_split root: {DATA_SPLIT_ROOT}")
    print(f"Using data subfolder: {DATA_SUBFOLDER}")
    print(f"Shuffle train samples: {SHUFFLE_TRAIN_SAMPLES}")
    print(f"Shuffle train batches each epoch: {SHUFFLE_TRAIN_BATCHES_EACH_EPOCH}")
    print(f"Shuffle train batches: {SHUFFLE_TRAIN_BATCHES}")

    # 8) Load & sample splits for this run
    (
        train_loader,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        chosen_data,
    ) = load_cifar_splits_with_batch_subset(
        run_dir=DATA_SPLIT_ROOT,
        batch_size=config["dataset"]["batch_size"],
        forget_fraction=FORGET_FRACTION,
        rng_seed=RUN_INDEX,
        num_forget_subsets=NUM_FORGET_SUBSETS,
        primary_subset_weight=PRIMARY_SUBSET_WEIGHT,
        data_subfolder=DATA_SUBFOLDER,
        single_forget_subset=SINGLE_FORGET_SUBSET,
        shuffle_train_samples=SHUFFLE_TRAIN_SAMPLES,
        shuffle_train_batches_each_epoch=SHUFFLE_TRAIN_BATCHES_EACH_EPOCH,
        shuffle_train_batches=SHUFFLE_TRAIN_BATCHES,
    )

    # 9) Save chosen data (format depends on data type)
    if NUM_FORGET_SUBSETS is not None:
        # Synthetic data: save as JSON
        chosen_file = os.path.join(run_dir, "chosen_forget_batches.json")
        with open(chosen_file, "w") as f:
            # Convert set to list if needed, otherwise save dict directly
            if isinstance(chosen_data, set):
                json.dump(list(chosen_data), f, indent=2)
            else:
                json.dump(chosen_data, f, indent=2)
    else:
        # CIFAR data: save as numpy array
        chosen_file = os.path.join(run_dir, "chosen_forget_batches.npy")
        np.save(chosen_file, chosen_data)
    
    wandb.save(chosen_file)

    vars_file = os.path.join(run_dir, "run_vars.json")
    with open(vars_file, "w") as vf:
        json.dump({
            "forget_fraction": FORGET_FRACTION,
            "run_index": RUN_INDEX,
            "unlearn_seed": UNLEARN_SEED,
            "unlearn_seed_offset": UNLEARN_SEED_OFFSET,
            "num_forget_subsets": NUM_FORGET_SUBSETS,
            "primary_subset_weight": PRIMARY_SUBSET_WEIGHT,
            "data_subfolder": DATA_SUBFOLDER,
            "single_forget_subset": SINGLE_FORGET_SUBSET,
            "shuffle_train_samples": SHUFFLE_TRAIN_SAMPLES,
            "shuffle_train_batches_each_epoch": SHUFFLE_TRAIN_BATCHES_EACH_EPOCH,
            "shuffle_train_batches": SHUFFLE_TRAIN_BATCHES,
        }, vf, indent=2)
    wandb.save(vars_file)

    # 10) Build model & trainer
    model = ModelFactory.create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["n_classes"],
    )
    
    # Use global_seed for training (consistent across runs if same global_seed)
    train_key = jax.random.PRNGKey(config["global_seed"])

    trainer = Trainer(
        config=config,
        model=model,
        key=train_key,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        forget_dataloader=forget_loader,
        retain_dataloader=retain_loader,
        use_pretrained=config["training"]["pretrained"],
        run_dir=run_dir,
    )

    # 11) Train with training seed
    state = trainer.fit()

    # 12) Unlearn
    if config["unlearning"]["algorithm"] in ("ascent_descent", "retain_finetune"):
        from src.training.unlearn_ascent_descent import unlearn_ascent_descent
        from src.training import unlearn_retain_finetune as retain_finetune_module

        if config["unlearning"]["algorithm"] == "ascent_descent":
            state = unlearn_ascent_descent(
                state=state,
                forget_dataloader=forget_loader,
                retain_dataloader=retain_loader,
                config=config,
                val_dataloader=val_loader,
                validate_fn=trainer.validate,
                save_checkpoint_fn=lambda s, n, **kw: trainer.save_checkpoint(
                    s, n, unlearn=True, **kw
                ),
                run_dir=run_dir,
            )
        else:
            state = retain_finetune_module.unlearn_retain_finetune(
                state=state,
                forget_dataloader=forget_loader,
                retain_dataloader=retain_loader,
                config=config,
                val_dataloader=val_loader,
                validate_fn=trainer.validate,
                save_checkpoint_fn=lambda s, n, **kw: trainer.save_checkpoint(
                    s, n, unlearn=True, **kw
                ),
                run_dir=run_dir,
            )
    else:
        # Update trainer key to use independent unlearn seed for unlearning
        unlearn_key = jax.random.PRNGKey(UNLEARN_SEED)
        trainer.key = unlearn_key
        logger.info(
            "Using independent unlearn seed: %s (run_index=%s + offset=%s)",
            UNLEARN_SEED, RUN_INDEX, UNLEARN_SEED_OFFSET,
        )
        state = trainer.unlearn(state)

    # 13) Test
    trainer.test(state=state)

    # Clean up run_index mapping (used by parallel runner to find log on failure)
    try:
        os.remove(os.path.join(RESULTS_DIR, f"_run_{RUN_INDEX}.dir"))
    except OSError:
        pass


if __name__ == "__main__":
    main()
