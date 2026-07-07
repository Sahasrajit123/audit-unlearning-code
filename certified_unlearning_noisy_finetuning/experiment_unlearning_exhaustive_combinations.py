# -*- coding: utf-8 -*-
"""
Exhaustive forget-combination runs: when the number of forget batches n is small,
iterate over all C(n,k) combinations. For each combination:
  - 1 shared training run (fixed global_seed, chosen_idx => identical across runs)
  - num_unlearn_per_combo independent unlearning trials (different seeds), each
    saved to ckpt_trial_0/, ckpt_trial_1/, ...

Requires num_forget_subsets is None (single forget directory). Uses
load_cifar_splits_with_batch_subset(..., chosen_idx_override=...).

Backward compatible: does not modify experiment_unlearning_random_forget_main.py
or existing launchers.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
import shutil
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

logging.getLogger("absl").setLevel(logging.ERROR)

import jax
import numpy as np
import wandb

from src.models.model import ModelFactory
from src.training.trainer import Trainer
from src.utils.utils import load_config, logger
from src.utils.data_cache import load_cifar_splits_with_batch_subset

# ---------- Defaults ---------------------------------------------------------
DEFAULT_FORGET_FRACTION = 0.50
DEFAULT_NUM_UNLEARN_PER_COMBO = 5
DEFAULT_UNLEARN_SEED_OFFSET = 10000
DEFAULT_EXHAUSTIVE_MAX_COMBOS = 500

n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -----------------------------------------------------------------------------


def _resolve_root(results_dir: str, data_subfolder: str, config_path: str = None) -> Path:
    """Return path for data_split/<data_subfolder>.
    
    data_subfolder must be explicitly specified (e.g., 'cifar_750' for CIFAR-10 or 'cifar100_bs_128' for CIFAR-100).
    
    Lookup order:
    1. Try: {results_dir}/data_split/{data_subfolder}
        2. If not found, try fallback(s) based on data_subfolder naming:
             - CIFAR-100 (if "cifar100" in data_subfolder): data/cifar100/data_split/{data_subfolder}
             - CIFAR-10 (if "cifar10" in data_subfolder): data/cifar10/data_split/{data_subfolder}
         - Otherwise: raise ValueError (unsupported data_subfolder naming)
    """
    if not data_subfolder:
        raise ValueError("data_subfolder must be explicitly specified (e.g., 'cifar_750', 'cifar100_bs_128')")
    
    ds_lower = data_subfolder.lower()
    # Primary location: {results_dir}/data_split/{data_subfolder}
    primary_root = Path(results_dir) / "data_split" / data_subfolder
    primary_forget_dir = primary_root / "forget"
    if primary_forget_dir.is_dir():
        return primary_root

    from src.utils.utils import load_config
    if config_path is None:
        raise FileNotFoundError(f"Could not find config file to determine dataset fallback for {data_subfolder}.")
    config = load_config(config_path)
    dataset_name = config.get("dataset", {}).get("name", "").lower()

    checked_fallbacks = []
    if dataset_name == "cifar100":
        fallback_bases = ["data/cifar100"]
    elif dataset_name == "cifar10":
        fallback_bases = ["data/cifar10"]
    else:
        raise ValueError(f"Unsupported dataset name '{dataset_name}' in config. Only 'cifar10' or 'cifar100' supported.")

    for base in fallback_bases:
        fallback_root = Path(base) / "data_split" / data_subfolder
        fallback_forget_dir = fallback_root / "forget"
        checked_fallbacks.append(str(fallback_forget_dir))
        if fallback_forget_dir.is_dir():
            return fallback_root

    # Neither location found
    raise FileNotFoundError(
        f"Forget dir not found in either location:\n"
        f"  Primary: {primary_forget_dir}\n"
        f"  Fallback(s):\n    " + "\n    ".join(checked_fallbacks) + "\n"
        f"Ensure data_split/{data_subfolder}/forget/ exists in either location."
    )


def _compute_num_combos(results_dir: str, forget_fraction: float, data_subfolder: str, config_path: str = None) -> Tuple[int, int, int]:
    """Return (n_forget, k_forget, num_combos)."""
    # config_path is required for fallback logic
    if config_path is None:
        raise FileNotFoundError("config_path must be provided to _compute_num_combos. No fallback or sys.argv search will be performed.")
    root = _resolve_root(results_dir, data_subfolder, config_path)
    forget_dir = root / "forget"
    forget_files = sorted(forget_dir.glob("batch_*.pkl"))
    n_forget = len(forget_files)
    k_forget = int(round(forget_fraction * n_forget))
    k_forget = max(0, min(k_forget, n_forget))
    num_combos = math.comb(n_forget, k_forget) if k_forget <= n_forget else 0
    return n_forget, k_forget, num_combos


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Exhaustive forget-combination runs: 1 train + N unlearn trials per combo.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (required unless --print_num_combos).")
    parser.add_argument("--results_dir", type=str, required=True, help="Base directory for runs.")
    parser.add_argument("--combo_idx", type=int, default=0, help="Combo index in [0, num_combos-1]. Ignored if --print_num_combos.")
    parser.add_argument("--forget_fraction", type=float, default=DEFAULT_FORGET_FRACTION)
    parser.add_argument("--data_subfolder", type=str, required=True, help="Data subfolder name (e.g., 'cifar_750', 'cifar100_bs_128'). REQUIRED.")
    parser.add_argument("--num_unlearn_per_combo", type=int, default=DEFAULT_NUM_UNLEARN_PER_COMBO)
    parser.add_argument("--unlearn_seed_offset", type=int, default=DEFAULT_UNLEARN_SEED_OFFSET)
    parser.add_argument("--exhaustive_max_combos", type=int, default=DEFAULT_EXHAUSTIVE_MAX_COMBOS)
    parser.add_argument("--print_num_combos", action="store_true", help="Print num_combos and exit.")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--shuffle_train_samples",
        action="store_true",
        help="Shuffle at sample level once (reproducible via rng_seed); batch order same every epoch.",
    )
    parser.add_argument(
        "--shuffle_train_batches_each_epoch",
        action="store_true",
        help="Shuffle batch order at the start of each epoch (different order every epoch; uses rng_seed).",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --print_num_combos: no wandb, no config load
    if args.print_num_combos:
        n_forget, k_forget, num_combos = _compute_num_combos(
            args.results_dir, args.forget_fraction, args.data_subfolder, args.config
        )

        ##raise ValueError("Came here for testing") 
        print(num_combos)
        return num_combos

    if not args.config:
        raise ValueError("--config is required when not using --print_num_combos")
    config = load_config(args.config)
    n_forget, k_forget, num_combos = _compute_num_combos(
        args.results_dir, args.forget_fraction, args.data_subfolder, args.config
    )
    if num_combos > args.exhaustive_max_combos:
        raise ValueError(
            f"num_combos={num_combos} > exhaustive_max_combos={args.exhaustive_max_combos}. "
            "Use smaller forget_fraction or increase --exhaustive_max_combos."
        )
    if args.combo_idx < 0 or args.combo_idx >= num_combos:
        raise ValueError(f"combo_idx={args.combo_idx} must be in [0, {num_combos}-1].")

    combo_idx = args.combo_idx
    RESULTS_DIR = args.results_dir
    FORGET_FRACTION = args.forget_fraction
    DATA_SUBFOLDER = args.data_subfolder
    NUM_UNLEARN_PER_COMBO = args.num_unlearn_per_combo
    UNLEARN_SEED_OFFSET = args.unlearn_seed_offset
    # Data_split base: get parent of resolved root (e.g., data/cifar100/data_split)
    # since load_cifar_splits_with_batch_subset appends data_subfolder to run_dir
    data_split_base = _resolve_root(RESULTS_DIR, DATA_SUBFOLDER, args.config).parent

    # Enumerate combo
    all_combos = list(itertools.combinations(range(n_forget), k_forget))
    chosen_idx = np.array(sorted(all_combos[combo_idx]), dtype=np.int64)

    # Redirect logs (same pattern as experiment_unlearning_random_forget_main)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pre_path = os.path.join(RESULTS_DIR, f"_run_{combo_idx}_pre.log")
    pre_file = open(pre_path, "w", buffering=1)
    sys.stdout = pre_file
    sys.stderr = pre_file
    os.dup2(pre_file.fileno(), 1)
    os.dup2(pre_file.fileno(), 2)

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config,
        dir=config["wandb"]["log_dir"],
    )

    run_dir = os.path.join(RESULTS_DIR, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    pre_file.flush()
    pre_file.close()
    with open(pre_path, "r") as f:
        pre_content = f.read()
    try:
        os.remove(pre_path)
    except OSError:
        pass

    train_log_path = os.path.join(run_dir, "run_train.log")
    train_log_file = open(train_log_path, "w", buffering=1)
    train_log_file.write(pre_content)
    train_log_file.flush()
    sys.stdout = train_log_file
    sys.stderr = train_log_file
    os.dup2(train_log_file.fileno(), 1)
    os.dup2(train_log_file.fileno(), 2)

    with open(os.path.join(RESULTS_DIR, f"_run_{combo_idx}.dir"), "w") as f:
        f.write(run_dir)
    print("\n=== EXHAUSTIVE COMBO (combo_idx={}) ===\n".format(combo_idx))

    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main_config_path = os.path.join(RESULTS_DIR, "config.yaml")
    if not os.path.exists(main_config_path):
        shutil.copy(args.config, main_config_path)

    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    config["checkpoint_path"] = ckpt_dir

    if args.debug >= 1:
        logger.setLevel(logging.DEBUG)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data with explicit chosen_idx (num_forget_subsets stays None)
    (
        train_loader,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        chosen_data,
    ) = load_cifar_splits_with_batch_subset(
        run_dir=str(data_split_base),
        batch_size=config["dataset"]["batch_size"],
        forget_fraction=FORGET_FRACTION,
        rng_seed=0,
        data_subfolder=DATA_SUBFOLDER,
        chosen_idx_override=chosen_idx,
        shuffle_train_samples=args.shuffle_train_samples,
        shuffle_train_batches_each_epoch=args.shuffle_train_batches_each_epoch,
    )

    # chosen_data is chosen_idx (CIFAR path when num_forget_subsets is None)
    chosen_file = os.path.join(run_dir, "chosen_forget_batches.npy")
    np.save(chosen_file, chosen_data)
    wandb.save(chosen_file)

    # Unlearn seeds: unique per (combo_idx, trial)
    unlearn_seeds = [UNLEARN_SEED_OFFSET + combo_idx * 1000 + j for j in range(NUM_UNLEARN_PER_COMBO)]

    run_vars = {
        "forget_fraction": FORGET_FRACTION,
        "combo_idx": combo_idx,
        "chosen_idx": chosen_idx.tolist(),
        "n_forget": n_forget,
        "k_forget": k_forget,
        "num_combos": num_combos,
        "num_unlearn_per_combo": NUM_UNLEARN_PER_COMBO,
        "unlearn_seed_offset": UNLEARN_SEED_OFFSET,
        "data_subfolder": DATA_SUBFOLDER,
        "mode": "exhaustive_combinations",
        "unlearn_seeds": unlearn_seeds,
        "shuffle_train_samples": args.shuffle_train_samples,
        "shuffle_train_batches_each_epoch": args.shuffle_train_batches_each_epoch,
    }
    with open(os.path.join(run_dir, "run_vars.json"), "w") as vf:
        json.dump(run_vars, vf, indent=2)
    wandb.save(os.path.join(run_dir, "run_vars.json"))

    # ----- 1) Train once (shared for this combo) -----
    model = ModelFactory.create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["n_classes"],
    )
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

    state = trainer.fit()
    train_epochs = config["training"]["epochs"]
    trainer.test(state)  # Test accuracy after training, before unlearning

    # ----- 2) Multiple unlearn trials (ckpt_trial_0, ckpt_trial_1, ...) -----
    for j in range(NUM_UNLEARN_PER_COMBO):
        ckpt_trial_j = os.path.join(run_dir, f"ckpt_trial_{j}")
        os.makedirs(ckpt_trial_j, exist_ok=True)

        # Per-trial log: run_unlearn_train_{j}.log
        trial_log_path = os.path.join(run_dir, f"run_unlearn_trial_{j}.log")
        trial_log_file = open(trial_log_path, "w", buffering=1)
        prev_stdout, prev_stderr = sys.stdout, sys.stderr
        sys.stdout = trial_log_file
        sys.stderr = trial_log_file
        os.dup2(trial_log_file.fileno(), 1)
        os.dup2(trial_log_file.fileno(), 2)

        config_trial = {**config, "checkpoint_path": ckpt_trial_j}
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
            run_dir=run_dir,
        )

        # Load trained state from ckpt/ (final epoch)
        state_template = trainer_j.train_setup()
        state_loaded = Trainer.load_checkpoint(ckpt_dir, state_template, train_epochs)

        if config["unlearning"]["algorithm"] in ("ascent_descent", "retain_finetune"):
            from src.training.unlearn_ascent_descent import unlearn_ascent_descent
            from src.training import unlearn_retain_finetune as retain_finetune_module

            if config["unlearning"]["algorithm"] == "ascent_descent":
                state_loaded = unlearn_ascent_descent(
                    state=state_loaded,
                    forget_dataloader=forget_loader,
                    retain_dataloader=retain_loader,
                    config=config_trial,
                    val_dataloader=val_loader,
                    validate_fn=trainer_j.validate,
                    save_checkpoint_fn=lambda s, n, **kw: trainer_j.save_checkpoint(s, n, unlearn=True, **kw),
                    run_dir=run_dir,
                )
                print("--- Trial {} unlearning complete (ascent_descent). See above for metrics. ---".format(j))
            else:
                state_loaded = retain_finetune_module.unlearn_retain_finetune(
                    state=state_loaded,
                    forget_dataloader=forget_loader,
                    retain_dataloader=retain_loader,
                    config=config_trial,
                    val_dataloader=val_loader,
                    validate_fn=trainer_j.validate,
                    save_checkpoint_fn=lambda s, n, **kw: trainer_j.save_checkpoint(s, n, unlearn=True, **kw),
                    run_dir=run_dir,
                )
                print("--- Trial {} unlearning complete (retain_finetune). See above for metrics. ---".format(j))
        else:
            logger.info("Trial %s: unlearn_seed=%s", j, unlearn_seeds[j])
            state_loaded = trainer_j.unlearn(state_loaded)
            # One-line summary: unlearning stop (trainer already logged details above)
            print("--- Trial {} unlearning stopped. See above for stop step/reason and metrics. ---".format(j))

        trainer_j.test(state=state_loaded)

        trial_log_file.flush()
        trial_log_file.close()
        sys.stdout = train_log_file
        sys.stderr = train_log_file
        os.dup2(train_log_file.fileno(), 1)
        os.dup2(train_log_file.fileno(), 2)
        print("Trial {} done. Logs in {}.".format(j, trial_log_path))

    # Cleanup
    try:
        os.remove(os.path.join(RESULTS_DIR, f"_run_{combo_idx}.dir"))
    except OSError:
        pass


if __name__ == "__main__":
    main()
