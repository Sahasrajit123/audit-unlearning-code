#!/usr/bin/env python3
"""
single_run_unlearning.py - Execute a single unlearning run

This script handles one run with a specific run_id and GPU assignment.
Called by the orchestrator for parallel execution.

Usage:
  python single_run_unlearning.py --run_id 0 --experiment_config config.json --gpu 0
"""

import argparse
import json
import os
import random
import shutil
import time
import logging
import sys
from pathlib import Path
import numpy as np
import torch

from data_loader import load_data_splits
from trainer_utils import (
    load_config,
    get_device,
    build_model,
    build_optimizer,
    build_scheduler,
)
from engine import evaluate


def setup_run_logger(run_dir, logger_name="run_logger"):
    """Setup logger for a run."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    log_file = run_dir / "run.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_experiment_config(config_path):
    """Load experiment config from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_text_file(path):
    """Load text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text_file(path, text):
    """Save text to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_forget_files(data_dir):
    """Get all forget_*.txt files in order. Supports both structures:
    - Old: data_dir/forget_0.txt, forget_1.txt, etc.
    - New: data_dir/forget/forget_0.txt, etc.
    Returns relative paths that work with Path(data_dir) / fname"""
    data_path = Path(data_dir)

    # Try new structure: forget/ subdirectory
    forget_dir = data_path / "forget"
    if forget_dir.exists() and forget_dir.is_dir():
        forget_files = sorted([p.name for p in forget_dir.glob("forget_*.txt")])
        if forget_files:
            return ["forget/" + f for f in forget_files]

    # Fallback to old structure: top-level forget_*.txt files
    forget_files = sorted([p.name for p in data_path.glob("forget_*.txt")])
    if not forget_files:
        if (data_path / "forget.txt").exists():
            forget_files = ["forget.txt"]
    return forget_files


def setup_run_directory(run_dir, run_config):
    """Create run directory and save configuration."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    with open(str(config_path), "w") as f:
        json.dump(run_config, f, indent=2)


def compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay=0.0):
    """Compute CE loss + L2 weight decay penalty."""
    # Handle case where model returns tuple (logits, hidden_state) or similar
    if isinstance(logits, tuple):
        logits = logits[0]

    ce_loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
    if weight_decay > 0:
        l2_penalty = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
        return ce_loss + weight_decay * l2_penalty
    return ce_loss


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return None


def train_initial_phase(model, train_loader, val_loader, criterion, device, cfg, run_dir, logger, retain_loader, forget_loader):
    """Phase 1: Train model on combined (retain + forget) data. Common to all strategies."""
    logger.info("="*80)
    logger.info("Phase 1: Training on combined data...")
    logger.info("="*80)

    weight_decay = cfg.get("weight_decay", 0.0)
    logger.info("\nPhase 1 Config: epochs={}, lr={}, batch_size={}, weight_decay={}".format(
        cfg['epochs'], cfg['lr'], cfg['batch_size'], weight_decay))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_epochs=cfg["epochs"])
    best_eval_acc = 0.0
    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss, train_acc, num_batches = 0.0, 0.0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            # Handle tuple output from model
            logits = output[0] if isinstance(output, tuple) else output
            loss = compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
            optimizer.step()
            train_loss += loss.item()
            acc = (logits.argmax(-1) == batch_y).float().mean().item()
            train_acc += acc
            num_batches += 1

        train_loss /= num_batches
        train_acc /= num_batches

        model.eval()
        val_loss, val_acc, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                # Handle tuple output from model
                logits = output[0] if isinstance(output, tuple) else output
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                val_loss += loss.item()
                acc = (logits.argmax(-1) == batch_y).float().mean().item()
                val_acc += acc
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        best_eval_acc = max(best_eval_acc, val_acc)
        torch.save(model.state_dict(), str(run_dir / "model_trained.pt"))
        current_lr = get_learning_rate(optimizer)
        logger.info("Epoch {:3d} | LR: {:.6f} | TrLoss: {:.4f} | TrAcc: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED_LATEST".format(
            epoch, current_lr, train_loss, train_acc, val_loss, val_acc))

        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

    phase1_time = time.time() - start_time
    logger.info("\nPhase 1 completed in {:.1f}s, best_val_acc={:.4f}\n".format(phase1_time, best_eval_acc))

    # Evaluate on forget, retain, and cumulative train sets
    model.load_state_dict(torch.load(str(run_dir / "model_trained.pt"), map_location=device))
    model.eval()

    forget_loss, forget_acc, forget_ppl = evaluate(model, forget_loader, criterion, device)
    retain_loss, retain_acc, retain_ppl = evaluate(model, retain_loader, criterion, device)
    combined_loss, combined_acc, combined_ppl = evaluate(model, train_loader, criterion, device)

    logger.info("\nPhase 1 - Val Metrics on Data Splits:")
    logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(forget_loss, forget_acc, forget_ppl))
    logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(retain_loss, retain_acc, retain_ppl))
    logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(combined_loss, combined_acc, combined_ppl))

    return model, phase1_time


from unlearning_scripts.finetune_retain_unlearning import unlearn_finetune_retain




from unlearning_scripts.ascent_descent_unlearning import unlearn_ascent_descent
from unlearning_scripts.hessian_unlearning import unlearn_hessian_style


def run_single_experiment(run_id, run_dir, exp_config, device):
    """Run a single unlearning experiment."""

    logger = setup_run_logger(run_dir, logger_name="run_{}".format(run_id))

    logger.info("="*80)
    logger.info("RUN {}: Starting on {}".format(run_id, device))
    logger.info("="*80)

    # Setup config for this run
    training_seed = exp_config["training"]["seed"]
    forget_sampling_seed = training_seed + 1000 + run_id
    unlearning_seed = training_seed + 2000 + run_id

    # Set training randomness (IDENTICAL for all runs)
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    random.seed(training_seed)

    cfg = json.loads(json.dumps(exp_config["training"]))
    exp_cfg = exp_config.get("experiment", {})
    strategy = exp_cfg["strategy"]
    forget_prob = exp_cfg["forget_prob"]
    dataset = exp_cfg["dataset"]

    logger.info("\nRun Configuration:")
    logger.info("  Strategy: {}".format(strategy))
    logger.info("  Forget prob: {}".format(forget_prob))
    logger.info("  Training seed: {} (FIXED)".format(training_seed))
    logger.info("  Forget sampling seed: {}".format(forget_sampling_seed))
    logger.info("  Unlearning seed: {}".format(unlearning_seed))

    # Load data (using FIXED training seed)
    loaders, metadata = load_data_splits(
        dataset,
        cfg.get("batch_size", 256),
        training_seed,
        shuffle_train_samples=True,
        shuffle_train_batches_each_epoch=False,
    )

    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Load forget sets
    forget_files = get_forget_files(dataset)
    forget_texts = [load_text_file(str(Path(dataset) / fname)) for fname in forget_files]
    retain_text = load_text_file(str(Path(dataset) / "retain.txt"))

    # Sample forget batches (using RUN-SPECIFIC seed)
    np.random.seed(forget_sampling_seed)
    random.seed(forget_sampling_seed)
    forget_indices = sorted(list(np.random.choice(
        len(forget_texts),
        size=max(1, int(forget_prob * len(forget_texts))),
        replace=False
    )))

    # Restore training seed after forget sampling
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    random.seed(training_seed)

    sampled_forget_text = "\n".join([forget_texts[i] for i in forget_indices])
    combined_text = retain_text + "\n" + sampled_forget_text

    logger.info("\nDataset:")
    logger.info("  Forget indices: {}".format(forget_indices))
    logger.info("  Combined train size: {:,} chars".format(len(combined_text)))

    # Create temp data directory
    temp_data_dir = run_dir / "data_temp"
    temp_data_dir.mkdir(exist_ok=True)

    shutil.copy(str(Path(dataset) / "meta.json"), str(temp_data_dir / "meta.json"))
    shutil.copy(str(Path(dataset) / "retain.txt"), str(temp_data_dir / "retain.txt"))
    shutil.copy(str(Path(dataset) / "val.txt"), str(temp_data_dir / "val.txt"))
    shutil.copy(str(Path(dataset) / "test.txt"), str(temp_data_dir / "test.txt"))
    save_text_file(str(temp_data_dir / "train.txt"), combined_text)
    save_text_file(str(temp_data_dir / "forget.txt"), sampled_forget_text)

    # Copy all original forget_*.txt files (for data_loader compatibility)
    for forget_file in get_forget_files(dataset):
        src = Path(dataset) / forget_file
        dst = temp_data_dir / forget_file
        dst.parent.mkdir(parents=True, exist_ok=True)  # Create directories if needed
        shutil.copy(str(src), str(dst))

    # Load loaders
    train_loaders, _ = load_data_splits(str(temp_data_dir), cfg.get("batch_size", 256), training_seed)
    train_loader = train_loaders["train"]
    retain_loader = train_loaders["retain"]
    forget_loader = train_loaders.get("forget", train_loader)

    # Build model
    vocab_size = metadata["vocab_size"]
    model = build_model(vocab_size, exp_config["model"], device, compile=False)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("\nStarting training...")

    # Phase 1: Train on combined data (common for all strategies)
    model, phase1_time = train_initial_phase(model, train_loader, val_loader, criterion, device, cfg, run_dir, logger, retain_loader, forget_loader)

    logger.info("\nStarting unlearning...")

    # Phase 2: Apply unlearning strategy
    unlearning_config = exp_config.get("unlearning", {})

    if strategy == "finetune_retain":
        model, metrics = unlearn_finetune_retain(
            model, retain_loader, val_loader, train_loader, forget_loader, criterion, device, run_dir, logger,
            unlearning_cfg=unlearning_config
        )
    elif strategy == "ascent_descent":
        model, metrics = unlearn_ascent_descent(
            model, forget_loader, retain_loader, train_loader, val_loader, criterion, device, run_dir, logger,
            unlearning_cfg=unlearning_config
        )
    elif strategy == "hessian_unlearning":
        # Use unlearning_seed as forget_seed for reproducibility
        model, metrics = unlearn_hessian_style(
            model, forget_loader, retain_loader, train_loader, val_loader, criterion, device, run_dir, logger,
            unlearning_seed,
            unlearning_cfg=unlearning_config
        )
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

    metrics["phase_1_duration"] = phase1_time

    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)
    model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))
    test_loss, test_acc, test_ppl = evaluate(model, test_loader, criterion, device)

    logger.info("Test loss:      {:.4f}".format(test_loss))
    logger.info("Test accuracy:  {:.4f}".format(test_acc))
    logger.info("Test perplexity: {:.2f}".format(test_ppl))

    # Save metrics
    metrics["test_loss"] = test_loss
    metrics["test_accuracy"] = test_acc
    metrics["test_perplexity"] = test_ppl
    metrics["forget_indices"] = [int(i) for i in forget_indices]  # Convert int64 to int

    metrics_path = run_dir / "metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save forget indices separately
    forget_indices_path = run_dir / "forget_indices.json"
    with open(str(forget_indices_path), "w") as f:
        json.dump({"forget_indices": [int(i) for i in forget_indices], "forget_prob": forget_prob}, f, indent=2)

    # Cleanup
    shutil.rmtree(str(temp_data_dir))

    logger.info("")
    logger.info("="*80)
    logger.info("RUN {}: COMPLETED".format(run_id))
    logger.info("="*80)
    logger.info("Results: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(test_loss, test_acc, test_ppl))

    return True


def main():
    parser = argparse.ArgumentParser(description="Execute a single unlearning run")
    parser.add_argument("--run_id", type=int, required=True, help="Run ID")
    parser.add_argument("--experiment_config", type=str, required=True, help="Experiment config JSON")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")

    args = parser.parse_args()

    # Load config
    exp_config = load_experiment_config(args.experiment_config)

    # Extract experiment settings
    exp_cfg = exp_config.get("experiment", {})
    run_folder = Path(exp_cfg["run_folder"])
    run_dir = run_folder / "run_{}".format(args.run_id)

    # Compute seeds for this run
    training_seed = exp_config["training"]["seed"]
    forget_sampling_seed = training_seed + 1000 + args.run_id
    unlearning_seed = training_seed + 2000 + args.run_id

    # Setup run directory and config
    setup_run_directory(run_dir, {
        "run_id": args.run_id,
        "gpu": args.gpu,
        "strategy": exp_cfg["strategy"],
        "forget_prob": exp_cfg["forget_prob"],
        "training_seed": training_seed,
        "forget_sampling_seed": forget_sampling_seed,
        "unlearning_seed": unlearning_seed,
    })

    # Get device
    if args.gpu is not None:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logger early for error handling
    logger = setup_run_logger(run_dir, logger_name=f"run_{args.run_id}")

    # Run
    try:
        run_single_experiment(args.run_id, run_dir, exp_config, device)
    except Exception as e:
        import traceback
        # Log error to run.log
        logger.error("Exception occurred: {}".format(e))
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
