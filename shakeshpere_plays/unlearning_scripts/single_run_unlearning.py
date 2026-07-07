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

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

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
    """Get all forget_*.txt files in order."""
    data_path = Path(data_dir)
    forget_files = sorted([p.name for p in data_path.glob("forget_*.txt")])
    if not forget_files:
        forget_files = ["forget.txt"] if (data_path / "forget.txt").exists() else []
    return forget_files


def setup_run_directory(run_dir, run_config):
    """Create run directory and save configuration."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    with open(str(config_path), "w") as f:
        json.dump(run_config, f, indent=2)


def compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay=0.0):
    """Compute CE loss + L2 weight decay penalty."""
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


def train_finetune_retain(model, train_loader, retain_loader, val_loader, criterion, device, cfg, run_dir, logger):
    """Strategy 1: Train on combined, then finetune on retain."""
    logger.info("="*80)
    logger.info("Strategy: finetune_retain - Starting training...")
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
            logits = model(batch_x)
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
                logits = model(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                val_loss += loss.item()
                acc = (logits.argmax(-1) == batch_y).float().mean().item()
                val_acc += acc
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        if val_acc > best_eval_acc:
            best_eval_acc = val_acc
            torch.save(model.state_dict(), str(run_dir / "model_trained.pt"))
            current_lr = get_learning_rate(optimizer)
            logger.info("Epoch {:3d} | LR: {:.6f} | TrLoss: {:.4f} | TrAcc: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED".format(
                epoch, current_lr, train_loss, train_acc, val_loss, val_acc))
        else:
            current_lr = get_learning_rate(optimizer)
            logger.info("Epoch {:3d} | LR: {:.6f} | TrLoss: {:.4f} | TrAcc: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f}".format(
                epoch, current_lr, train_loss, train_acc, val_loss, val_acc))

        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

    phase1_time = time.time() - start_time
    logger.info("\nPhase 1 completed in {:.1f}s, best_val_acc={:.4f}\n".format(phase1_time, best_eval_acc))

    # Phase 2: Finetune on retain
    logger.info("Phase 2: Finetuning on retain only...")
    cfg_finetune = json.loads(json.dumps(cfg))
    cfg_finetune["epochs"] = max(1, cfg["epochs"] // 2)
    cfg_finetune["lr"] = cfg["lr"] / 2
    logger.info("Phase 2 Config: epochs={}, lr={}, weight_decay={}".format(
        cfg_finetune['epochs'], cfg_finetune['lr'], weight_decay))

    optimizer = build_optimizer(model, cfg_finetune)
    scheduler = build_scheduler(optimizer, cfg_finetune, num_epochs=cfg_finetune["epochs"])
    best_eval_acc_p2 = 0.0
    start_time = time.time()

    for epoch in range(1, cfg_finetune["epochs"] + 1):
        model.train()
        train_loss, train_acc, num_batches = 0.0, 0.0, 0

        for batch_x, batch_y in retain_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_finetune.get("clip", 1.0))
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
                logits = model(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                val_loss += loss.item()
                acc = (logits.argmax(-1) == batch_y).float().mean().item()
                val_acc += acc
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        if val_acc > best_eval_acc_p2:
            best_eval_acc_p2 = val_acc
            torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
            current_lr = get_learning_rate(optimizer)
            logger.info("Epoch {:3d} | LR: {:.6f} | TrLoss: {:.4f} | TrAcc: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED".format(
                epoch, current_lr, train_loss, train_acc, val_loss, val_acc))
        else:
            current_lr = get_learning_rate(optimizer)
            logger.info("Epoch {:3d} | LR: {:.6f} | TrLoss: {:.4f} | TrAcc: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f}".format(
                epoch, current_lr, train_loss, train_acc, val_loss, val_acc))

        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

    phase2_time = time.time() - start_time
    logger.info("\nPhase 2 completed in {:.1f}s, best_val_acc={:.4f}".format(phase2_time, best_eval_acc_p2))

    model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))

    metrics = {
        "strategy": "finetune_retain",
        "phase_1_best_val_acc": best_eval_acc,
        "phase_1_duration": phase1_time,
        "phase_2_best_val_acc": best_eval_acc_p2,
        "phase_2_duration": phase2_time,
    }

    return model, metrics


def train_ascent_descent(model, forget_loader, retain_loader, val_loader, criterion, device, cfg, run_dir, logger, unlearning_cfg=None):
    """Strategy 2: Ascent-descent unlearning."""
    logger.info("="*80)
    logger.info("Strategy: ascent_descent - Starting unlearning...")
    logger.info("="*80)

    if unlearning_cfg is None:
        unlearning_cfg = {}

    cfg_unlearn = json.loads(json.dumps(cfg))
    ad_config = unlearning_cfg.get("ascent_descent", {})

    num_epochs = int(ad_config.get("epochs", max(2, cfg["epochs"] // 2)))
    q = ad_config.get("q")
    lambda_coef = float(ad_config.get("lambda_coef", 0.5))
    forget_epochs_ratio = float(ad_config.get("forget_epochs_ratio", 0.5))
    forget_epochs = max(1, int(forget_epochs_ratio * num_epochs))
    weight_decay = float(ad_config.get("weight_decay", 0.0))

    # q=None triggers pure two-phase: ascent-only then descent-only
    pure_two_phase = q is None
    if not pure_two_phase:
        q = float(q)

    cfg_unlearn["epochs"] = num_epochs
    clip = cfg_unlearn.get("clip", 1.0)

    def _eval_val(model):
        val_loss, val_acc, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += criterion(logits.view(-1, logits.size(-1)), by.view(-1)).item()
                val_acc += (logits.argmax(-1) == by).float().mean().item()
                val_batches += 1
        return val_loss / val_batches, val_acc / val_batches

    def _retain_descent_epoch(model, optimizer, epoch_label):
        """Pure descent on retain set — shared between both modes."""
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch_x, batch_y in retain_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        model.eval()
        val_loss, val_acc = _eval_val(model)
        current_lr = get_learning_rate(optimizer)
        avg_loss = epoch_loss / steps if steps else 0
        logger.info("{} | LR: {:.6f} | Loss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED_LATEST".format(
            epoch_label, current_lr, avg_loss, val_loss, val_acc))
        return val_loss

    start_time = time.time()
    best_val_loss = float("inf")

    # ------------------------------------------------------------------ #
    # Pure two-phase mode: q=null -> ascent-only then descent-only        #
    # ------------------------------------------------------------------ #
    if pure_two_phase:
        ascent_lr = float(ad_config.get("ascent_lr", cfg_unlearn["lr"]))
        retain_epochs = num_epochs - forget_epochs

        logger.info("\nPure Ascent-Descent Config (q=None):")
        logger.info("  total_epochs={}, ascent_epochs={}, descent_epochs={}, ascent_lr={}, descent_lr={}, weight_decay={}".format(
            num_epochs, forget_epochs, retain_epochs, ascent_lr, cfg_unlearn["lr"], weight_decay))

        # Phase 1: gradient ascent on forget set
        logger.info("\n--- Phase 1: Gradient Ascent on Forget Set ({} epochs) ---".format(forget_epochs))
        cfg_ascent = dict(cfg_unlearn)
        cfg_ascent["lr"] = ascent_lr
        cfg_ascent["epochs"] = forget_epochs
        optimizer = build_optimizer(model, cfg_ascent)
        scheduler = build_scheduler(optimizer, cfg_ascent, num_epochs=forget_epochs)

        for epoch in range(1, forget_epochs + 1):
            model.train()
            epoch_loss, steps = 0.0, 0
            for batch_x, batch_y in forget_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay)
                (-loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            model.eval()
            val_loss, val_acc = _eval_val(model)
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
            current_lr = get_learning_rate(optimizer)
            logger.info("Ascent Epoch {:3d} | LR: {:.6f} | ForgetLoss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED_LATEST".format(
                epoch, current_lr, epoch_loss / steps if steps else 0, val_loss, val_acc))
            if scheduler is not None:
                scheduler.step()

        torch.save(model.state_dict(), str(run_dir / "model_unlearnt_after_forget_phase.pt"))
        logger.info("Saved checkpoint after ascent phase.")

        # Phase 2: gradient descent on retain set — identical to standard retain-only loop
        logger.info("\n--- Phase 2: Gradient Descent on Retain Set ({} epochs) ---".format(retain_epochs))
        if retain_epochs > 0:
            cfg_descent = dict(cfg_unlearn)
            cfg_descent["epochs"] = retain_epochs
            optimizer = build_optimizer(model, cfg_descent)
            scheduler = build_scheduler(optimizer, cfg_descent, num_epochs=retain_epochs)
            for epoch in range(1, retain_epochs + 1):
                val_loss = _retain_descent_epoch(model, optimizer, "Descent Epoch {:3d}".format(epoch))
                best_val_loss = min(best_val_loss, val_loss)
                torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
                if scheduler is not None:
                    scheduler.step()

        elapsed = time.time() - start_time
        logger.info("\nPure ascent-descent completed in {:.1f}s".format(elapsed))
        model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))

        return model, {
            "strategy": "ascent_descent",
            "mode": "pure_two_phase",
            "epochs": num_epochs,
            "forget_epochs": forget_epochs,
            "retain_epochs": retain_epochs,
            "ascent_lr": ascent_lr,
            "descent_lr": cfg_unlearn["lr"],
            "best_val_loss": best_val_loss,
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------ #
    # Standard interleaved mode: q is a numeric ratio                     #
    # ------------------------------------------------------------------ #
    logger.info("\nAscent-Descent Config:")
    logger.info("  epochs={}, forget_epochs={}, q={}, lambda={}, weight_decay={}".format(
        num_epochs, forget_epochs, q, lambda_coef, weight_decay))

    optimizer = build_optimizer(model, cfg_unlearn)

    steps_per_epoch = len(retain_loader)
    forget_steps = forget_epochs * steps_per_epoch

    scheduler = build_scheduler(optimizer, cfg_unlearn, num_epochs=num_epochs)

    forget_iter = iter(forget_loader)

    def should_do_combined_step(combined_count, retain_count, q):
        if q >= 1:
            return retain_count >= q * (combined_count + 1)
        else:
            return combined_count < (retain_count + 1) / q

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        combined_steps = 0
        retain_steps = 0

        for step, (batch_x, batch_y) in enumerate(retain_loader):
            global_step = (epoch - 1) * steps_per_epoch + step
            in_forget_phase = global_step < forget_steps

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if in_forget_phase and should_do_combined_step(combined_steps, retain_steps, q):
                try:
                    forget_batch_x, forget_batch_y = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch_x, forget_batch_y = next(forget_iter)

                forget_batch_x = forget_batch_x.to(device)
                forget_batch_y = forget_batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss_retain = compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay)
                loss_retain.backward()
                retain_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

                optimizer.zero_grad()
                logits_f = model(forget_batch_x)
                loss_forget = compute_loss_with_weight_decay(criterion, logits_f, forget_batch_y, model, weight_decay)
                loss_forget.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None and n in retain_grads:
                        p.grad = retain_grads[n] - lambda_coef * p.grad

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                epoch_loss += (loss_retain.item() + loss_forget.item())
                combined_steps += 1
            else:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = compute_loss_with_weight_decay(criterion, logits, batch_y, model, weight_decay)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss += loss.item()
                retain_steps += 1

        model.eval()
        val_loss, val_acc = _eval_val(model)
        avg_loss = epoch_loss / (retain_steps + combined_steps) if (retain_steps + combined_steps) > 0 else 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
            current_lr = get_learning_rate(optimizer)
            logger.info("Epoch {:3d} | LR: {:.6f} | Loss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} | Combined: {}, Retain: {} -> SAVED".format(
                epoch, current_lr, avg_loss, val_loss, val_acc, combined_steps, retain_steps))
        else:
            current_lr = get_learning_rate(optimizer)
            logger.info("Epoch {:3d} | LR: {:.6f} | Loss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} | Combined: {}, Retain: {}".format(
                epoch, current_lr, avg_loss, val_loss, val_acc, combined_steps, retain_steps))

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))
    elapsed = time.time() - start_time
    logger.info("\nAscent-descent completed in {:.1f}s".format(elapsed))

    return model, {
        "strategy": "ascent_descent",
        "epochs": num_epochs,
        "forget_epochs": forget_epochs,
        "q": q,
        "lambda_coef": lambda_coef,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
    }


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
    strategy = exp_config.get("strategy", "finetune_retain")
    forget_prob = exp_config.get("forget_prob", 0.5)
    dataset = exp_config.get("dataset", "data_splits_speakers300_fs10")

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

    # Run strategy
    if strategy == "finetune_retain":
        model, metrics = train_finetune_retain(
            model, train_loader, retain_loader, val_loader, criterion, device, cfg, run_dir, logger
        )
    elif strategy == "ascent_descent":
        model, metrics = train_ascent_descent(
            model, forget_loader, retain_loader, val_loader, criterion, device, cfg, run_dir, logger,
            unlearning_cfg=exp_config.get("unlearning", {})
        )
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

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
    metrics["forget_indices"] = forget_indices

    metrics_path = run_dir / "metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)

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
    run_folder = Path(exp_cfg.get("run_folder", "runs_unlearning"))
    run_dir = run_folder / "run_{}".format(args.run_id)

    # Compute seeds for this run
    training_seed = exp_config["training"]["seed"]
    forget_sampling_seed = training_seed + 1000 + args.run_id
    unlearning_seed = training_seed + 2000 + args.run_id

    # Setup run directory and config
    setup_run_directory(run_dir, {
        "run_id": args.run_id,
        "gpu": args.gpu,
        "strategy": exp_cfg.get("strategy", "finetune_retain"),
        "forget_prob": exp_cfg.get("forget_prob", 0.5),
        "training_seed": training_seed,
        "forget_sampling_seed": forget_sampling_seed,
        "unlearning_seed": unlearning_seed,
    })

    # Get device
    if args.gpu is not None:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run
    try:
        run_single_experiment(args.run_id, run_dir, exp_config, device)
        print("Run {} completed successfully".format(args.run_id))
    except Exception as e:
        print("Run {} failed: {}".format(args.run_id, e))
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
