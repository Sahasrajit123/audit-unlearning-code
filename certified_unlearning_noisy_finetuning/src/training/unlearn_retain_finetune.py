# -*- coding: utf-8 -*-
"""
Retain-only finetuning unlearning: repeated gradient descent on retain set only.

Uses JAX/Flax and the same loss (CE + weight decay) as the main Trainer.
Optimization hyperparameters are read from config["post_unlearning"].
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training import common_utils, train_state
from torch.utils.data import DataLoader

from src.utils.utils import logger


def _build_lr_schedule(schedule_name: str, max_lr: float, total_steps: int):
    steps = max(int(total_steps), 1)
    if schedule_name == "constant":
        return optax.constant_schedule(max_lr)
    if schedule_name == "cos":
        return optax.cosine_decay_schedule(
            init_value=max_lr,
            decay_steps=steps,
        )
    if schedule_name == "onecycle":
        return optax.linear_onecycle_schedule(
            transition_steps=steps,
            peak_value=max_lr,
        )
    raise ValueError(f"Invalid lr_schedule for retain_finetune: {schedule_name}")


def _batch_to_jax(batch: Tuple[Any, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    images, labels = batch
    return jnp.asarray(np.asarray(images)), jnp.asarray(np.asarray(labels))


def _next_batch(dl: DataLoader, it: Any) -> Tuple[Any, Tuple[jnp.ndarray, jnp.ndarray]]:
    if it is None:
        it = iter(dl)
    try:
        b = next(it)
    except StopIteration:
        it = iter(dl)
        b = next(it)
    return it, _batch_to_jax(b)


def _loss_with_wd(
    apply_fn,
    num_classes: int,
    params,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    weight_decay: float,
):
    logits = apply_fn({"params": params}, images, train=True)
    one_hot = common_utils.onehot(labels, num_classes=num_classes)
    ce = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
    l2 = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    return ce + weight_decay * l2, logits


def unlearn_retain_finetune(
    state: train_state.TrainState,
    forget_dataloader: DataLoader,
    retain_dataloader: DataLoader,
    config: Dict[str, Any],
    *,
    val_dataloader: Optional[DataLoader] = None,
    validate_fn: Optional[Callable[..., None]] = None,
    save_checkpoint_fn: Optional[Callable[..., None]] = None,
    run_dir: Optional[str] = None,
) -> train_state.TrainState:
    """
    Run retain-only finetuning for unlearning. No ascent/forget-gradient step.

    Expects optimization hyperparameters in config["post_unlearning"]:
      - max_lr
      - weight_decay
      - optim
      - momentum
      - nesterov (optional)
      - lr_schedule
    """
    del run_dir
    unlearn_cfg = config["unlearning"]
    post_cfg = config["post_unlearning"]

    num_epochs = int(unlearn_cfg["epochs"])
    lr = float(post_cfg["max_lr"])
    lr_schedule_name = str(post_cfg.get("lr_schedule", "constant"))
    weight_decay = float(post_cfg.get("weight_decay", 0.0))
    optim_name = str(post_cfg.get("optim", "SGD")).upper()
    momentum = float(post_cfg.get("momentum", 0.0))
    nesterov = bool(post_cfg.get("nesterov", False))
    num_classes = int(config["model"]["n_classes"])

    steps_per_epoch = len(retain_dataloader)
    total_steps = num_epochs * steps_per_epoch

    logger.info(
        "Retain-finetune unlearning: num_epochs=%s, lr=%s, lr_schedule=%s, wd=%s",
        num_epochs,
        lr,
        lr_schedule_name,
        weight_decay,
    )
    logger.info("Steps: total=%s", total_steps)

    if optim_name != "SGD":
        raise ValueError(f"Retain-finetune currently supports only SGD, got {optim_name}")

    lr_schedule = _build_lr_schedule(lr_schedule_name, lr, total_steps)
    tx = optax.sgd(learning_rate=lr_schedule, momentum=momentum, nesterov=nesterov)
    state = train_state.TrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=tx,
    )

    apply_fn = state.apply_fn

    def retain_grad_metrics(images, labels):
        def loss_fn(params):
            return _loss_with_wd(apply_fn, num_classes, params, images, labels, weight_decay)

        (loss_val, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        return grads, float(np.asarray(loss_val)), float(np.asarray(acc))

    res_iter: Any = None
    start = time.time()

    for step in range(total_steps):
        epoch_1based = (step // steps_per_epoch) + 1
        res_iter, batch_r = _next_batch(retain_dataloader, res_iter)
        grads, loss_val, acc = retain_grad_metrics(batch_r[0], batch_r[1])
        state = state.apply_gradients(grads=grads)

        wandb.log(
            {
                "loss": loss_val,
                "accuracy": acc,
                "step_type": "retain",
                "retain_finetune_step": step,
                "retain_finetune_epoch": epoch_1based,
                "eta_t": float(np.asarray(lr_schedule(state.step))),
            }
        )

        steps_in_epoch = (step + 1) % steps_per_epoch
        if steps_in_epoch == 0 and step + 1 < total_steps:
            epoch_done = epoch_1based
            if validate_fn is not None:
                if val_dataloader is not None:
                    validate_fn(state, val_dataloader, epoch_done, "Val (RetainFinetune)")
                validate_fn(state, forget_dataloader, epoch_done, "Forget (RetainFinetune)")
                validate_fn(state, retain_dataloader, epoch_done, "Retain (RetainFinetune)")
            if save_checkpoint_fn is not None:
                try:
                    save_checkpoint_fn(state, epoch_done, phase="epoch")
                except Exception as e:
                    logger.warning("save_checkpoint_fn failed: %s", e)

    if validate_fn is not None:
        if val_dataloader is not None:
            validate_fn(state, val_dataloader, num_epochs, "Val (RetainFinetune final)")
        validate_fn(state, forget_dataloader, num_epochs, "Forget (RetainFinetune final)")
        validate_fn(state, retain_dataloader, num_epochs, "Retain (RetainFinetune final)")

    if save_checkpoint_fn is not None:
        try:
            save_checkpoint_fn(state, num_epochs, phase="epoch")
        except Exception as e:
            logger.warning("save_checkpoint_fn failed: %s", e)

    elapsed = time.time() - start
    logger.info("Retain-finetune unlearning finished in %.1f s", elapsed)
    return state
