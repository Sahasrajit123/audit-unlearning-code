# -*- coding: utf-8 -*-
"""
Ascent–descent unlearning: alternate gradient ascent on forget set and
gradient descent on retain set.

Based on /lfs/mercury1/0/sahasras/unlearning-gradient-ascent-descent/unlearn.py:
  - For the first `forget_epochs` epochs: every `q` retain-only steps, do one
    combined step: g = gr - λ*gf (descent on retain, ascent on forget).
  - After `forget_epochs` epochs: only retain steps (gradient descent on retain).

Uses JAX/Flax and the same loss (CE + weight decay) as the main Trainer.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training import train_state, common_utils
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
    raise ValueError(f"Invalid lr_schedule for ascent_descent: {schedule_name}")


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


def unlearn_ascent_descent(
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
    Run ascent–descent unlearning with support for:
      - Standard ascent-descent: For the first `forget_epochs` epochs, every `q` retain-only steps, do one combined step: g = gr - λ*gf (descent on retain, ascent on forget). After `forget_epochs`, only retain steps (gradient descent).
      - Forget-only ascent phase (when q is None): For the first `forget_epochs` epochs, perform only gradient ascent on the forget set (using negative gradients and a custom learning rate, `initial_forget_lr`). After that, switch to standard gradient descent on the retain set with the configured schedule.

    Expects under config["unlearning"]:
        - epochs: int

    Optimization hyperparameters are read from config["post_unlearning"] first
    (max_lr/weight_decay/optim/momentum/lr_schedule), then fall back to
    config["unlearning"].

    Ascent-descent specific options are read from:
        - config["post_unlearning"]["ascent_descent"] (preferred)
        - config["unlearning"]["ascent_descent"] (fallback)

        - ascent_descent:
          - q: int or None (retain steps before a combined step; if None, do forget-only ascent phase)
          - lambda_coef: float (coefficient for forget gradient in combined step)
          - forget_epochs: int (epochs with combined steps or forget-only phase)
          - initial_forget_lr: float (learning rate for forget-only ascent phase; required if q is None)

    During the forget-only ascent phase (q is None), only the forget set is used, and gradient ascent is performed by negating the gradients. After this phase, training switches to standard gradient descent on the retain set.

    Optional: validate_fn(state, dataloader, epoch, prefix), save_checkpoint_fn(state, number, **).
    """
    # ...existing code...
    # Disable online wandb logging for test runs
    try:
        wandb.init(mode="disabled")
    except Exception:
        pass
    unlearn_cfg = config["unlearning"]
    post_cfg = config.get("post_unlearning", {})
    ad = post_cfg.get("ascent_descent") or unlearn_cfg.get("ascent_descent", {})
    num_epochs = int(unlearn_cfg["epochs"])
    lr = float(post_cfg.get("max_lr", unlearn_cfg["max_lr"]))
    lr_schedule_name = str(post_cfg.get("lr_schedule", unlearn_cfg.get("lr_schedule", "constant")))
    weight_decay = float(post_cfg.get("weight_decay", unlearn_cfg.get("weight_decay", 0.0)))
    optim_name = str(post_cfg.get("optim", unlearn_cfg.get("optim", "SGD"))).upper()
    momentum = float(post_cfg.get("momentum", unlearn_cfg.get("momentum", 0.0)))
    nesterov = bool(post_cfg.get("nesterov", unlearn_cfg.get("nesterov", False)))
    q = ad.get("q", 9)
    lambda_coef = float(ad.get("lambda_coef", 0.5))
    forget_epochs = int(ad.get("forget_epochs", min(5, num_epochs)))
    initial_forget_lr = ad.get("initial_forget_lr", None)
    if (q is None or q == "None") and initial_forget_lr is None:
        raise ValueError(
            "q=None requires initial_forget_lr to be set (forget-only ascent phase needs a fixed LR). "
            "Set ascent_descent.initial_forget_lr in your config."
        )
    # Defensive: ensure initial_forget_lr is a float if provided
    if initial_forget_lr is not None:
        try:
            initial_forget_lr = float(initial_forget_lr)
        except Exception as e:
            raise ValueError(f"initial_forget_lr must be a float, got {initial_forget_lr} (type {type(initial_forget_lr)}): {e}")
    num_classes = int(config["model"]["n_classes"])

    steps_per_epoch = len(retain_dataloader)
    total_steps = num_epochs * steps_per_epoch
    forget_steps = forget_epochs * steps_per_epoch

    logger.info(
        "Ascent–descent unlearning: num_epochs=%s, forget_epochs=%s, q=%s, lambda_coef=%s, lr=%s, lr_schedule=%s, wd=%s",
        num_epochs, forget_epochs, q, lambda_coef, lr, lr_schedule_name, weight_decay,
    )
    logger.info(
        "Steps: total=%s, forget_phase=%s, retain_only_phase=%s",
        total_steps, forget_steps, total_steps - forget_steps,
    )

    if optim_name != "SGD":
        raise ValueError(f"Ascent-descent currently supports only SGD, got {optim_name}")

    # Support for initial forget-only ascent phase with custom LR
    if initial_forget_lr is not None:
        # Use constant LR for forget phase if specified
        if not isinstance(initial_forget_lr, float):
            raise ValueError(f"initial_forget_lr must be a float, got {type(initial_forget_lr)}")
        tx_forget = optax.sgd(learning_rate=initial_forget_lr, momentum=momentum, nesterov=nesterov)
        if q is None or q == "None":
            retain_lr_schedule = _build_lr_schedule(lr_schedule_name, lr, num_epochs * len(retain_dataloader) - forget_epochs * len(forget_dataloader))
        else:
            retain_lr_schedule = _build_lr_schedule(lr_schedule_name, lr, num_epochs * len(retain_dataloader) - forget_epochs * len(retain_dataloader))
        if not callable(retain_lr_schedule):
            raise ValueError("retain_lr_schedule must be a callable schedule object")
        tx_retain = optax.sgd(learning_rate=retain_lr_schedule, momentum=momentum, nesterov=nesterov)
        state = train_state.TrainState.create(
            apply_fn=state.apply_fn,
            params=state.params,
            tx=tx_forget,
        )
    else:
        # Use schedule for all phases
        lr_schedule = _build_lr_schedule(lr_schedule_name, lr, num_epochs * len(retain_dataloader))
        if not callable(lr_schedule):
            raise ValueError("lr_schedule must be a callable schedule object")
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

    def combined_grad_metrics(images_r, labels_r, images_f, labels_f):
        def loss_r(params):
            return _loss_with_wd(apply_fn, num_classes, params, images_r, labels_r, weight_decay)

        def loss_f(params):
            return _loss_with_wd(apply_fn, num_classes, params, images_f, labels_f, weight_decay)

        (loss_r_val, logits_r), gr = jax.value_and_grad(loss_r, has_aux=True)(state.params)
        (loss_f_val, logits_f), gf = jax.value_and_grad(loss_f, has_aux=True)(state.params)
        combined = jax.tree_util.tree_map(lambda a, b: a - lambda_coef * b, gr, gf)
        acc_r = jnp.mean(jnp.argmax(logits_r, -1) == labels_r)
        acc_f = jnp.mean(jnp.argmax(logits_f, -1) == labels_f)
        return (
            combined,
            float(np.asarray(loss_r_val)),
            float(np.asarray(loss_f_val)),
            float(np.asarray(acc_r)),
            float(np.asarray(acc_f)),
        )

    res_iter: Any = None
    unl_iter: Any = None
    start = time.time()

    # --- Forget phase ---
    if q is None or q == "None":
        # Forget-only ascent for forget_epochs
        for epoch in range(forget_epochs):
            # Log learning rate at the start of each epoch
            logger.info(
                f"[FORGET PHASE][Epoch {epoch+1}] Starting epoch with learning rate: {initial_forget_lr if initial_forget_lr is not None else 'schedule'}"
            )
            for batch_idx, batch_f in enumerate(forget_dataloader):
                images_f, labels_f = _batch_to_jax(batch_f)
                def loss_f(params):
                    return _loss_with_wd(apply_fn, num_classes, params, images_f, labels_f, weight_decay)
                (loss_f_val, logits_f), gf = jax.value_and_grad(loss_f, has_aux=True)(state.params)
                neg_gf = jax.tree_util.tree_map(lambda x: -x, gf)
                state = state.apply_gradients(grads=neg_gf)
                acc_f = jnp.mean(jnp.argmax(logits_f, -1) == labels_f)
                metrics = {
                    "loss_forget": float(np.asarray(loss_f_val)),
                    "accuracy_forget": float(np.asarray(acc_f)),
                    "step_type": "forget_ascent",
                    "ascent_descent_step": epoch * len(forget_dataloader) + batch_idx,
                    "ascent_descent_epoch": epoch + 1,
                    "eta_t": float(initial_forget_lr) if initial_forget_lr is not None else float(np.asarray(lr_schedule(state.step))),
                }
                wandb.log(metrics)
            # Validation/checkpoint at epoch end
            if validate_fn is not None:
                if val_dataloader is not None:
                    val_metrics = validate_fn(state, val_dataloader, epoch + 1, "Val (AscentDescent)")
                    logger.info(f"[VAL][Epoch {epoch+1}] {val_metrics}")
                forget_metrics = validate_fn(state, forget_dataloader, epoch + 1, "Forget (AscentDescent)")
                logger.info(f"[FORGET][Epoch {epoch+1}] {forget_metrics}")
                retain_metrics = validate_fn(state, retain_dataloader, epoch + 1, "Retain (AscentDescent)")
                logger.info(f"[RETAIN][Epoch {epoch+1}] {retain_metrics}")
            if save_checkpoint_fn is not None:
                try:
                    save_checkpoint_fn(state, epoch + 1, phase="epoch")
                except Exception as e:
                    logger.warning("save_checkpoint_fn failed: %s", e)
    else:
        # Standard ascent-descent for forget_epochs
        retain_count = 0
        for epoch in range(forget_epochs):
            # Log learning rate at the start of each epoch
            lr_val = float(initial_forget_lr) if initial_forget_lr is not None else float(np.asarray(lr_schedule(state.step)))
            logger.info(f"[FORGET PHASE][Epoch {epoch+1}] Starting epoch with learning rate: {lr_val}")
            res_iter = None
            unl_iter = None
            step_in_epoch = 0
            while step_in_epoch < len(retain_dataloader):
                # q retain steps, then 1 combined step
                for _ in range(q):
                    if step_in_epoch >= len(retain_dataloader):
                        break
                    res_iter, batch_r = _next_batch(retain_dataloader, res_iter)
                    grads, loss_val, acc = retain_grad_metrics(batch_r[0], batch_r[1])
                    state = state.apply_gradients(grads=grads)
                    metrics = {
                        "loss": loss_val,
                        "accuracy": acc,
                        "step_type": "retain",
                        "ascent_descent_step": epoch * len(retain_dataloader) + step_in_epoch,
                        "ascent_descent_epoch": epoch + 1,
                        "eta_t": lr_val,
                    }
                    wandb.log(metrics)
                    step_in_epoch += 1
                if step_in_epoch >= len(retain_dataloader):
                    break
                # Combined step
                res_iter, batch_r = _next_batch(retain_dataloader, res_iter)
                unl_iter, batch_f = _next_batch(forget_dataloader, unl_iter)
                combined, lr_val_comb, lf_val, ar, af = combined_grad_metrics(
                    batch_r[0], batch_r[1], batch_f[0], batch_f[1]
                )
                state = state.apply_gradients(grads=combined)
                metrics = {
                    "loss_retain": lr_val_comb,
                    "loss_forget": lf_val,
                    "accuracy_retain": ar,
                    "accuracy_forget": af,
                    "step_type": "combined",
                    "ascent_descent_step": epoch * len(retain_dataloader) + step_in_epoch,
                    "ascent_descent_epoch": epoch + 1,
                    "eta_t": lr_val,
                }
                wandb.log(metrics)
                step_in_epoch += 1
            # Validation/checkpoint at epoch end
            if validate_fn is not None:
                if val_dataloader is not None:
                    validate_fn(state, val_dataloader, epoch + 1, "Val (AscentDescent)")
                validate_fn(state, forget_dataloader, epoch + 1, "Forget (AscentDescent)")
                validate_fn(state, retain_dataloader, epoch + 1, "Retain (AscentDescent)")
            if save_checkpoint_fn is not None:
                try:
                    save_checkpoint_fn(state, epoch + 1, phase="epoch")
                except Exception as e:
                    logger.warning("save_checkpoint_fn failed: %s", e)

    # --- Retain/finetune phase ---
    # Switch optimizer whenever there is a separate retain optimizer (initial_forget_lr is not None)
    if initial_forget_lr is not None and state.tx is not tx_retain:
        new_opt_state = tx_retain.init(state.params)
        state = train_state.TrainState(
            step=state.step,
            apply_fn=state.apply_fn,
            params=state.params,
            tx=tx_retain,
            opt_state=new_opt_state,
        )
    # Remaining epochs
    retain_step = 0  # tracks steps within the retain phase, matching the reinitialized optimizer count
    for epoch in range(forget_epochs, num_epochs):
        # Log learning rate at the start of each epoch
        if initial_forget_lr is not None:
            eta_t = float(np.asarray(retain_lr_schedule(retain_step)))
        else:
            eta_t = float(np.asarray(lr_schedule(state.step)))
        logger.info(f"[RETAIN PHASE][Epoch {epoch+1}] Starting epoch with learning rate: {eta_t}")
        res_iter = None
        for batch_idx, batch_r in enumerate(retain_dataloader):
            images_r, labels_r = _batch_to_jax(batch_r)
            grads, loss_val, acc = retain_grad_metrics(images_r, labels_r)
            state = state.apply_gradients(grads=grads)
            metrics = {
                "loss": loss_val,
                "accuracy": acc,
                "step_type": "retain",
                "ascent_descent_step": epoch * len(retain_dataloader) + batch_idx,
                "ascent_descent_epoch": epoch + 1,
                "eta_t": eta_t,
            }
            wandb.log(metrics)
        retain_step += len(retain_dataloader)
        # Validation/checkpoint at epoch end
        if validate_fn is not None:
            if val_dataloader is not None:
                val_metrics = validate_fn(state, val_dataloader, epoch + 1, "Val (AscentDescent)")
                logger.info(f"[VAL][Epoch {epoch+1}] {val_metrics}")
            forget_metrics = validate_fn(state, forget_dataloader, epoch + 1, "Forget (AscentDescent)")
            logger.info(f"[FORGET][Epoch {epoch+1}] {forget_metrics}")
            retain_metrics = validate_fn(state, retain_dataloader, epoch + 1, "Retain (AscentDescent)")
            logger.info(f"[RETAIN][Epoch {epoch+1}] {retain_metrics}")
        if save_checkpoint_fn is not None:
            try:
                save_checkpoint_fn(state, epoch + 1, phase="epoch")
            except Exception as e:
                logger.warning("save_checkpoint_fn failed: %s", e)

    # Final validation
    if validate_fn is not None:
        if val_dataloader is not None:
            val_metrics = validate_fn(state, val_dataloader, num_epochs, "Val (AscentDescent final)")
            logger.info(f"[VAL][Epoch {num_epochs}] {val_metrics}")
        forget_metrics = validate_fn(state, forget_dataloader, num_epochs, "Forget (AscentDescent final)")
        logger.info(f"[FORGET][Epoch {num_epochs}] {forget_metrics}")
        retain_metrics = validate_fn(state, retain_dataloader, num_epochs, "Retain (AscentDescent final)")
        logger.info(f"[RETAIN][Epoch {num_epochs}] {retain_metrics}")

    elapsed = time.time() - start
    logger.info("Ascent–descent unlearning finished in %.1f s", elapsed)
    return state
