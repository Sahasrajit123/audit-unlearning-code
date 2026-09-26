"""Fine-tuning on ``D(S) = D_r  U  D_f(S)``, and the shared training loop.

Deliberate choices:

* Every run starts from the same pretrained checkpoint (never the released TOFU
  "full" checkpoint), because each run has a different inclusion vector ``S``.
* The example order is the fixed global permutation restricted to the run's
  examples, consumed sequentially. We do NOT reshuffle per epoch by default, so the
  order the auditor is assumed to observe (paper's threat model: the adversary sees
  the shuffle order ``pi``) is exactly the order used.
* Loss is computed on answer tokens only; ``labels`` already carries ``-100``
  everywhere else.
* No generation during training.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .tofu_data import AnswerLossCollator, QAExample, encode_example

__all__ = ["encode_dataset", "train_model", "TrainResult"]


def encode_dataset(
    tokenizer: Any,
    examples: Sequence[QAExample],
    *,
    max_length: int = 512,
    append_eos: bool = True,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tokenize in the given order; the order is preserved."""
    out = []
    for e in examples:
        enc = encode_example(
            tokenizer,
            e.question,
            e.answer,
            max_length=max_length,
            append_eos=append_eos,
            system_prompt=system_prompt,
        )
        enc["author_id"] = e.author_id
        enc["qa_id"] = e.qa_id
        out.append(enc)
    return out


class TrainResult(dict):
    """Training metrics; a dict subclass so it serializes directly to JSON."""


def _lr_at(step: int, total: int, base_lr: float, warmup_ratio: float, schedule: str) -> float:
    warmup = max(1, int(total * warmup_ratio)) if warmup_ratio > 0 else 0
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "constant":
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        return base_lr * (1.0 - progress)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}")


def train_model(
    model: Any,
    tokenizer: Any,
    examples: Sequence[QAExample],
    cfg: Dict[str, Any],
    *,
    seed: int,
    logger: Optional[Callable[[str], None]] = None,
    loss_fn: Optional[Callable[..., Any]] = None,
    label: str = "train",
    metrics_sink: Optional[Any] = None,
) -> TrainResult:
    """Run the fine-tuning loop in place on ``model``.

    ``loss_fn(model, batch) -> (loss, extra_dict)`` allows the unlearning methods to
    reuse this loop with a different objective (e.g. NPO) while keeping the data
    ordering, scheduling, accumulation and logging identical.

    ``metrics_sink`` is an optional object with ``.log(dict, step=int)`` -- a
    :class:`audit_tofu.wandb_logger.WandbRun` in practice. It is called at the same
    cadence as the text log. It must never raise; WandbRun guarantees that.
    """
    import torch

    from .modeling import build_optimizer, peak_memory_stats, reset_peak_memory

    log = logger or (lambda msg: None)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate = AnswerLossCollator(pad_token_id=pad_id)

    encoded = encode_dataset(
        tokenizer,
        examples,
        max_length=int(cfg.get("max_seq_length", 512)),
        append_eos=bool(cfg.get("append_eos", True)),
        system_prompt=cfg.get("system_prompt"),
    )
    if not encoded:
        raise ValueError(f"{label}: empty dataset")

    micro_bs = int(cfg.get("micro_batch_size", 2))
    accum = int(cfg.get("gradient_accumulation_steps", 8))
    epochs = int(cfg.get("epochs", 5))
    # Intra-epoch progress. An epoch here is ~1900 micro-steps, so without this a run
    # is silent for many minutes at a time and a stall is indistinguishable from
    # slow progress on a shared GPU.
    log_every = int(cfg.get("log_every_n_optimizer_steps", 25))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    base_lr = float(cfg.get("learning_rate", 1e-5))
    schedule = cfg.get("lr_schedule", "linear")
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))

    optimizer, opt_info = build_optimizer(model, cfg)

    micro_per_epoch = math.ceil(len(encoded) / micro_bs)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    total_steps = steps_per_epoch * epochs

    log(
        f"[{label}] n={len(encoded)} micro_bs={micro_bs} accum={accum} "
        f"eff_bs={micro_bs * accum} epochs={epochs} opt_steps={total_steps} "
        f"lr={base_lr} sched={schedule} {opt_info}"
    )

    reset_peak_memory()
    model.train()
    t0 = time.time()

    global_step = 0
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        # The global order is fixed; reshuffling is opt-in and off by default so the
        # realized order matches the order the threat model assumes is observable.
        if cfg.get("reshuffle_each_epoch", False):
            import numpy as np

            rng = np.random.default_rng(seed + epoch)
            idx = rng.permutation(len(encoded))
            epoch_data = [encoded[i] for i in idx]
        else:
            epoch_data = encoded

        micro_batches = [
            epoch_data[i : i + micro_bs] for i in range(0, len(epoch_data), micro_bs)
        ]

        running, n_running = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for mi, chunk in enumerate(micro_batches):
            batch = collate(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}

            if loss_fn is not None:
                loss, extra = loss_fn(model, batch)
            else:
                out = model(**batch)
                loss, extra = out.loss, {}

            (loss / accum).backward()
            running += float(loss.detach().item())
            n_running += 1

            is_last = mi == len(micro_batches) - 1
            if (mi + 1) % accum == 0 or is_last:
                lr = _lr_at(global_step, total_steps, base_lr, warmup_ratio, schedule)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if log_every > 0 and global_step % log_every == 0:
                    elapsed = time.time() - t0
                    frac = global_step / max(1, total_steps)
                    eta = elapsed / frac - elapsed if frac > 0 else float("nan")
                    cur_loss = running / max(1, n_running)
                    log(
                        f"[{label}] step {global_step}/{total_steps} "
                        f"({100 * frac:.0f}%) loss={cur_loss:.4f} "
                        f"lr={lr:.2e} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
                    )
                    if metrics_sink is not None:
                        metrics_sink.log({
                            f"{label}/loss": cur_loss,
                            f"{label}/lr": lr,
                            f"{label}/epoch": epoch + frac,
                            f"{label}/progress": frac,
                            f"{label}/elapsed_min": elapsed / 60,
                        }, step=global_step)

        epoch_loss = running / max(1, n_running)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        log(f"[{label}] epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")
        if metrics_sink is not None:
            metrics_sink.log({f"{label}/epoch_loss": epoch_loss,
                              f"{label}/epoch_done": epoch + 1}, step=global_step)

    duration = time.time() - t0
    mem = peak_memory_stats()

    return TrainResult(
        {
            "label": label,
            "num_examples": len(encoded),
            "epochs": epochs,
            "optimizer_steps": global_step,
            "effective_batch_size": micro_bs * accum,
            "final_loss": history[-1]["loss"] if history else None,
            "history": history,
            "duration_seconds": duration,
            "seed": int(seed),
            **{f"{k}": v for k, v in mem.items()},
            **opt_info,
        }
    )
