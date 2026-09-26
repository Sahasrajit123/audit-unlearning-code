"""
trainer_utils.py — shared training utilities used by both main.py and train.py
"""

import json
import time
import torch
from model import ShakespeareLSTM
from engine import train_one_epoch, evaluate, generate


def load_config(path="config.json"):
    """Load config and flatten nested sections."""
    with open(path, "r") as f:
        cfg = json.load(f)
    flat = {}
    for section in cfg.values():
        flat.update(section)
    return flat


def get_device(gpu_id=None):
    """Get device. If gpu_id given, use that GPU; else cuda:0 > mps > cpu."""
    if torch.cuda.is_available():
        idx = gpu_id if gpu_id is not None else 0
        return torch.device(f"cuda:{idx}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_header(cols):
    """Print table header with underline."""
    line = "  ".join(f"{c:>10}" for c in cols)
    bar = "-" * len(line)
    print(f"\n{bar}\n{line}\n{bar}")
    return bar


def print_row(values):
    """Print table row."""
    print("  ".join(f"{v:>10}" for v in values))


def build_model(vocab_size, cfg, device, compile=False):
    """Create model on the specified device."""
    model = ShakespeareLSTM(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
    ).to(device)
    print(f"[model] parameters: {model.count_parameters():,}")
    if compile:
        model = torch.compile(model)
        print("[model] torch.compile enabled (may reduce reproducibility)")
    return model


def build_optimizer(model, cfg):
    """Create optimizer from config."""
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"])


def build_scheduler(optimizer, cfg, num_epochs=None):
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: torch optimizer
        cfg: config dict with keys like 'scheduler', 'scheduler_step_size', 'scheduler_gamma', etc.
        num_epochs: total number of epochs (required for cosine and linear schedulers)

    Returns:
        scheduler object or None if no scheduler configured
    """
    scheduler_type = cfg.get("scheduler", "constant")

    if scheduler_type == "constant":
        return None
    elif scheduler_type == "cosine":
        if num_epochs is None and "epochs" not in cfg:
            raise ValueError("num_epochs must be provided or 'epochs' must be in config for cosine scheduler")
        T_max = num_epochs if num_epochs is not None else cfg["epochs"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "linear":
        if num_epochs is None and "epochs" not in cfg:
            raise ValueError("num_epochs must be provided or 'epochs' must be in config for linear scheduler")
        total_iters = num_epochs if num_epochs is not None else cfg["epochs"]
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_iters
        )
    elif scheduler_type == "step":
        if "scheduler_step_size" not in cfg:
            raise ValueError("'scheduler_step_size' must be specified in config for step scheduler")
        step_size = cfg["scheduler_step_size"]
        gamma = cfg.get("scheduler_gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")



def train_loop(
    model,
    train_loader,
    eval_loader,
    optimizer,
    criterion,
    device,
    cfg,
    save_path,
    eval_split_name="test",
):
    """
    Run the training loop.

    Returns:
        best_eval_acc : best accuracy achieved on eval set
    """
    cols = ["epoch", "tr_loss", "tr_acc", "tr_ppl", "ev_loss", "ev_acc", "ev_ppl", "lr", "ep_time", "cum_time"]
    bar = print_header(cols)
    best_eval_acc = 0.0
    cumulative_time = 0.0

    scheduler = build_scheduler(optimizer, cfg, num_epochs=cfg["epochs"])

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_ppl = train_one_epoch(
            model, train_loader, optimizer, criterion, device, clip=cfg["clip"]
        )
        ev_loss, ev_acc, ev_ppl = evaluate(model, eval_loader, criterion, device)
        epoch_time = time.time() - t0
        cumulative_time += epoch_time

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        print_row(
            [
                epoch,
                f"{tr_loss:.4f}",
                f"{tr_acc:.2%}",
                f"{tr_ppl:.2f}",
                f"{ev_loss:.4f}",
                f"{ev_acc:.2%}",
                f"{ev_ppl:.2f}",
                f"{current_lr:.2e}",
                f"{epoch_time:.1f}s",
                f"{cumulative_time:.1f}s",
            ]
        )

        if ev_acc > best_eval_acc:
            best_eval_acc = ev_acc
            torch.save(model.state_dict(), save_path)

        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

    print(f"\n[done] best {eval_split_name} accuracy : {best_eval_acc:.2%}")
    print(f"       paper target              : ~54%  (FedAvg non-IID baseline)")
    return best_eval_acc


def generate_text(model, cfg, char2idx, idx2char, device):
    """Generate text from trained model."""
    print("\n[generate] loading best weights...")
    # Note: save_path should be loaded before calling this

    output = generate(
        model,
        cfg["seed_text"],
        char2idx,
        idx2char,
        n_chars=cfg["n_chars"],
        temperature=cfg["temperature"],
        device=device,
    )
    print(f"\n--- seed: '{cfg['seed_text']}' ---\n")
    print(output)
