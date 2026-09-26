"""
Stage 2 of the pipeline: ordinary supervised training of the model that will later be
unlearned. The resulting weights become the "competent teacher" used in unlearn.py.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def _make_optimizer(model, optimizer_type, lr, weight_decay, momentum):
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}. Choose 'adam' or 'sgd'.")


def _make_scheduler(optimizer, scheduler_type, epochs, scheduler_step_size=None, scheduler_gamma=0.1):
    if scheduler_type is None or scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_type == "step":
        step_size = scheduler_step_size or max(1, epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=scheduler_gamma)
    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}. Choose 'cosine', 'step', or None.")


@torch.no_grad()
def evaluate(model, dataset_or_loader, device, batch_size=256):
    """
    Return (avg_loss, accuracy_pct) of `model` over a dataset or DataLoader.
    Accepts either so callers can reuse an existing loader or just pass a raw dataset.
    """
    loader = dataset_or_loader if isinstance(dataset_or_loader, DataLoader) else DataLoader(
        dataset_or_loader, batch_size=batch_size, shuffle=False, num_workers=0
    )
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    model.train(was_training)
    return total_loss / total, 100.0 * correct / total


def train(model, train_set, val_set, device, epochs=20, batch_size=128, lr=1e-3,
          weight_decay=5e-4, optimizer_type="adam", momentum=0.9, shuffle_generator=None,
          num_workers=0, scheduler_type=None, scheduler_step_size=None, scheduler_gamma=0.1):
    """
    Standard cross-entropy training loop.

    `shuffle_generator` (an already-seeded `torch.Generator`) is used once, before the
    first epoch, to draw a single random permutation of `train_set` -- pass one seeded
    from the run's `--seed` to make it reproducible. The DataLoader itself does not
    reshuffle after that, so every epoch sees the exact same batch order/composition
    (only which parameters are being updated changes epoch to epoch, not the data order).

    `scheduler_type`: None/"none" (default, constant `lr`), "cosine" (anneals `lr` to 0
    over `epochs`, stepped once per epoch), or "step" (decays by `scheduler_gamma` every
    `scheduler_step_size` epochs, default step size `epochs // 3`).

    Tracks the best validation-loss checkpoint in memory and restores it onto `model`
    before returning, so the returned model is the best-on-val snapshot rather than
    necessarily the last epoch.

    Returns:
        model: the trained model (same object, modified in place).
        history: list of per-epoch dicts with train/val loss, accuracy, and the LR used.
    """
    if shuffle_generator is not None:
        permutation = torch.randperm(len(train_set), generator=shuffle_generator).tolist()
        train_set = Subset(train_set, permutation)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, optimizer_type, lr, weight_decay, momentum)
    scheduler = _make_scheduler(optimizer, scheduler_type, epochs, scheduler_step_size, scheduler_gamma)

    best_val_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        train_loss /= total
        train_acc = 100.0 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })
        print(f"[train] epoch {epoch + 1}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%  lr={current_lr:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print("[train] finished, restored best-val-loss checkpoint")
    return model, history
