"""
The "bad teacher" (gated knowledge-distillation) unlearning method -- one of the two
methods registered in `unlearning/__init__.py` (see `--unlearn-method badteacher`).

This is a direct adaptation of the official paper implementation:
    Chundawat, Tarun, Mandal & Kankanhalli, "Can Bad Teaching Induce Forgetting?
    Unlearning in Deep Networks Using an Incompetent Teacher", AAAI 2023.
    Paper:  https://arxiv.org/abs/2205.08096
    Code:   https://github.com/vikram2000b/bad-teaching-unlearning
            (unlearn.py::UnlearnerLoss/unlearning_step/blindspot_unlearner and
            dataset.py::UnLearningData -- MIT licensed, Copyright (c) 2022 Vikram Singh
            Chundawat)

`UnLearningData`, `UnlearnerLoss`, `unlearning_step` and `blindspot_unlearner` below are
ported near-verbatim from that repo (same variable names, same gated KL-divergence
formula, same default `F.kl_div` reduction ('mean') as published -- note this divides by
`batch_size * num_classes`, not just `batch_size`, so the reported/effective loss is
~num_classes times smaller than the textbook per-sample KL divergence; kept as-is to match
the paper's published hyperparameters, which were tuned against this same scaling). The
only other changes are:
  - `unlearning_teacher` is built from this project's `model.build_model` factory instead
    of a hardcoded ResNet18,
  - its random init plus the unlearning DataLoader's shuffle order are drawn from an
    explicit, independent seed (via `torch.random.fork_rng` / a seeded `torch.Generator`)
    instead of ambient global RNG state -- required by this project's train/unlearn
    seed-independence contract (see train_and_unlearn.py's module docstring), and
  - intermediate checkpointing and optional per-epoch multi-split evaluation are exposed
    through the same `checkpoint_every` / `eval_sets` arguments the shared unlearning
    interface uses for every method (see `unlearning/__init__.py`).

Method recap: `model` (the student) starts as a copy of the fully-trained network. It is
fine-tuned so that on retain samples (label 0) its output distribution stays close to the
"full_trained_teacher" (the frozen trained model), and on forget samples (label 1) it is
pulled toward the "unlearning_teacher" -- a frozen, randomly-initialized, never-trained
network. Chasing an untrained network's output on the forget set is what "forgetting"
means under this method.
"""
import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.seeding import derive_seed
from core.train import evaluate


class UnLearningData(Dataset):
    """Combines forget_data (labeled 1) and retain_data (labeled 0) for the unlearning step.

    Port of dataset.py::UnLearningData from the official repo.
    """

    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    """
    Gated KL-divergence loss. Port of unlearn.py::UnlearnerLoss from the official repo.

    label 1 (forget sample) -> student is pulled toward the incompetent ("unlearn") teacher.
    label 0 (retain sample) -> student is pulled toward the competent ("full") teacher.
    """
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample, label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader,
                     optimizer, device, KL_temperature):
    """One epoch of gated-KD unlearning. Port of unlearn.py::unlearning_step."""
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits,
                              unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return float(np.mean(losses))


def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                         epochs=10, optimizer="adam", lr=0.01, batch_size=256, num_workers=0,
                         device="cuda", KL_temperature=1, generator=None,
                         eval_sets: Optional[List[Tuple[str, Dataset]]] = None,
                         eval_batch_size: int = 256, checkpoint_every: Optional[int] = None
                         ) -> Tuple[List[dict], Dict[int, dict]]:
    """
    The "blind spot" / bad-teacher unlearning loop. Port of unlearn.py::blindspot_unlearner.

    Differences from the official version: `num_workers` defaults to 0 (single-process,
    reproducible loading, matching the rest of this project instead of the original's 32),
    and it accepts an explicit `generator` so the unlearning DataLoader's shuffle order can
    be pinned to this run's unlearn-seed. It also always prints/records forget-set accuracy
    after every epoch (not just the gated-KD loss), so progress toward actually forgetting
    is visible epoch-by-epoch, not just at the final before/after evaluation.

    `eval_sets`, if given, is a list of (name, dataset) pairs additionally evaluated with
    `train.evaluate` after every epoch, recorded as `{name}_loss` / `{name}_acc`. This is
    on top of the always-present `forget_accuracy`, so the history schema is unchanged when
    it is None. Off by default -- each entry costs a full evaluation pass every epoch.

    `checkpoint_every`, if given, snapshots `model.state_dict()` (detached, moved to CPU)
    every `checkpoint_every` epochs, skipping the final epoch (the caller already has that
    one -- it's whatever `model` ends up as when this returns). `checkpoint_every=1` is the
    "every intermediate epoch" behaviour this method used before the two unlearning methods
    were merged behind one interface. This function does no file I/O itself -- consistent
    with the rest of this package, saving checkpoints to disk is left to the caller
    (train_and_unlearn.py / run_sweep.py), which is where run-directory paths live.

    Returns (history, checkpoints), matching the shared interface.
    """
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, generator=generator)

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # else: assume `optimizer` was already passed in as a constructed optimizer instance.

    history = []
    checkpoints: Dict[int, dict] = {}
    for epoch in range(epochs):
        loss = unlearning_step(model=model, unlearning_teacher=unlearning_teacher,
                                full_trained_teacher=full_trained_teacher,
                                unlearn_data_loader=unlearning_loader, optimizer=optimizer,
                                device=device, KL_temperature=KL_temperature)
        _, forget_acc = evaluate(model, forget_data, device)
        print(f"[unlearn] epoch {epoch + 1}/{epochs}  unlearning loss: {loss:.4f}  forget_acc: {forget_acc:.2f}%")
        epoch_num = epoch + 1
        epoch_record = {"epoch": epoch_num, "gated_kd_loss": loss, "forget_accuracy": forget_acc}
        if eval_sets is not None:
            for name, dataset in eval_sets:
                eval_loss, eval_acc = evaluate(model, dataset, device, batch_size=eval_batch_size)
                epoch_record[f"{name}_loss"] = eval_loss
                epoch_record[f"{name}_acc"] = eval_acc
                print(f"    {name:>14s}: loss={eval_loss:.4f} acc={eval_acc:.2f}%")
        if checkpoint_every is not None and checkpoint_every > 0 \
                and epoch_num % checkpoint_every == 0 and epoch_num < epochs:
            checkpoints[epoch_num] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[unlearn] epoch {epoch_num}/{epochs}  snapshotted intermediate checkpoint")
        history.append(epoch_record)
    return history, checkpoints


def build_incompetent_teacher(model_fn: Callable[[], torch.nn.Module], device: torch.device,
                               seed: Optional[int] = None) -> torch.nn.Module:
    """
    Build the "incompetent teacher": same architecture as the model being unlearned, with
    fresh random weights, never trained on any data, and frozen.

    The official repo builds this with `ResNet18(pretrained=False)` under whatever the
    ambient global RNG state happens to be at that point in the notebook. Here we instead
    fork the RNG (via `torch.random.fork_rng`) so the random init can be pinned to an
    independent seed without perturbing the training seed's reproducibility.
    """
    fork_devices = [] if device.type == "cpu" else [device]
    with torch.random.fork_rng(devices=fork_devices):
        if seed is not None:
            torch.manual_seed(seed)
        incompetent_teacher = model_fn().to(device)
    incompetent_teacher.eval()
    for p in incompetent_teacher.parameters():
        p.requires_grad_(False)
    return incompetent_teacher


def unlearn(trained_model: torch.nn.Module, model_fn: Callable[[], torch.nn.Module], retain_set,
            forget_set, device: torch.device, *, batch_size: int = 256, epochs: int = 1,
            lr: float = 1e-4, seed: Optional[int] = None, num_workers: int = 0,
            eval_sets: Optional[List[Tuple[str, Dataset]]] = None, eval_batch_size: int = 256,
            checkpoint_every: Optional[int] = None, temperature: float = 1.0):
    """
    Bad-teacher's implementation of this package's shared unlearning interface (see
    `unlearning/__init__.py` for the contract every method implements), mirroring the
    official repo's notebook usage:

        unlearning_teacher = ResNet18(pretrained=False).eval()
        student_model = ResNet18(pretrained=False); student_model.load_state_dict(trained_weights)
        blindspot_unlearner(model=student_model, unlearning_teacher=unlearning_teacher,
                             full_trained_teacher=trained_model, retain_data=..., forget_data=...)

    Deep-copies `trained_model` into a student (so `trained_model` itself is left
    untouched and can still serve as the frozen "full_trained_teacher"), builds the
    incompetent teacher via `model_fn`, and runs `blindspot_unlearner`. Unlike DELETE,
    this method genuinely uses both `retain_set` and `model_fn`.

    `temperature` is bad-teacher's one method-specific hyperparameter: the KL-divergence
    distillation temperature applied to both teachers and the student.

    `seed` seeds everything specific to this stage -- incompetent-teacher init and the
    unlearning DataLoader's shuffle order -- via independent sub-seeds (see
    `seeding.derive_seed`), so both can be pinned without correlating with `--seed`
    (training) or with each other. If None, both draw from ambient (non-reproducible)
    randomness.

    `eval_sets` and `checkpoint_every` are passed straight through to
    `blindspot_unlearner` (see its docstring).

    Returns (student, history, checkpoints).
    """
    teacher_seed = derive_seed(seed, "incompetent_teacher") if seed is not None else None
    shuffle_seed = derive_seed(seed, "unlearn_shuffle") if seed is not None else None

    full_trained_teacher = copy.deepcopy(trained_model).to(device).eval()
    for p in full_trained_teacher.parameters():
        p.requires_grad_(False)

    unlearning_teacher = build_incompetent_teacher(model_fn, device, seed=teacher_seed)

    student = copy.deepcopy(trained_model).to(device)
    student.train()

    generator = torch.Generator()
    if shuffle_seed is not None:
        generator.manual_seed(shuffle_seed)
    else:
        generator.seed()

    print(f"[unlearn] retain={len(retain_set)} forget={len(forget_set)} "
          f"batch_size={batch_size} epochs={epochs} lr={lr} temperature={temperature} "
          f"teacher_seed={teacher_seed} shuffle_seed={shuffle_seed}")

    history, checkpoints = blindspot_unlearner(
        model=student, unlearning_teacher=unlearning_teacher, full_trained_teacher=full_trained_teacher,
        retain_data=retain_set, forget_data=forget_set, epochs=epochs, optimizer="adam", lr=lr,
        batch_size=batch_size, num_workers=num_workers, device=device, KL_temperature=temperature,
        generator=generator, eval_sets=eval_sets, eval_batch_size=eval_batch_size,
        checkpoint_every=checkpoint_every,
    )

    student.eval()
    print("[unlearn] finished")
    return student, history, checkpoints
