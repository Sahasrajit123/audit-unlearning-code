"""
SCRUB (SCalable Remembering and Unlearning unBound) and SCRUB+R -- two of the methods
registered in `unlearning/__init__.py` (see `--unlearn-method scrub` / `scrub_r`).

This is a direct adaptation of the official paper implementation:
    Kurmanji, Triantafillou, Hayes & Ampatzoglou, "Towards Unbounded Machine Unlearning",
    NeurIPS 2023.
    Paper: https://arxiv.org/abs/2302.09880
    Code:  https://github.com/meghdadk/SCRUB
           (thirdparty/repdistiller/distiller_zoo/KD.py::DistillKL,
            thirdparty/repdistiller/helper/loops.py::train_distill,
            thirdparty/repdistiller/helper/util.py::adjust_learning_rate, and the driver
            loop in the `scrub unlearning` cell of small_scale_unlearning.ipynb /
            large_scale_unlearning.ipynb. The RepDistiller half is BSD-2-Clause,
            Copyright (c) 2019 Yonglong Tian.)

`DistillKL`, `sgda_adjust_learning_rate` and `train_distill` below are ported
near-verbatim (same variable names, same loss algebra, same per-epoch structure). The
changes:
  - `train_distill` keeps only the `opt.distill == 'kd'` branch of the original's
    fourteen-way dispatch. Every other branch needs intermediate features
    (`model(x, is_feat=True)`), which neither the original SCRUB configs nor this
    project's models provide -- the official notebooks all set `args.distill = 'kd'`,
    which makes `loss_kd = 0` and drops the `beta` term entirely. `beta` is therefore
    not exposed as a hyperparameter here: it multiplies a term that is identically zero.
  - `F.kl_div(..., size_average=False)` is written as the modern, numerically identical
    `reduction="sum"` (the original's spelling has been deprecated since torch 0.4 and
    now emits a warning on every batch).
  - the two DataLoaders' shuffle orders are drawn from explicit, independent seeded
    `torch.Generator`s instead of ambient global RNG state -- this project's
    train/unlearn seed-independence contract (see train_and_unlearn.py's module
    docstring), the same treatment unlearning/delete.py and unlearning/badteacher.py
    give their own loaders.
  - the original's run-management plumbing (wandb, `Logger`, tqdm, matplotlib training
    plots, `AverageMeter`, its `validate` helper) is dropped in favour of this project's
    `train.evaluate` and the shared `eval_sets` / `checkpoint_every` arguments every
    method here takes (see `unlearning/__init__.py`).
  - the original measures retain/forget error at the *start* of each epoch (plus once
    more after the loop); this port measures after each epoch instead. Those are the
    same measurements, shifted by one index -- but indexed this way each error belongs
    to the checkpoint taken at that same epoch, which is what SCRUB+R's rewinding needs.

Method recap: the frozen trained model is the *teacher* and the student starts as a copy
of it. One epoch over the forget set maximizes the student-teacher KL divergence (the
"max-step"), then one epoch over the retain set minimizes `gamma * cross-entropy +
alpha * KL-to-teacher` (the "min-step"). Max-steps run only for the first `msteps`
epochs; the remaining epochs are min-steps alone, which is what repairs the retain
damage the max-steps caused (Algorithm 1 in the paper). Unlike DELETE, and like
bad-teacher, SCRUB genuinely needs the retain set; unlike bad-teacher, it needs no
second network -- the teacher is the trained model itself.

SCRUB+R (`--unlearn-method scrub_r`) adds the paper's Section 3.2 "rewinding" step on
top. Plain SCRUB drives the forget error as high as it will go, and an *uncharacteristically*
high forget error is itself a membership signal -- exactly the signal this project's
audit measures. So SCRUB+R keeps every epoch's checkpoint, builds a reference error from
held-out validation data drawn from the same classes as the forget set, and returns the
checkpoint whose forget error is closest to that reference instead of the last one. See
`select_rewind_epoch` and `class_matched_val_subset`.
"""
import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from core.seeding import derive_seed
from core.train import evaluate as _evaluate_split


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network.

    Port of thirdparty/repdistiller/distiller_zoo/KD.py::DistillKL from the official repo.
    `size_average=False` there is spelled `reduction="sum"` here -- the same reduction,
    without the deprecation warning. Note the `T**2` factor and the division by the batch
    size (not by batch_size * num_classes): SCRUB's published learning rates are tuned
    against this scaling.
    """

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T ** 2) / y_s.shape[0]
        return loss


def sgda_adjust_learning_rate(epoch, optimizer, sgda_learning_rate, lr_decay_epochs, lr_decay_rate):
    """
    SCRUB's LR schedule. Port of thirdparty/repdistiller/helper/util.py::adjust_learning_rate,
    which the official notebooks import as `sgda_adjust_learning_rate` and call once per
    epoch with `epoch` counting from 1.

    Decays by `lr_decay_rate` once per passed milestone in `lr_decay_epochs`. Kept verbatim
    down to the quirk that the optimizer's param groups are only written to once at least
    one milestone has passed -- before that the optimizer keeps the LR it was constructed
    with, which is the same `sgda_learning_rate` anyway.
    """
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    return new_lr


def train_distill(train_loader, model_s, model_t, criterion_cls, criterion_div, optimizer,
                  device, split, gamma, alpha):
    """
    One SCRUB epoch, in either direction. Port of
    thirdparty/repdistiller/helper/loops.py::train_distill restricted to `opt.distill == 'kd'`
    (see the module docstring).

    split == "minimize": one epoch over the retain set, descending
        `gamma * CE(student, y) + alpha * KL(student || teacher)`.
    split == "maximize": one epoch over the forget set, descending `-KL(student || teacher)`
        -- i.e. ascending the divergence from the teacher. Note the original applies no
        `alpha` to the max-step: its weight is fixed at 1, so `alpha` only ever trades the
        min-step's distillation term against its cross-entropy term.

    Returns (top1_accuracy, mean_loss) for "minimize" and mean_loss for "maximize",
    matching the original's two return shapes. Both means are weighted by batch size, as
    the original's `AverageMeter` updates are.
    """
    model_s.train()
    model_t.eval()

    loss_sum, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.float().to(device), y.to(device)

        logit_s = model_s(x)
        with torch.no_grad():
            logit_t = model_t(x)

        loss_cls = criterion_cls(logit_s, y)
        loss_div = criterion_div(logit_s, logit_t)

        if split == "minimize":
            # The original's `+ beta * loss_kd` is dropped: loss_kd is identically 0 for
            # distill='kd', which is what every official SCRUB config uses.
            loss = gamma * loss_cls + alpha * loss_div
        elif split == "maximize":
            loss = -loss_div
        else:
            raise ValueError(f"split must be 'minimize' or 'maximize', got {split!r}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        if split == "minimize":
            correct += (logit_s.argmax(dim=1) == y).sum().item()

    mean_loss = loss_sum / max(total, 1)
    if split == "minimize":
        return 100.0 * correct / max(total, 1), mean_loss
    return mean_loss


def _labels_of(dataset: Dataset) -> torch.Tensor:
    """
    The label tensor of a dataset, without a full pass where that can be avoided.

    Handles the shapes this project actually produces -- `TensorDataset` (data_utils.py)
    and `Subset` of one -- and falls back to iterating for anything else.
    """
    if isinstance(dataset, Subset):
        return _labels_of(dataset.dataset)[torch.as_tensor(dataset.indices, dtype=torch.long)]
    tensors = getattr(dataset, "tensors", None)
    if tensors is not None and len(tensors) >= 2:
        return tensors[1]
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return torch.as_tensor(targets)
    return torch.as_tensor([int(dataset[i][1]) for i in range(len(dataset))])


def class_matched_val_subset(val_set: Dataset, forget_set: Dataset) -> Subset:
    """
    The slice of `val_set` whose labels appear in `forget_set` -- SCRUB+R's reference
    population.

    From the paper (Section 3.2): "we create a validation set of the same distribution as
    the forget set. For instance, if the forget set has only examples of class 0, we keep
    only examples of class 0 in the validation set too." This project's forget split is
    class-centric (cifar100_bs_1's forget set covers 10 of the 100 classes), so the
    restriction is substantive: the error the full validation set would report is
    dominated by the 90 classes nobody is forgetting.
    """
    forget_labels = torch.unique(_labels_of(forget_set))
    val_labels = _labels_of(val_set)
    keep = torch.nonzero(torch.isin(val_labels, forget_labels), as_tuple=False).flatten()
    return Subset(val_set, keep.tolist())


def select_rewind_epoch(forget_errors: Dict[int, float], reference_error: float,
                        tie_break: str = "earliest") -> int:
    """
    SCRUB+R's rewind point: the epoch "where the forget error is closest to that reference
    point" (paper, Section 3.2). `forget_errors` maps epoch number -> that checkpoint's
    forget-set error in percent.

    The paper does not say what to do when several epochs are equally close. That matters
    whenever the criterion saturates: if the unlearned model gets *nothing* right on
    held-out examples of the forgotten classes, the reference is 100%, the forget error is
    100% at every epoch, and the tie-break alone decides which model you keep. An
    undertrained teacher does exactly that. A properly trained one does not -- on this
    project's 200-epoch CIFAR-100 sweep the reference lands around 92-94% and the
    per-epoch forget errors around 88-96%, so the selection is genuinely per-run and the
    tie-break changes about one run in sixty. Hence `--scrub-rewind-tie-break`, which
    exists so the saturated case is a visible choice rather than a silent one:

      "earliest" (default): the first epoch that got close enough, i.e. the reading of
          "rewind" and of "just high enough" that the paper's own wording suggests. Least
          perturbation of the trained model, but it also discards the trailing min-steps
          SCRUB prescribes for repairing retain accuracy.
      "latest": rewind only when some epoch is *strictly* closer, so SCRUB+R degenerates
          to SCRUB whenever the criterion is indifferent.
    """
    epochs = sorted(forget_errors)
    if tie_break == "latest":
        epochs = list(reversed(epochs))
    elif tie_break != "earliest":
        raise ValueError(f"tie_break must be 'earliest' or 'latest', got {tie_break!r}")
    return min(epochs, key=lambda e: abs(forget_errors[e] - reference_error))


def _make_optimizer(params, optim_name: str, lr: float, momentum: float, weight_decay: float):
    """SCRUB's optimizer choice, ported from the official notebooks' `args.optim` switch.
    Adam takes no momentum there, exactly as below."""
    optim_name = optim_name.lower()
    if optim_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optim_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optim_name in ("rmsp", "rmsprop"):
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown scrub_optim {optim_name!r}; choose 'sgd', 'adam' or 'rmsprop'")


def _as_milestones(lr_decay_epochs) -> Tuple[int, ...]:
    """Accept `[3, 5, 9]`, `(3, 5, 9)`, `"3,5,9"` or `3` -- YAML, CLI and the registry
    default all reach this from different directions."""
    if lr_decay_epochs is None:
        return ()
    if isinstance(lr_decay_epochs, str):
        return tuple(int(part) for part in lr_decay_epochs.split(",") if part.strip())
    if isinstance(lr_decay_epochs, (int, np.integer)):
        return (int(lr_decay_epochs),)
    return tuple(int(e) for e in lr_decay_epochs)


def scrub(student, teacher, retain_set, forget_set, device, *, epochs: int, lr: float,
          msteps: int, alpha: float, gamma: float, kd_temperature: float, optim_name: str,
          momentum: float, weight_decay: float, retain_batch_size: int, forget_batch_size: int,
          lr_decay_epochs: Sequence[int], lr_decay_rate: float, num_workers: int = 0,
          retain_generator=None, forget_generator=None, track_forget_error: bool = False,
          eval_sets: Optional[List[Tuple[str, Dataset]]] = None, eval_batch_size: int = 256,
          checkpoint_every: Optional[int] = None, keep_every_epoch_state: bool = False
          ) -> Tuple[List[dict], Dict[int, dict], Dict[int, dict]]:
    """
    SCRUB's alternating max/min loop (Algorithm 1). Port of the official notebooks' driver
    cell.

    Per epoch: adjust the LR, then -- while `epoch <= msteps` -- one max-step epoch over
    the forget set, then always one min-step epoch over the retain set. Both steps share a
    single optimizer, as in the original ("We use the same optimizer for both min and max
    steps").

    `retain_batch_size` and `forget_batch_size` are deliberately separate: the paper tunes
    them per experiment "to control the number of iterations in each direction" (Table 3).

    `track_forget_error` adds a forget-set evaluation after every epoch, recorded as
    `forget_accuracy` / `forget_error`. `_unlearn` passes True for both methods: SCRUB+R's
    rewinding is defined in terms of it, and for plain SCRUB it is cheap next to a
    retain-set epoch and is the one number that says whether unlearning is happening.

    `eval_sets` / `eval_batch_size` / `checkpoint_every` behave exactly as they do for
    every other method in this package (see `unlearning/__init__.py`); this function does
    no file I/O.

    `keep_every_epoch_state` additionally retains *every* epoch's state_dict in the third
    return value, including the final one, which is what SCRUB+R rewinds through. It is
    independent of `checkpoint_every`, which decides what the caller writes to disk.

    Returns (history, checkpoints, epoch_states).
    """
    milestones = _as_milestones(lr_decay_epochs)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_div = DistillKL(kd_temperature).to(device)
    optimizer = _make_optimizer(student.parameters(), optim_name, lr, momentum, weight_decay)

    retain_loader = DataLoader(retain_set, batch_size=retain_batch_size, shuffle=True,
                                num_workers=num_workers, generator=retain_generator)
    forget_loader = DataLoader(forget_set, batch_size=forget_batch_size, shuffle=True,
                                num_workers=num_workers, generator=forget_generator)

    history: List[dict] = []
    checkpoints: Dict[int, dict] = {}
    epoch_states: Dict[int, dict] = {}
    for epoch in range(1, epochs + 1):
        epoch_lr = sgda_adjust_learning_rate(epoch, optimizer, lr, milestones, lr_decay_rate)

        maximize_loss = None
        if epoch <= msteps:
            maximize_loss = train_distill(forget_loader, student, teacher, criterion_cls,
                                           criterion_div, optimizer, device, "maximize",
                                           gamma, alpha)
        retain_acc, minimize_loss = train_distill(retain_loader, student, teacher, criterion_cls,
                                                   criterion_div, optimizer, device, "minimize",
                                                   gamma, alpha)

        epoch_record = {
            "epoch": epoch,
            "lr": epoch_lr,
            "scrub_max_loss": maximize_loss,   # None on the epochs that are min-step only
            "scrub_min_loss": minimize_loss,
            "retain_accuracy": retain_acc,     # measured during the min-step, not a second pass
        }
        msg = (f"[unlearn] epoch {epoch}/{epochs}  lr={epoch_lr:g}  "
               f"max_loss={'--' if maximize_loss is None else f'{maximize_loss:.4f}'}  "
               f"min_loss={minimize_loss:.4f}  retain_acc(train)={retain_acc:.2f}%")
        if track_forget_error:
            _, forget_acc = _evaluate_split(student, forget_set, device, batch_size=eval_batch_size)
            epoch_record["forget_accuracy"] = forget_acc
            epoch_record["forget_error"] = 100.0 - forget_acc
            msg += f"  forget_err={100.0 - forget_acc:.2f}%"
        print(msg)

        if eval_sets is not None:
            for name, dataset in eval_sets:
                eval_loss, eval_acc = _evaluate_split(student, dataset, device, batch_size=eval_batch_size)
                epoch_record[f"{name}_loss"] = eval_loss
                epoch_record[f"{name}_acc"] = eval_acc
                print(f"    {name:>14s}: loss={eval_loss:.4f} acc={eval_acc:.2f}%")

        # Same skip-the-final-epoch rule every method in this package follows: the caller
        # already holds the last epoch's weights as the returned model.
        save_to_disk = (checkpoint_every is not None and checkpoint_every > 0
                        and epoch % checkpoint_every == 0 and epoch < epochs)
        if keep_every_epoch_state or save_to_disk:
            state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            if keep_every_epoch_state:
                epoch_states[epoch] = state
            if save_to_disk:
                checkpoints[epoch] = state
                print(f"[unlearn] epoch {epoch}/{epochs}  snapshotted intermediate checkpoint")

        history.append(epoch_record)
    return history, checkpoints, epoch_states


def _unlearn(trained_model: torch.nn.Module, retain_set, forget_set, device: torch.device, *,
             rewind: bool, val_set=None, rewind_tie_break: str = "earliest",
             batch_size: int, epochs: int, lr: float,
             seed: Optional[int], num_workers: int, eval_sets, eval_batch_size: int,
             checkpoint_every: Optional[int], scrub_alpha: float, scrub_gamma: float,
             scrub_msteps: int, scrub_kd_temperature: float, scrub_optim: str,
             scrub_momentum: float, scrub_weight_decay: float,
             scrub_forget_batch_size: Optional[int], scrub_lr_decay_epochs, scrub_lr_decay_rate: float):
    """Shared body of `unlearn` (SCRUB) and `unlearn_rewind` (SCRUB+R)."""
    retain_shuffle_seed = derive_seed(seed, "scrub_retain_shuffle") if seed is not None else None
    forget_shuffle_seed = derive_seed(seed, "unlearn_shuffle") if seed is not None else None

    def _generator(sub_seed):
        g = torch.Generator()
        if sub_seed is not None:
            g.manual_seed(sub_seed)
        else:
            g.seed()
        return g

    teacher = copy.deepcopy(trained_model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(trained_model).to(device)
    student.train()

    forget_batch_size = scrub_forget_batch_size or batch_size
    print(f"[unlearn] retain={len(retain_set)} forget={len(forget_set)} "
          f"retain_bs={batch_size} forget_bs={forget_batch_size} epochs={epochs} "
          f"msteps={scrub_msteps} lr={lr} optim={scrub_optim} alpha={scrub_alpha} "
          f"gamma={scrub_gamma} kd_T={scrub_kd_temperature} rewind={rewind} "
          f"retain_shuffle_seed={retain_shuffle_seed} forget_shuffle_seed={forget_shuffle_seed}")

    history, checkpoints, epoch_states = scrub(
        student, teacher, retain_set, forget_set, device,
        epochs=epochs, lr=lr, msteps=scrub_msteps, alpha=scrub_alpha, gamma=scrub_gamma,
        kd_temperature=scrub_kd_temperature, optim_name=scrub_optim, momentum=scrub_momentum,
        weight_decay=scrub_weight_decay, retain_batch_size=batch_size,
        forget_batch_size=forget_batch_size, lr_decay_epochs=scrub_lr_decay_epochs,
        lr_decay_rate=scrub_lr_decay_rate, num_workers=num_workers,
        retain_generator=_generator(retain_shuffle_seed),
        forget_generator=_generator(forget_shuffle_seed),
        track_forget_error=True, eval_sets=eval_sets, eval_batch_size=eval_batch_size,
        checkpoint_every=checkpoint_every, keep_every_epoch_state=rewind,
    )

    if rewind:
        _rewind_to_reference(student, val_set, forget_set, device, history, checkpoints,
                             epoch_states, epochs, eval_batch_size, checkpoint_every,
                             tie_break=rewind_tie_break)

    student.eval()
    print("[unlearn] finished")
    return student, history, checkpoints


def _rewind_to_reference(student, val_set, forget_set, device, history, checkpoints,
                         epoch_states, epochs, eval_batch_size, checkpoint_every,
                         tie_break: str = "earliest") -> None:
    """
    SCRUB+R's Section 3.2 rewinding, applied in place to `student`, `history` and
    `checkpoints`.

    The reference error is the *final* SCRUB model's error on validation examples drawn
    from the forget set's own classes -- "the last step of unlearning approximates
    'maximally forgetting'", so its error on never-trained-on examples of those classes
    approximates the forget error a retrained-from-scratch model would have had. The
    student is then reloaded from the epoch whose forget error sits closest to it.

    The rewind summary is written onto `history`'s last entry (`rewind_*` keys) rather
    than appended as a pseudo-epoch, so `history` stays a list of one dict per epoch.
    """
    if val_set is None:
        raise ValueError(
            "scrub_r needs a validation set to build its rewind reference point, but "
            "val_set=None was passed. Both drivers supply one; a direct caller must too."
        )

    reference_set = class_matched_val_subset(val_set, forget_set)
    if len(reference_set) == 0:
        print("[unlearn] warning: no validation examples share a class with the forget set; "
              "falling back to the full validation set for the rewind reference")
        reference_set = val_set

    _, reference_acc = _evaluate_split(student, reference_set, device, batch_size=eval_batch_size)
    reference_error = 100.0 - reference_acc

    forget_errors = {rec["epoch"]: rec["forget_error"] for rec in history}
    selected = select_rewind_epoch(forget_errors, reference_error, tie_break=tie_break)
    best_distance = abs(forget_errors[selected] - reference_error)
    tied = [e for e, err in forget_errors.items() if abs(err - reference_error) == best_distance]

    print(f"[unlearn] SCRUB+R rewind: reference forget error {reference_error:.2f}% "
          f"(from {len(reference_set)} class-matched validation examples); per-epoch forget "
          f"error " + ", ".join(f"{e}:{forget_errors[e]:.2f}%" for e in sorted(forget_errors))
          + f" -> rewinding to epoch {selected}"
          + (f" ({len(tied)} epochs tied, tie_break={tie_break})" if len(tied) > 1 else ""))

    history[-1].update({
        "rewind_reference_error": reference_error,
        "rewind_reference_set_size": len(reference_set),
        "rewind_selected_epoch": selected,
        "rewind_selected_forget_error": forget_errors[selected],
        "rewind_tie_break": tie_break,
        "rewind_tied_epochs": sorted(tied),
    })

    if selected != epochs:
        # The un-rewound end state is no longer what the caller saves as the unlearned
        # model, but it is still the thing every earlier epoch was measured against, so
        # keep it alongside the other intermediate checkpoints when the caller wants them.
        if checkpoint_every is not None and checkpoint_every > 0:
            checkpoints[epochs] = epoch_states[epochs]
        student.load_state_dict(epoch_states[selected])
        student.to(device)


def unlearn(trained_model: torch.nn.Module, model_fn: Callable[[], torch.nn.Module], retain_set,
            forget_set, device: torch.device, *, batch_size: int = 128, epochs: int = 3,
            lr: float = 5e-4, seed: Optional[int] = None, num_workers: int = 0,
            eval_sets: Optional[List[Tuple[str, Dataset]]] = None, eval_batch_size: int = 256,
            checkpoint_every: Optional[int] = None, scrub_alpha: float = 0.001,
            scrub_gamma: float = 0.99, scrub_msteps: int = 2, scrub_kd_temperature: float = 4.0,
            scrub_optim: str = "sgd", scrub_momentum: float = 0.9,
            scrub_weight_decay: float = 5e-4, scrub_forget_batch_size: Optional[int] = None,
            scrub_lr_decay_epochs=(3, 5, 9), scrub_lr_decay_rate: float = 0.1):
    """
    SCRUB's implementation of this package's shared unlearning interface (see
    `unlearning/__init__.py` for the contract every method implements), mirroring the
    official notebooks' usage:

        model_t = copy.deepcopy(teacher); model_s = copy.deepcopy(student)
        for epoch in range(1, args.sgda_epochs + 1):
            lr = sgda_adjust_learning_rate(epoch, args, optimizer)
            if epoch <= args.msteps:
                train_distill(epoch, forget_loader, module_list, ..., "maximize")
            train_distill(epoch, retain_loader, module_list, ..., "minimize")

    Deep-copies `trained_model` into both the student and the frozen teacher, so
    `trained_model` itself is left untouched.

    The common `batch_size` is the *retain* (min-step) batch size; `scrub_forget_batch_size`
    is the max-step's, defaulting to the same value. `epochs` is the paper's STEPS and
    `scrub_msteps` its MAX-STEPS.

    `model_fn` is accepted only because it is part of the shared interface -- SCRUB never
    builds a second network (bad-teacher does).

    `seed` seeds both DataLoaders' shuffle orders via independent sub-seeds (see
    `seeding.derive_seed`), kept separate from `--seed` (training). If None, both draw
    from ambient (non-reproducible) randomness.

    Returns (student, history, checkpoints).
    """
    del model_fn  # unused by SCRUB; part of the shared interface, see docstring
    return _unlearn(
        trained_model, retain_set, forget_set, device, rewind=False, batch_size=batch_size,
        epochs=epochs, lr=lr, seed=seed, num_workers=num_workers, eval_sets=eval_sets,
        eval_batch_size=eval_batch_size, checkpoint_every=checkpoint_every,
        scrub_alpha=scrub_alpha, scrub_gamma=scrub_gamma, scrub_msteps=scrub_msteps,
        scrub_kd_temperature=scrub_kd_temperature, scrub_optim=scrub_optim,
        scrub_momentum=scrub_momentum, scrub_weight_decay=scrub_weight_decay,
        scrub_forget_batch_size=scrub_forget_batch_size,
        scrub_lr_decay_epochs=scrub_lr_decay_epochs, scrub_lr_decay_rate=scrub_lr_decay_rate,
    )


def unlearn_rewind(trained_model: torch.nn.Module, model_fn: Callable[[], torch.nn.Module],
                   retain_set, forget_set, device: torch.device, *, val_set=None,
                   batch_size: int = 128, epochs: int = 3, lr: float = 5e-4,
                   seed: Optional[int] = None, num_workers: int = 0,
                   eval_sets: Optional[List[Tuple[str, Dataset]]] = None, eval_batch_size: int = 256,
                   checkpoint_every: Optional[int] = None, scrub_alpha: float = 0.001,
                   scrub_gamma: float = 0.99, scrub_msteps: int = 2,
                   scrub_kd_temperature: float = 4.0, scrub_optim: str = "sgd",
                   scrub_momentum: float = 0.9, scrub_weight_decay: float = 5e-4,
                   scrub_forget_batch_size: Optional[int] = None,
                   scrub_lr_decay_epochs=(3, 5, 9), scrub_lr_decay_rate: float = 0.1,
                   scrub_rewind_tie_break: str = "earliest"):
    """
    SCRUB+R: exactly `unlearn` above, plus the paper's rewinding step (Section 3.2).

    Every epoch's weights are kept in memory, the final model's error on validation
    examples from the forget set's classes becomes the reference point, and the returned
    model is the epoch whose forget error is closest to it -- not necessarily the last.
    Same hyperparameters, same trajectory, same seeds: for a fixed seed SCRUB+R's
    trajectory is bit-identical to SCRUB's, and only the choice of which point on it to
    return differs.

    `val_set` is the extra input SCRUB+R declares in the registry (`extra_inputs`); both
    drivers pass their validation split. Passing None is an error.

    `scrub_rewind_tie_break` decides which epoch wins when several are equally close to
    the reference -- on a class-centric forget split that is the usual case, not a corner;
    see `select_rewind_epoch`.

    `history`'s last entry carries the `rewind_*` summary keys (reference error, reference
    set size, selected epoch, tie-break and which epochs tied). `checkpoints` still follows
    `checkpoint_every`, with the final SCRUB epoch added when rewinding actually moved the
    returned model off it.

    Returns (student, history, checkpoints).
    """
    del model_fn  # unused by SCRUB; part of the shared interface, see docstring
    return _unlearn(
        trained_model, retain_set, forget_set, device, rewind=True, val_set=val_set,
        rewind_tie_break=scrub_rewind_tie_break,
        batch_size=batch_size, epochs=epochs, lr=lr, seed=seed, num_workers=num_workers,
        eval_sets=eval_sets, eval_batch_size=eval_batch_size, checkpoint_every=checkpoint_every,
        scrub_alpha=scrub_alpha, scrub_gamma=scrub_gamma, scrub_msteps=scrub_msteps,
        scrub_kd_temperature=scrub_kd_temperature, scrub_optim=scrub_optim,
        scrub_momentum=scrub_momentum, scrub_weight_decay=scrub_weight_decay,
        scrub_forget_batch_size=scrub_forget_batch_size,
        scrub_lr_decay_epochs=scrub_lr_decay_epochs, scrub_lr_decay_rate=scrub_lr_decay_rate,
    )
