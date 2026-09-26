"""
The "DELETE" (mask-distillation) unlearning method -- one of the two methods registered
in `unlearning/__init__.py` (see `--unlearn-method delete`).

This is a direct adaptation of the official paper implementation:
    Zhou, Zheng, Mo, Lu, Lin & Zheng, "Decoupled Distillation to Erase: A General
    Unlearning Method for Any Class-centric Tasks", CVPR 2025 (Highlight).
    Paper: https://arxiv.org/abs/2503.23751
    Code:  https://github.com/shaaaaron/DELETE
           (method/delete.py::delete -- the unlearn_model/test_model setup, the
           `pred_label[arange(batch_size), y] = -1e10` masking trick, and the
           `nn.KLDivLoss(reduction='batchmean')` objective are ported near-verbatim.)

`delete_step` and `delete` below port the official `delete()` function's core
training loop (same variable roles: `unlearn_model` is the student being
fine-tuned, `test_model` is a frozen deep-copy of the trained model, serving as the
*sole* teacher -- unlike bad-teacher unlearning, DELETE needs no second,
randomly-initialized "incompetent teacher" network). The only changes:
  - all of the official repo's run-management plumbing (tqdm progress bar,
    per-epoch accuracy evaluation on four loaders, matplotlib plotting, its own
    logging framework) is dropped -- this project's train_and_unlearn.py /
    run_sweep.py already do before/after evaluation and logging themselves, the
    same way they do for every unlearning stage;
  - the unlearning DataLoader's shuffle order is drawn from an explicit, seeded
    `torch.Generator` instead of ambient global RNG state, matching this project's
    train/unlearn seed-independence contract (see train_and_unlearn.py's module
    docstring) -- the same treatment unlearning/badteacher.py gives its own
    unlearning dataloader.

Method recap: `unlearn_model` (the student) starts as a copy of the fully-trained
network. For every forget-set sample (x, y), a frozen copy of that *same* trained
network ("test_model") produces logits for x, and the true-class logit y is masked
to -1e10 before softmax -- collapsing that class's probability to ~0 while leaving
the *relative* probabilities ("dark knowledge") among every other class untouched.
`unlearn_model` is then fine-tuned to match that masked distribution via
KL-divergence. Because the mask only ever touches each sample's own forgotten
class, a single loss term simultaneously drives the forgetting objective (true
class -> ~0 probability) and the retention objective (every other class's relative
ranking is preserved by distilling from the model's own pre-unlearning knowledge)
-- no retain-set pass and no second teacher network are needed.
"""
import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.seeding import derive_seed
from core.train import evaluate as _evaluate_split


def delete_step(unlearn_model, test_model, forget_loader, optimizer, device, disable_bn=False):
    """One epoch of mask-distillation unlearning. Port of the inner loop of the
    official repo's method/delete.py::delete."""
    criterion = nn.KLDivLoss(reduction="batchmean")
    losses = []
    for x, y in forget_loader:
        x, y = x.to(device), y.to(device)
        unlearn_model.train()

        if disable_bn:
            for module in unlearn_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        unlearn_model.zero_grad()
        optimizer.zero_grad()

        test_model.eval()
        batch_size = x.shape[0]
        with torch.no_grad():
            pred_label = test_model(x)
        pred_label[torch.arange(batch_size), y] = -1e10  # mask each sample's own (forgotten) class

        student_logits = unlearn_model(x)
        student_out = F.log_softmax(student_logits, dim=1)
        teacher_out = F.softmax(pred_label, dim=1)

        loss = criterion(student_out, teacher_out)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return float(np.mean(losses))


def delete(unlearn_model, test_model, forget_loader, epochs, lr, device,
           disable_bn=False, generator=None, eval_sets: Optional[List[Tuple[str, Dataset]]] = None,
           eval_batch_size: int = 256, checkpoint_every: Optional[int] = None
           ) -> Tuple[List[dict], Dict[int, dict]]:
    """
    The mask-distillation unlearning loop. Port of method/delete.py::delete from
    the official repo. Only the forget set is used -- no retain-set pass, no
    separate incompetent-teacher network -- `unlearn_model` and `test_model` both
    start as copies of the same trained network, with `test_model` frozen as the
    sole teacher. `optimizer` is SGD(momentum=0.9), matching the official repo
    (not exposed as a choice -- the official repo hardcodes it too).

    `generator` is accepted so the forget DataLoader's shuffle order can be pinned
    to this run's unlearn-seed; unused by the official repo (which relies on
    ambient RNG state instead).

    `eval_sets`, if given, is a list of (name, dataset) pairs evaluated with
    `train.evaluate` after every epoch (not just once at the very end) --
    matching the official repo's own `evaluate_model_on_all_loaders` per-epoch
    check, which this port otherwise omits (see the module docstring) since it's
    extra evaluation passes over full datasets every epoch, not free. Off by
    default (None) so the common case (a large sweep) doesn't pay for it uninvited.

    `checkpoint_every`, if given, snapshots `unlearn_model.state_dict()` (detached,
    moved to CPU) every `checkpoint_every` epochs, skipping the final epoch (the
    caller already has that one -- it's whatever `unlearn_model` ends up as when
    this function returns). This function does no file I/O itself -- consistent
    with the rest of this package, saving checkpoints to disk is left to the caller
    (train_and_unlearn.py / run_sweep.py), which is where run-directory paths live.

    Returns (history, checkpoints): `history` is the usual list of per-epoch dicts
    (JSON-serializable, safe to dump straight into metrics.json); `checkpoints`
    maps epoch number -> state_dict for whichever epochs matched `checkpoint_every`.
    """
    del generator  # threaded through by `unlearn` via forget_loader's own construction
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=lr, momentum=0.9)
    test_model.eval()
    for p in test_model.parameters():
        p.requires_grad_(False)

    history = []
    checkpoints: Dict[int, dict] = {}
    for epoch in range(epochs):
        loss = delete_step(unlearn_model, test_model, forget_loader, optimizer, device, disable_bn=disable_bn)
        print(f"[unlearn] epoch {epoch + 1}/{epochs}  mask-distillation loss: {loss:.4f}")
        epoch_num = epoch + 1
        epoch_record = {"epoch": epoch_num, "mask_kd_loss": loss}
        if eval_sets is not None:
            for name, dataset in eval_sets:
                eval_loss, eval_acc = _evaluate_split(unlearn_model, dataset, device, batch_size=eval_batch_size)
                epoch_record[f"{name}_loss"] = eval_loss
                epoch_record[f"{name}_acc"] = eval_acc
                print(f"    {name:>14s}: loss={eval_loss:.4f} acc={eval_acc:.2f}%")
        if checkpoint_every is not None and checkpoint_every > 0 \
                and epoch_num % checkpoint_every == 0 and epoch_num < epochs:
            checkpoints[epoch_num] = {k: v.detach().cpu().clone() for k, v in unlearn_model.state_dict().items()}
            print(f"[unlearn] epoch {epoch_num}/{epochs}  snapshotted intermediate checkpoint")
        history.append(epoch_record)
    return history, checkpoints


def unlearn(trained_model: torch.nn.Module, model_fn: Callable[[], torch.nn.Module], retain_set,
            forget_set, device: torch.device, *, batch_size: int = 128, epochs: int = 20,
            lr: float = 1e-3, seed: Optional[int] = None, num_workers: int = 0,
            eval_sets: Optional[List[Tuple[str, Dataset]]] = None, eval_batch_size: int = 256,
            checkpoint_every: Optional[int] = None, disable_bn: bool = False):
    """
    DELETE's implementation of this package's shared unlearning interface (see
    `unlearning/__init__.py` for the contract every method implements), mirroring
    the official repo's `main.py` usage for `--method delete`:

        unlearn_model = method.delete(ori_model, train_forget_loader,
                                       unlearn_epoch, unlearn_rate, ...)

    Deep-copies `trained_model` into both the student ("unlearn_model") and the
    frozen teacher ("test_model") -- so `trained_model` itself is left untouched.

    `disable_bn` is DELETE's one method-specific hyperparameter: it freezes
    BatchNorm running stats during unlearning (the official repo only enables it
    for single-class forgetting on Tiny-ImageNet).

    `retain_set` and `model_fn` are accepted only because they are part of the
    shared interface -- DELETE's algorithm itself never reads either, matching the
    paper's claim of unlearning "without access to the remaining data or
    intervention." (bad-teacher unlearning needs both.)

    `seed` seeds the forget DataLoader's shuffle order via an independent sub-seed
    (see `seeding.derive_seed`), kept separate from `--seed` (training). If None,
    draws from ambient (non-reproducible) randomness.

    `eval_sets`, if given, is forwarded to `delete` to print/record loss+accuracy on
    every named split after every unlearning epoch (see `delete`'s docstring).

    `checkpoint_every`, if given, is forwarded to `delete` (see its docstring).

    Returns (student, history, checkpoints). `checkpoints` maps epoch number ->
    state_dict for whichever epochs matched `checkpoint_every`; an empty dict if
    `checkpoint_every` was None.
    """
    del retain_set, model_fn  # unused by DELETE; part of the shared interface, see docstring

    shuffle_seed = derive_seed(seed, "unlearn_shuffle") if seed is not None else None
    generator = torch.Generator()
    if shuffle_seed is not None:
        generator.manual_seed(shuffle_seed)
    else:
        generator.seed()

    test_model = copy.deepcopy(trained_model).to(device).eval()
    for p in test_model.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(trained_model).to(device)
    student.train()

    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, generator=generator)

    print(f"[unlearn] forget={len(forget_set)} batch_size={batch_size} epochs={epochs} "
          f"lr={lr} disable_bn={disable_bn} shuffle_seed={shuffle_seed}")

    history, checkpoints = delete(student, test_model, forget_loader, epochs, lr, device, disable_bn=disable_bn,
                                   eval_sets=eval_sets, eval_batch_size=eval_batch_size,
                                   checkpoint_every=checkpoint_every)

    student.eval()
    print("[unlearn] finished")
    return student, history, checkpoints
