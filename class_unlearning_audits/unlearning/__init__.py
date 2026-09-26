"""
Registry of the unlearning methods this project can run, and the interface they share.

Four methods live here, each a near-verbatim port of its paper's official implementation
(see each module's docstring for the citation and the exact list of deviations):

    delete      unlearning/delete.py     -- DELETE / decoupled distillation to erase
                                            (Zhou et al., CVPR 2025). Mask-distillation:
                                            one frozen copy of the trained model is the
                                            sole teacher, its true-class logit masked to
                                            -inf. Forget set only; no retain pass.
    badteacher  unlearning/badteacher.py -- "Can Bad Teaching Induce Forgetting?"
                                            (Chundawat et al., AAAI 2023). Gated KD
                                            between a competent teacher (the trained
                                            model) on retain and an incompetent teacher
                                            (a frozen random init) on forget.
    scrub       unlearning/scrub.py      -- SCRUB, "Towards Unbounded Machine Unlearning"
                                            (Kurmanji et al., NeurIPS 2023). Alternating
                                            max-steps (ascend KL from the trained model
                                            on forget) and min-steps (CE + KL to it on
                                            retain).
    scrub_r     unlearning/scrub.py      -- SCRUB+R: the same trajectory, returning the
                                            epoch whose forget error is closest to a
                                            reference built from class-matched validation
                                            data, rather than the last epoch (Section 3.2
                                            of the same paper). Its reason to exist is
                                            precisely the signal this project audits: a
                                            forget error that is *too* high is itself a
                                            membership signal.

They are interchangeable everywhere downstream -- train_and_unlearn.py, run_sweep.py and
the whole audit stack (audit_utils.py, run_cumulative_audit.py) only ever see a trained
checkpoint and an unlearned checkpoint, and neither depends on which method produced them.
Selecting a method is therefore purely a config decision: `--unlearn-method` or the
`unlearn_method:` key in a configs/*.yaml.

## The shared interface

Every method module exposes one function:

    unlearn(trained_model, model_fn, retain_set, forget_set, device, *,
            batch_size, epochs, lr, seed=None, num_workers=0,
            eval_sets=None, eval_batch_size=256, checkpoint_every=None,
            **extra_inputs, **method_hyperparams)
        -> (unlearned_model, history, checkpoints)

Positional arguments are the same for every method, even where a given method ignores
one (DELETE reads neither `retain_set` nor `model_fn` -- that is the substantive claim of
its paper, not an oversight). The keyword arguments split in three:

  - Common, handled identically by every method:
      batch_size, epochs, lr   the unlearning fine-tune's own hyperparameters
      seed                     this stage's seed; each method derives independent
                               sub-seeds from it via seeding.derive_seed, so nothing
                               inside unlearning correlates with the training seed
      eval_sets                optional [(name, dataset)] evaluated after every epoch and
                               recorded into `history` as {name}_loss / {name}_acc
      checkpoint_every         snapshot state_dict every N epochs, skipping the final
                               epoch (the caller already holds that one). Methods never
                               touch the filesystem; the caller decides where these land.
  - Extra data inputs, declared per method in `extra_inputs` below and passed by the
      drivers only to the methods that ask for them, so a method's call site stays
      exactly as narrow as its algorithm. Only `val_set` exists so far, and only
      scrub_r asks for it (it needs held-out data to build its rewind reference point).
  - Method-specific hyperparameters, declared per method in `hyperparams` below and
      forwarded by the drivers as **kwargs: `disable_bn` for delete, `temperature` for
      badteacher, the ten `scrub_*` knobs for scrub and scrub_r. One hyperparameter may
      be shared by several methods (scrub and scrub_r share all of theirs); the drivers
      warn only when you set one that *no* method you selected reads.

Return value is always the 3-tuple above: `history` is a list of JSON-serializable
per-epoch dicts (the per-epoch loss key is method-specific -- `mask_kd_loss` vs
`gated_kd_loss` vs SCRUB's `scrub_max_loss`/`scrub_min_loss` -- since they are different
objectives and conflating them in metrics.json would be a lie), and `checkpoints` maps
epoch number -> CPU state_dict.

## Defaults

`defaults` carries each method's published/tuned hyperparameters, and is what makes
`--unlearn-method X` alone sufficient to run X correctly. These are deliberately NOT
unified across methods: DELETE's 20 epochs of SGD(momentum=0.9) at lr 1e-3,
bad-teacher's 5 epochs of Adam at lr 3e-3 and SCRUB's 3 epochs of SGD at lr 5e-4 come
from their respective papers, and the sweeps already on disk under runs/ were produced
with exactly these values.

`sweep_defaults` overrides `defaults` for run_sweep.py only, where the methods
checkpoint at different granularities (DELETE every 3rd of 20 epochs; bad-teacher and
SCRUB every intermediate epoch of 5 and 3 respectively).

Precedence, lowest to highest: argparse default -> method `defaults` -> method
`sweep_defaults` (run_sweep.py only) -> configs/*.yaml -> explicit CLI flag. See config.py.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

from . import badteacher as _badteacher
from . import delete as _delete
from . import scrub as _scrub


@dataclass(frozen=True)
class UnlearningMethod:
    """One registered unlearning method: its entry point, its knobs, and its defaults."""

    name: str
    unlearn: Callable
    description: str
    #: Names of this method's own hyperparameters, beyond the common ones every method
    #: takes. The drivers expose each as a CLI flag and forward it as a keyword argument.
    hyperparams: Tuple[str, ...]
    #: Per-method defaults for any setting whose right value depends on the method.
    defaults: Dict[str, Any] = field(default_factory=dict)
    #: Overrides applied on top of `defaults` by run_sweep.py only.
    sweep_defaults: Dict[str, Any] = field(default_factory=dict)
    #: Optional data inputs this method needs beyond retain/forget, passed by the drivers
    #: as keyword arguments only to the methods that declare them. Supported: "val_set".
    extra_inputs: Tuple[str, ...] = ()


#: SCRUB and SCRUB+R are the same algorithm up to which checkpoint gets returned, so they
#: share this knob list and the defaults below. SCRUB+R adds to both: the validation split
#: it rewinds against (`extra_inputs`) and `scrub_rewind_tie_break`, neither of which
#: means anything until you are choosing between checkpoints.
_SCRUB_HYPERPARAMS: Tuple[str, ...] = (
    "scrub_alpha", "scrub_gamma", "scrub_msteps", "scrub_kd_temperature", "scrub_optim",
    "scrub_momentum", "scrub_weight_decay", "scrub_forget_batch_size",
    "scrub_lr_decay_epochs", "scrub_lr_decay_rate",
)

#: The official repo's large-scale CIFAR-10 *class* unlearning setting for ResNet
#: (large_scale_unlearning.ipynb's hyperparameter cell, plus Table 3 of the paper): the
#: closest published configuration to this project's class-centric CIFAR-100 forget split.
#: The retain (min-step) batch size is the shared --unlearn-batch-size, whose own default
#: is --batch-size = 128, which is also the paper's value for this setting.
_SCRUB_DEFAULTS: Dict[str, Any] = {
    "unlearn_epochs": 3,          # args.sgda_epochs
    "unlearn_lr": 5e-4,           # args.sgda_learning_rate
    "scrub_alpha": 0.001,
    "scrub_gamma": 0.99,
    "scrub_msteps": 2,
    "scrub_kd_temperature": 4.0,  # args.kd_T
    "scrub_optim": "sgd",
    "scrub_momentum": 0.9,
    "scrub_weight_decay": 5e-4,
    "scrub_forget_batch_size": 512,
    "scrub_lr_decay_epochs": (3, 5, 9),
    "scrub_lr_decay_rate": 0.1,
    # SCRUB reports forget error every epoch on its own (and SCRUB+R needs it), so the
    # extra full-split passes are off by default, as for badteacher.
    "eval_every_unlearn_epoch": False,
    "unlearn_checkpoint_every": None,
}


METHODS: Dict[str, UnlearningMethod] = {
    "delete": UnlearningMethod(
        name="delete",
        unlearn=_delete.unlearn,
        description="DELETE: mask-distillation from a frozen copy of the trained model "
                    "(Zhou et al., CVPR 2025). Forget set only.",
        hyperparams=("disable_bn",),
        defaults={
            # The official repo's recommended CIFAR-10 ResNet18 setting for --method delete:
            # SGD(momentum=0.9), lr 1e-3, 20 epochs (scripts.sh).
            "unlearn_epochs": 20,
            "unlearn_lr": 1e-3,
            "disable_bn": False,
            # Cheap relative to 20 epochs of distillation, and the DELETE sweeps on disk
            # were all run with it on.
            "eval_every_unlearn_epoch": True,
            "unlearn_checkpoint_every": None,
        },
        sweep_defaults={"unlearn_checkpoint_every": 3},
    ),
    "badteacher": UnlearningMethod(
        name="badteacher",
        unlearn=_badteacher.unlearn,
        description="Bad teacher: gated KD between the trained model (on retain) and a "
                    "frozen random init (on forget) (Chundawat et al., AAAI 2023).",
        hyperparams=("temperature",),
        defaults={
            "unlearn_epochs": 5,
            "unlearn_lr": 3e-3,
            "temperature": 1.0,
            # This method already evaluates forget accuracy every epoch on its own, and
            # its sweeps on disk were run without the extra full-split passes.
            "eval_every_unlearn_epoch": False,
            "unlearn_checkpoint_every": None,
        },
        sweep_defaults={"unlearn_checkpoint_every": 1},
    ),
    "scrub": UnlearningMethod(
        name="scrub",
        unlearn=_scrub.unlearn,
        description="SCRUB: alternating max-steps on the forget set and min-steps on the "
                    "retain set, both against the trained model as teacher "
                    "(Kurmanji et al., NeurIPS 2023).",
        hyperparams=_SCRUB_HYPERPARAMS,
        defaults=dict(_SCRUB_DEFAULTS),
        sweep_defaults={"unlearn_checkpoint_every": 1},
    ),
    "scrub_r": UnlearningMethod(
        name="scrub_r",
        unlearn=_scrub.unlearn_rewind,
        description="SCRUB+R: SCRUB, then rewind to the epoch whose forget error is "
                    "closest to a reference measured on class-matched validation data "
                    "(Kurmanji et al., NeurIPS 2023, Sec. 3.2).",
        # Everything SCRUB has, plus the one knob that only means something once you are
        # choosing between checkpoints.
        hyperparams=_SCRUB_HYPERPARAMS + ("scrub_rewind_tie_break",),
        defaults=dict(_SCRUB_DEFAULTS, scrub_rewind_tie_break="earliest"),
        sweep_defaults={"unlearn_checkpoint_every": 1},
        # The rewind reference point is the final model's error on held-out examples of
        # the forget set's classes, so SCRUB+R -- alone among the four -- needs the
        # validation split handed to it.
        extra_inputs=("val_set",),
    ),
}

#: Every method-specific hyperparameter across all methods, mapped to the methods that
#: read it, so the drivers can register one CLI flag per knob and warn when one is set
#: that the chosen method ignores. A knob can have several owners (scrub and scrub_r
#: share all ten of theirs).
ALL_HYPERPARAMS: Dict[str, Tuple[str, ...]] = {
    hp: tuple(m.name for m in METHODS.values() if hp in m.hyperparams)
    for method in METHODS.values() for hp in method.hyperparams
}


def get_method(name: str) -> UnlearningMethod:
    """Look up a registered method, with a useful error listing the valid names."""
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(
            f"unknown unlearning method {name!r}; choose one of {sorted(METHODS)}"
        ) from None


__all__ = ["METHODS", "ALL_HYPERPARAMS", "UnlearningMethod", "get_method"]
