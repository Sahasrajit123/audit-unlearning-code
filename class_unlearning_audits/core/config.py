"""
Configuration resolution shared by train_and_unlearn.py and run_sweep.py.

Both drivers take the same two selection arguments:

    --unlearn-method {badteacher,delete,scrub,scrub_r}
                                           which algorithm from the `unlearning` package
    --config path/to/config.yaml           a file of settings (see configs/)

and everything else can come from either the file or the command line. The resolution
order, lowest priority to highest, is:

    1. the argparse default declared on the flag
    2. the chosen method's `defaults`      (unlearning/__init__.py)
    3. the chosen method's `sweep_defaults` (run_sweep.py only)
    4. the YAML file given by --config
    5. an explicitly typed CLI flag

One combination is rejected rather than resolved: a `--config` whose `unlearn_method:`
disagrees with an explicit `--unlearn-method`. Those config files bundle a method with
its own paper's tuned hyperparameters, so blending them would quietly run one method on
the other's settings. Pick the matching config, or drop the flag.

Layers 2-3 are what make `--unlearn-method badteacher` alone do the right thing: the
methods' published hyperparameters genuinely differ (20 epochs of SGD at lr 1e-3 vs
5 of Adam at lr 3e-3 vs SCRUB's 3 of SGD at lr 5e-4), so there is no single sensible
argparse default for
`--unlearn-epochs`. Layer 5 sitting above layer 4 is what makes a config file a starting
point rather than a straitjacket -- `--config configs/delete.yaml --unlearn-lr 3e-3`
means what it looks like it means.

Telling layer 5 apart from layer 1 is the only subtle part, since argparse gives you a
namespace where "the user typed --unlearn-lr 1e-3" and "the default happened to be 1e-3"
are indistinguishable. `explicitly_passed` below re-parses the same argv with every
default swapped for `argparse.SUPPRESS`, which makes argparse populate a dest only when
the flag actually appeared.

YAML schema: flat keys matching the CLI flag names, with either dashes or underscores
(`unlearn-epochs` and `unlearn_epochs` are the same key). Unknown keys are an error, so a
typo fails loudly instead of being silently ignored. Two nested keys are special, and
hold the settings that exist in only one of the two drivers:

    sweep:        read only by run_sweep.py         (num_runs, gpus, runs_per_gpu, ...)
    single:       read only by train_and_unlearn.py (out_dir, run_name, unlearn_seed, ...)

Whichever block belongs to the other driver is skipped rather than rejected, so one file
can configure both.
"""
import argparse
import os
from typing import Any, Dict, Iterable, Optional

import yaml

from unlearning import ALL_HYPERPARAMS, METHODS, get_method

#: Nested YAML blocks, each consumed by exactly one driver and skipped by the other.
SWEEP_SECTION = "sweep"
SINGLE_SECTION = "single"
DRIVER_SECTIONS = (SWEEP_SECTION, SINGLE_SECTION)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --unlearn-method / --config. Call before the rest of a driver's arguments."""
    parser.add_argument(
        "--unlearn-method", choices=sorted(METHODS), default="delete",
        help="Which unlearning algorithm to run (default: delete). "
             + "  ".join(f"'{n}': {m.description}" for n, m in sorted(METHODS.items())),
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="YAML file of settings (see configs/). Values here override each method's "
             "built-in defaults, and are themselves overridden by any flag you type "
             "explicitly.",
    )


def _per_method_defaults(setting: str) -> str:
    """"delete: 20, badteacher: 5, ..." -- for help text that can't know the method yet."""
    return ", ".join(f"{name}: {method.defaults[setting]}"
                     for name, method in sorted(METHODS.items())
                     if setting in method.defaults)


def add_unlearning_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the unlearning-stage flags shared by both drivers.

    Defaults are left as None / SUPPRESS-like sentinels wherever the right value depends
    on the method; `resolve` fills those in from the registry. The help text says which
    method supplies what, since `--help` can't know which method you'll pick.
    """
    parser.add_argument("--unlearn-batch-size", type=int, default=None,
                         help="Default: same as --batch-size. For scrub/scrub_r this is the "
                              "retain-set (min-step) batch size; see --scrub-forget-batch-size "
                              "for the other direction.")
    parser.add_argument("--unlearn-epochs", type=int, default=None,
                         help="Unlearning fine-tune epochs (default: per method -- "
                              f"{_per_method_defaults('unlearn_epochs')})")
    parser.add_argument("--unlearn-lr", type=float, default=None,
                         help="Unlearning learning rate (default: per method -- "
                              f"{_per_method_defaults('unlearn_lr')}; each method's optimizer "
                              "is the one its paper hardcodes -- SGD(momentum=0.9) for delete, "
                              "Adam for badteacher, --scrub-optim for scrub/scrub_r)")
    parser.add_argument("--eval-every-unlearn-epoch", action=argparse.BooleanOptionalAction,
                         default=None,
                         help="Evaluate loss+accuracy on every split (train/retain/sampled_forget/"
                              "full_forget/val/test) after every unlearning epoch, not just once at "
                              "the end. Costs a full pass over every split each epoch. "
                              "Default: on for delete, off for badteacher (which always reports "
                              "forget accuracy per epoch regardless).")
    parser.add_argument("--unlearn-checkpoint-every", type=int, default=None,
                         help="Save an intermediate unlearned-model checkpoint every N unlearning "
                              "epochs (e.g. 5 -> epochs 5, 10, 15, ... saved as "
                              "unlearned_model_epoch_{N}.pth in the run directory), skipping the "
                              "final epoch (already saved as unlearned_model.pth). Pass 0 or a "
                              "negative value to disable. Default: none for a single run; for a "
                              "sweep, 3 for delete and 1 for badteacher/scrub/scrub_r.")

    # One flag per method-specific hyperparameter. All are always registered so that
    # --help is complete and a config file is portable; `resolve` warns if you set one
    # that the method you picked does not read.
    group = parser.add_argument_group(
        "method-specific unlearning hyperparameters",
        "Each is read by the --unlearn-method(s) named in brackets and ignored (with a "
        "warning) by the others.",
    )
    group.add_argument("--disable-bn", action=argparse.BooleanOptionalAction, default=None,
                        help="[delete] Freeze BatchNorm running stats during unlearning (the "
                             "official repo only enables this for single-class forgetting on "
                             "Tiny-ImageNet). Default: off.")
    group.add_argument("--temperature", type=float, default=None,
                        help="[badteacher] KL-divergence distillation temperature. Default: 1.0.")
    group.add_argument("--scrub-alpha", type=float, default=None,
                        help="[scrub, scrub_r] Weight on the min-step's KL-to-teacher term. "
                             "Default: 0.001.")
    group.add_argument("--scrub-gamma", type=float, default=None,
                        help="[scrub, scrub_r] Weight on the min-step's retain cross-entropy "
                             "term. Default: 0.99.")
    group.add_argument("--scrub-msteps", type=int, default=None,
                        help="[scrub, scrub_r] Number of leading epochs that get a max-step on "
                             "the forget set; later epochs are min-step only (the paper's "
                             "MAX-STEPS, vs --unlearn-epochs as its STEPS). Default: 2.")
    group.add_argument("--scrub-kd-temperature", type=float, default=None,
                        help="[scrub, scrub_r] Distillation temperature (the paper's kd_T), "
                             "applied to teacher and student in both directions. Default: 4.0.")
    group.add_argument("--scrub-optim", choices=["sgd", "adam", "rmsprop"], default=None,
                        help="[scrub, scrub_r] Optimizer, shared by the min- and max-steps as "
                             "in the original. Default: sgd (the paper's large-scale setting; "
                             "its small-scale experiments use adam).")
    group.add_argument("--scrub-momentum", type=float, default=None,
                        help="[scrub, scrub_r] Momentum for --scrub-optim sgd/rmsprop (adam "
                             "takes none, as in the original). Default: 0.9.")
    group.add_argument("--scrub-weight-decay", type=float, default=None,
                        help="[scrub, scrub_r] Weight decay during unlearning, separate from "
                             "training's --weight-decay. Default: 5e-4 (the paper's large-scale "
                             "value; 0.1 small-scale).")
    group.add_argument("--scrub-forget-batch-size", type=int, default=None,
                        help="[scrub, scrub_r] Max-step (forget set) batch size. The paper tunes "
                             "it separately from the retain batch size to control how many "
                             "iterations each direction gets. Default: 512.")
    group.add_argument("--scrub-lr-decay-epochs", type=_int_list, default=None,
                        metavar="E1,E2,...",
                        help="[scrub, scrub_r] Comma-separated epochs after which the unlearning "
                             "LR is multiplied by --scrub-lr-decay-rate, once per passed "
                             "milestone. Default: 3,5,9.")
    group.add_argument("--scrub-lr-decay-rate", type=float, default=None,
                        help="[scrub, scrub_r] Multiplier applied at each milestone above. "
                             "Default: 0.1.")
    group.add_argument("--scrub-rewind-tie-break", choices=["earliest", "latest"], default=None,
                        help="[scrub_r] Which epoch wins when several are equally close to the "
                             "rewind reference. 'earliest' rewinds to the first epoch that got "
                             "there; 'latest' rewinds only when some epoch is strictly closer, "
                             "making scrub_r reduce to scrub otherwise. Only decisive when the "
                             "criterion saturates (forget error and reference both at 100%%, as "
                             "with an undertrained teacher). Default: earliest.")


def _int_list(value: str) -> tuple:
    """argparse type for "3,5,9" -> (3, 5, 9). Matches the official SCRUB repo's own
    --lr_decay_epochs spelling; YAML may give a real list instead, which is fine too."""
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    try:
        return tuple(int(part) for part in str(value).split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of integers, got {value!r}"
        ) from None


def explicitly_passed(parser: argparse.ArgumentParser, argv: Optional[Iterable[str]]) -> set:
    """
    Return the set of dests the user actually typed on the command line.

    Re-parses `argv` with every action's default temporarily replaced by
    `argparse.SUPPRESS`, which tells argparse to leave a dest out of the namespace
    entirely unless the flag appeared. The defaults are restored before returning, so the
    parser is unchanged for any later use.
    """
    saved = {}
    try:
        for action in parser._actions:
            if action.dest == "help":
                continue
            saved[action] = action.default
            action.default = argparse.SUPPRESS
        return set(vars(parser.parse_args(list(argv) if argv is not None else None)))
    finally:
        for action, default in saved.items():
            action.default = default


def _normalize_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """YAML keys may use dashes or underscores; argparse dests always use underscores."""
    return {str(k).replace("-", "_"): v for k, v in raw.items()}


def load_yaml_config(path: str, valid_dests: Iterable[str], section: str) -> Dict[str, Any]:
    """
    Read a config file into a flat dict of argparse dests.

    `valid_dests` is the calling driver's set of known dests; any other top-level key is
    an error rather than a silent no-op, so `unlearn_epoch: 20` (missing the 's') fails
    immediately instead of quietly running 20-epoch-by-default.

    `section` names this driver's own nested block (`sweep` or `single`); it is merged on
    top of the shared top-level keys, and the other driver's block is skipped rather than
    rejected -- that is what lets one file configure both.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}")

    raw = _normalize_keys(raw)
    blocks = {name: _normalize_keys(raw.pop(name, {}) or {}) for name in DRIVER_SECTIONS}

    valid = set(valid_dests)
    merged: Dict[str, Any] = {}
    for label, values in (("top level", raw), (f"'{section}:' block", blocks[section])):
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(
                f"{path}: unknown {label} key(s) {unknown}. "
                f"Valid keys for this command: {sorted(valid)}"
            )
        merged.update(values)
    return merged


def resolve(parser: argparse.ArgumentParser, argv: Optional[Iterable[str]] = None,
            *, sweep: bool = False) -> argparse.Namespace:
    """
    Parse `argv` and apply the full precedence chain described in this module's docstring.

    Returns a Namespace with every value resolved, plus two extras the drivers record in
    their config.json: `resolved_config_path` (the --config file used, or None) and
    `unlearn_method` (always concrete).
    """
    args = parser.parse_args(list(argv) if argv is not None else None)
    typed = explicitly_passed(parser, argv)
    valid_dests = {a.dest for a in parser._actions if a.dest != "help"}

    # Layer 1: the plain argparse defaults.
    resolved: Dict[str, Any] = {
        a.dest: a.default for a in parser._actions if a.dest != "help"
    }

    # The config file can name the method, but an explicit --unlearn-method still wins,
    # so the file has to be read before the method's defaults can be applied.
    file_values: Dict[str, Any] = {}
    if args.config is not None:
        file_values = load_yaml_config(
            args.config, valid_dests, section=SWEEP_SECTION if sweep else SINGLE_SECTION
        )

    # A config file that names a method is a bundle FOR that method -- its
    # unlearn_epochs/unlearn_lr are that paper's tuned values. Pointing at one while
    # asking for the other method would otherwise silently produce a hybrid (badteacher
    # running DELETE's 20 epochs), which is never what anyone means. Refuse instead.
    file_method = file_values.get("unlearn_method")
    if "unlearn_method" in typed and file_method is not None and file_method != args.unlearn_method:
        raise ValueError(
            f"--unlearn-method {args.unlearn_method} contradicts "
            f"'unlearn_method: {file_method}' in {args.config}. That config file carries "
            f"{file_method}'s own tuned hyperparameters, so mixing the two would run "
            f"{args.unlearn_method} with {file_method}'s settings. Use "
            f"configs/{args.unlearn_method}.yaml, or drop --unlearn-method."
        )
    if "unlearn_method" in typed:
        method_name = args.unlearn_method
    elif file_method is not None:
        method_name = file_method
    else:
        method_name = resolved["unlearn_method"]
    method = get_method(method_name)

    # Layers 2-3: the method's own defaults, then its sweep-only overrides.
    resolved.update(method.defaults)
    if sweep:
        resolved.update(method.sweep_defaults)

    # Layer 4: the config file.
    resolved.update(file_values)

    # Layer 5: anything typed on the command line.
    resolved.update({dest: getattr(args, dest) for dest in typed if dest in valid_dests})

    resolved["unlearn_method"] = method_name
    resolved["resolved_config_path"] = args.config

    final = argparse.Namespace(**resolved)

    # A hyperparameter belonging to a method you didn't pick is almost always a copy
    # -paste slip. Say so, once, rather than dropping it silently.
    for hyperparam, owners in sorted(ALL_HYPERPARAMS.items()):
        if method_name in owners:
            continue
        if hyperparam in typed or hyperparam in file_values:
            print(f"[config] warning: --{hyperparam.replace('_', '-')} only applies to "
                  f"--unlearn-method {' / '.join(owners)}; ignored for {method_name}.")
        setattr(final, hyperparam, None)

    if getattr(final, "unlearn_batch_size", None) is None:
        final.unlearn_batch_size = final.batch_size
    return final


def method_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The chosen method's own hyperparameters, ready to splat into its `unlearn` call."""
    method = get_method(args.unlearn_method)
    return {hp: getattr(args, hp) for hp in method.hyperparams}


def unlearn_call_kwargs(args: argparse.Namespace, eval_sets, val_set=None) -> Dict[str, Any]:
    """
    Build the complete keyword-argument set for a method's `unlearn` -- common, plus any
    extra data inputs it declares, plus its own hyperparameters -- so both drivers wire
    the unlearning stage up identically.

    `eval_sets` is passed through as-is when --eval-every-unlearn-epoch is on, and
    replaced by None when it is off.

    `val_set` is forwarded only to methods whose registry entry lists "val_set" in
    `extra_inputs` (scrub_r, which needs held-out data for its rewind reference point).
    Every other method's call site is left exactly as it was.
    """
    method = get_method(args.unlearn_method)
    kwargs = {
        "batch_size": args.unlearn_batch_size,
        "epochs": args.unlearn_epochs,
        "lr": args.unlearn_lr,
        "eval_sets": eval_sets if args.eval_every_unlearn_epoch else None,
        "checkpoint_every": args.unlearn_checkpoint_every,
    }
    available_inputs = {"val_set": val_set}
    for name in method.extra_inputs:
        if name not in available_inputs:
            raise KeyError(f"method {method.name!r} declares an unknown extra input {name!r}; "
                           f"this driver can supply {sorted(available_inputs)}")
        if available_inputs[name] is None:
            raise ValueError(f"method {method.name!r} needs {name}, but the caller passed None")
        kwargs[name] = available_inputs[name]
    kwargs.update(method_kwargs(args))
    return kwargs


def describe(args: argparse.Namespace) -> str:
    """One-line summary of the resolved unlearning configuration, for logs."""
    method = get_method(args.unlearn_method)
    specific = " ".join(f"{hp}={getattr(args, hp)}" for hp in method.hyperparams)
    source = os.path.basename(args.resolved_config_path) if args.resolved_config_path else "CLI/defaults"
    return (f"method={args.unlearn_method} epochs={args.unlearn_epochs} lr={args.unlearn_lr:g} "
            f"batch_size={args.unlearn_batch_size} {specific} "
            f"checkpoint_every={args.unlearn_checkpoint_every} "
            f"eval_every_epoch={args.eval_every_unlearn_epoch} [config: {source}]")
