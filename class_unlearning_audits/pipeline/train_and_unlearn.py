"""
Train-then-unlearn pipeline: one run, one unlearning algorithm, chosen by config.

Which algorithm runs is a configuration decision, not a code decision -- see the
`unlearning` package for the registered methods and the interface they share:

    --unlearn-method delete      DELETE / decoupled distillation to erase
                                 (Zhou et al., CVPR 2025, arXiv:2503.23751).
    --unlearn-method badteacher  "Can Bad Teaching Induce Forgetting?"
                                 (Chundawat et al., AAAI 2023, arXiv:2205.08096).
    --unlearn-method scrub       SCRUB, "Towards Unbounded Machine Unlearning"
                                 (Kurmanji et al., NeurIPS 2023, arXiv:2302.09880).
    --unlearn-method scrub_r     SCRUB+R: the same run, rewound to the epoch whose
                                 forget error matches a class-matched validation
                                 reference (Sec. 3.2 of the same paper).

All are near-verbatim ports of their papers' official implementations; each module's
docstring carries the citation and the exact list of deviations. Everything downstream --
the checkpoints written here, and the whole audit stack that reads them -- is identical
either way, which is the point: the methods are meant to be compared on the same
pipeline, the same trained models, and the same audit.

The three stages:

    1. data_utils.sample_forget_batches  -- randomly keep whole batches of the forget set
       for this run (the rest of the forget set is simply not used); retain/val/test are
       untouched.
    2. train.train                        -- ordinary training on retain + sampled-forget.
       This becomes the network the unlearning stage will unlearn from (and doubles
       as the teacher for badteacher, scrub and scrub_r).
    3. <method>.unlearn                   -- fine-tune a copy of the trained model so it
       forgets the sampled forget batches. What "forget" means here is exactly what
       differs between the methods; see the `unlearning` package.

Seeding contract:
    --seed          controls ONLY model weight initialization and the training set's
                    batch order. The order is shuffled once (before epoch 1) from this
                    seed and then held fixed for every subsequent epoch. A fixed --seed
                    always reproduces the same trained model.
    --unlearn-seed  controls forget-batch sampling and everything inside the unlearning
                    stage (each method derives its own independent sub-seeds from it via
                    seeding.derive_seed -- every method's forget DataLoader batch order,
                    plus the incompetent teacher's init for badteacher and the retain
                    DataLoader's order for scrub/scrub_r). Independent of
                    --seed: if not given, a fresh seed is drawn from OS randomness on
                    every run (and printed + saved to config.json so the run can be
                    reproduced later by passing --unlearn-seed explicitly).

Usage:
    python -m pipeline.train_and_unlearn --config configs/delete.yaml
    python -m pipeline.train_and_unlearn --config configs/badteacher.yaml --unlearn-epochs 1
    python -m pipeline.train_and_unlearn --config configs/scrub_r.yaml
    python -m pipeline.train_and_unlearn --unlearn-method delete --unlearn-lr 3e-3
"""
from __future__ import print_function

import argparse
import json
import os

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from core.config import (add_selection_arguments, add_unlearning_arguments, describe, resolve,
                    unlearn_call_kwargs)
from core.data_utils import sample_forget_batches
from core.model import build_model, dataset_defaults
from core.seeding import derive_seed, fresh_seed, set_training_determinism
from core.train import evaluate, train
from unlearning import get_method

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "cifar100", "data_split", "cifar100_bs_1")

#: The splits reported before/after unlearning, and (when --eval-every-unlearn-epoch is
#: on) after every unlearning epoch too. One definition, used everywhere in this file.
EVAL_SPLIT_NAMES = ("train", "retain", "sampled_forget", "full_forget", "val", "test")


def run_train_and_unlearn(args: argparse.Namespace) -> dict:
    """
    Orchestrates the full pipeline in order: sample forget batches -> train -> unlearn.

    Returns a dict with the trained/unlearned models, forget indices, and before/after
    accuracy metrics -- this is also what gets written to disk under args.out_dir.
    """
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    method = get_method(args.unlearn_method)

    # --unlearn-seed governs forget sampling + unlearning; drawn fresh if not pinned, so it
    # is independent of --seed (training) by default. Sub-seeds keep the random decisions
    # inside the unlearning stage from correlating with each other.
    unlearn_seed = args.unlearn_seed if args.unlearn_seed is not None else fresh_seed()
    forget_sample_seed = derive_seed(unlearn_seed, "forget_sample")

    print(f"Using device: {device}")
    print(f"Unlearning method: {args.unlearn_method} -- {method.description}")
    print(f"Unlearning config: {describe(args)}")
    print(f"Training seed (--seed): {args.seed}")
    print(f"Unlearning seed (--unlearn-seed): {unlearn_seed} "
          f"({'pinned' if args.unlearn_seed is not None else 'drawn fresh this run'})")

    # ---- Stage 1: sample forget batches (independent of the training seed) ----
    train_set, retain_set, forget_set, full_forget_set, val_set, test_set, forget_indices = \
        sample_forget_batches(
            args.dataset, args.data_dir, args.batch_size,
            forget_prob=args.forget_prob, seed=forget_sample_seed,
        )

    # ---- Resolve model/architecture config ----
    default_input_size, default_num_classes, default_model_name = dataset_defaults(args.dataset)
    input_size = default_input_size
    num_classes = args.num_classes if args.num_classes is not None else default_num_classes
    model_name = args.model if args.model is not None else default_model_name

    def model_fn():
        return build_model(model_name, num_classes, input_size=input_size, filters_percentage=args.filters)

    # ---- Seed training: model init + the one-time batch shuffle, independent of unlearn_seed ----
    set_training_determinism(args.seed, device)
    model = model_fn().to(device)
    train_shuffle_generator = torch.Generator().manual_seed(args.seed)  # used once; train() fixes the order after

    # ---- Stage 2: train the model that will be unlearned ----
    print("\n[STAGE 2] Training on retain + sampled-forget...")
    trained_model, train_history = train(
        model, train_set, val_set, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, optimizer_type=args.optimizer, momentum=args.momentum,
        shuffle_generator=train_shuffle_generator, scheduler_type=args.lr_scheduler,
        scheduler_step_size=args.scheduler_step_size, scheduler_gamma=args.scheduler_gamma,
    )

    eval_sets = list(zip(EVAL_SPLIT_NAMES,
                         (train_set, retain_set, forget_set, full_forget_set, val_set, test_set)))

    print("\n[EVALUATION - after training]")
    metrics_before = {}
    for name, dataset in eval_sets:
        loss, acc = evaluate(trained_model, dataset, device)
        metrics_before[name] = {"loss": loss, "accuracy": acc}
        print(f"  {name:>14s}: loss={loss:.4f} acc={acc:.2f}%")

    # ---- Stage 3: unlearn the sampled forget batches (independent of --seed) ----
    print(f"\n[STAGE 3] Unlearning sampled forget batches ({args.unlearn_method})...")
    unlearned_model, unlearn_history, unlearn_checkpoints = method.unlearn(
        trained_model, model_fn, retain_set, forget_set, device,
        seed=unlearn_seed, **unlearn_call_kwargs(args, eval_sets, val_set),
    )

    print("\n[EVALUATION - after unlearning]")
    metrics_after = {}
    for name, dataset in eval_sets:
        loss, acc = evaluate(unlearned_model, dataset, device)
        metrics_after[name] = {"loss": loss, "accuracy": acc}
        print(f"  {name:>14s}: loss={loss:.4f} acc={acc:.2f}%")

    return {
        "device": device,
        "unlearn_seed": unlearn_seed,
        "forget_sample_seed": forget_sample_seed,
        "trained_model": trained_model,
        "unlearned_model": unlearned_model,
        "train_history": train_history,
        "unlearn_history": unlearn_history,
        "unlearn_checkpoints": unlearn_checkpoints,
        "forget_indices": forget_indices,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "model_name": model_name,
        "num_classes": num_classes,
        "input_size": input_size,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    add_selection_arguments(parser)

    parser.add_argument("--dataset", default="cifar100", help="Dataset name (default: cifar100)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                         help="Root folder containing retain/forget/val/test subfolders "
                              "(either flat .pt files or the cifar100_bs_1-style .pkl batch layout, "
                              "auto-detected; default: data/cifar100/data_split/cifar100_bs_1)")
    parser.add_argument("--out-dir", default="runs", help="Where to save models/config/metrics (default: runs)")
    parser.add_argument("--run-name", default=None, help="Subfolder name under --out-dir (default: auto-generated)")

    parser.add_argument("--model", default=None,
                         help="Default: chosen per dataset (mnist->mlp, cifar10->tinynet, "
                              "cifar100->tinynet_cifar100 -- see model.py's dataset_defaults)")
    parser.add_argument("--num-classes", type=int, default=None, help="Default: chosen per dataset")
    parser.add_argument("--filters", type=float, default=1.0, help="Filter-count multiplier for conv models")

    # Stage 1: forget sampling
    parser.add_argument("--forget-prob", type=float, default=0.5,
                         help="Fraction of forget *batches* to keep for this run, e.g. 0.5 keeps "
                              "exactly half of them, chosen uniformly at random (default: 0.5)")

    # Stage 2: training -- identical for both unlearning methods
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=2e-2,
                         help="Training learning rate (default: 2e-2 -- since --lr-scheduler defaults to "
                              "cosine, which anneals back down to 0 by the end instead of holding the peak "
                              "LR for all --epochs)")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum for SGD")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "step"], default="cosine",
                         help="Training LR schedule (default: cosine, anneals to 0 over --epochs). "
                              "'none' keeps --lr constant; 'step' decays by --scheduler-gamma every "
                              "--scheduler-step-size epochs.")
    parser.add_argument("--scheduler-step-size", type=int, default=None,
                         help="Epoch interval for 'step' schedule (default: --epochs // 3)")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                         help="Decay factor for 'step' schedule (default: 0.1)")

    # Stage 3: unlearning -- common flags plus every method's own, defaults filled in per
    # method by config.resolve (see the `unlearning` registry).
    add_unlearning_arguments(parser)

    # Seeding
    parser.add_argument("--seed", type=int, default=42,
                         help="Seeds ONLY model init + training batch order (default: 42)")
    parser.add_argument("--unlearn-seed", type=int, default=None,
                         help="Seeds forget-batch sampling + unlearning. Independent of --seed; "
                              "drawn fresh from OS randomness if omitted.")

    parser.add_argument("--no-cuda", action="store_true", default=False)
    return parser


def main():
    args = resolve(_build_arg_parser())

    result = run_train_and_unlearn(args)

    run_name = args.run_name or (
        f"{args.dataset}_{result['model_name']}_{args.unlearn_method}"
        f"_seed{args.seed}_unlearnseed{result['unlearn_seed']}"
    )
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    trained_path = os.path.join(run_dir, "trained_model.pth")
    unlearned_path = os.path.join(run_dir, "unlearned_model.pth")
    indices_path = os.path.join(run_dir, "forget_indices.npy")
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.json")

    torch.save(result["trained_model"].state_dict(), trained_path)
    torch.save(result["unlearned_model"].state_dict(), unlearned_path)
    np.save(indices_path, result["forget_indices"])

    for epoch_num, state_dict in result["unlearn_checkpoints"].items():
        ckpt_path = os.path.join(run_dir, f"unlearned_model_epoch_{epoch_num}.pth")
        torch.save(state_dict, ckpt_path)
        print(f"Saved intermediate checkpoint (epoch {epoch_num}) to: {ckpt_path}")

    config = vars(args).copy()
    config["resolved_unlearn_seed"] = result["unlearn_seed"]
    config["resolved_forget_sample_seed"] = result["forget_sample_seed"]
    config["resolved_model_name"] = result["model_name"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump({
            "unlearn_method": args.unlearn_method,
            "before_unlearning": result["metrics_before"],
            "after_unlearning": result["metrics_after"],
            "train_history": result["train_history"],
            "unlearn_history": result["unlearn_history"],
        }, f, indent=2)

    print(f"\nSaved trained model to:   {trained_path}")
    print(f"Saved unlearned model to: {unlearned_path}")
    print(f"Saved forget indices to:  {indices_path}")
    print(f"Saved config to:          {config_path}")
    print(f"Saved metrics to:         {metrics_path}")


if __name__ == "__main__":
    main()
