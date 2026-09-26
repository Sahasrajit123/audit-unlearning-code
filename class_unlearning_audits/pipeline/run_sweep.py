"""
Launch many independent train+unlearn runs, all differing only in which forget points get
sampled -- the sweep the privacy audit consumes.

Which unlearning algorithm every run uses is a configuration decision (`--unlearn-method`,
or `unlearn_method:` in a configs/*.yaml); see train_and_unlearn.py's module docstring for
the pipeline and the `unlearning` package for the registered methods. A sweep is
single-method by construction: every run in a <run-dir> shares one config.json, and the
audit treats the whole directory as one mechanism to bound. Run one sweep per method to
compare them.

Seeding -- every per-run seed is a base plus the run index, so the whole sweep is
reproducible from three plain integers:
    --seed               fixed across every run in the sweep -- model init + the training
                         set's one-time batch shuffle are identical for all runs.
    --unlearn-seed-base  run i uses unlearn_seed = unlearn_seed_base + i, which seeds
                         everything inside the unlearning stage. Each method derives its
                         own independent sub-seeds from it via seeding.derive_seed (the
                         forget DataLoader's batch order for every method, plus the
                         incompetent teacher's random init for badteacher and the retain
                         DataLoader's order for scrub/scrub_r), so those never correlate
                         with each other.
    --forget-seed-base   run i uses forget_sample_seed = forget_seed_base + i, which seeds
                         which forget batches get sampled for that run.
Both bases default to disjoint ranges (1000, 2000) so their additive sequences never
collide, but there's no OS-randomness fallback here either way -- unlike a single
train_and_unlearn.py invocation without --unlearn-seed, every run in a sweep is always
fully reproducible from (--seed, --unlearn-seed-base, --forget-seed-base) alone.

Reusing trained models (--reuse-trained-from <sweep>): stage 2 (200 epochs x 60 runs) is
by far the most expensive part of a sweep, and it does not depend on ANY unlearning
hyperparameter -- run i's trained model is a function of (--seed, --forget-seed-base, the
dataset and the training hyperparameters) alone. So a sweep that only varies the
unlearning stage (a different --unlearn-lr, --unlearn-epochs, or even a different method)
can load run_i/trained_model.pth from an existing sweep instead of retraining it, turning
~1.5 hours into ~minutes. The source run folder is looked up as <sweep>/run_NN or
<sweep>/test_run/run_NN (split_test_run.py moves the held-out runs), and reuse is refused
unless every training-relevant setting in the source sweep's config.json matches this one
AND each run's re-sampled forget_indices are identical to the ones the source model was
actually trained on -- a trained model paired with the wrong forget sample would silently
invalidate the audit. The copy saved into this sweep's run_NN/trained_model.pth is the
source file's weights, so the audit's --use-trained-model stats still work here.

Parallelism: runs are dispatched across a persistent pool of worker processes (one per
GPU slot, via --gpus / --runs-per-gpu), using ProcessPoolExecutor's `initializer` so each
worker loads retain/val/test and claims its GPU exactly ONCE when it starts, then reuses
that for every run routed to it -- not once per run. This matters a lot for the default
cifar100_bs_1 data layout, where a cold load means opening ~46,000 small .pkl files
(subsequent loads hit data_utils.py's on-disk cache and are fast either way).

Ctrl+C / kill: SIGINT and SIGTERM are both caught and force-kill every worker process
immediately (see _kill_pool_workers) instead of the ProcessPoolExecutor default of waiting
for in-flight runs to finish naturally, which for long training runs could mean hours of
apparent hang -- previously leading to abandoned terminals/sessions and orphaned,
GPU-memory-holding worker processes with no live parent left to manage them.

Output, to keep the terminal readable across many parallel runs:
    - Terminal: only "[run NN] started on GPU g at TIME" / "[run NN] completed on GPU g
      at TIME (took Xs)" lines, printed by the dispatching wrapper OUTSIDE each run's
      redirected block.
    - <run-dir>/run_NN/run_NN.log: everything else -- forget-sampling summary, every
      training epoch's train loss/accuracy and val loss/accuracy, every unlearning
      epoch's loss (the method's own objective), and the full before/after evaluation
      (train/retain/forget/val/test loss + accuracy).

Layout:
    <run-dir>/
      config.json           -- sweep-wide settings shared by every run (unlearn_method,
                              dataset, model, hyperparameters, --seed, --unlearn-seed-base,
                              --forget-seed-base, --gpus, --runs-per-gpu, ...)
      run_01/  trained_model.pth  unlearned_model.pth  forget_indices.npy  metrics.json
               unlearned_model_epoch_{N}.pth  -- intermediate unlearning checkpoints, for
                                  each epoch matching --unlearn-checkpoint-every (the
                                  final epoch is always just unlearned_model.pth)
               run_01.log       -- this run's full training/unlearning output
               run_config.json  -- ONLY what's specific to this run_id: run_idx,
                                  unlearn_seed, forget_sample_seed (see config.json for
                                  everything else, which is identical across every run)
      run_02/  ...
      ...
      sweep_summary.json   -- one row per run: seeds + before/after accuracy for every split

Usage:
    python -m pipeline.run_sweep --config configs/delete.yaml     --run-dir runs/delete/cifar100_bs_1
    python -m pipeline.run_sweep --config configs/badteacher.yaml --run-dir runs/badteacher/cifar100_bs_1
    python -m pipeline.run_sweep --config configs/scrub.yaml      --run-dir runs/scrub/cifar100_bs_1
    python -m pipeline.run_sweep --config configs/scrub_r.yaml    --run-dir runs/scrub_r/cifar100_bs_1
    python -m pipeline.run_sweep --unlearn-method delete --run-dir runs/delete/quick --num-runs 4 --gpus 3,4
"""
import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from core.config import (add_selection_arguments, add_unlearning_arguments, describe, resolve,
                    unlearn_call_kwargs)
from core.data_utils import load_retain_val_test, sample_forget
from core.model import build_model, dataset_defaults
from core.seeding import set_training_determinism
from core.train import evaluate, train
from pipeline.train_and_unlearn import DEFAULT_DATA_DIR, EVAL_SPLIT_NAMES
from unlearning import get_method

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Populated once per worker process by _init_worker (persists across every run that
# worker is asked to handle -- see the module docstring's Parallelism section).
_worker = {}


def _init_worker(gpu_queue, dataset: str, data_dir: str):
    """Runs once when a worker process starts: claim a GPU slot, load data once."""
    gpu_id = gpu_queue.get() if gpu_queue is not None else None
    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else torch.device("cpu")
    retain_set, val_set, test_set = load_retain_val_test(dataset, data_dir)
    _worker.update(device=device, gpu_id=gpu_id, retain_set=retain_set, val_set=val_set, test_set=test_set)
    gpu_label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    print(f"[worker pid={os.getpid()}] ready on {gpu_label} "
          f"(retain={len(retain_set)} val={len(val_set)} test={len(test_set)})", flush=True)


# Training-stage settings a reused trained_model.pth depends on. Anything outside this
# list (every unlearn_* key, the method itself, the GPU layout) is free to differ -- that
# is the entire point of reuse. `unlearn_seed_base` is deliberately absent: it seeds only
# stage 3. `forget_seed_base` is present because it decides which forget points the model
# was trained on.
_REUSE_TRAINING_KEYS = (
    "dataset", "data_dir", "model", "num_classes", "input_size", "filters", "forget_prob",
    "batch_size", "epochs", "lr", "weight_decay", "optimizer", "momentum",
    "lr_scheduler", "scheduler_step_size", "scheduler_gamma", "seed", "forget_seed_base",
)


def _reused_train_history(source_run_dir: str):
    """
    The source run's per-epoch training history, so a reusing sweep's metrics.json keeps the
    same schema as a normally-trained one. Returns None if the source has no metrics.json
    (e.g. a run that crashed during unlearning) -- the trained model is still valid, only
    its history is missing, which is not worth failing a run over.
    """
    metrics_path = os.path.join(source_run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f).get("train_history")


def _reuse_source_run_dir(source_sweep: str, run_idx: int) -> str:
    """Locate run_NN inside a source sweep, whether or not it has been held out into test_run/."""
    name = f"run_{run_idx:02d}"
    candidates = [os.path.join(source_sweep, name), os.path.join(source_sweep, "test_run", name)]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "trained_model.pth")):
            return candidate
    raise FileNotFoundError(
        f"--reuse-trained-from: no trained_model.pth for run {run_idx:02d}; looked in "
        f"{' and '.join(candidates)}"
    )


def _check_reuse_compatible(source_sweep: str, sweep_config: dict) -> dict:
    """
    Refuse to reuse trained models from a sweep whose TRAINING stage differs from this one.

    Called once in main() before any worker starts, so a mismatched --reuse-trained-from
    fails in a second instead of 60 runs later. Per-run forget_indices are checked
    separately inside _run_one -- that is the check that actually guarantees each loaded
    model saw this run's forget sample.
    """
    source_config_path = os.path.join(source_sweep, "config.json")
    if not os.path.isfile(source_config_path):
        raise FileNotFoundError(f"--reuse-trained-from: {source_config_path} not found")
    with open(source_config_path, "r") as f:
        source_config = json.load(f)

    mismatched = {
        key: (source_config.get(key, "<missing>"), sweep_config[key])
        for key in _REUSE_TRAINING_KEYS
        if source_config.get(key, "<missing>") != sweep_config[key]
    }
    if mismatched:
        detail = "\n".join(f"    {key}: source={src!r} but this sweep wants {dst!r}"
                           for key, (src, dst) in sorted(mismatched.items()))
        raise ValueError(
            f"--reuse-trained-from {source_sweep}: its trained models were produced under "
            f"different training settings, so they are not the models this sweep would have "
            f"trained:\n{detail}\n"
            f"  Only the unlearning stage may differ between a sweep and the one it reuses."
        )
    return source_config


def _run_one(run_idx: int, args, retain_set, val_set, test_set, device, model_fn, run_dir: str) -> dict:
    """Sample forget -> train -> unlearn -> save. Everything here is meant to be captured
    by the per-run log file (see _dispatch_run, which wraps this in redirect_stdout)."""
    method = get_method(args.unlearn_method)
    unlearn_seed = args.unlearn_seed_base + run_idx
    forget_sample_seed = args.forget_seed_base + run_idx

    sampled_forget_set, full_forget_set, forget_indices = sample_forget(
        args.dataset, args.data_dir, args.batch_size,
        forget_prob=args.forget_prob, seed=forget_sample_seed,
    )
    train_set = ConcatDataset([retain_set, sampled_forget_set])

    # Save immediately -- these are fully determined already and don't depend on
    # training/unlearning succeeding, so a crash mid-run (OOM, killed job, ...) still
    # leaves a record of which forget batches and seeds this run_id was assigned.
    # Everything shared across the sweep (unlearn_method, dataset, model, hyperparameters,
    # --seed, --unlearn-seed-base, ...) lives once in the top-level <run-dir>/config.json
    # instead of being repeated in every run folder.
    np.save(os.path.join(run_dir, "forget_indices.npy"), forget_indices)
    run_config = {"run_idx": run_idx, "unlearn_seed": unlearn_seed, "forget_sample_seed": forget_sample_seed}
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Model init + training batch order: identical across every run in the sweep.
    set_training_determinism(args.seed, device)
    model = model_fn().to(device)

    reuse_source = getattr(args, "reuse_trained_from", None)
    if reuse_source is not None:
        source_run_dir = _reuse_source_run_dir(reuse_source, run_idx)

        # The load-bearing check: the source model was trained on retain + ITS forget
        # sample. If this run sampled a different one, pairing them would attack a model
        # that never saw these points -- wrong in a way no downstream stage could detect.
        source_forget_indices = np.load(os.path.join(source_run_dir, "forget_indices.npy"))
        if not np.array_equal(source_forget_indices, forget_indices):
            raise ValueError(
                f"[run {run_idx:02d}] --reuse-trained-from {source_run_dir}: that model was "
                f"trained on a different forget sample ({len(source_forget_indices)} points) "
                f"than this run draws ({len(forget_indices)} points at "
                f"forget_sample_seed={forget_sample_seed}); refusing to pair them."
            )

        model.load_state_dict(torch.load(os.path.join(source_run_dir, "trained_model.pth"),
                                          map_location=device, weights_only=False))
        trained_model = model
        train_history = _reused_train_history(source_run_dir)
        print(f"[reuse] loaded trained_model.pth from {source_run_dir} "
              f"(skipped {args.epochs} training epochs; train_history "
              f"{'copied from source' if train_history is not None else 'unavailable'})")
    else:
        shuffle_generator = torch.Generator().manual_seed(args.seed)
        trained_model, train_history = train(
            model, train_set, val_set, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, optimizer_type=args.optimizer, momentum=args.momentum,
            shuffle_generator=shuffle_generator, scheduler_type=args.lr_scheduler,
            scheduler_step_size=args.scheduler_step_size, scheduler_gamma=args.scheduler_gamma,
        )

    eval_sets = list(zip(EVAL_SPLIT_NAMES,
                         (train_set, retain_set, sampled_forget_set, full_forget_set, val_set, test_set)))
    metrics_before = {name: dict(zip(("loss", "accuracy"), evaluate(trained_model, ds, device)))
                       for name, ds in eval_sets}

    unlearned_model, unlearn_history, unlearn_checkpoints = method.unlearn(
        trained_model, model_fn, retain_set, sampled_forget_set, device,
        seed=unlearn_seed, **unlearn_call_kwargs(args, eval_sets, val_set),
    )
    metrics_after = {name: dict(zip(("loss", "accuracy"), evaluate(unlearned_model, ds, device)))
                      for name, ds in eval_sets}

    torch.save(trained_model.state_dict(), os.path.join(run_dir, "trained_model.pth"))
    torch.save(unlearned_model.state_dict(), os.path.join(run_dir, "unlearned_model.pth"))
    for epoch_num, state_dict in unlearn_checkpoints.items():
        ckpt_path = os.path.join(run_dir, f"unlearned_model_epoch_{epoch_num}.pth")
        torch.save(state_dict, ckpt_path)

    metrics_record = {
        "unlearn_method": args.unlearn_method,
        "before_unlearning": metrics_before, "after_unlearning": metrics_after,
        "train_history": train_history, "unlearn_history": unlearn_history,
    }
    if reuse_source is not None:
        # Provenance: this run's trained model was not trained here.
        metrics_record["trained_model_reused_from"] = source_run_dir
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"[run {run_idx:02d}] unlearn_seed={unlearn_seed} forget_kept={len(forget_indices)} | "
          f"retain {metrics_before['retain']['accuracy']:.2f}->{metrics_after['retain']['accuracy']:.2f}%  "
          f"forget {metrics_before['sampled_forget']['accuracy']:.2f}->{metrics_after['sampled_forget']['accuracy']:.2f}%  "
          f"test {metrics_before['test']['accuracy']:.2f}->{metrics_after['test']['accuracy']:.2f}%")

    return {
        "run": run_idx, "seed": args.seed, "unlearn_seed": unlearn_seed,
        "forget_sample_seed": forget_sample_seed, "num_forget_kept": int(len(forget_indices)),
        "metrics_before": metrics_before, "metrics_after": metrics_after,
    }


def _dispatch_run(run_idx: int, args) -> dict:
    """
    Runs inside a worker process (via the pool's persistent retain/val/test + device from
    _init_worker). Prints only two lines to the *actual* terminal -- start and completion,
    with GPU + timestamps -- and redirects everything _run_one prints to that run's own
    log file.
    """
    device, gpu_id = _worker["device"], _worker["gpu_id"]
    gpu_label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"

    run_dir = os.path.join(args.run_dir, f"run_{run_idx:02d}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"run_{run_idx:02d}.log")

    def model_fn():
        return build_model(args.resolved_model_name, args.resolved_num_classes,
                            input_size=args.resolved_input_size, filters_percentage=args.filters)

    start_time = time.time()
    print(f"[run {run_idx:02d}] started on {gpu_label} at {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    with open(log_path, "w", buffering=1) as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        result = _run_one(run_idx, args, _worker["retain_set"], _worker["val_set"], _worker["test_set"],
                           device, model_fn, run_dir)

    print(f"[run {run_idx:02d}] completed on {gpu_label} at {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"(took {time.time() - start_time:.1f}s)", flush=True)
    return result


def _resolve_gpu_slots(args) -> list:
    """Returns the list of GPU-slot assignments (one entry per worker), or [None, ...] for CPU workers."""
    if args.no_cuda or not torch.cuda.is_available():
        return [None] * max(1, args.cpu_workers)
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in str(args.gpus).split(",") if str(g).strip() != ""]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    return [gpu_id for gpu_id in gpu_ids for _ in range(args.runs_per_gpu)]


def _kill_pool_workers(executor: ProcessPoolExecutor, grace_period: float = 5.0):
    """
    Force-kill every live worker process in the pool.

    `ProcessPoolExecutor.shutdown()` (what the `with` statement calls on exit, including
    on KeyboardInterrupt) only waits for in-flight tasks to finish naturally -- for a
    200-epoch training run that can mean hours of apparent hang after Ctrl+C, which is
    exactly what led earlier sweeps to get abandoned (terminal closed / session killed)
    and leave orphaned, GPU-memory-holding worker processes with no live parent. This
    reaches into the executor's worker Process objects directly (there's no public
    ProcessPoolExecutor API for "kill everything now") and terminates them: SIGTERM first,
    then SIGKILL for anything still alive after `grace_period` seconds.
    """
    processes = list(getattr(executor, "_processes", {}).values())
    for p in processes:
        if p.is_alive():
            p.terminate()
    deadline = time.time() + grace_period
    for p in processes:
        remaining = deadline - time.time()
        if remaining > 0:
            p.join(timeout=remaining)
    for p in processes:
        if p.is_alive():
            p.kill()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    add_selection_arguments(parser)

    parser.add_argument("--run-dir", required=True, help="Top-level folder; run_01/, run_02/, ... go under it")
    parser.add_argument("--num-runs", type=int, default=60)
    parser.add_argument("--start-run", type=int, default=1, help="First run index (default: 1 -> run_01)")

    parser.add_argument("--reuse-trained-from", default=None,
                         help="Existing sweep dir whose run_NN/trained_model.pth are loaded instead "
                              "of training from scratch -- for sweeps that only vary the unlearning "
                              "stage. Refused unless every training setting matches (see the module "
                              "docstring). Held-out runs under <sweep>/test_run/ are found too.")

    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                         help=f"Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--model", default=None,
                         help="Default: chosen per dataset (mnist->mlp, cifar10->tinynet, "
                              "cifar100->tinynet_cifar100 -- see model.py's dataset_defaults)")
    parser.add_argument("--num-classes", type=int, default=None, help="Default: chosen per dataset")
    parser.add_argument("--filters", type=float, default=1.0)

    parser.add_argument("--forget-prob", type=float, default=0.5,
                         help="Fraction of forget *batches* to keep each run, e.g. 0.5 keeps "
                              "exactly half of them, chosen uniformly at random (default: 0.5)")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-2,
                         help="Training learning rate (default: 2e-2 -- since --lr-scheduler defaults to "
                              "cosine, which anneals back down to 0 by the end instead of holding the peak "
                              "LR for all --epochs)")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "step"], default="cosine",
                         help="Training LR schedule (default: cosine, anneals to 0 over --epochs). "
                              "'none' keeps --lr constant; 'step' decays by --scheduler-gamma every "
                              "--scheduler-step-size epochs.")
    parser.add_argument("--scheduler-step-size", type=int, default=None,
                         help="Epoch interval for 'step' schedule (default: --epochs // 3)")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                         help="Decay factor for 'step' schedule (default: 0.1)")

    add_unlearning_arguments(parser)

    parser.add_argument("--seed", type=int, default=42,
                         help="Fixed across every run in the sweep: model init + training batch order")
    parser.add_argument("--unlearn-seed-base", type=int, default=1000,
                         help="Run i uses unlearn_seed = unlearn_seed_base + i (default base: 1000)")
    parser.add_argument("--forget-seed-base", type=int, default=2000,
                         help="Run i uses forget_sample_seed = forget_seed_base + i (default base: 2000)")

    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--gpus", type=str, default=None,
                         help="Comma-separated GPU ids to use, e.g. '3,4,5' (default: all visible GPUs)")
    parser.add_argument("--runs-per-gpu", type=int, default=2,
                         help="Concurrent runs per GPU -- total worker processes = "
                              "len(gpus) * runs_per_gpu (default: 2)")
    parser.add_argument("--cpu-workers", type=int, default=1,
                         help="Concurrent worker processes when running on CPU (--no-cuda or no GPU available)")
    return parser


def main():
    args = resolve(_build_arg_parser(), sweep=True)

    os.makedirs(args.run_dir, exist_ok=True)

    default_input_size, default_num_classes, default_model_name = dataset_defaults(args.dataset)
    args.resolved_input_size = default_input_size
    args.resolved_num_classes = args.num_classes if args.num_classes is not None else default_num_classes
    args.resolved_model_name = args.model if args.model is not None else default_model_name

    method = get_method(args.unlearn_method)
    gpu_slots = _resolve_gpu_slots(args)
    num_workers = len(gpu_slots)
    gpu_ids_used = sorted(set(g for g in gpu_slots if g is not None))

    # Settings shared by every run in the sweep, saved once at the top level. Per-run
    # folders only get run_config.json (run_idx + that run's seeds) -- see _run_one.
    sweep_config = {
        "unlearn_method": args.unlearn_method, "config_file": args.resolved_config_path,
        "num_runs": args.num_runs, "start_run": args.start_run,
        "dataset": args.dataset, "data_dir": args.data_dir,
        "model": args.resolved_model_name, "num_classes": args.resolved_num_classes,
        "input_size": args.resolved_input_size, "filters": args.filters,
        "forget_prob": args.forget_prob,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
        "weight_decay": args.weight_decay, "optimizer": args.optimizer, "momentum": args.momentum,
        "lr_scheduler": args.lr_scheduler, "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "unlearn_batch_size": args.unlearn_batch_size, "unlearn_epochs": args.unlearn_epochs,
        "unlearn_lr": args.unlearn_lr,
        "eval_every_unlearn_epoch": args.eval_every_unlearn_epoch,
        "unlearn_checkpoint_every": args.unlearn_checkpoint_every,
        # Whichever hyperparameters belong to the method actually being run.
        **{hp: getattr(args, hp) for hp in method.hyperparams},
        "seed": args.seed, "unlearn_seed_base": args.unlearn_seed_base, "forget_seed_base": args.forget_seed_base,
        "reuse_trained_from": args.reuse_trained_from,
        "gpus": gpu_ids_used,
        "runs_per_gpu": args.runs_per_gpu, "cpu_workers": args.cpu_workers, "num_workers": num_workers,
    }

    # Fail in a second rather than 60 runs in, and make sure every run this sweep is about
    # to dispatch actually has a source model before any of them start.
    if args.reuse_trained_from is not None:
        _check_reuse_compatible(args.reuse_trained_from, sweep_config)
        for run_idx in range(args.start_run, args.start_run + args.num_runs):
            _reuse_source_run_dir(args.reuse_trained_from, run_idx)
        print(f"[reuse] trained models come from {args.reuse_trained_from} "
              f"(training settings verified identical; skipping {args.epochs} epochs x "
              f"{args.num_runs} runs)")

    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(sweep_config, f, indent=2)

    run_indices = list(range(args.start_run, args.start_run + args.num_runs))
    worker_desc = (f"{num_workers} worker(s) on GPUs {sweep_config['gpus']} (runs_per_gpu={args.runs_per_gpu})"
                   if sweep_config["gpus"] else f"{num_workers} CPU worker(s)")
    print(f"Running {args.num_runs} runs ({run_indices[0]}..{run_indices[-1]}) with {worker_desc}")
    print(f"Unlearning: {describe(args)}")
    print(f"seed={args.seed} (fixed) unlearn_seed_base={args.unlearn_seed_base} "
          f"forget_seed_base={args.forget_seed_base} (both: base + run_idx) forget_prob={args.forget_prob}\n")

    ctx = mp.get_context("spawn")  # required for CUDA safety across worker processes
    manager = ctx.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_slots:
        gpu_queue.put(gpu_id)

    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_init_worker,
                                    initargs=(gpu_queue, args.dataset, args.data_dir))

    def _handle_termination(signum, frame):
        # Ctrl+C (SIGINT) or `kill <pid>` (SIGTERM, e.g. against a nohup'd/backgrounded
        # sweep): force-kill every worker NOW instead of leaving them orphaned. Hard-exits
        # (os._exit) rather than raising, so we don't risk re-entering any blocking
        # shutdown/cleanup path that would just reproduce the original hang.
        print(f"\n[run_sweep] received signal {signum}; terminating {num_workers} worker process(es)...",
              flush=True)
        _kill_pool_workers(executor)
        manager.shutdown()
        os._exit(1)

    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)

    new_results = []
    with executor:
        futures = {executor.submit(_dispatch_run, run_idx, args): run_idx for run_idx in run_indices}
        for future in as_completed(futures):
            run_idx = futures[future]
            try:
                new_results.append(future.result())
            except Exception as exc:
                print(f"[run {run_idx:02d}] FAILED: {exc}", flush=True)

    # Merge into any existing sweep_summary.json (keyed by run index) instead of overwriting
    # it outright -- a --start-run/--num-runs invocation only ever computes results for its
    # own run_indices, so writing "w" unconditionally would silently discard every earlier
    # invocation's rows (e.g. resuming runs 41-60 after an earlier runs 1-40 pass would wipe
    # out those first 40 entries). Re-running a given run_idx overwrites just that row.
    summary_path = os.path.join(args.run_dir, "sweep_summary.json")
    existing_results = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[run_sweep] warning: couldn't read existing {summary_path} ({exc}); "
                  f"starting a fresh summary instead of merging", flush=True)

    merged_by_run = {row["run"]: row for row in existing_results}
    merged_by_run.update({row["run"]: row for row in new_results})
    summary = sorted(merged_by_run.values(), key=lambda r: r["run"])

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved sweep summary ({len(new_results)}/{len(run_indices)} runs succeeded this invocation, "
          f"{len(summary)} total run(s) recorded) to {summary_path}")


if __name__ == "__main__":
    main()
