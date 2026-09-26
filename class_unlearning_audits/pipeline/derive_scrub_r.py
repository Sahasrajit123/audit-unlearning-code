"""
Turn a finished `scrub` sweep into the `scrub_r` sweep it would have produced, without
retraining anything.

SCRUB+R does not change SCRUB's trajectory -- it changes which point on it you keep (see
unlearning/scrub.py, and Section 3.2 of Kurmanji et al., NeurIPS 2023). Given the same
`(--seed, --unlearn-seed-base, --forget-seed-base)`, a `scrub_r` sweep would train the
identical models, walk the identical max/min steps, and differ from the `scrub` sweep in
exactly one respect: `unlearned_model.pth` would be the rewind-selected epoch instead of
the last one. Every candidate epoch is already on disk when the sweep ran with
`unlearn_checkpoint_every: 1` (the default in configs/scrub.yaml), so re-running 60 x 200
training epochs to relabel checkpoints would be several GPU-hours spent to arrive at
files this script can assemble in about a minute.

There is a second, better reason to derive rather than re-run: the two sweeps then share
their trained models *exactly*, so a difference between their audit bounds is the
rewinding and nothing else. Two independent sweeps would also differ by training noise.

What it needs from each run of the source sweep:
  - `metrics.json` -> `unlearn_history`, which this project's SCRUB port records with a
    `forget_error` for every epoch. That is the trajectory the rewind selects over, so no
    forget-set re-evaluation is needed.
  - every epoch's weights: `unlearned_model_epoch_{1..T-1}.pth` plus `unlearned_model.pth`
    for the final epoch T. A sweep run with a coarser `--unlearn-checkpoint-every` is
    missing candidates and would silently produce a *different* selection than a real
    SCRUB+R run, so that is refused unless you pass --allow-missing-epochs.

What it computes: the reference point. That is the final model's error on validation
examples drawn from the forget set's own classes (413 of 5000 for cifar100_bs_1), which
needs one small forward pass per run.

What it writes, per run, mirroring exactly what run_sweep.py would have written for
`--unlearn-method scrub_r` (including the held-out `test_run/` layout if the source sweep
has already been split):
    trained_model.pth          copied unchanged
    unlearned_model.pth        the selected epoch's weights
    unlearned_model_epoch_N    the un-rewound SCRUB trajectory, including epoch T itself
                               whenever rewinding moved the answer off it
    forget_indices.npy         copied unchanged
    run_config.json            copied unchanged (same seeds -- same run)
    metrics.json               copied, with the `rewind_*` keys added to the last
                               unlearn_history entry and `after_unlearning` recomputed on
                               the rewound model (only for runs that actually rewound --
                               for the rest it is already correct)
and at the sweep level, a `config.json` with `unlearn_method: scrub_r` plus a
`derived_from` block recording this script's inputs, and a regenerated
`sweep_summary.json`.

The derived files are value-identical to a real scrub_r sweep, not byte-identical: an
intermediate checkpoint was serialized from CPU tensors while a real run's
`unlearned_model.pth` is serialized from CUDA tensors. Everything downstream loads with
`map_location`, so this is invisible to the audit.

Usage:
    python -m pipeline.derive_scrub_r --scrub-dir runs/scrub/cifar100_bs_1
    python -m pipeline.derive_scrub_r --scrub-dir runs/scrub/cifar100_bs_1 --tie-break latest \
        --out-dir runs/scrub_r_latest/cifar100_bs_1
    python -m pipeline.derive_scrub_r --scrub-dir runs/scrub/cifar100_bs_1 --dry-run

Then audit it exactly like any other sweep:
    ./scripts/run_full_audit.sh --k "500 1500" runs/scrub_r/cifar100_bs_1
"""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from core.data_utils import load_full_forget, load_retain_val_test
from core.model import build_model, dataset_defaults
from core.train import evaluate
from pipeline.train_and_unlearn import EVAL_SPLIT_NAMES
from unlearning.scrub import class_matched_val_subset, select_rewind_epoch


def _find_runs(sweep_dir: Path):
    """Every run folder of a sweep, as (relative_path, absolute_path) pairs.

    Handles both layouts: runs still directly under the sweep directory, and runs already
    moved into `test_run/` by split_test_run.py. The relative path is preserved in the
    output so a split sweep derives into a split sweep.
    """
    runs = [(Path(d.name), d) for d in sorted(sweep_dir.iterdir())
            if d.is_dir() and d.name.startswith("run_")]
    test_run_dir = sweep_dir / "test_run"
    if test_run_dir.is_dir():
        runs += [(Path("test_run") / d.name, d) for d in sorted(test_run_dir.iterdir())
                 if d.is_dir() and d.name.startswith("run_")]
    return runs


def _epoch_checkpoint(run_dir: Path, epoch: int, final_epoch: int) -> Path:
    """Where epoch `epoch`'s weights live in a scrub run directory."""
    if epoch == final_epoch:
        return run_dir / "unlearned_model.pth"
    return run_dir / f"unlearned_model_epoch_{epoch}.pth"


def _place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def derive(args) -> int:
    sweep_dir = Path(args.scrub_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir.parent.parent / "scrub_r" / sweep_dir.name

    config_path = sweep_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"{config_path} not found -- is {sweep_dir} a run_sweep.py sweep directory?")
    config = json.loads(config_path.read_text())

    source_method = config.get("unlearn_method")
    if source_method != "scrub" and not args.force:
        raise SystemExit(
            f"{config_path} says unlearn_method={source_method!r}, not 'scrub'. Rewinding is "
            f"only defined over a SCRUB trajectory; pass --force if you are certain."
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists and is not empty; pass --overwrite to replace it.")

    dataset = args.dataset or config["dataset"]
    data_dir = args.data_dir or config["data_dir"]
    final_epoch = config["unlearn_epochs"]

    default_input_size, default_num_classes, default_model_name = dataset_defaults(dataset)
    model_name = config.get("model") or default_model_name
    num_classes = config.get("num_classes") or default_num_classes
    input_size = config.get("input_size") or default_input_size
    filters = config.get("filters", 1.0)

    runs = _find_runs(sweep_dir)
    if not runs:
        raise SystemExit(f"No run_* subfolders found under {sweep_dir} (or its test_run/).")

    print(f"[derive] source:    {sweep_dir}  ({len(runs)} runs, {final_epoch} unlearning epochs)")
    print(f"[derive] target:    {out_dir}")
    print(f"[derive] tie-break: {args.tie_break}")

    device = torch.device(args.device)
    # Loaded once for every run: only the forget subset changes from run to run.
    retain_set, val_set, test_set = load_retain_val_test(dataset, data_dir)
    full_forget_set = load_full_forget(dataset, data_dir)
    forget_data, forget_labels = full_forget_set.tensors
    model = build_model(model_name, num_classes, input_size=input_size,
                        filters_percentage=filters).to(device)

    # ---- Pass 1: decide every run's rewind epoch, writing nothing. A sweep with one
    # unusable run should fail before it has scattered half a derived sweep on disk.
    plan, skipped = [], []
    for rel_path, run_dir in runs:
        metrics_path = run_dir / "metrics.json"
        indices_path = run_dir / "forget_indices.npy"
        if not metrics_path.exists() or not (run_dir / "unlearned_model.pth").exists():
            # run_sweep.py writes forget_indices.npy + run_config.json before training, so
            # a run that died mid-way leaves a folder behind with no model in it.
            if args.skip_incomplete:
                skipped.append(str(rel_path))
                continue
            raise SystemExit(
                f"{run_dir} has no metrics.json / unlearned_model.pth -- it looks like a run "
                f"that failed partway through. Re-run it (run_sweep.py --start-run), or pass "
                f"--skip-incomplete to derive from the runs that did finish."
            )
        if not indices_path.exists():
            raise SystemExit(f"{run_dir} is missing forget_indices.npy")

        metrics = json.loads(metrics_path.read_text())
        history = metrics.get("unlearn_history") or []
        forget_errors = {rec["epoch"]: rec["forget_error"] for rec in history
                         if "forget_error" in rec}
        if len(forget_errors) != final_epoch:
            raise SystemExit(
                f"{metrics_path}: expected a per-epoch forget_error for all {final_epoch} "
                f"epochs, found {sorted(forget_errors)}. Was this sweep produced by "
                f"unlearning/scrub.py?"
            )

        missing = [e for e in sorted(forget_errors)
                   if not _epoch_checkpoint(run_dir, e, final_epoch).exists()]
        if missing:
            if not args.allow_missing_epochs:
                raise SystemExit(
                    f"{run_dir}: no saved weights for epoch(s) {missing}. The source sweep "
                    f"needs --unlearn-checkpoint-every 1 for every epoch to be a rewind "
                    f"candidate; selecting among a subset would not reproduce a real "
                    f"scrub_r run. Re-run with --allow-missing-epochs to select among the "
                    f"epochs that do exist."
                )
            for e in missing:
                forget_errors.pop(e)

        # The reference point: the FINAL model's error on held-out examples of the forget
        # set's classes (paper, Sec. 3.2) -- not the rewound model's.
        forget_indices = np.load(indices_path)
        sampled_forget_set = TensorDataset(forget_data[forget_indices], forget_labels[forget_indices])
        reference_set = class_matched_val_subset(val_set, sampled_forget_set)
        if len(reference_set) == 0:
            print(f"[derive] {rel_path}: no class-matched validation examples; "
                  f"falling back to the full validation set")
            reference_set = val_set

        final_state = torch.load(run_dir / "unlearned_model.pth", map_location=device,
                                 weights_only=False)
        model.load_state_dict(final_state)
        _, reference_acc = evaluate(model, reference_set, device)
        reference_error = 100.0 - reference_acc

        selected = select_rewind_epoch(forget_errors, reference_error, tie_break=args.tie_break)
        best = abs(forget_errors[selected] - reference_error)
        tied = sorted(e for e, err in forget_errors.items() if abs(err - reference_error) == best)

        print(f"[derive] {str(rel_path):<20s} reference={reference_error:6.2f}%  "
              f"forget_err=" + ",".join(f"{forget_errors[e]:.1f}" for e in sorted(forget_errors))
              + f"  -> epoch {selected}" + (f" ({len(tied)} tied)" if len(tied) > 1 else ""))

        plan.append({
            "rel_path": rel_path, "run_dir": run_dir, "metrics": metrics,
            "forget_indices": forget_indices, "sampled_forget_set": sampled_forget_set,
            "reference_error": reference_error, "reference_set_size": len(reference_set),
            "selected": selected, "tied": tied,
            "selected_forget_error": forget_errors[selected],
        })

    if skipped:
        print(f"[derive] skipped {len(skipped)} unfinished run(s): {', '.join(skipped)}")
    if not plan:
        raise SystemExit("No usable runs found -- nothing to derive.")

    selections = [d["selected"] for d in plan]
    histogram = Counter(selections)
    print("\n[derive] rewind epochs chosen: "
          + ", ".join(f"epoch {e}: {n} run(s)" for e, n in sorted(histogram.items())))
    if set(histogram) == {final_epoch}:
        print("[derive] note: every run kept its final epoch -- this scrub_r sweep is "
              "identical to the scrub sweep, and auditing both would measure the same "
              "mechanism twice.")

    if args.dry_run:
        print("\n[derive] --dry-run: nothing written")
        return 0

    # ---- Pass 2: materialize the sweep. Everything below is file copying plus, for the
    # runs that actually rewound, a re-evaluation of the model the audit will now see.
    summary_rows = []
    for decision in plan:
        rel_path, run_dir = decision["rel_path"], decision["run_dir"]
        metrics, selected = decision["metrics"], decision["selected"]
        forget_indices = decision["forget_indices"]
        sampled_forget_set = decision["sampled_forget_set"]

        dst_run = out_dir / rel_path
        dst_run.mkdir(parents=True, exist_ok=True)
        for name in ("trained_model.pth", "forget_indices.npy", "run_config.json"):
            src = run_dir / name
            if src.exists():
                _place(src, dst_run / name, args.link)

        _place(_epoch_checkpoint(run_dir, selected, final_epoch),
               dst_run / "unlearned_model.pth", args.link)
        # The source sweep's `unlearned_model_epoch_*.pth` files are exactly what its
        # --unlearn-checkpoint-every produced, and a scrub_r run with the same settings
        # writes the same set -- including the selected epoch, which therefore appears
        # both under its own name and as unlearned_model.pth.
        for src in sorted(run_dir.glob("unlearned_model_epoch_*.pth")):
            _place(src, dst_run / src.name, args.link)
        if selected != final_epoch:
            # Rewinding moved the answer off the last epoch, so that last epoch stops
            # being unlearned_model.pth and becomes an intermediate checkpoint instead.
            _place(run_dir / "unlearned_model.pth",
                   dst_run / f"unlearned_model_epoch_{final_epoch}.pth", args.link)

        metrics["unlearn_method"] = "scrub_r"
        metrics["unlearn_history"][-1].update({
            "rewind_reference_error": decision["reference_error"],
            "rewind_reference_set_size": decision["reference_set_size"],
            "rewind_selected_epoch": selected,
            "rewind_selected_forget_error": decision["selected_forget_error"],
            "rewind_tie_break": args.tie_break,
            "rewind_tied_epochs": decision["tied"],
        })
        if selected != final_epoch:
            # after_unlearning in the source describes the final SCRUB model, which is no
            # longer the model this run hands to the audit. Recompute it on the rewound one.
            model.load_state_dict(torch.load(dst_run / "unlearned_model.pth",
                                             map_location=device, weights_only=False))
            eval_sets = zip(EVAL_SPLIT_NAMES,
                            (ConcatDataset([retain_set, sampled_forget_set]), retain_set,
                             sampled_forget_set, full_forget_set, val_set, test_set))
            metrics["after_unlearning"] = {
                name: dict(zip(("loss", "accuracy"), evaluate(model, ds, device)))
                for name, ds in eval_sets
            }
        (dst_run / "metrics.json").write_text(json.dumps(metrics, indent=2))

        run_config = json.loads((run_dir / "run_config.json").read_text()) \
            if (run_dir / "run_config.json").exists() else {}
        summary_rows.append({
            "run": run_config.get("run_idx", int(rel_path.name.split("_")[-1])),
            "seed": config.get("seed"),
            "unlearn_seed": run_config.get("unlearn_seed"),
            "forget_sample_seed": run_config.get("forget_sample_seed"),
            "num_forget_kept": int(len(forget_indices)),
            "rewind_selected_epoch": selected,
            "rewind_reference_error": decision["reference_error"],
            "metrics_before": metrics.get("before_unlearning"),
            "metrics_after": metrics.get("after_unlearning"),
        })

    derived_config = dict(config)
    derived_config["unlearn_method"] = "scrub_r"
    derived_config["scrub_rewind_tie_break"] = args.tie_break
    derived_config["derived_from"] = {
        "sweep": str(sweep_dir),
        "script": Path(__file__).name,
        "tie_break": args.tie_break,
        "note": "Rewind-selected checkpoints of the scrub sweep above. SCRUB+R does not "
                "change SCRUB's trajectory, so this is value-identical to a scrub_r sweep "
                "run with the same seeds -- and shares its trained models exactly.",
        "rewind_epoch_histogram": {str(e): n for e, n in sorted(histogram.items())},
    }
    (out_dir / "config.json").write_text(json.dumps(derived_config, indent=2))
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(sorted(summary_rows, key=lambda r: r["run"]), indent=2))

    print(f"[derive] wrote {len(summary_rows)} runs to {out_dir}")
    print(f"[derive] next: ./scripts/run_full_audit.sh --k \"500 1500\" {out_dir}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scrub-dir", required=True,
                    help="A finished run_sweep.py sweep with unlearn_method: scrub")
    p.add_argument("--out-dir", default=None,
                    help="Default: runs/scrub_r/<same sweep name> alongside the source")
    p.add_argument("--tie-break", choices=["earliest", "latest"], default="earliest",
                    help="Which epoch wins when several are equally close to the reference. "
                         "On a class-centric forget split that is the usual case, not a "
                         "corner -- see unlearning/scrub.py::select_rewind_epoch. "
                         "Default: earliest, matching --scrub-rewind-tie-break's default.")
    p.add_argument("--dataset", default=None, help="Default: from the source sweep's config.json")
    p.add_argument("--data-dir", default=None, help="Default: from the source sweep's config.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--link", action="store_true", default=False,
                    help="Hard-link the checkpoints instead of copying them. Saves the "
                         "duplicate disk (~300MB for a 60-run sweep); safe because nothing "
                         "here rewrites a checkpoint in place, but the source sweep's files "
                         "then cannot be freed by deleting only one of the two directories.")
    p.add_argument("--skip-incomplete", action="store_true", default=False,
                    help="Skip source runs that never finished (folder present, no model) "
                         "instead of refusing. The derived sweep then has fewer runs than "
                         "the source sweep's num_runs.")
    p.add_argument("--allow-missing-epochs", action="store_true", default=False,
                    help="Select among whatever epochs were checkpointed instead of refusing. "
                         "The result is then NOT what a real scrub_r run would have produced.")
    p.add_argument("--overwrite", action="store_true", default=False,
                    help="Replace a non-empty --out-dir")
    p.add_argument("--force", action="store_true", default=False,
                    help="Proceed even if the source sweep is not unlearn_method: scrub")
    p.add_argument("--dry-run", action="store_true", default=False,
                    help="Print each run's rewind decision and write nothing")
    return p


if __name__ == "__main__":
    raise SystemExit(derive(_build_arg_parser().parse_args()))
