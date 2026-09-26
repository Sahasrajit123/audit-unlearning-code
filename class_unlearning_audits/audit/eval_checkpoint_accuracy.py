"""
Evaluate one checkpoint source across every run of a sweep and cache the mean
retain / sampled-forget / test accuracy next to the audit outputs.

This exists to fill one gap. A bound is not readable without the utility that goes with
it -- a mechanism that has destroyed the model attacks at chance and so reports an
excellent epsilon, which is privacy from having no model rather than from unlearning --
so make_comparison_table.py reports R/F/T accuracy alongside every bound. It can usually
get those for free:

  - at `unlearned`, from each sweep's `sweep_summary.json` (`metrics_after`);
  - at an intermediate `unlearn_epoch_N`, from each run's `metrics.json`, but only when
    that sweep ran with `--eval-every-unlearn-epoch` (DELETE did; bad-teacher's sweeps
    did not, and SCRUB's default is off too -- its history carries `forget_accuracy` and
    nothing else).

For the remaining case the numbers simply are not on disk, and the only honest way to get
them is to load each run's checkpoint and evaluate it. That is what this does, once,
writing `<sweep>/checkpoint_accuracy_<source>.json` so the table generator stays a pure
JSON reader and nobody pays for the evaluation twice.

Nothing here modifies an existing sweep artifact: the cache is a new file, and the
per-run `metrics.json` written by the sweep is left exactly as it was.

Usage:
    python -m audit.eval_checkpoint_accuracy --source unlearn_epoch_1 runs/badteacher/*
    python -m audit.eval_checkpoint_accuracy --source unlearn_epoch_3 --force runs/scrub/cifar100_bs_1
"""
import argparse
import json
import os
from pathlib import Path

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.utils.data import TensorDataset

from core.data_utils import load_full_forget, load_retain_val_test
from core.model import build_model, dataset_defaults
from core.train import evaluate

#: The three splits the table reports. `val` and `train` are deliberately left out -- the
#: table has room for R/F/T and nothing else, and these are the three that say whether a
#: bound came from a working model.
SPLITS = ("retain", "sampled_forget", "test")


def _model_filename(source: str) -> str:
    if source == "trained":
        return "trained_model.pth"
    if source == "unlearned":
        return "unlearned_model.pth"
    if source.startswith("unlearn_epoch_"):
        return f"unlearned_model_epoch_{source.rsplit('_', 1)[1]}.pth"
    raise ValueError(f"unknown source {source!r}; expected trained, unlearned or unlearn_epoch_N")


def _run_dirs(sweep: Path):
    """Every run of a sweep, held-out ones included -- the table averages over all of them,
    matching what sweep_summary.json reports at `unlearned`."""
    runs = sorted(d for d in sweep.iterdir() if d.is_dir() and d.name.startswith("run_"))
    test_run = sweep / "test_run"
    if test_run.is_dir():
        runs += sorted(d for d in test_run.iterdir() if d.is_dir() and d.name.startswith("run_"))
    return runs


def evaluate_sweep(sweep: Path, source: str, device: torch.device, batch_size: int = 512) -> dict:
    config = json.loads((sweep / "config.json").read_text())
    dataset, data_dir = config["dataset"], config["data_dir"]

    default_input_size, default_num_classes, default_model_name = dataset_defaults(dataset)
    model = build_model(config.get("model") or default_model_name,
                        config.get("num_classes") or default_num_classes,
                        input_size=config.get("input_size") or default_input_size,
                        filters_percentage=config.get("filters", 1.0)).to(device)

    retain_set, _, test_set = load_retain_val_test(dataset, data_dir)
    forget_data, forget_labels = load_full_forget(dataset, data_dir).tensors

    filename = _model_filename(source)
    per_run, missing = {}, []
    for run_dir in _run_dirs(sweep):
        path = run_dir / filename
        if not path.exists():
            missing.append(run_dir.name)
            continue
        indices = np.load(run_dir / "forget_indices.npy")
        sampled_forget = TensorDataset(forget_data[indices], forget_labels[indices])

        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        accs = {}
        for name, ds in zip(SPLITS, (retain_set, sampled_forget, test_set)):
            accs[name] = evaluate(model, ds, device, batch_size=batch_size)[1]
        per_run[run_dir.name] = accs
        print(f"  {run_dir.name}: " + "  ".join(f"{n}={accs[n]:.2f}%" for n in SPLITS), flush=True)

    if not per_run:
        raise SystemExit(f"{sweep}: no run has {filename}")
    if missing:
        print(f"  note: {len(missing)} run(s) have no {filename} and were skipped: "
              f"{', '.join(missing)}")

    return {
        "generated_by": Path(__file__).name,
        "sweep": str(sweep),
        "model_source": source,
        "model_filename": filename,
        "dataset": dataset,
        "data_dir": data_dir,
        "n_runs": len(per_run),
        "mean": {name: sum(a[name] for a in per_run.values()) / len(per_run) for name in SPLITS},
        "per_run": per_run,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="Sweep directories (runs/<method>/<sweep>)")
    p.add_argument("--source", required=True,
                   help="Checkpoint to evaluate: trained, unlearned, or unlearn_epoch_N")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--force", action="store_true", default=False,
                   help="Re-evaluate even if the cache file already exists")
    args = p.parse_args()

    device = torch.device(args.device)
    for sweep in (Path(s) for s in args.sweeps):
        out_path = sweep / f"checkpoint_accuracy_{args.source}.json"
        if out_path.exists() and not args.force:
            print(f"### {sweep}: cache exists, skipping ({out_path.name})")
            continue
        print(f"### {sweep} @ {args.source}")
        doc = evaluate_sweep(sweep, args.source, device, batch_size=args.batch_size)
        out_path.write_text(json.dumps(doc, indent=2))
        mean = doc["mean"]
        print(f"  -> {out_path}  mean over {doc['n_runs']} runs: "
              + " / ".join(f"{mean[n]:.1f}" for n in SPLITS) + " (R/F/T)")


if __name__ == "__main__":
    main()
