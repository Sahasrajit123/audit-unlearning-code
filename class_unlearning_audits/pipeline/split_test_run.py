"""
Move the last N runs of a run_sweep.py sweep into a `test_run/` subfolder, so they can be
held out as membership-inference attack targets (see membership_inference_attack.ipynb)
while the remaining runs are used to build the reference in/out stats via
audit_utils.compute_pointwise_forget_stats.

`compute_pointwise_forget_stats` only globs for `run_dir.iterdir()` entries whose name
starts with "run_", so once the held-out runs live under test_run/ instead of directly
under run_dir/, they're automatically excluded from the reference-stats sweep without any
other code changes.

Usage:
    python -m pipeline.split_test_run --run-dir runs_cifar100_bs_1 --num-test 10
    python -m pipeline.split_test_run --run-dir runs_cifar100_bs_1 --num-test 10 --dry-run
"""
import argparse
import shutil
from pathlib import Path


def split_test_run(run_dir: Path, num_test: int, dry_run: bool = False) -> None:
    run_dir = Path(run_dir)
    run_subdirs = sorted(
        (d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
        key=lambda d: d.name,
    )
    if len(run_subdirs) <= num_test:
        raise ValueError(
            f"{run_dir} has only {len(run_subdirs)} run_* dirs, not enough to hold out "
            f"{num_test} and still leave any for reference stats."
        )

    test_run_dir = run_dir / "test_run"
    to_move = run_subdirs[-num_test:]

    if test_run_dir.exists() and any(test_run_dir.iterdir()):
        raise FileExistsError(f"{test_run_dir} already exists and is non-empty; refusing to overwrite.")

    print(f"[{run_dir.name}] moving {len(to_move)} runs into {test_run_dir}: "
          f"{[d.name for d in to_move]}")
    if dry_run:
        print(f"[{run_dir.name}] --dry-run: no files moved")
        return

    test_run_dir.mkdir(exist_ok=True)
    for d in to_move:
        shutil.move(str(d), str(test_run_dir / d.name))
    print(f"[{run_dir.name}] done")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, action="append",
                         help="A run_sweep.py sweep directory (repeatable for multiple dirs)")
    parser.add_argument("--num-test", type=int, default=10, help="Number of trailing runs to hold out")
    parser.add_argument("--dry-run", action="store_true", default=False,
                         help="Print what would be moved without moving anything")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    for run_dir in args.run_dir:
        split_test_run(Path(run_dir), args.num_test, dry_run=args.dry_run)
