#!/usr/bin/env python3
"""Summarize measured runtime and peak memory from completed runs.

    python scripts/report_resources.py --config configs/pilot.yaml
    python scripts/report_resources.py --config configs/base.yaml --extrapolate 30

Reads each run's `metrics.json` `resources` block -- measured
`max_memory_allocated` / `max_memory_reserved`, not estimates -- and prints a table
suitable for pasting into docs/RESOURCE_ESTIMATE.md. `--extrapolate N` projects the
cost of N runs from the observed per-run mean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.config import load_config
from audit_tofu.run_manager import load_json


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--extrapolate", type=int, default=None,
                    help="project total cost for this many runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(cfg["experiment"]["output_root"])
    runs_dir = root / "runs"

    if not runs_dir.exists():
        raise SystemExit(f"no runs directory at {runs_dir}")

    found: List[Dict[str, Any]] = []
    for d in sorted(runs_dir.iterdir()):
        p = d / "metrics.json"
        if d.is_dir() and p.exists():
            try:
                found.append({"run_id": d.name, "metrics": load_json(p)})
            except Exception as exc:
                print(f"[report] skipping {d.name}: {exc}")

    if not found:
        raise SystemExit(
            f"no completed runs with metrics.json under {runs_dir}. "
            "Dry runs write dry_run.json only."
        )

    print(f"model      : {cfg['model']['id']}")
    print(f"output_root: {root}")
    print(f"runs found : {len(found)}\n")

    # Aggregate per stage across runs.
    per_stage: Dict[str, List[Dict[str, float]]] = {}
    for entry in found:
        stages = (entry["metrics"].get("resources") or {}).get("stages") or {}
        for name, rec in stages.items():
            per_stage.setdefault(name, []).append(rec)

    if not per_stage:
        print("No `resources.stages` recorded. Was this run completed with this version?")
        return 1

    w = max(len(s) for s in per_stage) + 2
    print(f"{'stage':<{w}}{'n':>3}  {'mean time':>12}  {'max time':>12}"
          f"  {'peak alloc':>11}  {'peak resv':>10}")
    print("-" * (w + 58))

    order = ["train"] + sorted(s for s in per_stage if s != "train")
    total_mean = 0.0
    for name in order:
        if name not in per_stage:
            continue
        recs = per_stage[name]
        times = [r.get("duration_seconds", 0.0) for r in recs]
        alloc = [r.get("peak_allocated_gib", 0.0) for r in recs]
        resv = [r.get("peak_reserved_gib", 0.0) for r in recs]
        mean_t = sum(times) / len(times)
        total_mean += mean_t
        print(f"{name:<{w}}{len(recs):>3}  {_fmt_duration(mean_t):>12}"
              f"  {_fmt_duration(max(times)):>12}"
              f"  {max(alloc):>8.2f} GiB  {max(resv):>6.2f} GiB")

    print("-" * (w + 58))
    print(f"{'per-run total (mean)':<{w}}     {_fmt_duration(total_mean):>12}")

    overall_alloc = max(
        (entry["metrics"].get("resources") or {}).get("overall_peak_allocated_gib", 0.0)
        for entry in found
    )
    overall_resv = max(
        (entry["metrics"].get("resources") or {}).get("overall_peak_reserved_gib", 0.0)
        for entry in found
    )
    print(f"{'overall peak allocated':<{w}}     {overall_alloc:>8.2f} GiB")
    print(f"{'overall peak reserved':<{w}}     {overall_resv:>8.2f} GiB")

    # Checkpoint footprint, if any survived the retention policy.
    ckpt_bytes = 0
    for entry in found:
        t = runs_dir / entry["run_id"] / "trained"
        if t.exists():
            ckpt_bytes += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
    if ckpt_bytes:
        print(f"{'trained checkpoints on disk':<{w}}     {ckpt_bytes / 1024**3:>8.2f} GiB")

    if args.extrapolate:
        n = args.extrapolate
        gpu_hours = total_mean * n / 3600.0
        print(f"\nProjection for {n} runs at the observed per-run mean:")
        print(f"  total GPU-hours   : {gpu_hours:.1f}")
        for g in (1, 4, 8):
            print(f"  wall-clock on {g} GPU{'s' if g > 1 else ' '} : {gpu_hours / g:.1f} h")
        print("  (assumes one run per GPU at a time and no contention; these GPUs are"
              " shared, so treat this as a floor)")

    print("\nNOTE: peak memory is measured (max_memory_allocated / max_memory_reserved),"
          "\n      not estimated. Reserved exceeds allocated by allocator caching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
