#!/bin/bash
# Re-launch specific run indices of an existing sweep (e.g. ones that crashed), without
# touching the runs that already succeeded.
#
# run_sweep.py can only express a *contiguous* range (--start-run/--num-runs), so each
# requested index is launched as its own single-run invocation, one per GPU, in parallel.
# Two things need care when doing that, both handled below:
#   - config.json: every invocation rewrites <run-dir>/config.json with its own
#     num_runs=1/start_run=i. The original is saved first and restored at the end, so the
#     sweep-wide record (num_runs: 60, start_run: 1) survives.
#   - sweep_summary.json: each invocation read-modify-writes it at exit, so near-simultaneous
#     finishes can drop a row. After everything completes, any missing row is rebuilt from
#     that run's own metrics.json / run_config.json / forget_indices.npy.
#
# The yaml is passed straight through to run_sweep.py --config, so the re-run uses the exact
# same hyperparameters as the original sweep.
#
# Usage:
#   ./rerun_runs.sh 2 39 59                                        # defaults below
#   CONFIG=configs/delete.yaml RUN_DIR=runs/delete/cifar100_bs_1 \
#     GPUS=6,7,8 ./rerun_runs.sh 4 11 27

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/scrub.yaml}"
RUN_DIR="${RUN_DIR:-runs/$(basename "$CONFIG" .yaml)/cifar100_bs_1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7,8,9}"
EXTRA="${EXTRA:-}"
PYTHON="${PYTHON:-python}"

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "usage: $0 <run-idx> [<run-idx> ...]   e.g. $0 2 39 59" >&2
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "error: config file not found: $CONFIG (relative to $PWD)" >&2
  exit 1
fi
if [ ! -d "$RUN_DIR" ]; then
  echo "error: run dir not found: $RUN_DIR -- this script re-runs indices of an existing sweep" >&2
  exit 1
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [ ${#RUNS[@]} -gt ${#GPU_ARR[@]} ]; then
  echo "error: ${#RUNS[@]} runs requested but only ${#GPU_ARR[@]} GPU(s) in GPUS=$GPUS" >&2
  echo "       (one run per GPU here; re-run in batches or widen GPUS)" >&2
  exit 1
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
SWEEP_JSON_BACKUP="${RUN_DIR}/config.json.bak.${STAMP}"
[ -f "${RUN_DIR}/config.json" ] && cp "${RUN_DIR}/config.json" "$SWEEP_JSON_BACKUP"

echo "[$(date '+%F %T')] re-running runs: ${RUNS[*]}"
echo "[$(date '+%F %T')] config=$CONFIG run-dir=$RUN_DIR gpus=$GPUS"

pids=()
for i in "${!RUNS[@]}"; do
  run_idx="${RUNS[$i]}"
  gpu="${GPU_ARR[$i]}"
  padded="$(printf '%02d' "$run_idx")"
  run_sub="${RUN_DIR}/run_${padded}"

  # run_sweep.py opens run_NN.log with "w" -- keep the failed attempt's log around.
  if [ -f "${run_sub}/run_${padded}.log" ]; then
    mv "${run_sub}/run_${padded}.log" "${run_sub}/run_${padded}.log.failed.${STAMP}"
  fi

  echo "[$(date '+%F %T')]   run_${padded} -> GPU ${gpu}"
  # shellcheck disable=SC2086
  "$PYTHON" -m pipeline.run_sweep --config "$CONFIG" --run-dir "$RUN_DIR" \
    --start-run "$run_idx" --num-runs 1 --gpus "$gpu" --runs-per-gpu 1 $EXTRA \
    > "${RUN_DIR}/rerun_${padded}_${STAMP}.out" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[$(date '+%F %T')] run_$(printf '%02d' "${RUNS[$i]}") invocation exited non-zero" >&2
    status=1
  fi
done

# Restore the sweep-wide config.json that the single-run invocations overwrote.
[ -f "$SWEEP_JSON_BACKUP" ] && mv "$SWEEP_JSON_BACKUP" "${RUN_DIR}/config.json"

# Repair sweep_summary.json for any run whose row lost the concurrent-write race.
"$PYTHON" - "$RUN_DIR" "${RUNS[@]}" <<'PY'
import json, os, sys
import numpy as np

run_dir, run_indices = sys.argv[1], [int(x) for x in sys.argv[2:]]
summary_path = os.path.join(run_dir, "sweep_summary.json")

with open(os.path.join(run_dir, "config.json")) as f:
    sweep_config = json.load(f)
rows = {}
if os.path.exists(summary_path):
    with open(summary_path) as f:
        rows = {r["run"]: r for r in json.load(f)}

for run_idx in run_indices:
    sub = os.path.join(run_dir, f"run_{run_idx:02d}")
    metrics_path = os.path.join(sub, "metrics.json")
    if run_idx in rows:
        print(f"run_{run_idx:02d}: present in sweep_summary.json")
        continue
    if not os.path.exists(metrics_path):
        print(f"run_{run_idx:02d}: MISSING -- no metrics.json, the run did not finish")
        continue
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(os.path.join(sub, "run_config.json")) as f:
        run_config = json.load(f)
    rows[run_idx] = {
        "run": run_idx,
        "seed": sweep_config["seed"],
        "unlearn_seed": run_config["unlearn_seed"],
        "forget_sample_seed": run_config["forget_sample_seed"],
        "num_forget_kept": int(len(np.load(os.path.join(sub, "forget_indices.npy")))),
        "metrics_before": metrics["before_unlearning"],
        "metrics_after": metrics["after_unlearning"],
    }
    print(f"run_{run_idx:02d}: row rebuilt from metrics.json (lost the summary write race)")

with open(summary_path, "w") as f:
    json.dump([rows[k] for k in sorted(rows)], f, indent=2)
print(f"sweep_summary.json now has {len(rows)} run(s): "
      f"missing {sorted(set(range(1, sweep_config['num_runs'] + 1)) - set(rows)) or 'none'}")
PY

echo "[$(date '+%F %T')] done (per-run logs: ${RUN_DIR}/run_NN/run_NN.log)"
exit "$status"
