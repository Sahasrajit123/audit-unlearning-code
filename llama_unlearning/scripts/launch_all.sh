#!/usr/bin/env bash
# Launch the audit's runs across local GPUs, one process per run.
#
#   scripts/launch_all.sh <config> <family> <gpu_list> [max_per_gpu]
#
#   scripts/launch_all.sh configs/base.yaml calibration 0,1,2,3
#   scripts/launch_all.sh configs/base.yaml evaluation  0,1,2,3 2
#   scripts/launch_all.sh configs/base.yaml all         0,1,2,3,4,5,6,7
#
# Runs are independently launchable by run_id, so this is just a scheduler: it keeps
# `max_per_gpu` processes alive per GPU and starts the next run_id as slots free up.
# Interrupting it is safe -- re-running skips completed methods, and each run's sign
# vector and seeds are pinned in run_state.json.
set -uo pipefail

CONFIG="${1:?usage: launch_all.sh <config> <family> <gpu_list> [max_per_gpu]}"
FAMILY="${2:?family must be calibration | evaluation | all}"
GPU_LIST="${3:?comma-separated GPU ids, e.g. 0,1,2,3}"
MAX_PER_GPU="${4:-1}"

cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mapfile -t RUN_IDS < <("$PY" - "$CONFIG" "$FAMILY" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
from audit_tofu.manifest import load_manifest
cfg = load_config(sys.argv[1])
man = load_manifest(cfg["experiment"]["manifest_path"])
fams = ["calibration", "evaluation"] if sys.argv[2] == "all" else [sys.argv[2]]
for f in fams:
    for rid in man["sign_vectors"][f]["run_ids"]:
        print(rid)
PYEOF
)

if [ "${#RUN_IDS[@]}" -eq 0 ]; then
  echo "No run ids found. Did you run scripts/build_manifest.py?" >&2
  exit 1
fi

SLOTS=$(( ${#GPUS[@]} * MAX_PER_GPU ))
LOGDIR="launch_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "config      : $CONFIG"
echo "family      : $FAMILY  (${#RUN_IDS[@]} runs)"
echo "gpus        : ${GPUS[*]}  (max $MAX_PER_GPU per gpu -> $SLOTS concurrent)"
echo "logs        : $LOGDIR"
echo

i=0
for RUN_ID in "${RUN_IDS[@]}"; do
  GPU="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  echo "[launch] $RUN_ID -> gpu $GPU"
  "$PY" scripts/run_single.py \
      --config "$CONFIG" --run_id "$RUN_ID" --gpu "$GPU" \
      > "$LOGDIR/$RUN_ID.log" 2>&1 &
  i=$(( i + 1 ))
  # Throttle to the configured concurrency.
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 20; done
done

wait
echo
echo "[launch] all $FAMILY runs finished. Logs in $LOGDIR"
echo "[launch] next: .venv/bin/python scripts/aggregate_audit.py --config $CONFIG --method noop"
