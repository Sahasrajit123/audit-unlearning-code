#!/usr/bin/env bash
#SBATCH --job-name=tofu-audit
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#
# Slurm array launch: one array task per audit run.
#
#   mkdir -p slurm_logs
#   sbatch --array=0-19 scripts/slurm_run.sbatch configs/base.yaml calibration
#   sbatch --array=0-9  scripts/slurm_run.sbatch configs/base.yaml evaluation
#
# SLURM_ARRAY_TASK_ID indexes into the family's run_ids, so each task is a single
# independently-launchable run. Re-submitting a failed task is safe: run_state.json
# pins that run's sign vector and seeds, and completed methods are skipped.
set -uo pipefail

CONFIG="${1:-configs/base.yaml}"
FAMILY="${2:-calibration}"

cd "$SLURM_SUBMIT_DIR" || exit 1
PY="${PYTHON:-.venv/bin/python}"

RUN_ID=$("$PY" - "$CONFIG" "$FAMILY" "$SLURM_ARRAY_TASK_ID" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
from audit_tofu.manifest import load_manifest
cfg = load_config(sys.argv[1])
man = load_manifest(cfg["experiment"]["manifest_path"])
ids = man["sign_vectors"][sys.argv[2]]["run_ids"]
idx = int(sys.argv[3])
if idx >= len(ids):
    sys.exit(f"array index {idx} exceeds {len(ids)} runs in {sys.argv[2]}")
print(ids[idx])
PYEOF
) || exit 1

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Slurm already scopes the GPU, so no --gpu flag here.
exec "$PY" scripts/run_single.py --config "$CONFIG" --run_id "$RUN_ID"
