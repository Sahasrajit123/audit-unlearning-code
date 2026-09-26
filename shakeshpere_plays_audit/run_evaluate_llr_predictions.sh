#!/usr/bin/env bash
# Runs evaluate_llr_predictions.py over multiple runs_dir folders.
# Edit RUNS_DIRS and DATA_DIR below, then just run this script.
set -euo pipefail

# One entry per run folder you want evaluated.
RUNS_DIRS=(
  "runs_ascent_descent_fs400_q_1"
  "runs_ascent_descent_fs400_q_1_var1"
  "runs_ascent_descent_fs400_q_1_var2"
  "runs_ascent_descent_fs400_q_1_var3"
  "runs_ascent_descent_fs400_mid_q"
  "runs_ascent_descent_fs400_q_9"
  "runs_ascent_descent_fs400_q_None"
  "runs_hessian_unlearning_fs400"
  "runs_finetune_fs400"
)

##RUNS_DIRS=("runs_finetune_fs400")

# Same data_dir used for every run_dir above.
DATA_DIR="data_splits_speakers300_fs400"

# Flags shared across all runs below; override here if needed.
MODEL_TYPE="unlearnt"
# Audit parameter r: TOTAL guess budget. The attack takes the top R/2 and the
# bottom R/2 by LLR, and v ranges over [0, R]. Must be even and <= #forget files.
# R is passed to the epsilon/rho/mu tests unchanged (no factor of 2 downstream).
R=100
# Number of test runs (L) to audit. Empty = use every run in test_run/.
# When set, outputs are suffixed _T<N> so sweeps don't overwrite each other.
T=10
CI_DELTA=0.05
THETA_MAX=50.0
AVG_DIRECTION="ge"
GAMMA_MAX=1e4
CONV_DELTA=1e-3

for i in "${!RUNS_DIRS[@]}"; do
  runs_dir="${RUNS_DIRS[$i]}"
  echo "=== [$((i+1))/${#RUNS_DIRS[@]}] runs_dir=${runs_dir} data_dir=${DATA_DIR} ==="
  cmd=(python evaluate_llr_predictions.py
    --runs_dir "${runs_dir}"
    --data_dir "${DATA_DIR}"
    --model_type "${MODEL_TYPE}"
    --r "${R}"
    --ci_delta "${CI_DELTA}"
    --theta_max "${THETA_MAX}"
    --avg_direction "${AVG_DIRECTION}"
    --gamma_max "${GAMMA_MAX}"
    --conv_delta "${CONV_DELTA}")
  [ -n "${T}" ] && cmd+=(--T "${T}")
  "${cmd[@]}"
done

echo "Done. Results written under each runs_dir as llr_epsilon_lb_${MODEL_TYPE}${T:+_T${T}}.json"
