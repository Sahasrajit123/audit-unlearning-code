#!/usr/bin/env bash
#
# End-to-end driver for the cumulative privacy audit: build whatever reference stats are
# missing, then run run_cumulative_audit.py over every sweep.
#
# The audit has two stages, and only the first is expensive:
#
#   Stage 1 (minutes to hours, once per sweep+checkpoint)
#     audit_utils.py  ->  <sweep>/forget_pointwise_stats_<source>.json
#     Scores all m forget points under all ~50 shadow runs to fit the per-point in/out
#     Gaussians. ~50 model loads per stats file, 6 files per sweep (trained, unlearned,
#     and one per intermediate unlearn epoch). Skipped for any stats file that already
#     exists -- delete the file or pass --force-stats to rebuild it.
#
#   Stage 2 (~2s per held-out run, all k and all three bounds included)
#     run_cumulative_audit.py  ->  <sweep>/audit_bounds_<source>_<metric>.json
#     Attacks the 10 held-out runs under <sweep>/test_run/ once per (sweep, source) and
#     turns the overlap scores into the eps / rho / mu lower bounds. This is the part you
#     will re-run often; it is cheap, and --from-json makes it free.
#
# Usage
#   ./run_full_audit.sh                      # stage 1 (missing only) + stage 2, all sweeps
#   ./run_full_audit.sh --dry-run            # print every command, run nothing
#   ./run_full_audit.sh --audit-only         # stage 2 only (stats must already exist)
#   ./run_full_audit.sh --stats-only         # stage 1 only
#   ./run_full_audit.sh --k "250 500 1000"   # sweep k in stage 2, one scoring pass
#   ./run_full_audit.sh --device cpu         # default is cuda:0
#   ./run_full_audit.sh --split              # also move the last 10 runs of a sweep that
#                                            # has no test_run/ into one (MOVES FILES)
#   ./run_full_audit.sh runs/delete/cifar100_bs_1        # just these sweeps
#   ./run_full_audit.sh runs/badteacher/*                # every bad-teacher sweep
#   ./run_full_audit.sh runs/scrub_r/*                   # every SCRUB+R sweep
#
# With no sweeps named, every runs/<method>/<sweep>/ directory holding run_* folders is
# audited -- every unlearning method in one pass, which is the comparison this project
# exists for. The audit itself is method-agnostic: it only ever reads checkpoints. (For
# scrub_r that means unlearned_model.pth is the rewound model and the per-epoch
# checkpoints are the un-rewound SCRUB trajectory -- see the README.)
#
# Environment overrides: PYTHON, DATASET, DATA_DIR, DEVICE, K, METRIC, SUMMARY, NUM_TEST.
#
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-cifar100}"
DATA_DIR="${DATA_DIR:-data/cifar100/data_split/cifar100_bs_1}"
DEVICE="${DEVICE:-cuda:0}"
K="${K:-500}"
METRIC="${METRIC:-phi}"
SUMMARY="${SUMMARY:-tables/audit_summary.json}"
NUM_TEST="${NUM_TEST:-10}"

DO_STATS=1
DO_AUDIT=1
FORCE_STATS=0
DO_SPLIT=0
DRY_RUN=0
SWEEPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stats-only)   DO_AUDIT=0; shift ;;
    --audit-only)   DO_STATS=0; shift ;;
    --force-stats)  FORCE_STATS=1; shift ;;
    --split)        DO_SPLIT=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --device)       DEVICE="$2"; shift 2 ;;
    --k)            K="$2"; shift 2 ;;
    --metric)       METRIC="$2"; shift 2 ;;
    --summary)      SUMMARY="$2"; shift 2 ;;
    -h|--help)      sed -n '2,37p' "$0"; exit 0 ;;
    -*)             echo "unknown flag: $1" >&2; exit 2 ;;
    *)              SWEEPS+=("$1"); shift ;;
  esac
done

cd "$(dirname "$0")/.."

# Default: every sweep directory that actually holds runs, under either method.
if [[ ${#SWEEPS[@]} -eq 0 ]]; then
  for d in runs/*/*; do
    [[ -d "$d" ]] && compgen -G "$d/run_*" >/dev/null && SWEEPS+=("$d")
  done
fi
if [[ ${#SWEEPS[@]} -eq 0 ]]; then
  echo "no sweep directories found (expected runs/<method>/<sweep>/run_*)" >&2
  exit 1
fi

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  [dry-run] %s\n' "$*"
  else
    printf '  + %s\n' "$*"
    "$@"
  fi
}

echo "sweeps      : ${SWEEPS[*]}"
echo "python      : $PYTHON"
echo "device      : $DEVICE   metric: $METRIC   k: $K"
echo "data        : $DATASET @ $DATA_DIR"
echo "stage 1 (stats): $([[ $DO_STATS -eq 1 ]] && echo yes || echo skipped)$([[ $FORCE_STATS -eq 1 ]] && echo ' (forced rebuild)' || true)"
echo "stage 2 (audit): $([[ $DO_AUDIT -eq 1 ]] && echo yes || echo skipped)"

# ---------------------------------------------------------------------------
# Stage 0: hold out the attack targets, only if asked (this MOVES run folders)
# ---------------------------------------------------------------------------
auditable=()
for sweep in "${SWEEPS[@]}"; do
  if [[ ! -d "$sweep/test_run" ]]; then
    if [[ $DO_SPLIT -eq 1 ]]; then
      echo
      echo "### $sweep: no test_run/, holding out the last $NUM_TEST run(s)"
      run "$PYTHON" -m pipeline.split_test_run --run-dir "$sweep" --num-test "$NUM_TEST"
    else
      echo
      echo "### $sweep: SKIPPED -- no test_run/ to attack."
      echo "    Pass --split to move its last $NUM_TEST runs into test_run/, or run:"
      echo "      $PYTHON -m pipeline.split_test_run --run-dir $sweep --num-test $NUM_TEST"
      continue
    fi
  fi
  auditable+=("$sweep")
done

if [[ ${#auditable[@]} -eq 0 ]]; then
  echo
  echo "nothing auditable: no sweep has a test_run/ directory." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 1: reference in/out stats, one file per (sweep, checkpoint), skip if present
# ---------------------------------------------------------------------------
if [[ $DO_STATS -eq 1 ]]; then
  for sweep in "${auditable[@]}"; do
    echo
    echo "### $sweep: reference stats"

    # Which intermediate unlearn-epoch checkpoints this sweep actually saved.
    first_run=$(find "$sweep" -maxdepth 1 -type d -name 'run_*' | sort | head -1)
    epochs=()
    for ckpt in "$first_run"/unlearned_model_epoch_*.pth; do
      [[ -e "$ckpt" ]] || continue
      n="${ckpt##*_epoch_}"; epochs+=("${n%.pth}")
    done

    # source label -> the flag that builds it (empty flag = the final unlearned model)
    declare -a labels=("trained" "unlearned")
    declare -a flags=("--use-trained-model" "")
    for n in "${epochs[@]}"; do
      labels+=("unlearn_epoch_$n"); flags+=("--unlearn-epoch $n")
    done

    for i in "${!labels[@]}"; do
      out="$sweep/forget_pointwise_stats_${labels[$i]}.json"
      if [[ -f "$out" && $FORCE_STATS -eq 0 ]]; then
        echo "  exists, skipping: $out"
        continue
      fi
      # shellcheck disable=SC2086
      run "$PYTHON" -m audit.audit_utils --run-dir "$sweep" --dataset "$DATASET" \
          --data-dir "$DATA_DIR" --device "$DEVICE" ${flags[$i]}
    done
    unset labels flags
  done
fi

# ---------------------------------------------------------------------------
# Stage 2: the cumulative audit -- one scoring pass per (sweep, source), all bounds
# ---------------------------------------------------------------------------
if [[ $DO_AUDIT -eq 1 ]]; then
  echo
  echo "### cumulative audit (eps / rho / mu, mean-v and median-v)"
  # shellcheck disable=SC2086
  run "$PYTHON" -m audit.run_cumulative_audit "${auditable[@]}" \
      --all-epochs --k $K --metric "$METRIC" --device "$DEVICE" \
      --summary-out "$SUMMARY"

  echo
  echo "per-sweep results : <sweep>/audit_bounds_<source>_${METRIC}.json"
  echo "combined summary  : $SUMMARY"
  echo
  echo "To re-derive the bounds later without touching a checkpoint (e.g. a different"
  echo "conv_delta or confidence level):"
  echo "  $PYTHON -m audit.run_cumulative_audit --from-json $SUMMARY --conv-delta 1e-5"
fi
