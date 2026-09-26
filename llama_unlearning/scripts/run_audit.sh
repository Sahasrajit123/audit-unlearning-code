#!/usr/bin/env bash
# Run the full audit: every calibration + evaluation run, every configured method.
#
#   scripts/run_audit.sh <config> <gpu_list> [runs_per_gpu] [extra --set args...]
#
#   scripts/run_audit.sh configs/base.yaml 0,1,2,3 2
#   scripts/run_audit.sh configs/qa_level.yaml 0,1,2,3,4,5 1
#   scripts/run_audit.sh configs/base.yaml 0,1 1 utility.enabled=false
#   scripts/run_audit.sh configs/base.yaml 0,1,2,3 2 wandb.mode=offline
#   STATUS_INTERVAL=3600 scripts/run_audit.sh configs/base.yaml 0,1,2,3 2
#   FAMILIES=evaluation scripts/run_audit.sh configs/forgetq5_pinned180.yaml 3,4 1
#
# Env vars: STATUS_INTERVAL (status tick seconds, 0 disables), FAMILIES (which run
# families to launch: calibration, evaluation, or both -- default both).
#
# Safe to interrupt and re-run. Each run's sign vector and seeds are pinned in
# run_state.json, and a method whose losses.json already exists is skipped, so
# re-running only does the work that is actually missing.
#
# A run whose trained checkpoint was already pruned but which needs a NEW method will
# re-fine-tune, because branching requires the checkpoint. That is why it is cheaper to
# decide the method list before launching than to add methods later.
#
# Logging, three levels:
#   terminal                      launch table, DONE/FAILED per run as it exits,
#                                 plus a periodic status tick (STATUS_INTERVAL,
#                                 default 1800s; 0 disables)
#   $LOGDIR/<run_id>.log          that run's stdout/stderr
#   <output_root>/runs/<id>/run.log   that run's own structured logger
set -uo pipefail

CONFIG="${1:?usage: run_audit.sh <config> <gpu_list> [runs_per_gpu] [extra --set args...]}"
GPU_LIST="${2:?comma-separated GPU ids, e.g. 0,1,2,3}"
RUNS_PER_GPU="${3:-1}"
shift 3 2>/dev/null || shift 2
EXTRA_SET=("$@")

cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SLOTS=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="run_logs/${STAMP}"
mkdir -p "$LOGDIR"
ASSIGN="$LOGDIR/assignments.tsv"
: > "$ASSIGN"

HELPER=".audit_pending.py"
cat > "$HELPER" <<'HELPER_EOF'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
from audit_tofu.manifest import load_manifest
from audit_tofu.run_manager import resolve_run_paths

mode, cfg_path = sys.argv[1], sys.argv[2]
# `snapshot` takes an output dir as argv[3]; every other mode treats the rest as
# --set overrides.
overrides = sys.argv[4:] if mode == "snapshot" else sys.argv[3:]
cfg = load_config(cfg_path, overrides)
man = load_manifest(cfg["experiment"]["manifest_path"])
methods = cfg["experiment"]["methods"]
root = cfg["experiment"]["output_root"]


import os

# FAMILIES restricts which run families are launched, e.g. FAMILIES=evaluation for a
# utility/forget-quality experiment whose manifest pins calibration vectors for a
# later epsilon pass but should not spend GPU time on them now. Unset = both.
FAMILIES = tuple(
    f.strip() for f in os.environ.get("FAMILIES", "calibration,evaluation").split(",")
    if f.strip()
)
_known = ("calibration", "evaluation")
_bad = [f for f in FAMILIES if f not in _known]
if _bad:
    raise SystemExit(f"FAMILIES contains unknown family {_bad}; expected {_known}")


def pending():
    out = []
    for fam in FAMILIES:
        for rid in man["sign_vectors"][fam]["run_ids"]:
            p = resolve_run_paths(root, rid)
            missing = [m for m in methods
                       if not (p.method_dir(m) / "losses.json").exists()]
            if missing:
                out.append((rid, missing))
    return out


if mode == "pending":
    for rid, missing in pending():
        print(f"{rid}\t{','.join(missing)}")

elif mode == "header":
    from audit_tofu.wandb_logger import preflight
    print(f"config       : {cfg_path}")
    print(f"model        : {cfg['model']['id']}")
    print(f"output_root  : {root}")
    print(f"manifest     : {man['split_hash'][:16]}...  m={man['split']['m']} "
          f"batching={man['split']['batching']}")
    print(f"methods      : {methods}")
    print(f"families     : {','.join(FAMILIES)}"
          + ("" if len(FAMILIES) == 2 else "   (FAMILIES env var; others skipped)"))
    print(f"retention    : {cfg['storage']['retention']}")
    print(f"utility      : enabled={cfg['utility']['enabled']} "
          f"(evaluation runs only: {not cfg['utility']['calibration_runs']})")
    ok, why = preflight(cfg)
    w = cfg.get("wandb") or {}
    print(f"wandb        : {'ON' if ok else 'off'} -- {why}")
    if ok and w.get("mode") != "offline":
        print(f"               project={w.get('project')}  "
              f"group={w.get('group') or '<name>-<hash>'}")
    elif not ok and w.get("enabled"):
        print("               WARNING: wandb is enabled but unusable. Runs will")
        print("               proceed with it disabled. Export WANDB_API_KEY, or")
        print("               pass wandb.mode=offline to silence this.")

elif mode == "report":
    print(f"\n{'method':<14}{'calibration':>13}{'evaluation':>12}")
    for m in methods:
        counts = []
        for fam in _known:
            ids = man["sign_vectors"][fam]["run_ids"]
            n = sum((resolve_run_paths(root, r).method_dir(m) / "losses.json").exists()
                    for r in ids)
            counts.append(f"{n}/{len(ids)}")
        print(f"{m:<14}{counts[0]:>13}{counts[1]:>12}")

elif mode == "methods":
    print(" ".join(methods))

elif mode == "snapshot":
    # Snapshot the launch's configuration next to its logs, so the whole audit --
    # not just each run -- is reconstructable later.
    import shutil
    from pathlib import Path
    from audit_tofu.run_manager import save_config_provenance
    out = Path(sys.argv[3]); overrides = sys.argv[4:]
    save_config_provenance(out, cfg, config_path=cfg_path, overrides=overrides,
                           argv=None)
    shutil.copyfile(cfg["experiment"]["manifest_path"], out / "manifest.snapshot.json")
    print(f"config snapshot -> {out}/config.effective.yaml (+ config_sources/, "
          f"invocation.json, manifest.snapshot.json)")
HELPER_EOF
trap 'rm -f "$HELPER"' EXIT

# Enumerate via a temp file, NOT process substitution: `mapfile < <(cmd)` returns the
# exit status of mapfile, not of cmd, so a crashing helper would yield an empty list
# and the script would cheerfully report "nothing to do". A missing manifest or a bad
# config must be a hard failure.
PENDING_FILE="$LOGDIR/pending.tsv"
if ! "$PY" "$HELPER" pending "$CONFIG" "${EXTRA_SET[@]}" > "$PENDING_FILE"; then
  echo >&2
  echo "ERROR: could not enumerate pending runs (see the traceback above)." >&2
  echo "  Most likely the manifest does not exist yet. Build it with:" >&2
  echo "    $PY scripts/build_manifest.py --config $CONFIG" >&2
  exit 1
fi
mapfile -t PENDING < "$PENDING_FILE"

"$PY" "$HELPER" header "$CONFIG" "${EXTRA_SET[@]}" || {
  echo "ERROR: config/manifest header failed" >&2; exit 1; }
echo "gpus         : ${GPUS[*]}  (${RUNS_PER_GPU} per gpu -> ${SLOTS} concurrent)"
echo "logs         : ${LOGDIR}/"
echo "pending      : ${#PENDING[@]} run(s) with missing methods"
"$PY" "$HELPER" snapshot "$CONFIG" "$LOGDIR" "${EXTRA_SET[@]}" | sed "s/^/               /"

if [ "${#PENDING[@]}" -eq 0 ]; then
  echo
  echo "Nothing to do -- every configured method already has losses.json."
  echo "Aggregate with: $PY scripts/aggregate_audit.py --config $CONFIG --method <m>"
  exit 0
fi

# --- planned assignment, printed before anything starts ----------------------
echo
echo "planned GPU assignment (round-robin over ${#GPUS[@]} gpu(s)):"
printf "  %-4s %-13s %-5s %s\n" "#" "run_id" "gpu" "methods to compute"
printf "  %-4s %-13s %-5s %s\n" "----" "-------------" "-----" "------------------"
j=0
for row in "${PENDING[@]}"; do
  RID="${row%%$'\t'*}"
  MISS="${row##*$'\t'}"
  printf "  %-4s %-13s %-5s %s\n" "$((j + 1))" "$RID" "${GPUS[$(( j % ${#GPUS[@]} ))]}" "$MISS"
  j=$(( j + 1 ))
done
echo

# --- launch, throttled to SLOTS ----------------------------------------------
# Each run is wrapped in a subshell that prints a DONE/FAILED line the moment it
# finishes, so completion and failure are reported immediately rather than waiting for
# the next status tick. That is what lets the tick interval be long. The subshell
# re-raises the child's exit code so the failure count below stays accurate.
TOTAL=${#PENDING[@]}
DONE_FILE="$LOGDIR/completed.tsv"
: > "$DONE_FILE"

START=$(date +%s)
i=0
declare -a PIDS=()
for row in "${PENDING[@]}"; do
  RUN_ID="${row%%$'\t'*}"
  MISSING="${row##*$'\t'}"
  GPU="${GPUS[$(( i % ${#GPUS[@]} ))]}"

  # Wait for a free slot before starting the next run.
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 15; done

  (
    r_start=$(date +%s)
    "$PY" scripts/run_single.py \
        --config "$CONFIG" --run_id "$RUN_ID" --gpu "$GPU" \
        ${EXTRA_SET:+--set "${EXTRA_SET[@]}"} \
        > "$LOGDIR/${RUN_ID}.log" 2>&1
    rc=$?
    r_dur=$(( $(date +%s) - r_start ))
    printf "%s\t%s\t%s\t%s\n" "$RUN_ID" "$GPU" "$rc" "$r_dur" >> "$DONE_FILE"
    n_done=$(wc -l < "$DONE_FILE" | tr -d ' ')
    if [ "$rc" -eq 0 ]; then
      printf "[%s] DONE    %-12s  GPU %-2s  %dm%02ds   (%s/%s complete)\n" \
          "$(date +%H:%M:%S)" "$RUN_ID" "$GPU" \
          $(( r_dur / 60 )) $(( r_dur % 60 )) "$n_done" "$TOTAL"
    else
      printf "[%s] FAILED  %-12s  GPU %-2s  exit %-3s after %dm%02ds  (%s/%s) -> %s\n" \
          "$(date +%H:%M:%S)" "$RUN_ID" "$GPU" "$rc" \
          $(( r_dur / 60 )) $(( r_dur % 60 )) "$n_done" "$TOTAL" \
          "$LOGDIR/${RUN_ID}.log"
    fi
    exit "$rc"
  ) &
  CHILD=$!
  PIDS+=("$CHILD")

  printf "[%s] LAUNCH  %-12s  GPU %-2s  pid %-8s  %s\n" \
      "$(date +%H:%M:%S)" "$RUN_ID" "$GPU" "$CHILD" "$LOGDIR/${RUN_ID}.log"
  printf "%s\t%s\t%s\t%s\n" "$RUN_ID" "$GPU" "$CHILD" "$LOGDIR/${RUN_ID}.log" >> "$ASSIGN"

  i=$(( i + 1 ))
  sleep 3   # stagger model loads so they do not all hit the HF cache at once
done

echo
echo "[$(date +%H:%M:%S)] all ${#PENDING[@]} runs launched; assignments in $ASSIGN"
echo "per-run detail:  tail -f $LOGDIR/<run_id>.log"
echo

# --- periodic status ticker ---------------------------------------------------
# Completion is already reported the instant a run exits (the DONE/FAILED lines
# above), so this is NOT the failure channel. Its remaining jobs are narrower:
#   * progress of still-running runs, and the ETA
#   * detecting a HUNG run -- one that neither finishes nor fails produces no line
#     at all, so the tick is the only thing that would reveal it
# Hence a long interval. 30 min gives a handful of snapshots over a ~4 h audit while
# bounding how long a stall can hide. Override with STATUS_INTERVAL (seconds), or set
# it to 0 to disable the ticker entirely.
STATUS_INTERVAL="${STATUS_INTERVAL:-1800}"
if [ "$STATUS_INTERVAL" -gt 0 ] 2>/dev/null; then
  echo "status ticks every $(( STATUS_INTERVAL / 60 )) min (STATUS_INTERVAL=$STATUS_INTERVAL, 0 disables)"
fi
(
  [ "$STATUS_INTERVAL" -gt 0 ] 2>/dev/null || exit 0
  while true; do
    sleep "$STATUS_INTERVAL"
    n_done=$(wc -l < "$DONE_FILE" 2>/dev/null | tr -d ' '); n_done=${n_done:-0}
    n_fail=$(awk -F'\t' '$3 != 0' "$DONE_FILE" 2>/dev/null | wc -l | tr -d ' ')
    el=$(( $(date +%s) - START ))
    # Extrapolate remaining time from the mean duration of finished runs.
    eta="?"
    if [ "$n_done" -gt 0 ] && [ "$n_done" -lt "$TOTAL" ]; then
      eta=$(awk -F'\t' -v n="$n_done" -v tot="$TOTAL" -v slots="$SLOTS" \
            '{s+=$4} END {if (n>0) printf "%.0fm", ((tot-n)/slots)*(s/n)/60}' \
            "$DONE_FILE" 2>/dev/null)
    fi
    printf "\n---- status %s   elapsed %dh%02dm   %s/%s done" \
        "$(date +%H:%M:%S)" $(( el / 3600 )) $(( (el % 3600) / 60 )) "$n_done" "$TOTAL"
    [ "${n_fail:-0}" -gt 0 ] && printf "   %s FAILED" "$n_fail"
    printf "   eta ~%s ----\n" "${eta:-?}"
    printf "  %-13s %-4s %-6s %s\n" "run_id" "gpu" "state" "progress"
    while IFS=$'\t' read -r rid gpu pid lg; do
      rcline=$(awk -F'\t' -v r="$rid" '$1==r {print $3}' "$DONE_FILE" 2>/dev/null | tail -1)
      if [ -n "$rcline" ]; then
        [ "$rcline" -eq 0 ] && st="done" || st="FAIL"
      elif kill -0 "$pid" 2>/dev/null; then
        st="RUN"
      else
        st="gone"
      fi
      prog=$(grep -oE '\[[a-z_]+\] (step [0-9]+/[0-9]+ \([0-9]+%\)|epoch [0-9]+/[0-9]+)' \
             "$lg" 2>/dev/null | tail -1)
      printf "  %-13s %-4s %-6s %s\n" "$rid" "$gpu" "$st" "${prog:-starting}"
    done < "$ASSIGN"
  done
) &
TICKER=$!

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=$(( FAILED + 1 ))
done
kill "$TICKER" 2>/dev/null || true
ELAPSED=$(( $(date +%s) - START ))

echo
echo "================================================================"
printf "finished in %dh %02dm  (%d run(s), %d nonzero exit)\n" \
    $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) "${#PENDING[@]}" "$FAILED"

# Per-run outcome table, slowest first, so the long tail is obvious.
if [ -s "$DONE_FILE" ]; then
  echo
  printf "  %-13s %-4s %-7s %s\n" "run_id" "gpu" "exit" "duration"
  printf "  %-13s %-4s %-7s %s\n" "-------------" "----" "-------" "--------"
  sort -t$'\t' -k4,4nr "$DONE_FILE" | while IFS=$'\t' read -r rid gpu rc dur; do
    printf "  %-13s %-4s %-7s %dm%02ds\n" \
        "$rid" "$gpu" "$([ "$rc" -eq 0 ] && echo ok || echo "FAIL($rc)")" \
        $(( dur / 60 )) $(( dur % 60 ))
  done
  awk -F'\t' '{s+=$4; if($4>mx)mx=$4} END {
    if (NR>0) printf "\n  mean %.1f min, slowest %.1f min, total %.1f GPU-hours\n",
                     (s/NR)/60, mx/60, s/3600 }' "$DONE_FILE"
fi

"$PY" "$HELPER" report "$CONFIG" "${EXTRA_SET[@]}"

echo
if [ "$FAILED" -gt 0 ]; then
  echo "WARNING: $FAILED run(s) exited nonzero. Check $LOGDIR/*.log, then re-run"
  echo "this same command -- completed methods are skipped."
  echo
fi
METHODS="$("$PY" "$HELPER" methods "$CONFIG" "${EXTRA_SET[@]}")"
echo "next:"
echo "  $PY scripts/collect_results.py --config $CONFIG"
echo "  $PY scripts/compare_methods.py --config $CONFIG"
echo "  for m in $METHODS; do"
echo "    $PY scripts/aggregate_audit.py --config $CONFIG --method \$m"
echo "  done"
