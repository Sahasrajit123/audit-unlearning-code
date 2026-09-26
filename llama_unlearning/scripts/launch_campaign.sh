#!/usr/bin/env bash
# Launch the whole pinned-split campaign: reference model, four audits, aggregation.
#
#   scripts/launch_campaign.sh <gpu_list> [runs_per_gpu]
#
#   scripts/launch_campaign.sh 5 2                  # one free GPU, 2 runs on it
#   scripts/launch_campaign.sh 2,5,7 2              # three GPUs, 6 concurrent
#   DRY_RUN=1 scripts/launch_campaign.sh 2,5,7 2    # print the plan, run nothing
#
# STAGE ORDER MATTERS, for one reason. The reference model is trained FIRST so that
# every audit run's utility suite can pick it up and record forget quality inline. If
# the audits run first, their utility.json files carry forget_quality: null and the
# number has to come from scripts/forget_quality.py afterwards. Both routes give
# IDENTICAL numbers -- they call the same audit_tofu.utility.forget_quality_slices --
# so this is about convenience, not correctness. Reference first is simply strictly
# better: ~20 minutes buys the metric in all 120 per-run files.
#
# Every stage is idempotent and safe to interrupt:
#   * build_manifest.py refuses to clobber an existing manifest
#   * train_reference.py reuses an existing reference checkpoint
#   * run_audit.sh skips any method whose losses.json exists
#   * aggregation is pure post-processing and re-runs cheaply
# So re-running this script after a failure or a Ctrl-C resumes rather than restarts.
#
# Env knobs:
#   CONFIGS="base base_batching_qa"   subset of audits (default: all four)
#   SKIP_REFERENCE=1                  assume the reference already exists
#   SKIP_AUDITS=1                     aggregation only
#   SKIP_AGGREGATE=1                  runs only
#   DRY_RUN=1                         print the plan and exit
#   STATUS_INTERVAL=1800              passed through to run_audit.sh (0 disables)
#   EXTRA_SET="utility.compute_rouge=true"   extra --set args for the audits
set -uo pipefail

GPU_LIST="${1:?usage: launch_campaign.sh <gpu_list> [runs_per_gpu]   e.g. 2,5,7 2}"
RUNS_PER_GPU="${2:-1}"

cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"
export STATUS_INTERVAL="${STATUS_INTERVAL:-1800}"

CONFIGS="${CONFIGS:-base base_batch_size_10 base_batch_size_4 base_batching_qa}"
REF_CONFIG="configs/base.yaml"   # any of them: all four share D_r, hence one reference
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
FIRST_GPU="${GPUS[0]}"
SLOTS=$(( ${#GPUS[@]} * RUNS_PER_GPU ))

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="campaign_logs/${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/campaign.tsv"
: > "$SUMMARY"

say() { printf '%s\n' "$*"; }
rule() { printf '%s\n' "==============================================================================="; }

# --- preflight ---------------------------------------------------------------
rule
say "PINNED-SPLIT CAMPAIGN   ${STAMP}"
rule
say "python        : $PY"
say "HF_HOME       : $HF_HOME"
say "gpus          : ${GPUS[*]}  (${RUNS_PER_GPU}/gpu -> ${SLOTS} concurrent)"
say "audit configs : $CONFIGS"
say "logs          : $LOGDIR/"
say ""

if [ ! -x "$PY" ]; then
  say "ERROR: $PY not found. Create the venv (see README) or set PYTHON=." >&2
  exit 1
fi

# Per-process footprint as nvidia-smi sees it: ~15.4 GiB peak PyTorch reserved
# (measured over 30 runs) PLUS the CUDA context, which the allocator statistics do
# not include and which costs a few hundred MiB per process. 16400 MiB is the
# budgeting figure; using the bare 15.4 GiB understates it by exactly the amount that
# turns "just fits" into an OOM.
PER_PROC_MIB=16400

# A GPU id may appear MORE THAN ONCE in the list, which is how you express uneven
# packing across cards with different amounts of free memory: run_audit.sh assigns
# runs round-robin over the list, so "5,5,2,7" puts two concurrent runs on gpu 5 and
# one each on 2 and 7. Slots per card is therefore (occurrences x RUNS_PER_GPU), not
# RUNS_PER_GPU, and the check below has to count occurrences or it understates demand.
if command -v nvidia-smi >/dev/null 2>&1; then
  say "gpu memory (~${PER_PROC_MIB} MiB per concurrent run, incl. CUDA context):"
  uniq_gpus=$(printf '%s\n' "${GPUS[@]}" | sort -u)
  tight=0
  for g in $uniq_gpus; do
    occ=$(printf '%s\n' "${GPUS[@]}" | grep -cx "$g")
    slots=$(( occ * RUNS_PER_GPU ))
    line=$(nvidia-smi --id="$g" --query-gpu=memory.total,memory.free,utilization.gpu \
           --format=csv,noheader 2>/dev/null)
    if [ -z "$line" ]; then say "  gpu $g: NOT FOUND"; tight=1; continue; fi
    tot=$(echo "$line" | awk -F'[, ]+' '{print $1}')
    free_mib=$(echo "$line" | awk -F'[, ]+' '{print $3}')
    util=$(echo "$line" | awk -F'[, ]+' '{print $5}')
    need_mib=$(( slots * PER_PROC_MIB ))
    flag=""
    if [ "$free_mib" -lt "$need_mib" ]; then
      flag="  <-- WILL NOT FIT (need ${need_mib})"; tight=1
    elif [ "$free_mib" -lt $(( need_mib + PER_PROC_MIB / 2 )) ]; then
      flag="  <-- tight (need ${need_mib})"
    fi
    say "  gpu $g: ${free_mib} MiB free of ${tot}, util ${util}%  -> ${slots} slot(s)${flag}"
  done
  if [ "$tight" -ne 0 ]; then
    say ""
    say "  At least one GPU cannot hold its assigned runs. Options:"
    say "    * fewer runs per gpu:   $0 $GPU_LIST 1"
    say "    * uneven packing:       $0 5,5,2,7 1   (2 on gpu 5, 1 each on 2 and 7)"
    say "    * smaller footprint:    EXTRA_SET='training.micro_batch_size=1' $0 ..."
    say "  An OOM is recoverable, not fatal: that run fails, run_audit.sh carries on,"
    say "  and re-running resumes it. But it wastes the fine-tune that preceded it."
  fi
  say ""
fi

# Disk. keep_all keeps 1 trained + 6 unlearned checkpoints per run, ~17 GiB/run.
n_cfg=$(echo "$CONFIGS" | wc -w | tr -d ' ')
need_gib=$(( n_cfg * 30 * 17 ))
mkdir -p runs; root_fs="runs"
avail_gib=$(df -BG --output=avail "$root_fs" 2>/dev/null | tail -1 | tr -dc '0-9')
say "disk          : need ~${need_gib} GiB for ${n_cfg} audit(s) at retention=keep_all"
if [ -n "$avail_gib" ]; then
  say "                ${avail_gib} GiB available on ${root_fs}"
  if [ "$avail_gib" -lt "$need_gib" ]; then
    say ""
    say "ERROR: not enough space. Either free some, or drop to keep_trained:" >&2
    say "  EXTRA_SET='storage.retention=keep_trained' $0 $GPU_LIST $RUNS_PER_GPU" >&2
    exit 1
  fi
fi
say ""

# --- wandb ------------------------------------------------------------------
# Checked HERE, not left to each stage, because the failure mode is invisible: every
# wandb call degrades to a no-op by design, so a whole 76-GPU-hour campaign can run
# with logging silently off and nothing looks wrong until you go looking for the runs.
# A common cause is an unreadable ~/.netrc (e.g. on a network home directory).
WB_STATUS=$("$PY" - "$REF_CONFIG" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
from audit_tofu.wandb_logger import preflight
cfg = load_config(sys.argv[1])
ok, why = preflight(cfg)
w = cfg.get("wandb") or {}
print(f"{'ON' if ok else 'off'}\t{why}\t{w.get('enabled')}\t{w.get('project')}")
PYEOF
)
wb_state=$(echo "$WB_STATUS" | cut -f1)
wb_why=$(echo "$WB_STATUS" | cut -f2)
wb_enabled=$(echo "$WB_STATUS" | cut -f3)
wb_project=$(echo "$WB_STATUS" | cut -f4)
say "wandb         : ${wb_state} -- ${wb_why}"
if [ "$wb_state" = "ON" ]; then
  say "                project=${wb_project}"
elif [ "$wb_enabled" = "True" ]; then
  say ""
  say "  wandb is ENABLED in the config but unusable, so all ${n_cfg} audit(s) and the"
  say "  reference would run with logging silently disabled. Results on disk are"
  say "  unaffected -- JSON under output_root is always authoritative -- but you would"
  say "  get no wandb runs. Fix before committing the GPU time:"
  say "    export WANDB_API_KEY=...                       (recommended here)"
  say "    EXTRA_SET='wandb.mode=offline' $0 ...          (sync later)"
  say "    EXTRA_SET='wandb.enabled=false' $0 ...         (accept it, no warnings)"
  say "  Continuing in 10s; Ctrl-C to abort."
  [ -z "${DRY_RUN:-}" ] && sleep 10
fi
say ""

# --- plan --------------------------------------------------------------------
say "plan:"
say "  1. validate/build manifests           (${n_cfg} config(s), no GPU)"
if [ -n "${SKIP_REFERENCE:-}" ]; then
  say "  2. reference model                    SKIPPED (SKIP_REFERENCE)"
else
  say "  2. train retain-only reference        gpu ${FIRST_GPU}, ~20 min, shared by all"
fi
if [ -n "${SKIP_AUDITS:-}" ]; then
  say "  3. audits                             SKIPPED (SKIP_AUDITS)"
else
  say "  3. run each audit (30 runs x 6 methods, resumable)"
  for c in $CONFIGS; do say "       configs/$c.yaml"; done
fi
if [ -n "${SKIP_AGGREGATE:-}" ]; then
  say "  4. aggregation                        SKIPPED (SKIP_AGGREGATE)"
else
  say "  4. collect_results + forget_quality + aggregate_audit per config"
fi
say ""
say "  ~19 GPU-h per audit (measured) -> ~$(( n_cfg * 19 )) GPU-h total,"
say "  ~$(( n_cfg * 19 / (SLOTS > 0 ? SLOTS : 1) )) h wall clock at ${SLOTS} concurrent."
say ""

if [ -n "${DRY_RUN:-}" ]; then
  say "DRY_RUN set; nothing executed."
  exit 0
fi

START=$(date +%s)
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$SUMMARY"; }

# --- stage 1: manifests ------------------------------------------------------
rule
say "STAGE 1: manifests"
rule
for c in $CONFIGS; do
  "$PY" scripts/build_manifest.py --config "configs/$c.yaml" \
      > "$LOGDIR/manifest_$c.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    say "  $c: FAILED (see $LOGDIR/manifest_$c.log)" >&2
    record manifest "$c" "$rc" 0
    exit 1
  fi
  # build_manifest.py prints the hash in two different layouts (fresh build vs
  # "already exists"), so pull the hex directly rather than by field position.
  hash=$(grep -oE '[0-9a-f]{64}' "$LOGDIR/manifest_$c.log" | head -1 | cut -c1-12)
  if grep -q "already exists" "$LOGDIR/manifest_$c.log"; then
    say "  $c: existing manifest validated  (${hash}...)"
  else
    say "  $c: built  (${hash}...)"
  fi
  record manifest "$c" 0 0
done
say ""

# --- stage 2: reference ------------------------------------------------------
REF_DIR=$("$PY" - "$REF_CONFIG" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from audit_tofu.config import load_config
cfg = load_config(sys.argv[1])
print(cfg["utility"].get("reference_dir")
      or (Path(cfg["experiment"]["output_root"]) / "reference"))
PYEOF
)

if [ -z "${SKIP_REFERENCE:-}" ]; then
  rule
  say "STAGE 2: retain-only reference model"
  rule
  say "  target: $REF_DIR"
  if [ -f "$REF_DIR/truth_ratios.json" ]; then
    say "  already present; skipping (delete truth_ratios.json to force a retrain)"
    record reference shared 0 0
  else
    t0=$(date +%s)
    # tee, not >. Redirecting hid this stage's own preflight output -- including its
    # wandb status -- so a reference that ran with wandb silently disabled looked
    # identical to one that logged fine. Stage 3 already tees for the same reason.
    "$PY" scripts/train_reference.py --config "$REF_CONFIG" --gpu "$FIRST_GPU" \
        2>&1 | tee "$LOGDIR/reference.log"
    rc=${PIPESTATUS[0]}
    dur=$(( $(date +%s) - t0 ))
    if [ "$rc" -ne 0 ]; then
      say "  FAILED after ${dur}s -- see $LOGDIR/reference.log" >&2
      say "  The audits can still run; forget quality would then come from" >&2
      say "  scripts/forget_quality.py afterwards. Aborting so you can decide." >&2
      record reference shared "$rc" "$dur"
      exit 1
    fi
    say "  done in $(( dur / 60 ))m$(( dur % 60 ))s"
    record reference shared 0 "$dur"
  fi
  say ""
fi

if [ ! -f "$REF_DIR/truth_ratios.json" ]; then
  say "NOTE: no reference at $REF_DIR."
  say "      Runs will record forget_quality: null; recover it later with"
  say "      scripts/forget_quality.py once a reference exists."
  say ""
fi

# --- stage 3: audits ---------------------------------------------------------
FAILED_CFGS=""
if [ -z "${SKIP_AUDITS:-}" ]; then
  for c in $CONFIGS; do
    rule
    say "STAGE 3: audit  configs/$c.yaml"
    rule
    t0=$(date +%s)
    # run_audit.sh does its own GPU scheduling, resume and status ticks.
    # shellcheck disable=SC2086
    scripts/run_audit.sh "configs/$c.yaml" "$GPU_LIST" "$RUNS_PER_GPU" ${EXTRA_SET:-} \
        2>&1 | tee "$LOGDIR/audit_$c.log"
    rc=${PIPESTATUS[0]}
    dur=$(( $(date +%s) - t0 ))
    record audit "$c" "$rc" "$dur"
    if [ "$rc" -ne 0 ]; then
      say ""
      say "  configs/$c.yaml exited $rc after $(( dur / 3600 ))h$(( (dur % 3600) / 60 ))m."
      say "  Continuing with the remaining configs; re-run this script to retry"
      say "  (completed methods are skipped)." >&2
      FAILED_CFGS="$FAILED_CFGS $c"
    else
      say "  configs/$c.yaml done in $(( dur / 3600 ))h$(( (dur % 3600) / 60 ))m"
    fi
    say ""
  done
fi

# --- stage 4: aggregation ----------------------------------------------------
if [ -z "${SKIP_AGGREGATE:-}" ]; then
  for c in $CONFIGS; do
    rule
    say "STAGE 4: aggregate  configs/$c.yaml"
    rule
    cfg="configs/$c.yaml"

    if "$PY" scripts/collect_results.py --config "$cfg" \
           > "$LOGDIR/collect_$c.log" 2>&1; then
      rows=$(grep -oE 'runs\.csv +[0-9,]+ rows' "$LOGDIR/collect_$c.log" | head -1)
      printf '  %-22s ok   %s\n' collect_results "$rows"
    else
      printf '  %-22s FAILED (see %s)\n' collect_results "$LOGDIR/collect_$c.log"
    fi

    if [ -f "$REF_DIR/truth_ratios.json" ]; then
      if "$PY" scripts/forget_quality.py --config "$cfg" \
             > "$LOGDIR/forgetq_$c.log" 2>&1; then
        printf '  %-22s ok\n' forget_quality
      else
        printf '  %-22s FAILED (see %s)\n' forget_quality "$LOGDIR/forgetq_$c.log"
      fi
    else
      printf '  %-22s skipped (no reference)\n' forget_quality
    fi

    METHODS=$("$PY" - "$cfg" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
print(" ".join(load_config(sys.argv[1])["experiment"]["methods"]))
PYEOF
)
    for m in $METHODS; do
      if "$PY" scripts/aggregate_audit.py --config "$cfg" --method "$m" \
             > "$LOGDIR/agg_${c}_${m}.log" 2>&1; then
        eps=$("$PY" - "$cfg" "$m" <<'PYEOF' 2>/dev/null
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from audit_tofu.config import load_config
cfg = load_config(sys.argv[1])
p = Path(cfg["experiment"]["output_root"]) / "audit" / sys.argv[2] / "audit_summary.json"
h = json.load(open(p)).get("headline", {})
v = h.get("epsilon_lb_mean")
print(f"eps_lb={v:.4f} overlap={h.get('mean_overlap')} r={h.get('r')}"
      if isinstance(v, (int, float)) else "eps_lb=None (overlap at chance)")
PYEOF
)
        printf '  %-22s ok   %s\n' "epsilon/$m" "$eps"
      else
        printf '  %-22s FAILED (see %s)\n' "epsilon/$m" "$LOGDIR/agg_${c}_${m}.log"
      fi
    done
    say ""
  done
fi

# --- summary -----------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
rule
printf 'CAMPAIGN FINISHED in %dh%02dm\n' $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 ))
rule
if [ -s "$SUMMARY" ]; then
  printf '  %-12s %-22s %-6s %s\n' stage target exit duration
  while IFS=$'\t' read -r st tgt rc dur; do
    printf '  %-12s %-22s %-6s %dm%02ds\n' "$st" "$tgt" \
        "$([ "$rc" -eq 0 ] && echo ok || echo "FAIL($rc)")" \
        $(( dur / 60 )) $(( dur % 60 ))
  done < "$SUMMARY"
fi
say ""
say "logs: $LOGDIR/"
if [ -n "$FAILED_CFGS" ]; then
  say ""
  say "WARNING: these audits exited nonzero:$FAILED_CFGS"
  say "Re-run this same command to retry; completed methods are skipped."
  exit 1
fi
say "results:"
for c in $CONFIGS; do
  ro=$("$PY" - "configs/$c.yaml" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_root"])
PYEOF
)
  say "  $c"
  say "    epsilon        $ro/audit/<method>/audit_summary.json"
  say "    forget quality $ro/forget_quality/forget_quality.csv"
  say "    tidy tables    $ro/collected/{runs,losses,truth_ratios}.csv"
done
