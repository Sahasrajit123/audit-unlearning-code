#!/usr/bin/env bash
# Re-aggregate finished experiments over the WIDER r grid, without disturbing the
# audits already on disk.
#
#   DRY_RUN=1 scripts/reaudit_r_grid.sh          # print the plan, run nothing
#   scripts/reaudit_r_grid.sh                    # every config x every method
#   CONFIGS="configs/base.yaml" scripts/reaudit_r_grid.sh
#   METHODS="noop npo" scripts/reaudit_r_grid.sh
#   OUT_SUBDIR=audit_r_grid_v2 scripts/reaudit_r_grid.sh
#
# Why no GPU is needed: r is a POST-PROCESSING knob. Training and scoring do not
# depend on it -- the per-example losses in each run's losses.json are the only model
# output the audit consumes. aggregate_audit.py re-fits the calibration Gaussians,
# re-predicts, reveals the labels and recomputes V for every r in attack.r_values. A
# new r grid is therefore a CPU re-aggregation, not a new experiment.
#
# How the existing audits survive: output goes to <output_root>/$OUT_SUBDIR/<method>/
# rather than <output_root>/audit/<method>/. Nothing under audit/ is read or written.
# Because the new grid is a strict superset of the old one, every previously reported
# r reappears with the same value, beside the new points.
#
# One thing to watch: with more candidates in the sweep, the calibration-only
# leave-one-out in aggregate_audit.py can freeze a DIFFERENT r than before. That stays
# valid -- selection never touches the evaluation runs -- but it means the headline r
# of the two trees may differ. Both are on disk; compare deliberately instead of
# assuming the new headline supersedes the old.
#
# Knobs (env): CONFIGS, METHODS, OUT_SUBDIR, SELECT_R, NO_PLOTS, DRY_RUN, PYTHON.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

CONFIGS="${CONFIGS:-configs/base.yaml configs/base_batch_size_4.yaml configs/base_batch_size_10.yaml configs/base_batching_qa.yaml}"
METHODS="${METHODS:-noop npo retain_ft grad_ascent grad_diff simnpo}"
OUT_SUBDIR="${OUT_SUBDIR:-audit_r_grid}"
SELECT_R="${SELECT_R:-calibration}"

EXTRA=()
[[ "${NO_PLOTS:-0}" == "1" ]] && EXTRA+=(--no_plots)

echo "[reaudit] configs   : $CONFIGS"
echo "[reaudit] methods   : $METHODS"
echo "[reaudit] out subdir: <output_root>/$OUT_SUBDIR/<method>   (audit/ left untouched)"
echo "[reaudit] select r  : $SELECT_R"

cfg_field() {  # cfg_field <config path> <dotted key>
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
sys.path.insert(0, ".")
from audit_tofu.config import load_config
cfg = load_config(sys.argv[1])
for part in sys.argv[2].split("."):
    cfg = cfg[part]
print(cfg)
PY
}

failed=()
done_count=0
for cfg in $CONFIGS; do
  [[ -f "$REPO/$cfg" ]] || { echo "[reaudit] SKIP $cfg (no such config)"; continue; }

  root="$(cd "$REPO" && cfg_field "$REPO/$cfg" experiment.output_root)"
  grid="$(cd "$REPO" && cfg_field "$REPO/$cfg" attack.r_values)"
  n_losses=$(find -L "$root" -name losses.json 2>/dev/null | wc -l)

  echo ""
  echo "=== $cfg"
  echo "    output_root : $root"
  echo "    r grid      : $grid"
  echo "    losses.json : $n_losses"
  if [[ "$n_losses" -eq 0 ]]; then
    echo "    SKIP: no scored runs under this output_root -- run the audit itself first"
    continue
  fi

  for method in $METHODS; do
    out="$root/$OUT_SUBDIR/$method"
    cmd=("$PYTHON" "$REPO/scripts/aggregate_audit.py"
         --config "$REPO/$cfg" --method "$method"
         --out_dir "$out" --select_r_from "$SELECT_R" "${EXTRA[@]}")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "    would run: ${cmd[*]}"
      continue
    fi
    mkdir -p "$out"
    echo "    --> $method"
    if ! (cd "$REPO" && "${cmd[@]}" > "$out/aggregate.log" 2>&1); then
      # A method with no losses at all exits non-zero; that is normal for a partially
      # run experiment, so keep going and report the set at the end.
      echo "        FAILED (see $out/aggregate.log): $(tail -2 "$out/aggregate.log" | tr '\n' ' ')"
      failed+=("$cfg/$method")
      continue
    fi
    done_count=$((done_count + 1))
    "$PYTHON" - "$out/audit_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
h = d["headline"]
rs = sorted(int(k) for k in d["per_r"])
print(f"        r sweep {rs}\n"
      f"        frozen r={d['frozen_r']}  eps_LB={h['epsilon_lb_mean']}  "
      f"rho_LB={h['rho_lb_mean']}  mu_LB={h['mu_lb_mean']}")
PY
  done
done

echo ""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[reaudit] dry run: nothing written"
  exit 0
fi
echo "[reaudit] $done_count aggregation(s) written"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "[reaudit] ${#failed[@]} failed: ${failed[*]}"
fi
cat <<EOF
[reaudit] old audits stay at : <output_root>/audit/<method>/
[reaudit] new audits land at : <output_root>/$OUT_SUBDIR/<method>/
[reaudit] one table over BOTH trees (the audit_tree column separates them):
          $PYTHON scripts/backfill_dp_bounds.py --scan runs --csv runs/dp_bounds_summary.csv
EOF
