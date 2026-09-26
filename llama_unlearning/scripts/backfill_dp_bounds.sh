#!/usr/bin/env bash
# Add the rho (zCDP) and mu (GDP) lower bounds to every finished audit on this box.
#
#   scripts/backfill_dp_bounds.sh                 # write dp_bounds.json + the CSV
#   DRY_RUN=1 scripts/backfill_dp_bounds.sh       # compute and print, write nothing
#   IN_PLACE=1 scripts/backfill_dp_bounds.sh      # also fold the fields into audit_summary.json
#   scripts/backfill_dp_bounds.sh /some/other/root ...   # extra roots to scan
#
# Nothing is re-run: the overlap vector V stored in each audit_summary.json is the
# audit's sufficient statistic, so all three bounds are closed-form from
# (m, r, {V}, zeta). No model, no losses, no GPU. The epsilon bound is recomputed
# only as a cross-check against the value each audit already stored.
#
# Knobs (env): DRY_RUN, IN_PLACE, ZETA, CONV_DELTA, CSV, PYTHON.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

# The folders that hold audit output. `runs` is the repo's symlink into scratch; the
# resolved path is scanned too, so passing either form finds the same audits once
# (the script de-duplicates by resolved path).
ROOTS=()
[[ -e "$REPO/runs" ]] && ROOTS+=("$REPO/runs")
for extra in "$@"; do ROOTS+=("$extra"); done
if [[ ${#ROOTS[@]} -eq 0 ]]; then
  echo "no roots to scan: $REPO/runs does not exist, and none were passed" >&2
  exit 1
fi

# The combined table lands beside the experiment folders, not in the repo.
CSV_DEFAULT="$(cd "$(dirname "${ROOTS[0]}")" && pwd)/$(basename "${ROOTS[0]}")/dp_bounds_summary.csv"
CSV="${CSV:-$CSV_DEFAULT}"

ARGS=(--scan "${ROOTS[@]}")
[[ -n "${ZETA:-}" ]]       && ARGS+=(--zeta "$ZETA")
[[ -n "${CONV_DELTA:-}" ]] && ARGS+=(--conv_delta "$CONV_DELTA")
[[ "${IN_PLACE:-0}" == "1" ]] && ARGS+=(--in_place)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry_run)
else
  ARGS+=(--csv "$CSV")
fi

echo "[backfill.sh] roots  : ${ROOTS[*]}"
echo "[backfill.sh] python : $PYTHON"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[backfill.sh] mode   : dry run (nothing written)"
else
  echo "[backfill.sh] writes : <audit dir>/dp_bounds.json  and  $CSV"
  [[ "${IN_PLACE:-0}" == "1" ]] && echo "[backfill.sh]          plus audit_summary.json, updated in place"
fi

exec "$PYTHON" "$REPO/scripts/backfill_dp_bounds.py" "${ARGS[@]}"
