#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
NAMELIST="${WRF_RUN}/namelist.input"
RUN_SCRIPT="${CASE_ROOT}/scripts/run_wrfchem.sh"
ARCHIVE_ROOT="${CASE_ROOT}/output/archive"

RUN_HOURS=24
WRF_NTASKS_VALUE="${WRF_NTASKS:-16}"
DUST_OPT_VALUE="keep"
SEAS_OPT_VALUE="keep"
DMSEMIS_OPT_VALUE="keep"
VARIANT_NAME="variant"
KEEP_NAMELIST=0

usage() {
  cat <<EOF
Usage: $0 [options]

Run a temporary WRF-Chem variant and restore namelist.input afterward.

Options:
  --variant-name NAME   Label used in run stamp and namelist archive.
  --run-hours HOURS    Duration from current namelist start time. Default: 24.
  --tasks N            MPI ranks. Default: WRF_NTASKS or 16.
  --dust-opt VALUE     Temporary dust_opt value, or keep. Default: keep.
  --seas-opt VALUE     Temporary seas_opt value, or keep. Default: keep.
  --dmsemis-opt VALUE  Temporary dmsemis_opt value, or keep. Default: keep.
  --keep-namelist      Do not restore namelist.input after the run.
  -h, --help           Show this help.

Examples:
  $0 --variant-name edgarpm_sector_v2_dustseas_24h --run-hours 24 --tasks 16 --dust-opt 13 --seas-opt 2
  $0 --variant-name edgarpm_sector_v2_dustseas_7d --run-hours 168 --tasks 16 --dust-opt 13 --seas-opt 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant-name)
      VARIANT_NAME="$2"
      shift 2
      ;;
    --run-hours)
      RUN_HOURS="$2"
      shift 2
      ;;
    --tasks)
      WRF_NTASKS_VALUE="$2"
      shift 2
      ;;
    --dust-opt)
      DUST_OPT_VALUE="$2"
      shift 2
      ;;
    --seas-opt)
      SEAS_OPT_VALUE="$2"
      shift 2
      ;;
    --dmsemis-opt)
      DMSEMIS_OPT_VALUE="$2"
      shift 2
      ;;
    --keep-namelist)
      KEEP_NAMELIST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! "${RUN_HOURS}" =~ ^[0-9]+$ || "${RUN_HOURS}" -le 0 ]]; then
  echo "--run-hours must be a positive integer" >&2
  exit 2
fi

STAMP="${WRFCHEM_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_${VARIANT_NAME}}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/namelist_${STAMP}"
mkdir -p "${ARCHIVE_DIR}"
cp -p "${NAMELIST}" "${ARCHIVE_DIR}/namelist.input.before_variant"

restore_namelist() {
  if [[ "${KEEP_NAMELIST}" -eq 0 && -f "${ARCHIVE_DIR}/namelist.input.before_variant" ]]; then
    cp -p "${ARCHIVE_DIR}/namelist.input.before_variant" "${NAMELIST}"
  fi
}
trap restore_namelist EXIT

python3 - "${NAMELIST}" "${RUN_HOURS}" "${DUST_OPT_VALUE}" "${SEAS_OPT_VALUE}" "${DMSEMIS_OPT_VALUE}" <<'PY'
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
run_hours_total = int(sys.argv[2])
overrides = {
    "dust_opt": sys.argv[3],
    "seas_opt": sys.argv[4],
    "dmsemis_opt": sys.argv[5],
}

text = path.read_text(encoding="utf-8")


def get_int(key: str) -> int:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9]+)", text, re.M)
    if not match:
        raise SystemExit(f"Could not find {key} in {path}")
    return int(match.group(1))


def set_int(text_in: str, key: str, value: int) -> str:
    pattern = rf"^(\s*{re.escape(key)}\s*=\s*)([^,\n]+)(,.*)$"
    replacement = rf"\g<1>{value}\g<3>"
    new_text, count = re.subn(pattern, replacement, text_in, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f"Could not update {key} in {path}")
    return new_text


start = datetime(
    get_int("start_year"),
    get_int("start_month"),
    get_int("start_day"),
    get_int("start_hour"),
    get_int("start_minute"),
    get_int("start_second"),
)
end = start + timedelta(hours=run_hours_total)
run_days, run_hours = divmod(run_hours_total, 24)

new_text = text
for key, value in [
    ("run_days", run_days),
    ("run_hours", run_hours),
    ("run_minutes", 0),
    ("run_seconds", 0),
    ("end_year", end.year),
    ("end_month", end.month),
    ("end_day", end.day),
    ("end_hour", end.hour),
    ("end_minute", end.minute),
    ("end_second", end.second),
]:
    new_text = set_int(new_text, key, value)

for key, value in overrides.items():
    if value.lower() != "keep":
        new_text = set_int(new_text, key, int(value))

path.write_text(new_text, encoding="utf-8")
PY

cp -p "${NAMELIST}" "${ARCHIVE_DIR}/namelist.input.active_variant"

echo "Running WRF-Chem variant:"
echo "  stamp=${STAMP}"
echo "  run_hours=${RUN_HOURS}"
echo "  WRF_NTASKS=${WRF_NTASKS_VALUE}"
echo "  dust_opt=${DUST_OPT_VALUE}"
echo "  seas_opt=${SEAS_OPT_VALUE}"
echo "  dmsemis_opt=${DMSEMIS_OPT_VALUE}"
echo "  namelist archive=${ARCHIVE_DIR}"

export WRFCHEM_RUN_STAMP="${STAMP}"
export WRF_NTASKS="${WRF_NTASKS_VALUE}"
bash "${RUN_SCRIPT}"
