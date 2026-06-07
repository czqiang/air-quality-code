#!/usr/bin/env bash

set -euo pipefail
ulimit -s unlimited || true

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WPS_RUN="${CASE_ROOT}/wps_run"
WRF_RUN="${CASE_ROOT}/wrf_run"
LOG_DIR="${CASE_ROOT}/logs"
NAMELIST="${WRF_RUN}/namelist.input"
APPLY_CAMS_ICBC="${APPLY_CAMS_ICBC:-1}"
ICBC_SCRIPT="${CASE_ROOT}/scripts/apply_cams_eac4_to_wrfchem_icbc.py"
CAMS_ROOT="${CASE_ROOT}/input_data/cams_eac4_chemical_icbc"

mkdir -p "${LOG_DIR}"

set_namelist_value() {
  local key="$1"
  local value="$2"
  python3 - "${NAMELIST}" "${key}" "${value}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
text = path.read_text(encoding="utf-8")
pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)[^,\n]*", re.IGNORECASE | re.MULTILINE)
new_text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
if count != 1:
    raise SystemExit(f"Could not find namelist key: {key}")
path.write_text(new_text, encoding="utf-8")
PY
}

shopt -s nullglob
met_files=( "${WPS_RUN}"/met_em.d01.* )
shopt -u nullglob

if [[ ${#met_files[@]} -eq 0 ]]; then
  echo "No met_em.d01.* files found in: ${WPS_RUN}"
  exit 1
fi

cd "${WRF_RUN}"

rm -f met_em.d01.*
for f in "${met_files[@]}"; do
  ln -snf "${f}" .
done

namelist_backup="${LOG_DIR}/namelist.input.before_run_real_$(date +%Y%m%d_%H%M%S)"
cp -p "${NAMELIST}" "${namelist_backup}"

set_namelist_value chem_in_opt 0
set_namelist_value have_bcs_chem .false.

echo "Stack size limit: $(ulimit -s)"
if ! ./real.exe > "${LOG_DIR}/real.log" 2>&1; then
  cp -p "${namelist_backup}" "${NAMELIST}"
  echo "real.exe failed. Restored namelist from:"
  echo "  ${namelist_backup}"
  echo "Log:"
  echo "  ${LOG_DIR}/real.log"
  exit 1
fi

echo "real.exe finished. Log:"
echo "  ${LOG_DIR}/real.log"

cp -p "${namelist_backup}" "${NAMELIST}"

if [[ "${APPLY_CAMS_ICBC}" == "1" && -f "${ICBC_SCRIPT}" && -d "${CAMS_ROOT}" ]]; then
  echo "Applying CAMS EAC4 chemical IC/BC after real.exe..."
  conda run --no-capture-output -n wrf-chem python "${ICBC_SCRIPT}" \
    > "${LOG_DIR}/apply_cams_eac4_icbc.log" 2>&1
  set_namelist_value chem_in_opt 1
  set_namelist_value have_bcs_chem .true.
  echo "CAMS EAC4 chemical IC/BC applied. Log:"
  echo "  ${LOG_DIR}/apply_cams_eac4_icbc.log"
else
  echo "Skipping CAMS EAC4 IC/BC apply. Set APPLY_CAMS_ICBC=1 and ensure data/script exist to enable."
fi
