#!/usr/bin/env bash

set -euo pipefail
ulimit -s unlimited || true

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
LOG_DIR="${CASE_ROOT}/logs"
MPI_LAUNCH="${MPI_LAUNCH:-mpirun}"
WRF_NTASKS="${WRF_NTASKS:-16}"
RUN_STAMP="${WRFCHEM_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export WRFCHEM_RUN_STAMP="${RUN_STAMP}"
LOG_FILE="${LOG_DIR}/wrfchem_${RUN_STAMP}.log"
COLLECT_SCRIPT="${CASE_ROOT}/scripts/collect_wrfchem_outputs.sh"
CHECK_SCRIPT="${CASE_ROOT}/scripts/check_wrfchem_inputs.sh"

usage() {
  cat <<EOF
Usage: $0

Run the prepared WRF-Chem case.

Environment overrides:
  MPI_LAUNCH   MPI launcher command. Default: mpirun
  WRF_NTASKS   Number of MPI ranks. Default: 16

This script sets "ulimit -s unlimited" before launching WRF-Chem.

Before using this script after rerunning real.exe, make sure CAMS EAC4 IC/BC
has been applied. The case run_real.sh does that automatically by default.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
fi

mkdir -p "${LOG_DIR}"
ln -snf "$(basename "${LOG_FILE}")" "${LOG_DIR}/wrfchem_latest.log"

bash "${CHECK_SCRIPT}" --strict

python3 - "${WRF_NTASKS}" "${WRF_RUN}/namelist.input" <<'PY'
from pathlib import Path
import math
import re
import sys

ntasks = int(sys.argv[1])
namelist = Path(sys.argv[2]).read_text(encoding="utf-8")

def get_int(key):
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9]+)", namelist, re.M)
    if not match:
        raise SystemExit(f"Could not find {key} in namelist.input")
    return int(match.group(1))

e_we = get_int("e_we")
e_sn = get_int("e_sn")
min_patch = 10

valid = []
for nx in range(1, ntasks + 1):
    if ntasks % nx:
        continue
    ny = ntasks // nx
    if e_we // nx >= min_patch and e_sn // ny >= min_patch:
        valid.append((nx, ny))

if not valid:
    possible = []
    for n in range(1, ntasks + 1):
        ok = False
        for nx in range(1, n + 1):
            if n % nx:
                continue
            ny = n // nx
            if e_we // nx >= min_patch and e_sn // ny >= min_patch:
                ok = True
                break
        if ok:
            possible.append(n)
    suggested = possible[-1] if possible else 1
    raise SystemExit(
        f"WRF_NTASKS={ntasks} is too high for e_we={e_we}, e_sn={e_sn}. "
        f"Use WRF_NTASKS<={suggested}; suggested WRF_NTASKS={suggested}."
    )
PY

shopt -s nullglob
existing_outputs=(
  "${WRF_RUN}"/wrfout_d*
  "${WRF_RUN}"/wrfrst_d*
  "${WRF_RUN}"/auxhist*_d*
)
shopt -u nullglob

for path in "${existing_outputs[@]}"; do
  if [[ -L "${path}" ]]; then
    rm -f "${path}"
  else
    echo "Existing non-symlink WRF output found in wrf_run: ${path}" >&2
    echo "Archive or remove stale outputs before launching a new run." >&2
    exit 1
  fi
done

cd "${WRF_RUN}"

rm -f rsl.out.* rsl.error.*

echo "Stack size limit: $(ulimit -s)"

status=0
set +e
"${MPI_LAUNCH}" -n "${WRF_NTASKS}" ./wrf.exe > "${LOG_FILE}" 2>&1
status=$?
set -e

if [[ -x "${COLLECT_SCRIPT}" ]]; then
  "${COLLECT_SCRIPT}"
fi

if [[ "${status}" -eq 0 ]]; then
  echo "wrf.exe WRF-Chem run finished. Log:"
else
  echo "wrf.exe WRF-Chem run exited with status ${status}. Log:"
fi
ln -snf "$(basename "${LOG_FILE}")" "${LOG_DIR}/wrfchem_latest.log"
echo "  ${LOG_FILE}"
echo "Latest log link:"
echo "  ${LOG_DIR}/wrfchem_latest.log"

exit "${status}"
