#!/usr/bin/env bash

set -euo pipefail
ulimit -s unlimited || true

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
LOG_DIR="${CASE_ROOT}/logs"
MPI_LAUNCH="${MPI_LAUNCH:-mpirun}"
WRF_NTASKS="${WRF_NTASKS:-16}"
COLLECT_SCRIPT="${CASE_ROOT}/scripts/collect_wrf_outputs.sh"

mkdir -p "${LOG_DIR}"

cd "${WRF_RUN}"

rm -f rsl.out.* rsl.error.*

status=0
set +e
"${MPI_LAUNCH}" -n "${WRF_NTASKS}" ./wrf.exe > "${LOG_DIR}/wrf.log" 2>&1
status=$?
set -e

if [[ -x "${COLLECT_SCRIPT}" ]]; then
  "${COLLECT_SCRIPT}"
fi

if [[ "${status}" -eq 0 ]]; then
  echo "wrf.exe finished. Log:"
else
  echo "wrf.exe exited with status ${status}. Log:"
fi
echo "  ${LOG_DIR}/wrf.log"

exit "${status}"
