#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
LOG_DIR="${CASE_ROOT}/logs"
OUT_ROOT="${CASE_ROOT}/output"
OUT_DIR="${OUT_ROOT}/wrfout"
ARCHIVE_ROOT="${OUT_ROOT}/archive"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${ARCHIVE_ROOT}/partial_run_${STAMP}"

mkdir -p "${ARCHIVE_DIR}" "${OUT_ROOT}"

if [[ -d "${OUT_DIR}" ]] && [[ -n "$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  mv "${OUT_DIR}" "${ARCHIVE_DIR}/wrfout"
fi

if [[ -d "${LOG_DIR}" ]] && [[ -n "$(find "${LOG_DIR}" -maxdepth 1 -type f -print -quit)" ]]; then
  mkdir -p "${ARCHIVE_DIR}/logs"
  find "${LOG_DIR}" -maxdepth 1 -type f -exec mv {} "${ARCHIVE_DIR}/logs/" \;
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

rm -f "${WRF_RUN}"/wrfout_d*
rm -f "${WRF_RUN}"/wrfrst_d*
rm -f "${WRF_RUN}"/auxhist*_d*
rm -f "${WRF_RUN}"/rsl.out.* "${WRF_RUN}"/rsl.error.*
rm -f "${WRF_RUN}"/wrfinput_d01 "${WRF_RUN}"/wrfbdy_d01
rm -f "${WRF_RUN}"/namelist.output

echo "Archived previous WRF outputs/logs to ${ARCHIVE_DIR}"
echo "Cleaned WRF run products from ${WRF_RUN}"
