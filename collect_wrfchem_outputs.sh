#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
OUT_ROOT="${CASE_ROOT}/output"
RUN_STAMP="${WRFCHEM_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_ROOT}/wrfchem/${RUN_STAMP}"
LATEST_LINK="${OUT_ROOT}/wrfchem_latest"

mkdir -p "${OUT_DIR}"

shopt -s nullglob
files=(
  "${WRF_RUN}"/wrfout_d*
  "${WRF_RUN}"/wrfrst_d*
  "${WRF_RUN}"/auxhist*_d*
)
shopt -u nullglob

for path in "${files[@]}"; do
  [[ -e "${path}" ]] || continue

  base="$(basename "${path}")"
  dest="${OUT_DIR}/${base}"

  if [[ -L "${path}" ]]; then
    continue
  fi

  if [[ -e "${dest}" ]]; then
    echo "Refusing to overwrite existing output: ${dest}" >&2
    exit 1
  fi

  mv "${path}" "${dest}"
  ln -snf "${dest}" "${path}"
done

ln -snf "${OUT_DIR}" "${LATEST_LINK}"
echo "Collected WRF-Chem outputs into ${OUT_DIR}"
echo "Latest WRF-Chem output link: ${LATEST_LINK}"
