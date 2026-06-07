#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
OUT_ROOT="${CASE_ROOT}/output"
OUT_DIR="${OUT_ROOT}/wrfout"

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
    rm -f "${path}"
    ln -snf "${dest}" "${path}"
    continue
  fi

  mv "${path}" "${dest}"
  ln -snf "${dest}" "${path}"
done

echo "Collected WRF outputs into ${OUT_DIR}"
