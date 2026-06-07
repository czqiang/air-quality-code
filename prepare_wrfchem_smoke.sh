#!/usr/bin/env bash

set -euo pipefail
ulimit -s unlimited || true

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
LOG_DIR="${CASE_ROOT}/logs"
OUT_DIR="${CASE_ROOT}/output/wrfchem"
ARCHIVE_DIR="${CASE_ROOT}/output/archive"
TEMPLATE="${CASE_ROOT}/templates/namelist.input.wrfchem_smoke"
RUN_REAL=0

usage() {
  cat <<EOF
Usage: $0 [--run-real] [--template PATH]

Prepare the case for a short WRF-Chem smoke test.

--run-real       After installing the chemistry namelist, rerun real.exe so
                 wrfinput_d01/wrfbdy_d01 are regenerated with chemistry fields.
--template PATH  Use a different namelist template.

This script does not run wrf.exe.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-real)
      RUN_REAL=1
      shift
      ;;
    --template)
      TEMPLATE="$2"
      shift 2
      ;;
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
done

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Template not found: ${TEMPLATE}"
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
setup_archive="${ARCHIVE_DIR}/wrfchem_setup_${stamp}"
mkdir -p "${setup_archive}" "${LOG_DIR}" "${ARCHIVE_DIR}"

if [[ -f "${WRF_RUN}/namelist.input" ]]; then
  cp -p "${WRF_RUN}/namelist.input" "${setup_archive}/namelist.input.before_wrfchem"
fi
cp -p "${TEMPLATE}" "${setup_archive}/namelist.input.template"
cp -p "${TEMPLATE}" "${WRF_RUN}/namelist.input"

if [[ -d "${OUT_DIR}" ]] && find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  mv "${OUT_DIR}" "${setup_archive}/previous_wrfchem_output"
fi
mkdir -p "${OUT_DIR}"

find "${WRF_RUN}" -maxdepth 1 \( -type l -o -type f \) \
  \( -name 'wrfout_d*' -o -name 'wrfrst_d*' -o -name 'auxhist*_d*' -o -name 'rsl.out.*' -o -name 'rsl.error.*' -o -name 'namelist.output' \) \
  -delete

if [[ "${RUN_REAL}" -eq 1 ]]; then
  inputs_archive="${setup_archive}/met_only_real_inputs"
  mkdir -p "${inputs_archive}"
  for input in wrfinput_d01 wrfbdy_d01; do
    if [[ -f "${WRF_RUN}/${input}" ]]; then
      mv "${WRF_RUN}/${input}" "${inputs_archive}/${input}"
    fi
  done
  bash "${CASE_ROOT}/scripts/run_real.sh"
else
  echo "Installed chemistry namelist but did not run real.exe."
  echo "Before running WRF-Chem, regenerate wrfinput/wrfbdy with:"
  echo "  bash ${CASE_ROOT}/scripts/prepare_wrfchem_smoke.sh --run-real"
fi

bash "${CASE_ROOT}/scripts/check_wrfchem_inputs.sh" --warn-only

echo "WRF-Chem smoke-test setup archive:"
echo "  ${setup_archive}"
