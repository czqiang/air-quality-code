#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_ROOT="$(cd "${CASE_ROOT}/../.." && pwd)"
OUT_DIR="${GDAS_RAW_DIR:-${WRF_ROOT}/shared_data/meteorology/gdas_fnl0p25/20240206_20240213}"
LOG_DIR="${CASE_ROOT}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

START_UTC="2024-02-06 00:00:00"
END_UTC="2024-02-13 00:00:00"

BASE_URL="https://osdf-director.osg-htc.org/ncar/gdex/d083003"
USER_AGENT="Mozilla/5.0"

download_one() {
  local ymdh="$1"
  local cycle="${ymdh:8:2}"
  local yyyy="${ymdh:0:4}"
  local yyyymm="${ymdh:0:6}"
  local outfile="${OUT_DIR}/gdas1.fnl0p25.${ymdh}.f00.grib2"
  local statefile="${outfile}.st"
  local url="${BASE_URL}/${yyyy}/${yyyymm}/gdas1.fnl0p25.${ymdh}.f00.grib2"

  case "${cycle}" in
    00|06|12|18) ;;
    *)
      echo "Refusing invalid GDAS cycle ${ymdh}" >&2
      return 1
      ;;
  esac

  if [[ -e "${statefile}" ]]; then
    echo "Found incomplete-state marker for ${outfile}; removing stale partial download"
    rm -f "${outfile}" "${statefile}"
  fi

  if [[ -s "${outfile}" ]]; then
    echo "Skip existing ${outfile}"
    return 0
  fi

  echo "Downloading ${outfile}"
  axel -a -n 24 -U "${USER_AGENT}" -o "${outfile}" "${url}"
}

start_epoch="$(date -u -d "${START_UTC} UTC" +%s)"
end_epoch="$(date -u -d "${END_UTC} UTC" +%s)"
step_seconds=21600

current_epoch="${start_epoch}"
while (( current_epoch <= end_epoch )); do
  ymdh="$(date -u -d "@${current_epoch}" +%Y%m%d%H)"
  download_one "${ymdh}" | tee -a "${LOG_DIR}/download_gdas_fnl0p25_gdex.log"
  current_epoch=$(( current_epoch + step_seconds ))
done

echo "GDEX GDAS/FNL 0.25 download loop finished."
echo "Files are in ${OUT_DIR}"
