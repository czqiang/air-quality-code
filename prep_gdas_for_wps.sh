#!/usr/bin/env bash

set -euo pipefail
ulimit -s unlimited || true

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_ROOT="$(cd "${CASE_ROOT}/../.." && pwd)"
SRC_DIR="${GDAS_RAW_DIR:-${WRF_ROOT}/shared_data/meteorology/gdas_fnl0p25/20240206_20240213}"
OUT_DIR="${CASE_ROOT}/input_data/gdas_wps"
LOG_DIR="${CASE_ROOT}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

WGRIB2="${WGRIB2:-/home/xeon/soft/bin/wgrib2}"

if [[ ! -x "${WGRIB2}" ]]; then
  echo "Missing wgrib2 executable: ${WGRIB2}"
  exit 1
fi

LEFT_LON=112
RIGHT_LON=128
BOTTOM_LAT=24
TOP_LAT=38

# This list mirrors the fields that WPS/real.exe actually need for the current
# GFS/GDAS/FNL workflow. The goal is to avoid feeding very large global files
# with many extra records into ungrib, which is where we observed crashes.
FIELD_REGEX='(:(HGT|TMP|UGRD|VGRD|RH):(1000|975|950|925|900|850|800|750|700|650|600|550|500|450|400|350|300|250|200|150|100|70|50|40|30|20|15|10|7|5|3|2|1) mb:)|(:PRMSL:mean sea level:)|(:PRES:surface:)|(:HGT:surface:)|(:TMP:surface:)|(:WEASD:surface:)|(:SNOD:surface:)|(:LAND:surface:)|(:ICEC:surface:)|(:TMP:2 m above ground:)|(:RH:2 m above ground:)|(:UGRD:10 m above ground:)|(:VGRD:10 m above ground:)|(:TSOIL:(0-0.1|0.1-0.4|0.4-1|1-2) m below ground:)|(:SOILW:(0-0.1|0.1-0.4|0.4-1|1-2) m below ground:)|(:PRES:tropopause:)|(:HGT:tropopause:)|(:TMP:tropopause:)|(:UGRD:tropopause:)|(:VGRD:tropopause:)|(:PRES:max wind:)|(:HGT:max wind:)|(:TMP:max wind:)|(:UGRD:max wind:)|(:VGRD:max wind:))'

process_one() {
  local infile="$1"
  local base
  local outfile
  base="$(basename "${infile}")"
  outfile="${OUT_DIR}/${base}"

  if [[ -s "${outfile}" ]]; then
    echo "Skip existing ${outfile}"
    return 0
  fi

  echo "Processing ${base}"
  "${WGRIB2}" "${infile}" \
    -match "${FIELD_REGEX}" \
    -small_grib "${LEFT_LON}:${RIGHT_LON}" "${BOTTOM_LAT}:${TOP_LAT}" \
    "${outfile}.tmp" >> "${LOG_DIR}/prep_gdas_for_wps.log" 2>&1
  mv "${outfile}.tmp" "${outfile}"
}

shopt -s nullglob
files=( "${SRC_DIR}"/gdas1.fnl0p25.*.grib2 )
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No GDAS source files found in ${SRC_DIR}"
  exit 1
fi

for f in "${files[@]}"; do
  process_one "${f}"
done

echo "Prepared GDAS subset files are in ${OUT_DIR}"
