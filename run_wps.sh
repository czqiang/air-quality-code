#!/usr/bin/env bash

set -euo pipefail
ulimit -s unlimited || true

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_ROOT="$(cd "${CASE_ROOT}/../.." && pwd)"
WPS_RUN="${CASE_ROOT}/wps_run"
GDAS_DIR="${GDAS_RAW_DIR:-${WRF_ROOT}/shared_data/meteorology/gdas_fnl0p25/20240206_20240213}"
GDAS_WPS_DIR="${CASE_ROOT}/input_data/gdas_wps"
GEOG_DIR="${GEOG_DIR:-/home/xeon/wrf/shared_data/WPS_GEOG}"
LOG_DIR="${CASE_ROOT}/logs"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${GEOG_DIR}" ]]; then
  echo "Missing WPS geographic data directory: ${GEOG_DIR}"
  exit 1
fi

if [[ ! -d "${GEOG_DIR}/modis_landuse_20class_30s_with_lakes" ]]; then
  echo "WPS geographic data directory exists but does not look complete: ${GEOG_DIR}"
  exit 1
fi

if [[ ! -d "${GDAS_WPS_DIR}" ]] || [[ -z "$(find "${GDAS_WPS_DIR}" -maxdepth 1 -name '*.grib2' -print -quit 2>/dev/null)" ]]; then
  "${CASE_ROOT}/scripts/prep_gdas_for_wps.sh"
fi

shopt -s nullglob
gdas_files=( "${GDAS_WPS_DIR}"/*.grib2 )
shopt -u nullglob

if [[ ${#gdas_files[@]} -eq 0 ]]; then
  echo "No prepared GDAS subset GRIB2 files found in: ${GDAS_WPS_DIR}"
  exit 1
fi

cd "${WPS_RUN}"

rm -f GRIBFILE.* FILE:* geo_em.d0* met_em.d0*

./link_grib.csh "${gdas_files[@]}"
./geogrid.exe > "${LOG_DIR}/geogrid.log" 2>&1
./ungrib.exe  > "${LOG_DIR}/ungrib.log" 2>&1
./metgrid.exe > "${LOG_DIR}/metgrid.log" 2>&1

echo "WPS finished. Logs:"
echo "  ${LOG_DIR}/geogrid.log"
echo "  ${LOG_DIR}/ungrib.log"
echo "  ${LOG_DIR}/metgrid.log"
