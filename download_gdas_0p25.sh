#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_ROOT="$(cd "${CASE_ROOT}/../.." && pwd)"
OUT_DIR="${GDAS_RAW_DIR:-${WRF_ROOT}/shared_data/meteorology/gdas_fnl0p25/20240206_20240213}"
LOG_DIR="${CASE_ROOT}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LEFT_LON=115
RIGHT_LON=125
TOP_LAT=35
BOTTOM_LAT=27

START_UTC="2024-02-06 00:00:00"
END_UTC="2024-02-13 00:00:00"

URL_BASE="https://nomads.ncep.noaa.gov/cgi-bin/filter_gdas_0p25.pl"
USER_AGENT="Mozilla/5.0"

# Conservative choice: request all variables and all levels for a small
# subregion. This is larger than a hand-tuned minimal list, but is much less
# fragile for getting WPS/real.exe running on a first pass.
QUERY_FIELDS=(
  "all_var=on"
  "all_lev=on"
  "subregion="
  "leftlon=${LEFT_LON}"
  "rightlon=${RIGHT_LON}"
  "toplat=${TOP_LAT}"
  "bottomlat=${BOTTOM_LAT}"
)

download_one() {
  local ymd="$1"
  local cyc="$2"
  local outfile="${OUT_DIR}/gdas_${ymd}_${cyc}.grib2"
  local file="gdas.t${cyc}z.pgrb2.0p25.f000"
  local dir="/gdas.${ymd}/${cyc}/atmos"
  local url="${URL_BASE}?file=${file}&$(IFS='&'; echo "${QUERY_FIELDS[*]}")&dir=$(python3 - <<PY
import urllib.parse
print(urllib.parse.quote('${dir}', safe=''))
PY
)"

  if [[ -s "${outfile}" ]]; then
    echo "Skip existing ${outfile}"
    return 0
  fi

  echo "Downloading ${outfile}"
  curl -fL --retry 8 --retry-delay 20 --connect-timeout 30 --max-time 0 \
       -A "${USER_AGENT}" \
       -o "${outfile}.part" \
       "${url}"
  mv "${outfile}.part" "${outfile}"
}

current="${START_UTC}"
while [[ "${current}" < "${END_UTC}" || "${current}" == "${END_UTC}" ]]; do
  ymd="$(date -u -d "${current}" +%Y%m%d)"
  cyc="$(date -u -d "${current}" +%H)"
  download_one "${ymd}" "${cyc}" | tee -a "${LOG_DIR}/download_gdas_0p25.log"
  sleep 10
  current="$(date -u -d "${current} + 6 hours" '+%Y-%m-%d %H:%M:%S')"
done

echo "GDAS download loop finished."
echo "Files are in ${OUT_DIR}"
