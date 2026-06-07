#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_RUN="${CASE_ROOT}/wrf_run"
WPS_RUN="${CASE_ROOT}/wps_run"
NAMELIST="${WRF_RUN}/namelist.input"
STRICT=1

usage() {
  cat <<EOF
Usage: $0 [--strict|--warn-only]

Checks whether the current case is ready for a WRF-Chem run.

--strict     Return non-zero if critical inputs are missing. Default.
--warn-only  Print errors as warnings and return zero.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
      ;;
    --warn-only)
      STRICT=0
      shift
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

errors=0
warnings=0

note() { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; warnings=$((warnings + 1)); }
err() { echo "[ERR ] $*"; errors=$((errors + 1)); }

namelist_value() {
  local key="$1"
  local value
  value="$(
    awk -v key="$(printf '%s' "$key" | tr '[:upper:]' '[:lower:]')" '
      {
        line=$0
        sub(/!.*/, "", line)
        low=tolower(line)
        pat="^[[:space:]]*" key "[[:space:]]*="
        if (low ~ pat) {
          split(line, a, "=")
          split(a[2], b, ",")
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", b[1])
          print b[1]
          exit
        }
      }
    ' "${NAMELIST}" 2>/dev/null || true
  )"
  printf '%s' "${value}"
}

as_int() {
  local raw="${1:-}"
  raw="${raw%%.*}"
  raw="${raw// /}"
  if [[ -z "${raw}" ]]; then
    printf '0'
  else
    printf '%s' "${raw}"
  fi
}

echo "WRF-Chem preflight for ${CASE_ROOT}"

[[ -x "${WRF_RUN}/wrf.exe" ]] && note "wrf.exe exists" || err "Missing executable: ${WRF_RUN}/wrf.exe"
[[ -x "${WRF_RUN}/real.exe" ]] && note "real.exe exists" || err "Missing executable: ${WRF_RUN}/real.exe"
[[ -f "${NAMELIST}" ]] && note "namelist.input exists" || err "Missing namelist: ${NAMELIST}"

if [[ -f "${NAMELIST}" ]]; then
  chem_opt="$(as_int "$(namelist_value chem_opt)")"
  emiss_inpt_opt="$(as_int "$(namelist_value emiss_inpt_opt)")"
  io_style_emissions="$(as_int "$(namelist_value io_style_emissions)")"
  chem_in_opt="$(as_int "$(namelist_value chem_in_opt)")"
  have_bcs_chem="$(namelist_value have_bcs_chem | tr '[:upper:]' '[:lower:]')"
  run_days="$(as_int "$(namelist_value run_days)")"
  run_hours="$(as_int "$(namelist_value run_hours)")"
  total_hours=$((run_days * 24 + run_hours))

  if [[ "${chem_opt}" -gt 0 ]]; then
    note "chem_opt=${chem_opt}"
  else
    err "chem_opt is ${chem_opt}; WRF-Chem is not enabled in namelist.input"
  fi

  if [[ "${total_hours}" -le 12 ]]; then
    note "run_days=${run_days}, run_hours=${run_hours}; short smoke-test length"
  else
    note "run_days=${run_days}, run_hours=${run_hours}; production-length run"
  fi

  if [[ "${chem_in_opt}" -eq 1 ]]; then
    note "chem_in_opt=1; wrfinput chemistry will be used"
  else
    warn "chem_in_opt=${chem_in_opt}; WRF-Chem may overwrite wrfinput chemistry with default profiles"
  fi

  if [[ "${have_bcs_chem}" == ".true." || "${have_bcs_chem}" == "true" ]]; then
    note "have_bcs_chem=.true.; chemical lateral boundaries are enabled"
  else
    warn "have_bcs_chem=${have_bcs_chem:-unset}; chemical lateral boundaries are not enabled"
  fi

  if [[ "${emiss_inpt_opt}" -eq 0 || "${io_style_emissions}" -eq 0 ]]; then
    warn "Anthropogenic emissions are disabled (emiss_inpt_opt=${emiss_inpt_opt}, io_style_emissions=${io_style_emissions}); this is technical smoke-test mode only"
  else
    shopt -s nullglob
    wrfchemi_files=( "${WRF_RUN}"/wrfchemi* )
    shopt -u nullglob
    if [[ ${#wrfchemi_files[@]} -gt 0 ]]; then
      note "Found ${#wrfchemi_files[@]} wrfchemi* emission file(s)"
    else
      err "emiss_inpt_opt=${emiss_inpt_opt} requires wrfchemi* files, but none were found in ${WRF_RUN}"
    fi
  fi
fi

shopt -s nullglob
met_files=( "${WPS_RUN}"/met_em.d01.* )
shopt -u nullglob
if [[ ${#met_files[@]} -gt 0 ]]; then
  note "Found ${#met_files[@]} met_em.d01.* files"
else
  err "No met_em.d01.* files found in ${WPS_RUN}"
fi

if [[ -f "${WRF_RUN}/wrfinput_d01" ]]; then
  header_tmp="$(mktemp)"
  ncdump -h "${WRF_RUN}/wrfinput_d01" > "${header_tmp}"
  if rg -q '^[[:space:]]*float[[:space:]]+(o3|co|no|no2|so2|so4aj|so4ai|PM2_5_DRY|pm25)' "${header_tmp}" ; then
    note "wrfinput_d01 appears to contain chemistry variables"
  else
    err "wrfinput_d01 does not appear to contain chemistry variables; rerun real.exe after installing a chemistry namelist"
  fi
  rm -f "${header_tmp}"
else
  err "Missing ${WRF_RUN}/wrfinput_d01"
fi

[[ -f "${WRF_RUN}/wrfbdy_d01" ]] && note "wrfbdy_d01 exists" || err "Missing ${WRF_RUN}/wrfbdy_d01"

required_tables=(
  GENPARM.TBL
  LANDUSE.TBL
  VEGPARM.TBL
  SOILPARM.TBL
  RRTMG_LW_DATA
  RRTMG_SW_DATA
  aerosol.formatted
  aerosol_lat.formatted
  aerosol_lon.formatted
  aerosol_plev.formatted
  ozone.formatted
  ozone_lat.formatted
  ozone_plev.formatted
)

for table in "${required_tables[@]}"; do
  if [[ -e "${WRF_RUN}/${table}" ]]; then
    note "Runtime table exists: ${table}"
  else
    err "Missing runtime table/link: ${WRF_RUN}/${table}"
  fi
done

echo "Preflight summary: ${errors} error(s), ${warnings} warning(s)"

if [[ "${errors}" -gt 0 && "${STRICT}" -eq 1 ]]; then
  exit 1
fi

exit 0
