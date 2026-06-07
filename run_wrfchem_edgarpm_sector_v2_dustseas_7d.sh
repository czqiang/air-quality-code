#!/usr/bin/env bash

set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP_DEFAULT="$(date +%Y%m%d_%H%M%S)_edgarpm_sector_v2_dustseas_7d"
export WRFCHEM_RUN_STAMP="${WRFCHEM_RUN_STAMP:-${STAMP_DEFAULT}}"

exec "${CASE_ROOT}/scripts/run_wrfchem_variant.sh" \
  --variant-name edgarpm_sector_v2_dustseas_7d \
  --run-hours 168 \
  --tasks "${WRF_NTASKS:-16}" \
  --dust-opt 13 \
  --seas-opt 2 \
  --dmsemis-opt 0 \
  "$@"
