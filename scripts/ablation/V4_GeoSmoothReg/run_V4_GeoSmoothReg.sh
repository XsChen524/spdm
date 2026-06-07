#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ablation_common.sh"

DEFAULT_VARIANTS=(GR)
VARIANTS=("${@:-${DEFAULT_VARIANTS[@]}}")
for VARIANT in "${VARIANTS[@]}"; do
    case $VARIANT in
        GR) export ABLATION_OUTPUT_DIR="./output/ablation/V4_geo_smooth_reg" ;;
        *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
    esac

    run_all_datasets \
        --ablation geo_smooth_reg \
        --des "ablation_geo_smooth_reg"
done
