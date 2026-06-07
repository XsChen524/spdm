#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ablation_common.sh"

DEFAULT_VARIANTS=(WD)
VARIANTS=("${@:-${DEFAULT_VARIANTS[@]}}")
for VARIANT in "${VARIANTS[@]}"; do
    case $VARIANT in
        WD) export ABLATION_OUTPUT_DIR="./output/ablation/V4_w_dt" ;;
        *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
    esac

    run_all_datasets \
        --ablation w_dt \
        --des "ablation_w_dt"
done
