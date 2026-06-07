#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ablation_common.sh"

DEFAULT_VARIANTS=(LI)
VARIANTS=("${@:-${DEFAULT_VARIANTS[@]}}")
for VARIANT in "${VARIANTS[@]}"; do
    case $VARIANT in
        LI) export ABLATION_OUTPUT_DIR="./output/ablation/V4_linear_interp" ;;
        *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
    esac

    run_all_datasets \
        --ablation linear_interp \
        --des "ablation_linear_interp"
done
