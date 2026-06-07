#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ablation_common.sh"

DEFAULT_VARIANTS=(TA)
VARIANTS=("${@:-${DEFAULT_VARIANTS[@]}}")
for VARIANT in "${VARIANTS[@]}"; do
    case $VARIANT in
        TA) export ABLATION_OUTPUT_DIR="./output/ablation/V4_tanh_alpha" ;;
        *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
    esac

    run_all_datasets \
        --ablation tanh_alpha \
        --des "ablation_tanh_alpha"
done
