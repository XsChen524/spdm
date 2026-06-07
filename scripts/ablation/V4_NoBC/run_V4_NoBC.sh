#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ablation_common.sh"

DEFAULT_VARIANTS=(NB)
VARIANTS=("${@:-${DEFAULT_VARIANTS[@]}}")
for VARIANT in "${VARIANTS[@]}"; do
    case $VARIANT in
        NB) export ABLATION_OUTPUT_DIR="./output/ablation/V4_no_bc" ;;
        *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
    esac

    run_all_datasets \
        --ablation no_bc \
        --des "ablation_no_bc"
done
