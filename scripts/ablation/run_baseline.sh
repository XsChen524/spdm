#!/bin/bash
set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_common.sh"

DEFAULT_VARIANTS=(V3)
VARIANTS=("${@:-${DEFAULT_VARIANTS[@]}}")

for VARIANT in "${VARIANTS[@]}"; do
    case $VARIANT in
        V3)
            export ABLATION_OUTPUT_DIR="./output/ablation/02_baseline_v3"
            run_all_datasets --des "ablation_V3"
            ;;
        *)
            echo "Unknown baseline variant: $VARIANT" >&2
            exit 1
            ;;
    esac
done
