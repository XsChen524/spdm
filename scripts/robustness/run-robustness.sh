#!/bin/bash

# =============================================================================
# MODE Robustness Testing Runner
# =============================================================================
# Runs robustness experiments with Gaussian noise at multiple levels.
#
# Usage:
#   bash scripts/robustness/run-robustness.sh [options] [dataset]
#
# Options:
#   --para N    Max concurrent experiments per GPU (default: 1)
#   --gpu ID    CUDA device(s), comma-separated (default: 0)
#
# Examples:
#   bash scripts/robustness/run-robustness.sh                       # ETTm2, sequential
#   bash scripts/robustness/run-robustness.sh --para 4              # 4 pred_lens in parallel
#   bash scripts/robustness/run-robustness.sh --para 2 ETTm2        # 2 concurrent
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Change to the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Set robustness output directory
export ROBUSTNESS_OUTPUT_DIR="./output/robustness"

# Defaults
PARA=1
GPU_IDS=("0")
DATASETS=("ETTm2")
PRED_LENS=("96" "192" "336" "720")
NOISE_LEVELS=(0 0.1 0.2 0.3 0.5)

# ---------------------------------------------------------------------------
# Parse options
# ---------------------------------------------------------------------------
dataset_arg="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --para)
            PARA="$2"
            shift 2
            ;;
        --gpu)
            IFS=',' read -ra GPU_IDS <<< "$2"
            shift 2
            ;;
        -h|--help|help)
            echo "Usage: $0 [--para N] [--gpu ID[,ID...]] [dataset]"
            echo ""
            echo "Options:"
            echo "  --para N     Max concurrent experiments (default: 1)"
            echo "  --gpu IDs    CUDA devices, comma-separated (default: 0)"
            echo "  dataset      Dataset to test: ${DATASETS[*]}, or 'all' (default: all)"
            echo ""
            echo "Examples:"
            echo "  $0                          # Sequential"
            echo "  $0 --para 4                 # 4 pred_lens in parallel"
            echo "  $0 --para 2 --gpu 0,1 ETTm2 # 2 per GPU across GPU 0,1"
            exit 0
            ;;
        *)
            dataset_arg="$1"
            shift
            ;;
    esac
done

echo -e "${BLUE}ManiMamba Robustness Testing Runner${NC}"
echo "Repository root: $REPO_ROOT"
echo "Output directory: $ROBUSTNESS_OUTPUT_DIR"
echo "Parallel slots:   ${PARA} per GPU"
echo "GPUs:             ${GPU_IDS[*]}"
echo ""

# ---------------------------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------------------------
validate_dataset() {
    local dataset=$1
    for d in "${DATASETS[@]}"; do
        if [[ "$d" == "$dataset" ]]; then return 0; fi
    done
    echo -e "${RED}Error: Unknown dataset: $dataset${NC}"
    echo "Available datasets: ${DATASETS[*]}"
    exit 1
}

# Determine datasets to process
datasets_to_run=()
if [[ "$dataset_arg" == "all" ]]; then
    datasets_to_run=("${DATASETS[@]}")
else
    validate_dataset "$dataset_arg"
    datasets_to_run=("$dataset_arg")
fi

echo -e "${YELLOW}Datasets:    ${datasets_to_run[*]}${NC}"
echo -e "${YELLOW}Pred lens:   ${PRED_LENS[*]}${NC}"
echo -e "${YELLOW}Noise levels:${NOISE_LEVELS[*]}${NC}"
echo ""

# ---------------------------------------------------------------------------
# Build job queue: each entry is "dataset pred_len noise_level"
# ---------------------------------------------------------------------------
JOBS=()
for dataset in "${datasets_to_run[@]}"; do
    for nl in "${NOISE_LEVELS[@]}"; do
        for pl in "${PRED_LENS[@]}"; do
            JOBS+=("$dataset $pl $nl")
        done
    done
done

total_jobs=${#JOBS[@]}
echo -e "${CYAN}Total experiments: ${total_jobs}${NC}"
echo ""

# ---------------------------------------------------------------------------
# Sliding-window dispatcher
# ---------------------------------------------------------------------------
NUM_GPUS=${#GPU_IDS[@]}
active_pids=()
active_gpus=()

_reap() {
    local new_pids=()
    local new_gpus=()
    for i in "${!active_pids[@]}"; do
        if kill -0 "${active_pids[$i]}" 2>/dev/null; then
            new_pids+=("${active_pids[$i]}")
            new_gpus+=("${active_gpus[$i]}")
        fi
    done
    active_pids=("${new_pids[@]}")
    active_gpus=("${new_gpus[@]}")
}

_gpu_active_count() {
    local gpu=$1
    local count=0
    for g in "${active_gpus[@]}"; do
        if [[ "$g" == "$gpu" ]]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

_wait_for_slot() {
    while true; do
        _reap
        for gpu in "${GPU_IDS[@]}"; do
            local cnt
            cnt=$(_gpu_active_count "$gpu")
            if [[ $cnt -lt $PARA ]]; then
                echo "$gpu"
                return
            fi
        done
        sleep 2
    done
}

completed=0
failed=0

for i in "${!JOBS[@]}"; do
    read -r dataset pl nl <<< "${JOBS[$i]}"
    job_num=$((i + 1))

    gpu=$(_wait_for_slot)

    echo -e "${BLUE}[$job_num/$total_jobs] ${dataset} pl=${pl} nl=${nl} → GPU ${gpu}${NC}"

    log_dir="${REPO_ROOT}/log/robustness"
    mkdir -p "$log_dir"
    log_file="${log_dir}/${dataset}_pl${pl}_nl${nl}.log"

    (
        CUDA_VISIBLE_DEVICES="$gpu" \
        noise_level="$nl" \
        bash "./scripts/robustness/${dataset}.sh" "$pl" \
            > "$log_file" 2>&1
    ) &

    active_pids+=($!)
    active_gpus+=("$gpu")
done

# Wait for all remaining jobs
echo -e "${YELLOW}Waiting for ${#active_pids[@]} remaining experiments...${NC}"

for pid in "${active_pids[@]}"; do
    if wait "$pid"; then
        completed=$((completed + 1))
    else
        failed=$((failed + 1))
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "\n${BLUE}=================================================================${NC}"
echo -e "${BLUE}Robustness Testing Summary${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "Total:     ${total_jobs}"
echo -e "Completed: ${GREEN}${completed}${NC}"
echo -e "Failed:    ${RED}${failed}${NC}"
echo -e "Logs:      ${REPO_ROOT}/log/robustness/"

unset noise_level

if [[ $failed -eq 0 ]]; then
    echo -e "\n${GREEN}All tests completed successfully! ✓${NC}"
    exit 0
else
    echo -e "\n${YELLOW}${failed} experiment(s) failed${NC}"
    exit 1
fi
