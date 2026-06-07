#!/bin/bash

# =============================================================================
# ManiMamba Increasing Lookback Experiments Runner
# =============================================================================
# This script runs lookback experiments for ManiMamba across datasets.
# Tests model performance with varying input sequence lengths.
#
# Primary Model: ManiMamba (geometry-aware manifold Mamba)
#
# Scheduling granularity: individual (dataset, seq_len) experiments.
# Uses FIFO semaphores + flock atomic round-robin (same mechanism as
# scripts/ablation).  Each GPU holds --para tokens; experiments acquire
# a token before starting and release it on completion.
#
# Usage:
#   bash run-lookback.sh [OPTIONS] [dataset] [model]
#
# Options:
#   --gpu IDS  Comma-separated GPU IDs (default: 0)
#   --para N   Max concurrent experiments per GPU (default: 1)
#   -h, --help Show this help message
#
# Arguments:
#   dataset    Dataset name (ETTm1, PEMS08, Weather, ECL) or 'all' (default: all)
#   model      Optional model name. If not specified, runs all models
#
# Examples:
#   scripts/increasing_lookback/run-lookback.sh                          # All datasets (GPU 0)
#   scripts/increasing_lookback/run-lookback.sh --para 2                 # 2 experiments in parallel on GPU 0
#   scripts/increasing_lookback/run-lookback.sh --gpu 0,1,2,3 --para 2   # 2 parallel per GPU across 4 GPUs
#   scripts/increasing_lookback/run-lookback.sh --gpu 0,1,2,3 ETTm1      # ETTm1 seq_lens round-robin across 4 GPUs
#   scripts/increasing_lookback/run-lookback.sh ETTm1 ManiMamba          # Run only ManiMamba on ETTm1
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# =============================================================================
# Configuration
# =============================================================================

DATASETS=(
    # "ETTm1"
    "PEMS08"
    # "Weather"
    # "ECL"
)

MODELS=(
    "ManiMamba"
)

# =============================================================================
# Parse arguments
# =============================================================================

PARA=1
GPU_STR="0"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_STR="$2"
            shift 2
            ;;
        --para)
            PARA="$2"
            shift 2
            ;;
        -h|--help|help)
            echo "Usage: $0 [--gpu IDS] [--para N] [dataset] [model]"
            echo ""
            echo "Options:"
            echo "  --gpu IDS  Comma-separated GPU IDs (default: 0)"
            echo "  --para N   Max concurrent experiments per GPU (default: 1)"
            echo ""
            echo "Arguments:"
            echo "  dataset   Dataset: ${DATASETS[*]}, or 'all' (default: all)"
            echo "  model     Model name. If not specified, runs all models"
            echo ""
            echo "Examples:"
            echo "  $0                                       # All datasets x all models"
            echo "  $0 --para 2                              # 2 experiments in parallel"
            echo "  $0 --gpu 0,1 --para 2                    # 2 parallel per GPU on GPU 0,1"
            echo "  $0 ETTm1 ManiMamba                       # ManiMamba on ETTm1"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

dataset_arg="${POSITIONAL_ARGS[0]:-all}"
model_arg="${POSITIONAL_ARGS[1]:-}"

IFS=',' read -ra GPU_IDS <<< "$GPU_STR"
NUM_GPUS=${#GPU_IDS[@]}

# =============================================================================
# Helper functions
# =============================================================================

validate_dataset() {
    local dataset=$1
    for d in "${DATASETS[@]}"; do
        if [[ "$d" == "$dataset" ]]; then
            return 0
        fi
    done
    echo "Error: Unknown dataset: $dataset"
    echo "Available datasets: ${DATASETS[*]}"
    exit 1
}

validate_model() {
    local model=$1
    for m in "${MODELS[@]}"; do
        if [[ "$m" == "$model" ]]; then
            return 0
        fi
    done
    echo "Error: Unknown model: $model"
    echo "Available models: ${MODELS[*]}"
    exit 1
}

get_model_script() {
    local dataset=$1
    local model=$2
    echo "./scripts/increasing_lookback/${dataset}/${model}.sh"
}

# =============================================================================
# Resolve datasets / models
# =============================================================================

datasets_to_run=()
if [[ "$dataset_arg" == "all" ]]; then
    datasets_to_run=("${DATASETS[@]}")
else
    validate_dataset "$dataset_arg"
    datasets_to_run=("$dataset_arg")
fi

models_to_run=()
if [[ -z "$model_arg" ]]; then
    models_to_run=("${MODELS[@]}")
else
    validate_model "$model_arg"
    models_to_run=("$model_arg")
fi

source "${SCRIPT_DIR}/lookback_common.sh"

declare -a TASK_LIST=()
skipped_experiments=0

for dataset in "${datasets_to_run[@]}"; do
    for model in "${models_to_run[@]}"; do
        script=$(get_model_script "$dataset" "$model")
        if [[ ! -f "$script" ]]; then
            echo "SKIP: script not found: $script"
            skipped_experiments=$((skipped_experiments + 1))
            continue
        fi
        read -ra sls <<< "$(get_seq_lens "$dataset")"
        for sl in "${sls[@]}"; do
            TASK_LIST+=("${dataset}|${sl}")
        done
    done
done

total_experiments=${#TASK_LIST[@]}

LOG_DIR="$REPO_ROOT/log/lookback"
TEMP_DIR="$REPO_ROOT/temp/lookback"
mkdir -p "$LOG_DIR" "$TEMP_DIR"

SCHEDULER_LOG="$LOG_DIR/scheduler.log"
TASK_FILE="$TEMP_DIR/tasks.txt"

LOOKBACK_OUTPUT_DIR="./output/lookback"

printf "%s\n" "${TASK_LIST[@]}" > "$TASK_FILE"

echo "Running ManiMamba lookback experiments (per-experiment dispatch)"
echo "Datasets: ${datasets_to_run[*]}"
echo "Models:   ${models_to_run[*]}"
echo "GPUs:     ${GPU_IDS[*]} ($NUM_GPUS)"
echo "Parallel: $PARA per GPU ($((NUM_GPUS * PARA)) max concurrent)"
echo "Total experiments: $total_experiments (skipped: $skipped_experiments)"
echo "Logs:    $LOG_DIR/"
echo ""

# =============================================================================
# Background scheduler — FIFO semaphore dispatch (same pattern as ablation)
# =============================================================================

export GPU_LIST="${GPU_IDS[*]}"
export PARA_PER_GPU="${PARA}"

nohup bash -c '
set -u
cd "'"$REPO_ROOT"'" || exit 1

SCRIPT_DIR="'"${SCRIPT_DIR}"'"
source "${SCRIPT_DIR}/lookback_common.sh"

LOG_DIR="'"${LOG_DIR}"'"
LOOKBACK_OUTPUT_DIR="'"${LOOKBACK_OUTPUT_DIR}"'"
total='"${total_experiments}"'
cur=0

run_task() {
    local dataset=$1 seq_len=$2 task_num=$3
    local gpu="${_CURRENT_GPU}"
    local fd="${_GPU_FD[$gpu]}"
    (
        eval "read -r _ <&${fd}"
        echo "[$(date "+%H:%M:%S")] ${task_num}/${total} START ${dataset} seq=${seq_len} → GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="$gpu" \
        LOOKBACK_OUTPUT_DIR="${LOOKBACK_OUTPUT_DIR}" \
        run_dataset_lookback "$dataset" "$seq_len" \
            > "${LOG_DIR}/${dataset}_seq${seq_len}.log" 2>&1
        local ec=$?
        if [[ $ec -eq 0 ]]; then
            echo "[$(date "+%H:%M:%S")] ${task_num}/${total} DONE  ${dataset} seq=${seq_len} ✓"
        else
            echo "[$(date "+%H:%M:%S")] ${task_num}/${total} FAIL  ${dataset} seq=${seq_len} (exit $ec)"
        fi
        eval "echo >&${fd}"
    ) &
}

while IFS="|" read -r dataset seq_len; do
    cur=$((cur + 1))
    _rotate_gpu
    run_task "$dataset" "$seq_len" "$cur"
done < "'"${TASK_FILE}"'"

wait
echo "[$(date "+%H:%M:%S")] All ${total} experiment(s) finished"
rm -f "${_GPU_COUNTER_FILE:-}"
' >> "$SCHEDULER_LOG" 2>&1 &

DISOWN_PID=$!
disown $DISOWN_PID 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Launched lookback runner as PID $DISOWN_PID"
echo "  GPUs: ${GPU_IDS[*]}  slots/GPU: ${PARA}  total: ${total_experiments}"
echo "  Log:  tail -f ${SCHEDULER_LOG}"
