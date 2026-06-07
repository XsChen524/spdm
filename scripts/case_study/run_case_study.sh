#!/bin/bash

################################################################################
# Case Study Experiments
#
# Usage:
#   bash scripts/case_study/run_case_study.sh [DATASET] [--gpus N]
#
# Examples:
#   bash scripts/case_study/run_case_study.sh ETTm2
#   bash scripts/case_study/run_case_study.sh ETTm2 --gpus 4
#
# The script detaches immediately.  All output is appended to:
#   log/case_study/scheduler.log              (overall)
#   log/case_study/<MODEL>.log                (per model)
#
# Example configuration:
#   EXPERIMENTS+=("ManiMamba.sh ManiMamba")       # Run ManiMamba model
#   # EXPERIMENTS+=("S-Mamba.sh S_Mamba")         # Skip S_Mamba model
#
################################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

DATASET="${1:-ETTm2}"
GPU_COUNT=4

if [[ "${2:-}" == "--gpus" && -n "${3:-}" ]]; then
    GPU_COUNT=$3
fi

# =============================================================================
# CONFIGURATION: Select experiments to run (comment/uncomment lines)
# =============================================================================

EXPERIMENTS=()
# EXPERIMENTS+=("ManiMamba.sh ManiMamba")
EXPERIMENTS+=("S-Mamba.sh S_Mamba")
EXPERIMENTS+=("iTransformer.sh iTransformer")
# EXPERIMENTS+=("PatchTST.sh PatchTST")

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

EXPERIMENT_SCRIPTS=()
EXPERIMENT_MODELS=()
for exp in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_SCRIPTS+=("${exp%% *}")
    EXPERIMENT_MODELS+=("${exp##* }")
done

if [ ${#EXPERIMENT_SCRIPTS[@]} -eq 0 ]; then
    echo "No experiments selected!"
    echo "Please uncomment at least one experiment in the CONFIGURATION section."
    exit 1
fi

if [ ${#EXPERIMENT_SCRIPTS[@]} -gt $GPU_COUNT ]; then
    echo "Warning: ${#EXPERIMENT_SCRIPTS[@]} experiments but only $GPU_COUNT GPUs."
    echo "Some experiments will share GPUs (round-robin)."
fi

for i in "${!EXPERIMENT_SCRIPTS[@]}"; do
    SCRIPT="${EXPERIMENT_SCRIPTS[$i]}"
    MODEL="${EXPERIMENT_MODELS[$i]}"
    SCRIPT_PATH="scripts/case_study/${DATASET}/${SCRIPT}"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Script not found: $SCRIPT_PATH"
        echo "Please check the script name for $MODEL in dataset $DATASET."
        exit 1
    fi
done

mkdir -p temp/results
mkdir -p "output/case_study/record_${DATASET}"
mkdir -p "log/case_study"

LOG_FILE="log/case_study/scheduler.log"
RECORD_DIR="output/case_study/record_${DATASET}"

SCRIPTS_STR="${EXPERIMENT_SCRIPTS[*]}"
MODELS_STR="${EXPERIMENT_MODELS[*]}"

echo "================================================================================"
echo "${DATASET} Case Study Experiments"
echo "================================================================================"
echo "Selected models (${#EXPERIMENT_MODELS[@]}): ${EXPERIMENT_MODELS[*]}"
echo "GPUs: $GPU_COUNT"
echo "================================================================================"
echo ""

nohup bash -c '
set -u
cd "'"$REPO_ROOT"'" || exit 1

DATASET="'"${DATASET}"'"
GPU_COUNT='"${GPU_COUNT}"'
RECORD_DIR="'"${RECORD_DIR}"'"

read -ra EXPERIMENT_SCRIPTS <<< "'"${SCRIPTS_STR}"'"
read -ra EXPERIMENT_MODELS  <<< "'"${MODELS_STR}"'"
PRED_LENS=(96 192 384)

export CASE_STUDY_MODE=1
export DATASET_NAME=$DATASET

OUTPUT_FILE="output/case_study/${DATASET}.txt"
> "$OUTPUT_FILE"
echo "${DATASET} Dataset Experiments" >> "$OUTPUT_FILE"
echo "=========================" >> "$OUTPUT_FILE"
echo "Started: $(date "+%Y-%m-%d %H:%M:%S")" >> "$OUTPUT_FILE"
echo "Models to run: ${EXPERIMENT_MODELS[*]}" >> "$OUTPUT_FILE"
echo "GPU count: $GPU_COUNT" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "[$(date "+%H:%M:%S")] Starting ${#EXPERIMENT_SCRIPTS[@]} experiments in parallel (one GPU per model)..."

PIDS=()

for i in "${!EXPERIMENT_SCRIPTS[@]}"; do
    SCRIPT="${EXPERIMENT_SCRIPTS[$i]}"
    MODEL="${EXPERIMENT_MODELS[$i]}"
    GPU_ID=$((i % GPU_COUNT))

    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    echo "Model: $MODEL | GPU: $GPU_ID | Started: $START_TIME" >> "$OUTPUT_FILE"

    echo "[$(date "+%H:%M:%S")] START $MODEL -> GPU $GPU_ID"

    (
        GPU_ID=$GPU_ID bash "scripts/case_study/${DATASET}/$SCRIPT"
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            for pred_len in "${PRED_LENS[@]}"; do
                for RESULT_DIR in ./temp/results/*"${MODEL}"*"${DATASET}"_*"${pred_len}"*; do
                    if [ -d "$RESULT_DIR" ]; then
                        NPY_FILE="${RESULT_DIR}/${MODEL}_96_${pred_len}_pred.npy"
                        if [ -f "$NPY_FILE" ]; then
                            cp "$NPY_FILE" "${RECORD_DIR}/${MODEL}_96_${pred_len}_pred.npy"
                        fi
                    fi
                done
            done
        fi

        exit $EXIT_CODE
    ) > "log/case_study/${MODEL}.log" 2>&1 &

    PIDS+=($!)
done

echo "[$(date "+%H:%M:%S")] Waiting for all experiments to complete..."

FAILED_MODELS=()
SUCCESSFUL_MODELS=()

for i in "${!PIDS[@]}"; do
    MODEL="${EXPERIMENT_MODELS[$i]}"
    PID="${PIDS[$i]}"

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date "+%H:%M:%S")] DONE  $MODEL"
        SUCCESSFUL_MODELS+=("$MODEL")
        echo "Status: SUCCESS | Completed: $(date "+%Y-%m-%d %H:%M:%S")" >> "$OUTPUT_FILE"
    else
        echo "[$(date "+%H:%M:%S")] FAIL  $MODEL (exit $EXIT_CODE)"
        FAILED_MODELS+=("$MODEL")
        echo "Status: FAILED | Exit Code: $EXIT_CODE" >> "$OUTPUT_FILE"
    fi

    echo "" >> "$OUTPUT_FILE"
done

echo ""
echo "================================================================================"
echo "Experiment Complete!"
echo "================================================================================"
echo "Results: $OUTPUT_FILE"
echo "Records: ${RECORD_DIR}/"
echo "Successful models (${#SUCCESSFUL_MODELS[@]}): ${SUCCESSFUL_MODELS[*]}"
echo "Failed models (${#FAILED_MODELS[@]}): ${FAILED_MODELS[*]}"
echo "Generated prediction files:"
ls -lh "${RECORD_DIR}/"*.npy 2>/dev/null || echo "  No prediction files found"
echo "================================================================================"

unset CASE_STUDY_MODE
unset DATASET_NAME
' >> "$LOG_FILE" 2>&1 &

DISOWN_PID=$!
disown $DISOWN_PID 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Launched case study runner as PID $DISOWN_PID"
echo "  GPUs: $GPU_COUNT  models: ${#EXPERIMENT_MODELS[@]}"
echo "  Log:  tail -f ${LOG_FILE}"
echo "  Per-model logs: log/case_study/<MODEL>.log"
