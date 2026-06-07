#!/bin/bash

set -e

DATASET="${1:-ETTm2}"
RECORD_DIR="output/case_study/record_${DATASET}"

MODELS=(ManiMamba S_Mamba iTransformer PatchTST)
PRED_LENS=(96 192 384)

mkdir -p "$RECORD_DIR"

echo "================================================================================"
echo "Collecting case study predictions for ${DATASET}"
echo "Target: ${RECORD_DIR}/"
echo "================================================================================"

COPIED=0
MISSING=0

for model in "${MODELS[@]}"; do
    for pred_len in "${PRED_LENS[@]}"; do
        DIR_PATTERN="temp/results/${DATASET}_96_${pred_len}_${model}_CASESTUDY_l*_itr*"
        RESULT_DIR=$(ls -d $DIR_PATTERN 2>/dev/null | head -1)

        if [ -z "$RESULT_DIR" ]; then
            echo "  [MISS] ${model} pred_len=${pred_len} — no result dir found"
            MISSING=$((MISSING + 1))
            continue
        fi

        PRED_FILE="${RESULT_DIR}/${model}_96_${pred_len}_pred.npy"
        TRUE_FILE="${RESULT_DIR}/${model}_96_${pred_len}_true.npy"

        if [ -f "$PRED_FILE" ]; then
            cp "$PRED_FILE" "${RECORD_DIR}/${model}_96_${pred_len}_pred.npy"
            echo "  [PRED] ${model}_96_${pred_len}_pred.npy"
            COPIED=$((COPIED + 1))
        else
            echo "  [MISS] ${model}_96_${pred_len}_pred.npy — file not found in ${RESULT_DIR}"
            MISSING=$((MISSING + 1))
        fi

        if [ -f "$TRUE_FILE" ] && [ ! -f "${RECORD_DIR}/true_${pred_len}.npy" ]; then
            cp "$TRUE_FILE" "${RECORD_DIR}/true_${pred_len}.npy"
            echo "  [TRUE] true_${pred_len}.npy (from ${model})"
        fi
    done
done

echo ""
echo "================================================================================"
echo "Done: ${COPIED} files copied, ${MISSING} missing"
echo "================================================================================"
echo ""
ls -lh "${RECORD_DIR}/"*.npy 2>/dev/null || echo "No .npy files found"
