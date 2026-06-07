#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/efficiency/run_efficiency.py \
    --dataset Weather \
    --d_model 128 \
    --d_ff 256 \
    --e_layers 2 \
    --lradj type1 \
    --learning_rate 1e-04

python scripts/efficiency/run_efficiency.py \
    --dataset Weather \
    --d_model 128 \
    --d_ff 256 \
    --e_layers 2 \
    --lradj type1 \
    --learning_rate 8e-05

python scripts/efficiency/run_efficiency.py \
    --dataset Weather \
    --d_model 128 \
    --d_ff 256 \
    --e_layers 2 \
    --lradj type1 \
    --learning_rate 6e-05
