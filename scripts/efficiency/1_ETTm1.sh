#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/efficiency/run_efficiency.py \
    --dataset ETTm1 \
    --d_model 256 \
    --d_ff 512 \
    --geo_d_model 256 \
    --e_layers 2 \
    --learning_rate 9.5e-06

python scripts/efficiency/run_efficiency.py \
    --dataset ETTm1 \
    --d_model 256 \
    --d_ff 512 \
    --geo_d_model 256 \
    --e_layers 2 \
    --learning_rate 9e-06

python scripts/efficiency/run_efficiency.py \
    --dataset ETTm1 \
    --d_model 256 \
    --d_ff 512 \
    --geo_d_model 256 \
    --e_layers 2 \
    --learning_rate 8.5e-06

python scripts/efficiency/run_efficiency.py \
    --dataset ETTm1 \
    --d_model 256 \
    --d_ff 512 \
    --geo_d_model 256 \
    --e_layers 2 \
    --learning_rate 8e-06

python scripts/efficiency/run_efficiency.py \
    --dataset ETTm1 \
    --d_model 256 \
    --d_ff 512 \
    --geo_d_model 256 \
    --e_layers 2 \
    --learning_rate 1.1e-05
