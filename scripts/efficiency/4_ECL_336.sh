#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/efficiency/run_efficiency.py \
    --dataset ECL \
    --d_model 256 \
    --d_ff 512 \
    --e_layers 2 \
    --lradj type1 \
    --learning_rate 3.3e-4
