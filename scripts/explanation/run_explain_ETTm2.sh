#!/bin/bash
# Mani-Mamba interpretability data capture on ETTm2
# Best hyperparameters from Optuna (ManiMamba_ETTm2_pl96, trial 123, MSE=0.1798)
# Output: output/exp_plot/{setting}/spd_*.npy, layer*_*.npy, meta.json
# Then run output/exp_plot/geometry_plots.ipynb to generate figures.

set -e

python -u scripts/run.py \
    --is_training 1 \
    --model ManiMamba \
    --model_id ETTm2_96_96 \
    --data ETTm2 \
    --des "GEOMETRY_EXPLAIN" \
    --root_path ./data/ETT-small/ \
    --data_path ETTm2.csv \
    --features M \
    --freq t \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 3 \
    --enc_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 512 \
    --d_state 8 \
    --expand 2 \
    --epsilon 1e-05 \
    --cov_window 16 \
    --cov_stride 4 \
    --cov_rank 0 \
    --geo_d_model 256 \
    --geo_d_state 8 \
    --geo_d_conv 2 \
    --geo_expand 2 \
    --batch_size 16 \
    --dropout 0.2 \
    --learning_rate 7.707736980647645e-06 \
    --optim "Adam" \
    --weight_decay 0 \
    --train_epochs 15 \
    --patience 5 \
    --use_amp \
    --explain

echo "=== Done. Check output/exp_plot/ for geometry intermediates ==="
echo "Run output/exp_plot/geometry_plots.ipynb to generate figures."
