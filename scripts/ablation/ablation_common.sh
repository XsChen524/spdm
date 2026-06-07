#!/bin/bash
# Shared functions and hyperparameters for ManiMamba ablation scripts.
# Source this file: source scripts/ablation/ablation_common.sh

EPOCHS=15
WARMUP=3
PARALLEL=2
CNT=0

# GPU rotation — when GPU_LIST is set (by run_all_ablation.sh), each
# dataset loop in run_all_datasets calls _rotate_gpu once, then all
# pred_lens for that dataset share the same GPU.
# When GPU_LIST is unset (running a group script standalone), falls back
# to the current CUDA_VISIBLE_DEVICES with PARALLEL concurrency.
if [[ -n "${GPU_LIST:-}" ]]; then
    read -ra _GPU_ARR <<< "${GPU_LIST}"
    _GPU_IDX=0
    _GPU_CNT=${#_GPU_ARR[@]}
    _PARA_PER_GPU="${PARA_PER_GPU:-1}"
    _CURRENT_GPU=""
    declare -A _GPU_FD=()
    _next_fd=20
    for _g in "${_GPU_ARR[@]}"; do
        if [[ -z "${_GPU_FD[$_g]+x}" ]]; then
            _sem="/tmp/_manimamba_gpu_sem_${_g}_$$"
            mkfifo "$_sem"
            eval "exec ${_next_fd}<>\"$_sem\""
            rm -f "$_sem"
            _GPU_FD[$_g]=${_next_fd}
            for ((i = 0; i < _PARA_PER_GPU; i++)); do
                eval "echo >&${_next_fd}"
            done
            ((_next_fd++))
        fi
    done
fi

_rotate_gpu() {
    if [[ -n "${GPU_LIST:-}" ]]; then
        if [[ -n "${_GPU_COUNTER_FILE:-}" && -f "${_GPU_COUNTER_FILE:-}" ]]; then
            local idx
            idx=$(flock "${_GPU_COUNTER_FILE}" bash -c 'read -r v < "$1"; echo $((v+1)) > "$1"; printf "%d" "$v"' _ "${_GPU_COUNTER_FILE}")
            _CURRENT_GPU="${_GPU_ARR[$((idx % _GPU_CNT))]}"
        else
            _CURRENT_GPU="${_GPU_ARR[$((_GPU_IDX % _GPU_CNT))]}"
            _GPU_IDX=$(( _GPU_IDX + 1 ))
        fi
    fi
}

_dispatch() {
    local cmd=("$@")
    if [[ -n "${GPU_LIST:-}" ]]; then
        local gpu="${_CURRENT_GPU}"
        local fd="${_GPU_FD[$gpu]}"
        (
            eval "read -r _ <&${fd}"
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
            eval "echo >&${fd}"
        ) &
    else
        "${cmd[@]}" &
        CNT=$((CNT + 1))
        if [ $((CNT % PARALLEL)) -eq 0 ]; then wait; fi
    fi
}

_wait_all() {
    wait
}

# _run wraps a python invocation for GPU-rotation dispatch.
# Usage:  _run python -u scripts/run.py [args...]
_run() {
    _dispatch "$@"
}

# ============================================================
# ETTm1 params (from optuna best trials, d_ff capped 512, bs=64, lr×2)
# ============================================================

run_ettm1_96() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_96 \
        --data ETTm1 \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 2 \
        --d_model 128 \
        --d_ff 256 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 4 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 2e-04 \
        --weight_decay 1e-6 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_ettm1_192() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_192 \
        --data ETTm1 \
        --seq_len 96 \
        --pred_len 192 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 2 \
        --d_model 128 \
        --d_ff 256 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 2e-04 \
        --weight_decay 1e-6 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_ettm1_336() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_336 \
        --data ETTm1 \
        --seq_len 96 \
        --pred_len 336 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 2 \
        --d_model 128 \
        --d_ff 256 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 2e-04 \
        --weight_decay 1e-6 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_ettm1_720() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_720 \
        --data ETTm1 \
        --seq_len 96 \
        --pred_len 720 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 2 \
        --d_model 128\
        --d_ff 256 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 2e-04 \
        --weight_decay 1e-6 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

# ============================================================
# Weather params (d_ff capped 512, bs=64, lr×2)
# ============================================================

run_weather_96() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 21 \
        --c_out 21 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 8e-05 \
        --weight_decay 1e-6 \
        --dropout 0.1 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq t \
        "${extra_args[@]}"
}

run_weather_192() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/weather/ \
        --data_path weather.csv \
        --model_id weather_96_192 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 192 \
        --enc_in 21 \
        --c_out 21 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 8e-05 \
        --weight_decay 1e-6 \
        --dropout 0.1 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq t \
        "${extra_args[@]}"
}

run_weather_336() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/weather/ \
        --data_path weather.csv \
        --model_id weather_96_336 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 336 \
        --enc_in 21 \
        --c_out 21 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 5e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq t \
        "${extra_args[@]}"
}

run_weather_720() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/weather/ \
        --data_path weather.csv \
        --model_id weather_96_720 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 720 \
        --enc_in 21 \
        --c_out 21 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 5e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq t \
        "${extra_args[@]}"
}

# ============================================================
# ECL params (d_ff capped 512, bs=64, lr×2)
# ============================================================

run_ecl_96() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_96 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 321 \
        --c_out 321 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --cov_rank 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 5e-05 \
        --weight_decay 1e-6 \
        --dropout 0.1 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq h \
        "${extra_args[@]}"
}

run_ecl_192() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_192 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 192 \
        --enc_in 321 \
        --c_out 321 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --cov_rank 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 5e-05 \
        --weight_decay 1e-6 \
        --dropout 0.1 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq h \
        "${extra_args[@]}"
}

run_ecl_336() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_336 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 336 \
        --enc_in 321 \
        --c_out 321 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --cov_rank 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 5e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq h \
        "${extra_args[@]}"
}

run_ecl_720() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_720 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 720 \
        --enc_in 321 \
        --c_out 321 \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 8 \
        --cov_rank 16 \
        --geo_d_model 512 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 5e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        --freq h \
        "${extra_args[@]}"
}

# ============================================================
# ETTh2 params (from optuna best trials)
# ============================================================

run_etth2_96() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --data ETTh2 \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 3 \
        --d_model 128 \
        --d_ff 256 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 16 \
        --cov_rank 0 \
        --geo_d_model 256 \
        --geo_d_state 4 \
        --geo_d_conv 2 \
        --geo_expand 2 \
        --learning_rate 9.24e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_etth2_192() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_192 \
        --data ETTh2 \
        --seq_len 96 \
        --pred_len 192 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 8 \
        --expand 2 \
        --epsilon 1e-05 \
        --cov_window 32 \
        --cov_stride 16 \
        --cov_rank 0 \
        --geo_d_model 512 \
        --geo_d_state 8 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --learning_rate 6.72e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_etth2_336() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_336 \
        --data ETTh2 \
        --seq_len 96 \
        --pred_len 336 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-03 \
        --cov_window 32 \
        --cov_stride 16 \
        --cov_rank 0 \
        --geo_d_model 128 \
        --geo_d_state 4 \
        --geo_d_conv 2 \
        --geo_expand 2 \
        --learning_rate 4.18e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_etth2_720() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_720 \
        --data ETTh2 \
        --seq_len 96 \
        --pred_len 720 \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 2 \
        --expand 2 \
        --epsilon 1e-03 \
        --cov_window 32 \
        --cov_stride 16 \
        --cov_rank 0 \
        --geo_d_model 256 \
        --geo_d_state 8 \
        --geo_d_conv 2 \
        --geo_expand 2 \
        --learning_rate 4.18e-05 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 64 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

# ============================================================
# PEMS08 params (from optuna best trials)
# ============================================================

run_pems08_12() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/PEMS/ \
        --data_path PEMS08.npz \
        --model_id PEMS08_96_12 \
        --data PEMS \
        --seq_len 96 \
        --pred_len 12 \
        --enc_in 170 \
        --c_out 170 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 8 \
        --expand 1 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 8 \
        --cov_rank 16 \
        --geo_d_model 128 \
        --geo_d_state 4 \
        --geo_d_conv 2 \
        --geo_expand 1 \
        --learning_rate 1e-03 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 32 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_pems08_24() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/PEMS/ \
        --data_path PEMS08.npz \
        --model_id PEMS08_96_24 \
        --data PEMS \
        --seq_len 96 \
        --pred_len 24 \
        --enc_in 170 \
        --c_out 170 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 2 \
        --expand 1 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 16 \
        --cov_rank 16 \
        --geo_d_model 64 \
        --geo_d_state 8 \
        --geo_d_conv 2 \
        --geo_expand 1 \
        --learning_rate 1e-03 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 32 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_pems08_48() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/PEMS/ \
        --data_path PEMS08.npz \
        --model_id PEMS08_96_48 \
        --data PEMS \
        --seq_len 96 \
        --pred_len 48 \
        --enc_in 170 \
        --c_out 170 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 16 \
        --expand 1 \
        --epsilon 1e-05 \
        --cov_window 16 \
        --cov_stride 4 \
        --cov_rank 16 \
        --geo_d_model 128 \
        --geo_d_state 8 \
        --geo_d_conv 4 \
        --geo_expand 1 \
        --learning_rate 8.532e-04 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 32 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

run_pems08_96() {
    local extra_args=("$@")
    _run python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/PEMS/ \
        --data_path PEMS08.npz \
        --model_id PEMS08_96_96 \
        --data PEMS \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 170 \
        --c_out 170 \
        --e_layers 3 \
        --d_model 256 \
        --d_ff 512 \
        --d_state 8 \
        --expand 1 \
        --epsilon 1e-06 \
        --cov_window 16 \
        --cov_stride 8 \
        --cov_rank 16 \
        --geo_d_model 128 \
        --geo_d_state 16 \
        --geo_d_conv 2 \
        --geo_expand 1 \
        --learning_rate 9.991e-04 \
        --weight_decay 1e-6 \
        --dropout 0.2 \
        --patience 5 \
        --batch_size 32 \
        --warmup_epochs ${WARMUP} \
        --train_epochs ${EPOCHS} \
        "${extra_args[@]}"
}

# ============================================================
# Dispatchers
# ============================================================

run_ettm1() {
    local pl=$1
    shift
    "run_ettm1_${pl}" "$@"
}

run_weather() {
    local pl=$1
    shift
    "run_weather_${pl}" "$@"
}

run_ecl() {
    local pl=$1
    shift
    "run_ecl_${pl}" "$@"
}

run_etth2() {
    local pl=$1
    shift
    "run_etth2_${pl}" "$@"
}

run_pems08() {
    local pl=$1
    shift
    "run_pems08_${pl}" "$@"
}

run_all_datasets() {
    local extra_args=("$@")
    for pl in 96 192 336 720; do
        _rotate_gpu
        run_ettm1 ${pl} "${extra_args[@]}"
    done
    for pl in 96 192 336 720; do
        _rotate_gpu
        run_weather ${pl} "${extra_args[@]}"
    done
    for pl in 96 192 336 720; do
        _rotate_gpu
        run_ecl ${pl} "${extra_args[@]}"
    done
    for pl in 96 192 336 720; do
        _rotate_gpu
        run_etth2 ${pl} "${extra_args[@]}"
    done
    for pl in 12 24 48 96; do
        _rotate_gpu
        run_pems08 ${pl} "${extra_args[@]}"
    done
    _wait_all
}
