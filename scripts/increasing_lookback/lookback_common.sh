#!/bin/bash
# Shared parameters, run functions, and GPU semaphore infrastructure
# for ManiMamba increasing lookback experiments.
#
# Source this file: source scripts/increasing_lookback/lookback_common.sh
#
# Part A (params + functions) is always loaded.
# Part B (GPU semaphores) activates only when GPU_LIST is set (by run-lookback.sh).

# ============================================================
# Part A: Seq_len lists and per-dataset run functions
# ============================================================

ETTM1_SEQ_LENS=(48 96 192 336 720)
WEATHER_SEQ_LENS=(48 96 192 336 720)
PEMS08_SEQ_LENS=(48 96 144 192)
ECL_SEQ_LENS=(48 96 192 336 720)

# Mapping from dataset name (as in DATASETS array) to seq_len array variable
get_seq_lens() {
    local dataset=$1
    case "$dataset" in
        ETTm1)   echo "${ETTM1_SEQ_LENS[*]}" ;;
        Weather) echo "${WEATHER_SEQ_LENS[*]}" ;;
        PEMS08)  echo "${PEMS08_SEQ_LENS[*]}" ;;
        ECL)     echo "${ECL_SEQ_LENS[*]}" ;;
    esac
}

# --- ETTm1 ---
run_ettm1_lookback() {
    local seq_len=$1
    local d_model d_ff lr
    case $seq_len in
        48)  d_model=128; d_ff=256;  lr=2e-4 ;;
        96)  d_model=128; d_ff=256;  lr=3e-5 ;;
        192) d_model=256; d_ff=512;  lr=3e-5 ;;
        336) d_model=256; d_ff=512;  lr=2e-5 ;;
        720) d_model=256; d_ff=512;  lr=2e-5 ;;
        *)   echo "Unknown seq_len for ETTm1: $seq_len" >&2; return 1 ;;
    esac
    python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_${seq_len}_96 \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in 7 \
        --c_out 7 \
        --des LOOKBACK_MANIMAMBA \
        --d_model $d_model \
        --d_ff $d_ff \
        --d_state 8 \
        --expand 1 \
        --epsilon 1e-4 \
        --cov_window 16 \
        --cov_stride 16 \
        --cov_rank 0 \
        --geo_d_model 64 \
        --geo_d_state 16 \
        --geo_d_conv 4 \
        --geo_expand 1 \
        --dropout 0.2 \
        --learning_rate $lr \
        --weight_decay 0 \
        --warmup_epochs 3 \
        --train_epochs 15 \
        --patience 5 \
        --batch_size 64
}

# --- Weather ---
run_weather_lookback() {
    local seq_len=$1
    local d_model d_ff lr
    case $seq_len in
        48)  d_model=128; d_ff=256;  lr=1e-4 ;;
        96)  d_model=256; d_ff=512;  lr=4.89e-5 ;;
        192) d_model=256; d_ff=512;  lr=5e-5 ;;
        336) d_model=512; d_ff=768;  lr=5e-5 ;;
        720) d_model=512; d_ff=768;  lr=4e-5 ;;
        *)   echo "Unknown seq_len for Weather: $seq_len" >&2; return 1 ;;
    esac
    python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/weather/ \
        --data_path weather.csv \
        --model_id Weather_${seq_len}_96 \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in 21 \
        --c_out 21 \
        --des LOOKBACK_MANIMAMBA \
        --d_model $d_model \
        --d_ff $d_ff \
        --d_state 16 \
        --expand 2 \
        --epsilon 1e-4 \
        --cov_window 32 \
        --cov_stride 16 \
        --cov_rank 0 \
        --geo_d_model 128 \
        --geo_d_state 8 \
        --geo_d_conv 2 \
        --geo_expand 2 \
        --dropout 0.2 \
        --learning_rate $lr \
        --weight_decay 1e-6 \
        --warmup_epochs 3 \
        --train_epochs 15 \
        --patience 5 \
        --freq t \
        --batch_size 32
}

# --- PEMS08 ---
run_pems08_lookback() {
    local seq_len=$1
    LOOKBACK_SEQ_LEN=$seq_len bash scripts/increasing_lookback/PEMS08/ManiMamba.sh
}

# --- ECL ---
run_ecl_lookback() {
    local seq_len=$1
    local d_model d_ff cov_rank batch_size lr
    case $seq_len in
        48)  d_model=128; d_ff=256;  cov_rank=0;  batch_size=32; lr=1e-4 ;;
        96)  d_model=256; d_ff=512;  cov_rank=0;  batch_size=32; lr=7.3e-4 ;;
        192) d_model=256; d_ff=512;  cov_rank=16; batch_size=16; lr=5e-4 ;;
        336) d_model=256; d_ff=512;  cov_rank=16; batch_size=16; lr=5e-5 ;;
        720) d_model=256; d_ff=512;  cov_rank=32; batch_size=8;  lr=5e-5 ;;
        *)   echo "Unknown seq_len for ECL: $seq_len" >&2; return 1 ;;
    esac
    python -u scripts/run.py \
        --is_training 1 \
        --model ManiMamba \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_${seq_len}_96 \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in 321 \
        --c_out 321 \
        --des LOOKBACK_MANIMAMBA \
        --d_model $d_model \
        --d_ff $d_ff \
        --d_state 8 \
        --expand 2 \
        --epsilon 1e-5 \
        --cov_window 16 \
        --cov_stride 16 \
        --cov_rank $cov_rank \
        --geo_d_model 512 \
        --geo_d_state 4 \
        --geo_d_conv 4 \
        --geo_expand 2 \
        --dropout 0.2 \
        --learning_rate $lr \
        --weight_decay 1e-6 \
        --warmup_epochs 3 \
        --train_epochs 15 \
        --patience 5 \
        --freq h \
        --batch_size $batch_size
}

# --- Unified dispatcher ---
run_dataset_lookback() {
    local dataset=$1 seq_len=$2
    case "$dataset" in
        ETTm1)   run_ettm1_lookback "$seq_len" ;;
        Weather) run_weather_lookback "$seq_len" ;;
        PEMS08)  run_pems08_lookback "$seq_len" ;;
        ECL)     run_ecl_lookback "$seq_len" ;;
        *)       echo "Unknown dataset: $dataset" >&2; return 1 ;;
    esac
}

# ============================================================
# Part B: GPU semaphore infrastructure
# Activates only when GPU_LIST is set (by run-lookback.sh).
# ============================================================

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
            _sem="/tmp/_lookback_gpu_sem_${_g}_$$"
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

    _GPU_COUNTER_FILE="/tmp/_lookback_gpu_counter_$$"
    echo 0 > "$_GPU_COUNTER_FILE"
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
