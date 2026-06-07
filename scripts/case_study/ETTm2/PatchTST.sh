#!/bin/bash

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
model_name=${model_name:-PatchTST}

python -u scripts/run.py \
	--model_id ETTm2_96_96 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm2.csv \
	--model $model_name \
	--data ETTm2 \
	--features M \
	--seq_len 96 \
	--pred_len 96 \
	--e_layers 3 \
	--enc_in 7 \
	--c_out 7 \
	--batch_size 128 \
	--des 'CASESTUDY' \
	--d_model 128 \
	--d_ff 256 \
	--n_heads 16 \
	--dropout 0.2 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--patch_len 16 \
	--stride 8 \
	--patience 20 \
	--pct_start 0.4 \
	--learning_rate 0.0001 \
	--train_epochs 100 \
	--itr 1

python -u scripts/run.py \
	--model_id ETTm2_96_192 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm2.csv \
	--model $model_name \
	--data ETTm2 \
	--features M \
	--seq_len 96 \
	--pred_len 192 \
	--e_layers 3 \
	--enc_in 7 \
	--c_out 7 \
	--batch_size 128 \
	--des 'CASESTUDY' \
	--d_model 128 \
	--d_ff 256 \
	--n_heads 16 \
	--dropout 0.2 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--patch_len 16 \
	--stride 8 \
	--patience 20 \
	--pct_start 0.4 \
	--learning_rate 0.0001 \
	--train_epochs 100 \
	--itr 1

python -u scripts/run.py \
	--model_id ETTm2_96_384 \
	--is_training 1 \
	--root_path ./data/ETT-small/ \
	--data_path ETTm2.csv \
	--model $model_name \
	--data ETTm2 \
	--features M \
	--seq_len 96 \
	--pred_len 384 \
	--e_layers 3 \
	--enc_in 7 \
	--c_out 7 \
	--batch_size 128 \
	--des 'CASESTUDY' \
	--d_model 128 \
	--d_ff 256 \
	--n_heads 16 \
	--dropout 0.2 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--patch_len 16 \
	--stride 8 \
	--patience 20 \
	--pct_start 0.4 \
	--learning_rate 0.0001 \
	--train_epochs 100 \
	--itr 1
