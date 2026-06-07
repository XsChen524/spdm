#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-ManiMamba}

noise_level=${noise_level:-0.1}

run_pl96() {
	python -u scripts/run.py \
		--is_training 1 \
		--root_path ./data/ETT-small/ \
		--data_path ETTm2.csv \
		--model_id ETTm2_96_96 \
		--model $model_name \
		--data ETTm2 \
		--features M \
		--seq_len 96 \
		--pred_len 96 \
		--e_layers 3 \
		--enc_in 7 \
		--dec_in 7 \
		--c_out 7 \
		--des 'ROBUSTNESS' \
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
		--learning_rate 7.708e-06 \
		--weight_decay 0 \
		--dropout 0.2 \
		--warmup_epochs 3 \
		--patience 5 \
		--train_epochs 15 \
		--optim "Adam" \
		--seed 2023 \
		--itr 1 \
		--noise_level $noise_level
}

run_pl192() {
	python -u scripts/run.py \
		--is_training 1 \
		--root_path ./data/ETT-small/ \
		--data_path ETTm2.csv \
		--model_id ETTm2_96_192 \
		--model $model_name \
		--data ETTm2 \
		--features M \
		--seq_len 96 \
		--pred_len 192 \
		--e_layers 2 \
		--enc_in 7 \
		--dec_in 7 \
		--c_out 7 \
		--des 'ROBUSTNESS' \
		--d_model 256 \
		--d_ff 512 \
		--d_state 2 \
		--expand 1 \
		--epsilon 1e-05 \
		--cov_window 32 \
		--cov_stride 16 \
		--cov_rank 0 \
		--geo_d_model 256 \
		--geo_d_state 4 \
		--geo_d_conv 4 \
		--geo_expand 1 \
		--batch_size 16 \
		--learning_rate 1.349e-05 \
		--weight_decay 0 \
		--dropout 0.2 \
		--warmup_epochs 3 \
		--patience 5 \
		--train_epochs 15 \
		--optim "Adam" \
		--seed 2023 \
		--itr 1 \
		--noise_level $noise_level
}

run_pl336() {
	python -u scripts/run.py \
		--is_training 1 \
		--root_path ./data/ETT-small/ \
		--data_path ETTm2.csv \
		--model_id ETTm2_96_336 \
		--model $model_name \
		--data ETTm2 \
		--features M \
		--seq_len 96 \
		--pred_len 336 \
		--e_layers 3 \
		--enc_in 7 \
		--dec_in 7 \
		--c_out 7 \
		--des 'ROBUSTNESS' \
		--d_model 512 \
		--d_ff 1024 \
		--d_state 8 \
		--expand 1 \
		--epsilon 1e-05 \
		--cov_window 8 \
		--cov_stride 16 \
		--cov_rank 0 \
		--geo_d_model 512 \
		--geo_d_state 16 \
		--geo_d_conv 4 \
		--geo_expand 2 \
		--batch_size 16 \
		--learning_rate 5.874e-06 \
		--weight_decay 0 \
		--dropout 0.2 \
		--warmup_epochs 3 \
		--patience 5 \
		--train_epochs 15 \
		--optim "Adam" \
		--seed 2023 \
		--itr 1 \
		--noise_level $noise_level
}

run_pl720() {
	python -u scripts/run.py \
		--is_training 1 \
		--root_path ./data/ETT-small/ \
		--data_path ETTm2.csv \
		--model_id ETTm2_96_720 \
		--model $model_name \
		--data ETTm2 \
		--features M \
		--seq_len 96 \
		--pred_len 720 \
		--e_layers 2 \
		--enc_in 7 \
		--dec_in 7 \
		--c_out 7 \
		--des 'ROBUSTNESS' \
		--d_model 256 \
		--d_ff 512 \
		--d_state 2 \
		--expand 2 \
		--epsilon 1e-04 \
		--cov_window 32 \
		--cov_stride 16 \
		--cov_rank 0 \
		--geo_d_model 128 \
		--geo_d_state 8 \
		--geo_d_conv 2 \
		--geo_expand 2 \
		--batch_size 64 \
		--learning_rate 1.17e-05 \
		--weight_decay 1e-06 \
		--dropout 0.2 \
		--warmup_epochs 3 \
		--patience 5 \
		--train_epochs 15 \
		--optim "Adam" \
		--seed 2023 \
		--itr 1 \
		--noise_level $noise_level
}

if [[ "${1:-all}" == "all" ]]; then
	run_pl96
	run_pl192
	run_pl336
	run_pl720
else
	"run_pl${1}"
fi
