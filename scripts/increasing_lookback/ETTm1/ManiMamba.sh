#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=${model_name:-ManiMamba}

seq_lens=(48 96 192 336 720)
d_models=(128 128 256 256 256)
d_ffs=(256 256 512 512 512)
learning_rates=(2e-4 3e-5 3e-5 2e-5 2e-5)
pred_len=96

for i in "${!seq_lens[@]}"; do
	seq_len=${seq_lens[$i]}
	d_model=${d_models[$i]}
	d_ff=${d_ffs[$i]}
	lr=${learning_rates[$i]}

	python -u scripts/run.py \
		--is_training 1 \
		--model $model_name \
		--root_path ./data/ETT-small/ \
		--data_path ETTm1.csv \
		--model_id ETTm1_${seq_len}_${pred_len} \
		--data ETTm1 \
		--features M \
		--seq_len $seq_len \
		--pred_len $pred_len \
		--e_layers 3 \
		--enc_in 7 \
		--c_out 7 \
		--des 'LOOKBACK_MANIMAMBA' \
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

done
