#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=${model_name:-ManiMamba}

seq_lens=(48 96 192 336 720)
d_models=(128 256 256 512 512)
d_ffs=(256 512 512 768 768)
learning_rates=(1e-4 4.89e-5 5e-5 5e-5 4e-5)
pred_len=96

for i in "${!seq_lens[@]}"; do
	seq_len=${seq_lens[$i]}
	d_model=${d_models[$i]}
	d_ff=${d_ffs[$i]}
	lr=${learning_rates[$i]}

	python -u scripts/run.py \
		--is_training 1 \
		--model $model_name \
		--root_path ./data/weather/ \
		--data_path weather.csv \
		--model_id Weather_${seq_len}_${pred_len} \
		--data custom \
		--features M \
		--seq_len $seq_len \
		--pred_len $pred_len \
		--e_layers 3 \
		--enc_in 21 \
		--c_out 21 \
		--des 'LOOKBACK_MANIMAMBA' \
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

done
