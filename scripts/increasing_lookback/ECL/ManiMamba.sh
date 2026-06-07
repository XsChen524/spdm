#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=${model_name:-ManiMamba}

seq_lens=(48 96 192 336 720)
d_models=(128 256 256 256 256)
d_ffs=(256 512 512 512 512)
cov_ranks=(0 0 16 16 32)
batch_sizes=(32 32 16 16 8)
learning_rates=(1e-4 7.3e-4 5e-4 5e-5 5e-5)
pred_len=96

for i in "${!seq_lens[@]}"; do
	seq_len=${seq_lens[$i]}
	d_model=${d_models[$i]}
	d_ff=${d_ffs[$i]}
	cov_rank=${cov_ranks[$i]}
	bs=${batch_sizes[$i]}
	lr=${learning_rates[$i]}

	python -u scripts/run.py \
		--is_training 1 \
		--model $model_name \
		--root_path ./data/electricity/ \
		--data_path electricity.csv \
		--model_id ECL_${seq_len}_${pred_len} \
		--data custom \
		--features M \
		--seq_len $seq_len \
		--pred_len $pred_len \
		--e_layers 3 \
		--enc_in 321 \
		--c_out 321 \
		--des 'LOOKBACK_MANIMAMBA' \
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
		--batch_size $bs

done
