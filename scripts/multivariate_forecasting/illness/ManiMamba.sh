#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-ManiMamba}

pred_lens=(24 36 48 60)
e_layers=(2 2 2 2)
d_models=(256 256 256 256)
d_ffs=(512 512 512 512)
d_states=(16 16 16 16)
expands=(1 1 1 1)
epsilons=(1e-4 1e-4 1e-4 1e-4)
cov_windows=(16 16 16 16)
cov_strides=(8 8 8 8)
cov_ranks=(0 0 0 0)
geo_d_models=(64 64 64 64)
geo_d_states=(16 16 16 16)
geo_d_convs=(4 4 4 4)
geo_expands=(1 1 1 1)

run_indices=(0 1 2 3)

for i in "${run_indices[@]}"; do

	python -u scripts/run.py \
		--is_training 1 \
		--des 'MULTI_PRED' \
		--root_path ./data/illness/ \
		--data_path national_illness.csv \
		--model_id ili_36_${pred_lens[$i]} \
		--model $model_name \
		--data custom \
		--features M \
		--seq_len 36 \
		--label_len 18 \
		--pred_len ${pred_lens[$i]} \
		--e_layers ${e_layers[$i]} \
		--batch_size 32 \
		--enc_in 7 \
		--c_out 7 \
		--d_model ${d_models[$i]} \
		--d_ff ${d_ffs[$i]} \
		--d_state ${d_states[$i]} \
		--expand ${expands[$i]} \
		--epsilon ${epsilons[$i]} \
		--cov_window ${cov_windows[$i]} \
		--cov_stride ${cov_strides[$i]} \
		--cov_rank ${cov_ranks[$i]} \
		--geo_d_model ${geo_d_models[$i]} \
		--geo_d_state ${geo_d_states[$i]} \
		--geo_d_conv ${geo_d_convs[$i]} \
		--geo_expand ${geo_expands[$i]} \
		--learning_rate 1e-4 \
		--optim Adam \
		--lradj type1 \
		--weight_decay 1e-6 \
		--dropout 0.2 \
		--patience 5 \
		--train_epochs 15 \
		--freq w

done
