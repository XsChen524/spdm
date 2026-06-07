#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-ManiMamba}

pred_lens=(96 192 336 720)
e_layers=(3 2 3 2)
batch_size=(64 64 64 32)
d_models=(512 128 512 128)
d_ffs=(1024 256 1024 256)
d_states=(2 2 2 2)
expands=(2 2 2 2)
epsilons=(1e-4 1e-5 1e-3 1e-4)
cov_windows=(32 32 32 16)
cov_strides=(8 4 4 16)
geo_d_models=(128 128 256 128)
geo_d_states=(16 16 4 16)
geo_d_convs=(2 4 4 2)
geo_expands=(2 2 2 2)
learning_rates=(1.37e-5 9.24e-5 2.21e-5 6.72e-5)
weight_decays=(1e-6 1e-6 1e-6 1e-6)
dropouts=(0.2 0.2 0.2 0.2)
patiences=(5 5 5 5)

run_indices=(0 1 2 3)

for i in "${run_indices[@]}"; do

	python -u scripts/run.py \
		--is_training 1 \
		--des 'MULTI_PRED' \
		--root_path ./data/exchange_rate/ \
		--data_path exchange_rate.csv \
		--model_id Exchange_96_${pred_lens[$i]} \
		--model $model_name \
		--data custom \
		--features M \
		--seq_len 96 \
		--pred_len ${pred_lens[$i]} \
		--e_layers ${e_layers[$i]} \
		--batch_size ${batch_size[$i]} \
		--enc_in 8 \
		--c_out 8 \
		--d_model ${d_models[$i]} \
		--d_ff ${d_ffs[$i]} \
		--d_state ${d_states[$i]} \
		--expand ${expands[$i]} \
		--epsilon ${epsilons[$i]} \
		--cov_window ${cov_windows[$i]} \
		--cov_stride ${cov_strides[$i]} \
		--cov_rank 0 \
		--geo_d_model ${geo_d_models[$i]} \
		--geo_d_state ${geo_d_states[$i]} \
		--geo_d_conv ${geo_d_convs[$i]} \
		--geo_expand ${geo_expands[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--weight_decay ${weight_decays[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--warmup_epochs 3 \
		--train_epochs 15 \
		--freq t

done
