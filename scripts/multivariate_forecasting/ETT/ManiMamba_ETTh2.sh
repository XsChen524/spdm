#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-ManiMamba}

pred_lens=(96 192 336 720)
e_layers=(3 3 3 3)
batch_size=(64 64 64 64)
d_models=(128 256 256 256)
d_ffs=(256 512 512 512)
d_states=(2 8 2 2)
expands=(2 2 2 2)
epsilons=(1e-5 1e-5 1e-3 1e-3)
cov_windows=(16 32 32 32)
cov_strides=(16 16 16 16)
geo_d_models=(256 512 128 256)
geo_d_states=(4 8 4 8)
geo_d_convs=(2 4 2 2)
geo_expands=(2 2 2 2)
learning_rates=(9.24e-5 6.72e-5 4.18e-5 4.18e-5)
weight_decays=(1e-6 1e-6 1e-6 1e-6)
dropouts=(0.2 0.2 0.2 0.2)
patiences=(5 5 5 5)

run_indices=(0 1 2 3)

for i in "${run_indices[@]}"; do

	python -u scripts/run.py \
		--is_training 1 \
		--des 'MULTI_PRED' \
		--root_path ./data/ETT-small/ \
		--data_path ETTh2.csv \
		--model_id ETTh2_96_${pred_lens[$i]} \
		--model $model_name \
		--data ETTh2 \
		--seq_len 96 \
		--pred_len ${pred_lens[$i]} \
		--e_layers ${e_layers[$i]} \
		--batch_size ${batch_size[$i]} \
		--enc_in 7 \
		--c_out 7 \
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
		--train_epochs 15

done
