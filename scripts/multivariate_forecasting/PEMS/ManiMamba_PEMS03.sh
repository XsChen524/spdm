#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-ManiMamba}

pred_lens=(12 24 48 96)
e_layers=(2 2 2 2)
batch_size=(32 32 32 32)
d_models=(256 256 256 256)
d_ffs=(512 512 512 512)
learning_rates=(5e-4 5e-4 5e-4 5e-4)
weight_decays=(1e-6 1e-6 1e-6 1e-6)
dropouts=(0.1 0.1 0.2 0.2)
patiences=(7 7 5 5)

run_indices=(0 1 2 3)

for i in "${run_indices[@]}"; do

	python -u scripts/run.py \
		--is_training 1 \
		--des 'MULTI_PRED' \
		--root_path ./data/PEMS/ \
		--data_path PEMS03.npz \
		--model_id PEMS03_96_${pred_lens[$i]} \
		--model $model_name \
		--data PEMS \
		--seq_len 96 \
		--pred_len ${pred_lens[$i]} \
		--e_layers ${e_layers[$i]} \
		--batch_size ${batch_size[$i]} \
		--enc_in 358 \
		--c_out 358 \
		--d_model ${d_models[$i]} \
		--d_ff ${d_ffs[$i]} \
		--d_state 16 \
		--expand 1 \
		--epsilon 1e-4 \
		--cov_window 16 \
		--cov_stride 8 \
		--cov_rank 32 \
		--geo_d_model 64 \
		--geo_d_state 16 \
		--geo_d_conv 4 \
		--geo_expand 1 \
		--learning_rate ${learning_rates[$i]} \
		--weight_decay ${weight_decays[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--warmup_epochs 5 \
		--train_epochs 50

done
