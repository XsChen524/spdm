#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-iTransformer}

pred_lens=(24 36 48 60)
e_layers=(2 2 2 2)
d_models=(512 512 512 512)
d_ffs=(1024 1024 1024 1024)
n_heads=(8 8 8 8)

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
		--n_heads ${n_heads[$i]} \
		--optim Adam \
		--lradj type1 \
		--learning_rate 5e-4 \
		--dropout 0.2 \
		--patience 5 \
		--train_epochs 15 \
		--freq w

done
