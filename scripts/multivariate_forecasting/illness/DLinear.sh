#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=${model_name:-DLinear}

for pred_len in 24 36 48 60; do

	python -u scripts/run.py \
		--is_training 1 \
		--des 'MULTI_PRED' \
		--root_path ./data/illness/ \
		--data_path national_illness.csv \
		--model_id ili_36_${pred_len} \
		--model $model_name \
		--data custom \
		--features M \
		--seq_len 36 \
		--label_len 18 \
		--pred_len ${pred_len} \
		--enc_in 7 \
		--c_out 7 \
		--moving_avg 13 \
        --dropout 0.1 \
		--optim Adam \
		--lradj type1 \
		--learning_rate 1e-3 \
		--patience 5 \
		--train_epochs 15 \
		--batch_size 32

done
