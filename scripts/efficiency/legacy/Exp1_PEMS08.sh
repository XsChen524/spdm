#!/bin/bash

# Hi_Mamba on PEMS08
python scripts/efficiency/run_efficiency.py --dataset PEMS08

# S_Mamba on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model S_Mamba \
	--dataset PEMS08 \
	--d_state 32 \
	--learning_rate 0.001

# PatchTST on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model PatchTST \
	--dataset PEMS08 \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.001

# DLinear on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model DLinear \
	--dataset PEMS08 \
	--individual \
	--learning_rate 0.001

# iTransformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model iTransformer \
	--dataset PEMS08 \
	--learning_rate 0.001

# Transformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model Transformer \
	--dataset PEMS08 \
	--learning_rate 0.001

# Autoformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model Autoformer \
	--dataset PEMS08 \
	--learning_rate 0.001

# Flowformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model Flowformer \
	--dataset PEMS08 \
	--learning_rate 0.001

# Informer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model Informer \
	--dataset PEMS08 \
	--learning_rate 0.001

# Reformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model Reformer \
	--dataset PEMS08 \
	--learning_rate 0.001

# iFlashformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlashformer \
	--dataset PEMS08 \
	--learning_rate 0.001

# iFlowformer on PEMS08
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlowformer \
	--dataset PEMS08 \
	--learning_rate 0.001
