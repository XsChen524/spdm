#!/bin/bash

# Hi_Mamba on Weather
python scripts/efficiency/run_efficiency.py --dataset Weather

# UniMamba on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model UniMamba \
	--dataset weather \
	--d_state 8 \
	--d_conv 4 \
	--tcn_kernel_size 4 \
	--tcn_dropout 0.1 \
	--st_attention_dim 64 \
	--st_dropout 0.1 \
	--learning_rate 0.00005

# PatchTST on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model PatchTST \
	--dataset weather \
	--patch_len 16 \
	--stride 8 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.0001

# S_Mamba on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model S_Mamba \
	--dataset weather \
	--d_state 2 \
	--learning_rate 0.00005

# DLinear on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model DLinear \
	--dataset weather \
	--individual \
	--learning_rate 0.00005

# iTransformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model iTransformer \
	--dataset weather \
	--learning_rate 0.00005

# Transformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model Transformer \
	--dataset weather \
	--learning_rate 0.00005

# Autoformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model Autoformer \
	--dataset weather \
	--learning_rate 0.00005

# Flowformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model Flowformer \
	--dataset weather \
	--learning_rate 0.00005

# Informer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model Informer \
	--dataset weather \
	--learning_rate 0.00005

# Reformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model Reformer \
	--dataset weather \
	--learning_rate 0.00005

# iFlashformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlashformer \
	--dataset weather \
	--learning_rate 0.00005

# iFlowformer on Weather
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlowformer \
	--dataset weather \
	--learning_rate 0.00005
