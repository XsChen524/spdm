#!/bin/bash

# Hi_Mamba on ETTm1
python scripts/efficiency/run_efficiency.py --dataset ETTm1

# UniMamba on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model UniMamba \
	--dataset ETTm1 \
	--d_state 2 \
	--d_conv 4 \
	--tcn_kernel_size 3 \
	--tcn_dropout 0.2 \
	--st_attention_dim 64 \
	--st_dropout 0.1 \
	--learning_rate 0.00007

# PatchTST on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model PatchTST \
	--dataset ETTm1 \
	--patch_len 16 \
	--stride 8 \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.0001

# S_Mamba on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model S_Mamba \
	--dataset ETTm1 \
	--d_state 2 \
	--learning_rate 0.00005

# BiMamba4TS on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model BiMamba4TS \
	--dataset ETTm1 \
	--d_state 2 \
	--d_conv 2 \
	--e_fact 2 \
	--bi_dir 1 \
	--residual 1 \
	--ch_ind 1 \
	--embed_type 1 \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--learning_rate 0.00005

# DLinear on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model DLinear \
	--dataset ETTm1 \
	--individual \
	--learning_rate 0.00005

# iTransformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model iTransformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Transformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model Transformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Autoformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model Autoformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Flowformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model Flowformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Informer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model Informer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# Reformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model Reformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# iFlashformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlashformer \
	--dataset ETTm1 \
	--learning_rate 0.00005

# iFlowformer on ETTm1
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlowformer \
	--dataset ETTm1 \
	--learning_rate 0.00005
