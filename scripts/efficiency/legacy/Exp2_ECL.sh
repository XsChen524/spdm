#!/bin/bash

# Hi_Mamba on ECL
python scripts/efficiency/run_efficiency.py --dataset ECL

# S_Mamba on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model S_Mamba \
	--dataset ECL \
	--d_state 32 \
	--learning_rate 0.0005

# PatchTST on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model PatchTST \
	--dataset ECL \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--fc_dropout 0.2 \
	--head_dropout 0 \
	--pct_start 0.4 \
	--learning_rate 0.0005

# DLinear on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model DLinear \
	--dataset ECL \
	--individual \
	--learning_rate 0.0005

# iTransformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model iTransformer \
	--dataset ECL \
	--learning_rate 0.0005

# Transformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model Transformer \
	--dataset ECL \
	--learning_rate 0.0005

# Autoformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model Autoformer \
	--dataset ECL \
	--learning_rate 0.0005

# Flowformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model Flowformer \
	--dataset ECL \
	--learning_rate 0.0005

# Informer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model Informer \
	--dataset ECL \
	--learning_rate 0.0005

# Reformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model Reformer \
	--dataset ECL \
	--learning_rate 0.0005

# iFlashformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlashformer \
	--dataset ECL \
	--learning_rate 0.0005

# iFlowformer on ECL
python scripts/efficiency/run_efficiency_legacy.py \
	--model iFlowformer \
	--dataset ECL \
	--learning_rate 0.0005
