#!/bin/bash

# # BiMamba4TS on ETTm1
# # Source: references/Bi-Mamba4TS/scripts/BiMamba4TS/ETT.sh (ETTm1 pl96)
# python scripts/efficiency/legacy/run_efficiency_legacy.py \
# 	--model BiMamba4TS \
# 	--dataset ETTm1 \
# 	--d_model 64 \
# 	--d_ff 128 \
# 	--e_layers 2 \
# 	--d_state 8 \
# 	--d_conv 2 \
# 	--e_fact 1 \
# 	--bi_dir 1 \
# 	--residual 1 \
# 	--ch_ind 1 \
# 	--embed_type 0 \
# 	--patch_len 4 \
# 	--stride 2 \
# 	--padding_patch end \
#     --optim "Adam" \
# 	--lradj type1 \
# 	--learning_rate 5e-5 \
# 	--dropout 0.2 \
# 	--train_epochs 10 \
# 	--batch_size 16 \
#     --patience 5

# # BiMamba4TS on Weather
# # Source: references/Bi-Mamba4TS/scripts/BiMamba4TS/weather.sh (pl96)
# python scripts/efficiency/legacy/run_efficiency_legacy.py \
# 	--model BiMamba4TS \
# 	--dataset weather \
# 	--d_model 64 \
# 	--d_ff 128 \
# 	--e_layers 2 \
# 	--d_state 8 \
# 	--d_conv 2 \
# 	--e_fact 1 \
# 	--bi_dir 1 \
# 	--residual 1 \
# 	--ch_ind 0 \
# 	--embed_type 0 \
# 	--patch_len 4 \
# 	--stride 2 \
# 	--padding_patch end \
# 	--lradj type1 \
#     --optim "Adam" \
# 	--learning_rate 5e-5 \
# 	--dropout 0.0 \
# 	--train_epochs 10 \
# 	--batch_size 16 \
#     --patience 5

# # interPDN on ETTm1
# # Source: references/interPDN/scripts/interPDN_search/ETTm1.sh (pl96)
# # Note: interPDN does not use d_model/d_ff/e_layers
# python scripts/efficiency/legacy/run_efficiency_legacy.py \
# 	--model interPDN \
# 	--dataset ETTm1 \
#     --seq_len 96 \
#     --pred_len 96 \
# 	--patch_len 16 \
# 	--stride 8 \
# 	--padding_patch end \
# 	--ma_type ema \
# 	--alpha 0.3 \
# 	--beta 0.3 \
# 	--con_cls_1 0.3 \
# 	--con_cls_2 0.3 \
# 	--con_time 0.2 \
# 	--lradj sigmoid \
#     --optim "Adam" \
# 	--learning_rate 5e-5 \
# 	--dropout 0.1 \
# 	--train_epochs 10 \
# 	--batch_size 16 \
#     --patience 5

# # interPDN on Weather
# # Source: references/interPDN/scripts/interPDN_search/Weather.sh (pl96)
# python scripts/efficiency/legacy/run_efficiency_legacy.py \
# 	--model interPDN \
# 	--dataset weather \
# 	--seq_len 96 \
#     --pred_len 96 \
# 	--patch_len 16 \
# 	--stride 8 \
# 	--padding_patch end \
# 	--ma_type ema \
# 	--alpha 0.3 \
# 	--beta 0.3 \
# 	--con_cls_1 0.1 \
# 	--con_cls_2 0.1 \
# 	--con_time 0.2 \
# 	--lradj sigmoid \
#     --optim "Adam" \
# 	--learning_rate 5e-5 \
# 	--dropout 0.1 \
# 	--train_epochs 10 \
# 	--batch_size 16 \
#     --patience 5

# # BiMamba4TS on ECL
# # Source: scripts/increasing_lookback/ECL/BiMamba4TS.sh (pl336 via DATASET_CONFIGS)
# python scripts/efficiency/legacy/run_efficiency_legacy.py \
# 	--model BiMamba4TS \
# 	--dataset ECL \
# 	--d_model 128 \
# 	--d_ff 256 \
# 	--e_layers 3 \
# 	--d_state 32 \
# 	--d_conv 2 \
# 	--e_fact 1 \
# 	--bi_dir 1 \
# 	--residual 1 \
# 	--ch_ind 0 \
# 	--embed_type 0 \
# 	--patch_len 24 \
# 	--stride 12 \
# 	--padding_patch end \
#     --optim "Adam" \
# 	--lradj type1 \
# 	--learning_rate 5e-5 \
# 	--dropout 0.3 \
# 	--train_epochs 10 \
# 	--batch_size 16 \
#     --patience 5

# interPDN on ECL
# Source: scripts/increasing_lookback/ECL/interPDN.sh (pl336 via DATASET_CONFIGS)
python scripts/efficiency/legacy/run_efficiency_legacy.py \
	--model interPDN \
	--dataset ECL \
	--patch_len 16 \
	--stride 8 \
	--padding_patch end \
	--ma_type ema \
	--alpha 0.3 \
	--beta 0.3 \
	--con_cls_1 0.1 \
	--con_cls_2 0.1 \
	--con_time 0.2 \
	--lradj sigmoid \
    --optim "Adam" \
	--learning_rate 5e-5 \
	--dropout 0.1 \
	--train_epochs 10 \
	--batch_size 16 \
    --patience 5
