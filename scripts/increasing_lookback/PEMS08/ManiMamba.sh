#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=${model_name:-ManiMamba}

seq_lens=(48 96 144 192)
d_models=(128 128 128 128)
d_ffs=(256 256 256 256)
cov_ranks=(16 16 16 16)
batch_sizes=(32 32 16 16)
learning_rates=(1e-4 1e-4 1e-4 1e-4)
pred_len=12

_run_for_index() {
	local i=$1
	seq_len=${seq_lens[$i]}
	d_model=${d_models[$i]}
	d_ff=${d_ffs[$i]}
	cov_rank=${cov_ranks[$i]}
	bs=${batch_sizes[$i]}
	lr=${learning_rates[$i]}

	python -u scripts/run.py \
		--is_training 1 \
		--model $model_name \
		--root_path ./data/PEMS/ \
		--data_path PEMS08.npz \
		--model_id PEMS08_${seq_len}_${pred_len} \
		--data PEMS \
		--features M \
		--seq_len $seq_len \
		--pred_len $pred_len \
		--e_layers 3 \
		--enc_in 170 \
		--c_out 170 \
		--des 'LOOKBACK_MANIMAMBA' \
		--d_model $d_model \
		--d_ff $d_ff \
		--d_state 8 \
		--expand 1 \
		--epsilon 1e-5 \
		--cov_window 16 \
		--cov_stride 8 \
		--cov_rank $cov_rank \
		--geo_d_model 64 \
		--geo_d_state 4 \
		--geo_d_conv 2 \
		--geo_expand 1 \
		--dropout 0.2 \
		--learning_rate $lr \
		--weight_decay 1e-6 \
		--warmup_epochs 3 \
		--train_epochs 15 \
		--patience 5 \
		--freq s \
		--batch_size $bs
}

if [[ -n "${LOOKBACK_SEQ_LEN:-}" ]]; then
	for i in "${!seq_lens[@]}"; do
		if [[ ${seq_lens[$i]} -eq $LOOKBACK_SEQ_LEN ]]; then
			_run_for_index $i
			exit $?
		fi
	done
	echo "Unknown LOOKBACK_SEQ_LEN for PEMS08: $LOOKBACK_SEQ_LEN" >&2
	exit 1
else
	for i in "${!seq_lens[@]}"; do
		_run_for_index $i
	done
fi
