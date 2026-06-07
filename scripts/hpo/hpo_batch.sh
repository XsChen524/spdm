#!/usr/bin/env bash
set -euo pipefail

N_GPUS=4
STUDIES=()
EPOCH=0
SAVE_TMP=1

usage() {
	cat <<'EOF'
Usage:
  bash scripts/hpo/hpo_batch.sh --n_gpus 4 \
    --study "ETTm1:192:200" --study "ETTm1:720:200" ...

  Each study is formatted as:  dataset:pred_len:trials[:target_gpu]

Options:
  --n_gpus, -N   Number of GPUs (default: 4)
  --study, -s    A study definition (repeatable)
  --epoch        Override training epochs (0 = use hpo_tune default)
  --save_tmp     Save temp files: 1=save (default), 0=skip
  --help, -h     Show help

GPU assignment:
  If target_gpu is specified in the study string, that GPU is used directly.
  Otherwise, round-robin across GPUs (0,1,...,N-1,0,1,...).
Each study is launched as an independent background process (fire-and-forget).
Per-study PID files are created for poe hpo-stop compatibility.
EOF
	exit 0
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--n_gpus|-N) N_GPUS="$2"; shift 2 ;;
		--study|-s) STUDIES+=("$2"); shift 2 ;;
		--epoch) EPOCH="$2"; shift 2 ;;
		--save_tmp) SAVE_TMP="$2"; shift 2 ;;
		--help|-h) usage ;;
		*) echo "Unknown option: $1"; usage ;;
	esac
done

if [ ${#STUDIES[@]} -eq 0 ]; then
	echo "Error: no studies provided. Use --study <dataset:pred_len:trials[:target_gpu]>"
	exit 1
fi

PID_DIR="./temp/optuna/pids"
mkdir -p "$PID_DIR" "./log"

echo "[$(date +%H:%M:%S)] === hpo_batch: ${#STUDIES[@]} studies across ${N_GPUS} GPUs (round-robin) ==="
echo ""

EXTRA_ARGS=""
[ "$EPOCH" -gt 0 ] && EXTRA_ARGS="$EXTRA_ARGS --train_epochs $EPOCH"
[ "$SAVE_TMP" -eq 0 ] && EXTRA_ARGS="$EXTRA_ARGS --save_tmp 0"

for ((i = 0; i < ${#STUDIES[@]}; i++)); do
	line="${STUDIES[$i]}"

	ds=$(echo "$line" | cut -d: -f1)
	pl=$(echo "$line" | cut -d: -f2)
	trials=$(echo "$line" | cut -d: -f3)
	target_gpu=$(echo "$line" | cut -d: -f4)

	if [ -n "$target_gpu" ]; then
		if [ "$target_gpu" -ge "$N_GPUS" ] || [ "$target_gpu" -lt 0 ]; then
			echo "  [SKIP] ${ds}_pl${pl}: target_gpu=$target_gpu out of range [0,$((N_GPUS-1))]"
			continue
		fi
		gpu=$target_gpu
	else
		gpu=$((i % N_GPUS))
	fi

	tag="${ds}_pl${pl}"
	pidfile="$PID_DIR/${tag}.pid"
	log_file="log/${ds}_${pl}_study.log"

	if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
		echo "  [SKIP] ${tag} already running (PID $(cat "$pidfile"))"
		continue
	fi

	(
		export CUDA_VISIBLE_DEVICES="$gpu"
		exec python scripts/hpo/hpo_tune.py \
			--dataset "$ds" --pred_len "$pl" --n_trials "$trials" \
			--gpu 0 $EXTRA_ARGS \
			>> "$log_file" 2>&1
	) &
	study_pid=$!
	echo "$study_pid" > "$pidfile"
	if [ -n "$target_gpu" ]; then
		echo "  [GPU $gpu (pinned)] ${tag} launched (PID $study_pid, ${trials} trials)"
	else
		echo "  [GPU $gpu] ${tag} launched (PID $study_pid, ${trials} trials)"
	fi
done

echo ""
echo "[$(date +%H:%M:%S)] All studies launched. Monitor: poe hpo-list"
