#!/bin/bash
set -euo pipefail

PRED_LENS_ETT="96 192 336 720"
PRED_LENS_PEMS="12 24 48 96"
PID_DIR="./temp/optuna/pids"

usage() {
	cat <<'EOF'
Usage:
	# Tuning mode (default, foreground)
	bash scripts/hpo/hpo_parallel.sh --dataset ETTh1 --pl 96 --trials 50
	bash scripts/hpo/hpo_parallel.sh --dataset ETTh1 --pl all --trials 30 --gpu 0
	bash scripts/hpo/hpo_parallel.sh --dataset multi --trials 30

	# Daemon mode (survives terminal close, log to log/)
	bash scripts/hpo/hpo_parallel.sh --dataset ETTh1 --pl 96 --trials 50 --bg_mode 1

	# Process management
	bash scripts/hpo/hpo_parallel.sh --list
	bash scripts/hpo/hpo_parallel.sh --stop --dataset ETTh1 --pl 96
	bash scripts/hpo/hpo_parallel.sh --stop-all

	# Dashboard mode (launch Optuna Dashboard only)
	bash scripts/hpo/hpo_parallel.sh --dataset ETTh1 --pl 96 --dashboard
	bash scripts/hpo/hpo_parallel.sh --dataset ETTh1 --pl 96 --dashboard --port 59102

Options:
	--dataset, -d   Dataset name (ETTh1, Weather, ECL, ..., multi)
	--pl, -p        Prediction length or "all" (default: all)
	--trials, -n    Number of Optuna trials (default: 50)
	--gpu, -g       GPU ID (default: 0)
	--epoch         Override training epochs (default: 0 = use 50)
	--patience      Override early stopping patience (default: 0 = use 7)
	--bg_mode       Daemon mode: 0=foreground (default), 1=background with nohup
	--nodes         Number of GPUs for multi-GPU mode (default: 1)
	--stagger       Delay (seconds) between launching background studies (default: 0)
	--list          List running daemon processes
	--stop          Stop a running daemon (--dataset and --pl required)
	--stop-all      Stop all running daemons
	--dashboard     Launch Optuna Dashboard instead of tuning
	--port          Dashboard port (default: 58231)
	--help, -h      Show this help
EOF
}

DATASET=""
PL="all"
N_TRIALS=50
GPU=0
EPOCH=0
PATIENCE=5
DASHBOARD=false
NOHUP_VAL=0
NODES=1
STAGGER=0
PORT=${OPTUNA_PORT:-58231}
LIST_MODE=false
STOP_MODE=false
STOP_ALL=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dataset|-d)
			DATASET="$2"
			shift 2
			;;
		--pl|-p)
			PL="$2"
			shift 2
			;;
		--trials|-n)
			N_TRIALS="$2"
			shift 2
			;;
		--gpu|-g)
			GPU="$2"
			shift 2
			;;
		--bg_mode)
			NOHUP_VAL="$2"
			shift 2
			;;
		--epoch)
			EPOCH="$2"
			shift 2
			;;
		--patience)
			PATIENCE="$2"
			shift 2
			;;
		--nodes)
			NODES="$2"
			shift 2
			;;
		--stagger)
			STAGGER="$2"
			shift 2
			;;
		--dashboard)
			DASHBOARD=true
			shift
			;;
		--port)
			PORT="$2"
			shift 2
			;;
		--list)
			LIST_MODE=true
			shift
			;;
		--stop)
			STOP_MODE=true
			shift
			;;
		--stop-all)
			STOP_ALL=true
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			usage
			exit 1
			;;
	esac
done

mkdir -p "$PID_DIR"

pid_file() {
	echo "${PID_DIR}/${1}_pl${2}.pid"
}

log_file() {
	echo "log/${1}_${2}_study.log"
}

check_pid_alive() {
	local pid=$1
	[[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

do_list() {
	local found=0
	for pf in "$PID_DIR"/*.pid; do
		[[ -f "$pf" ]] || continue
		local base
		base=$(basename "$pf" .pid)
		local pid
		pid=$(cat "$pf")
		local log
		log="log/${base}_study.log"
		if check_pid_alive "$pid"; then
			local elapsed=""
			if command -v ps &>/dev/null; then
				elapsed=$(ps -o etimes= -p "$pid" 2>/dev/null | xargs 2>/dev/null || true)
				[[ -n "$elapsed" ]] && elapsed=" (running ${elapsed}s)"
			fi
			echo "RUNNING  PID=$pid  $base  log=$log${elapsed}"
			found=1
		else
			echo "DEAD     PID=$pid  $base  (stale pid file removed)"
			rm -f "$pf"
		fi
	done
	[[ "$found" -eq 0 ]] && echo "No daemon processes found."
}

do_stop() {
	local ds=$1
	local pl=$2
	local pf
	pf=$(pid_file "$ds" "$pl")
	if [[ ! -f "$pf" ]]; then
		echo "No PID file for ${ds}_pl${pl}"
		return 1
	fi
	local pid
	pid=$(cat "$pf")
	local study_name="ManiMamba_${ds}_pl${pl}"
	if check_pid_alive "$pid"; then
		kill "$pid"
		echo "Sent SIGTERM to PID $pid (${ds}_pl${pl})"
		local i
		for i in $(seq 1 15); do
			check_pid_alive "$pid" || break
			sleep 1
		done
		if check_pid_alive "$pid"; then
			kill -9 "$pid" 2>/dev/null || true
			echo "Process did not exit after 15s, sent SIGKILL"
		fi
	else
		echo "Process $pid already dead (cleaning up)"
	fi
	python scripts/hpo/gpu_budget_reset.py --release-pid "$pid" 2>/dev/null || true
	python scripts/hpo/gpu_budget_reset.py --prune-study "$study_name" 2>/dev/null || true
	rm -f "$pf"
}

do_stop_all() {
	for pf in "$PID_DIR"/*.pid; do
		[[ -f "$pf" ]] || continue
		local base
		base=$(basename "$pf" .pid)
		local ds
		ds=$(echo "$base" | sed 's/_pl[0-9]*$//')
		local pl
		pl=$(echo "$base" | sed 's/.*_pl//')
		do_stop "$ds" "$pl"
	done
	echo "--- Auto GPU budget reset ---"
	python scripts/hpo/gpu_budget_reset.py --yes
}

_release_gpu_budget() {
	python scripts/hpo/gpu_budget_reset.py --yes 2>/dev/null
}

MULTI_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange Solar PEMS08 PEMS03 PEMS04 ECL PEMS07 Traffic illness)

if [[ "$LIST_MODE" == true ]]; then
	do_list
	exit 0
fi

if [[ "$STOP_ALL" == true ]]; then
	do_stop_all
	exit 0
fi

if [[ "$STOP_MODE" == true ]]; then
	if [[ -z "$DATASET" ]]; then
		echo "Error: --stop requires --dataset"
		exit 1
	fi
	if [[ "$PL" == "all" ]]; then
		echo "Error: --stop requires a specific --pl value (not 'all')"
		exit 1
	fi
	do_stop "$DATASET" "$PL"
	exit 0
fi

if [[ -z "$DATASET" ]]; then
	echo "Error: --dataset is required"
	echo ""
	usage
	exit 1
fi

study_name() {
	echo "ManiMamba_${1}_pl${2}"
}

launch_dashboard() {
	local STUDY
	STUDY=$(study_name "$1" "$2")
	local DB_FILE="./temp/optuna/${STUDY}.db"
	local STORAGE="sqlite:///./temp/optuna/${STUDY}.db"

	if [[ ! -f "$DB_FILE" ]]; then
		echo "Warning: database not found: $DB_FILE"
		echo "The dashboard will start, but the study may not exist yet."
		echo ""
	fi

	echo "Launching Optuna Dashboard..."
	echo "  Study:      $STUDY"
	echo "  Storage:    $STORAGE"
	echo "  Access URL: http://localhost:$PORT"
	echo ""
	echo "Press Ctrl+C to stop."

	optuna-dashboard "$STORAGE" --port "$PORT"
}

mkdir -p temp/optuna output/optuna log

# ── Dashboard mode ──
if [[ "$DASHBOARD" == true ]]; then
	if [[ "$PL" == "all" ]]; then
		echo "Error: --dashboard requires a specific --pl value (not 'all')"
		echo "Example: bash scripts/hpo/hpo_parallel.sh --dataset ETTh1 --pl 96 --dashboard"
		exit 1
	fi
	launch_dashboard "$DATASET" "$PL"
	exit 0
fi

# ── Tuning mode ──

launch_bg() {
	local ds=$1
	local pl=$2
	local lf
	lf=$(log_file "$ds" "$pl")
	local pf
	pf=$(pid_file "$ds" "$pl")

	if [[ -f "$pf" ]]; then
		local old_pid
		old_pid=$(cat "$pf")
		if check_pid_alive "$old_pid"; then
			echo "Already running: ${ds}_pl${pl} (PID $old_pid). Use --stop first."
			return 0
		fi
		rm -f "$pf"
	fi

	echo "Starting daemon: ${ds}_pl${pl}  log=$lf"
	local nodes_arg=""
	if [[ "$NODES" -gt 1 ]]; then
		nodes_arg="--nodes $NODES"
	fi
	setsid nohup python scripts/hpo/hpo_tune.py \
		--dataset "$ds" --pred_len "$pl" \
		--n_trials "$N_TRIALS" --gpu "$GPU" \
		--train_epochs "$EPOCH" --patience "$PATIENCE" \
		$nodes_arg \
		--bg \
		>> "$lf" 2>&1 &
	local pid=$!
	echo "$pid" > "$pf"
	echo "PID=$pid"
}

launch_fg() {
	local ds=$1
	local pl=$2
	local nodes_arg=""
	if [[ "$NODES" -gt 1 ]]; then
		nodes_arg="--nodes $NODES"
	fi
	CUDA_VISIBLE_DEVICES=$GPU python scripts/hpo/hpo_tune.py \
		--dataset "$ds" --pred_len "$pl" \
		--n_trials "$N_TRIALS" --gpu "$GPU" \
		--train_epochs "$EPOCH" --patience "$PATIENCE" \
		$nodes_arg
}

if [[ "$DATASET" == "multi" ]]; then
	echo "=== Multi-Dataset Mode: ${#MULTI_DATASETS[@]} datasets, pl=${PL} ==="

	if [[ "$NOHUP_VAL" == "1" ]]; then
		for ds in "${MULTI_DATASETS[@]}"; do
			launch_bg "$ds" "$PL"
			if [[ "$STAGGER" -gt 0 ]]; then
				echo "Staggering ${STAGGER}s before next dataset..."
				sleep "$STAGGER"
			fi
		done
		echo "=== All ${#MULTI_DATASETS[@]} dataset daemons launched ==="
	else
		for ds in "${MULTI_DATASETS[@]}"; do
			echo "--- Starting ${ds} (pl=${PL}) ---"
			launch_fg "$ds" "$PL"
		done
		echo "=== All ${#MULTI_DATASETS[@]} datasets complete ==="
	fi
else
	if [[ "$NOHUP_VAL" == "1" ]]; then
		if [[ "$PL" == "all" ]]; then
			PRED_LENS_ARR=()
			case "$DATASET" in
				ETTh1|ETTh2|ETTm1|ETTm2|Weather|ECL|Traffic|Exchange|Solar)
					PRED_LENS_ARR=(96 192 336 720) ;;
				PEMS03|PEMS04|PEMS07|PEMS08)
					PRED_LENS_ARR=(12 24 48 96) ;;
				illness)
					PRED_LENS_ARR=(24 36 48 60) ;;
				*) PRED_LENS_ARR=(96 192 336 720) ;;
			esac
			for pl_val in "${PRED_LENS_ARR[@]}"; do
				launch_bg "$DATASET" "$pl_val"
				if [[ "$STAGGER" -gt 0 ]]; then
					echo "Staggering ${STAGGER}s before next study..."
					sleep "$STAGGER"
				fi
			done
			echo "=== All pred_len daemons launched ==="
		else
			launch_bg "$DATASET" "$PL"
		fi
	else
		launch_fg "$DATASET" "$PL"
	fi
fi
