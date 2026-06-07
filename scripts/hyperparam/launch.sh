#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$PROJECT_ROOT/scripts/hyperparam/run_sensitivity.py"

DATASET="Weather"
# PRED_LENS="96 336"
PRED_LENS="96"
NUM_GPUS=4
PARALLEL_PER_GPU=2
# PARAMS=("d_model" "epsilon" "geo_d_model" "cov_rank")
PARAMS=("cov_rank")
declare -A RANGES
RANGES[d_model]="128 256 512 1024"
RANGES[epsilon]="1e-5 1e-4 1e-3 1e-2"
RANGES[geo_d_model]="64 128 256 512"
# RANGES[cov_rank]="0 8 16 32"
RANGES[cov_rank]="32"

cd "$PROJECT_ROOT"
mkdir -p output/hyperparam log

RESULTS_XLSX="output/hyperparam/sensitivity_results.xlsx"

JOBS=()
for param in "${PARAMS[@]}"; do
    for pl in $PRED_LENS; do
        for val in ${RANGES[$param]}; do
            JOBS+=("$param|$pl|$val")
        done
    done
done

TOTAL=${#JOBS[@]}
TOTAL_WORKERS=$((NUM_GPUS * PARALLEL_PER_GPU))
echo "============================================================"
echo "Hyperparameter Sensitivity — $DATASET"
echo "  Total experiments: $TOTAL"
echo "  GPUs: $NUM_GPUS x $PARALLEL_PER_GPU parallel = $TOTAL_WORKERS workers"
echo "  Params: ${PARAMS[*]}"
echo "  Pred lens: $PRED_LENS"
echo "  Output: $RESULTS_XLSX"
echo "============================================================"

QUEUE_FILE=$(mktemp /tmp/hparam_queue.XXXXXX)
for job in "${JOBS[@]}"; do
    echo "$job" >> "$QUEUE_FILE"
done

LOCK_FILE="${QUEUE_FILE}.lock"
RUN_DIR="$(mktemp -d /tmp/hparam_run.XXXXXX)"
LOG_DIR="$PROJECT_ROOT/log/hyperparam_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cat <<EOF > "$RUN_DIR/config"
QUEUE_FILE=$QUEUE_FILE
LOCK_FILE=$LOCK_FILE
LOG_DIR=$LOG_DIR
RESULTS_XLSX=$RESULTS_XLSX
TOTAL=$TOTAL
TOTAL_WORKERS=$TOTAL_WORKERS
DATASET=$DATASET
PY=$PY
PARALLEL_PER_GPU=$PARALLEL_PER_GPU
EOF

WORKER_SCRIPT="$RUN_DIR/worker.sh"
cat <<'WORKER_EOF' > "$WORKER_SCRIPT"
#!/bin/bash
set -euo pipefail

source "$1"
worker_id=$2
gpu=$((worker_id / PARALLEL_PER_GPU))
LOG_FILE="$LOG_DIR/worker_${worker_id}_gpu${gpu}.log"

{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker $worker_id started on GPU $gpu"
    while true; do
        job=""
        {
            flock 9
            job=$(head -n1 "$QUEUE_FILE" 2>/dev/null || true)
            if [ -n "$job" ]; then
                sed -i '1d' "$QUEUE_FILE"
            fi
        } 9>"$LOCK_FILE"

        [ -z "$job" ] && break

        IFS='|' read -r param pl val <<< "$job"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Worker $worker_id GPU $gpu] $param=$val pl=$pl"
        CUDA_VISIBLE_DEVICES=$gpu python -u "$PY" \
            --sweep_param "$param" \
            --sweep_value "$val" \
            --pred_len "$pl" \
            --gpu 0 \
            2>&1 | while IFS= read -r line; do
                echo "[W${worker_id} G${gpu}] $line"
            done
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Worker $worker_id GPU $gpu] Done."
} > "$LOG_FILE" 2>&1
WORKER_EOF
chmod +x "$WORKER_SCRIPT"

PID_FILE="$RUN_DIR/pids"
> "$PID_FILE"

for ((w = 0; w < TOTAL_WORKERS; w++)); do
    nohup bash "$WORKER_SCRIPT" "$RUN_DIR/config" "$w" </dev/null &>/dev/null &
    echo $! >> "$PID_FILE"
done

MONITOR_SCRIPT="$RUN_DIR/monitor.sh"
cat <<'MONITOR_EOF' > "$MONITOR_SCRIPT"
#!/bin/bash
set -euo pipefail
source "$1"

PIDS=($(cat "${RUN_DIR}/pids"))
for pid in "${PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
    done
done

REMAINING=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo "0")
rm -f "$QUEUE_FILE" "$LOCK_FILE"

DONE=$(python3 -c "import openpyxl; print(openpyxl.load_workbook('$RESULTS_XLSX').active.max_row - 1)" 2>/dev/null || echo "0")
echo "" >> "$LOG_DIR/summary.log"
echo "============================================================" >> "$LOG_DIR/summary.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All workers finished." >> "$LOG_DIR/summary.log"
echo "Results: $RESULTS_XLSX" >> "$LOG_DIR/summary.log"
echo "Logs: $LOG_DIR/" >> "$LOG_DIR/summary.log"
echo "============================================================" >> "$LOG_DIR/summary.log"
MONITOR_EOF
chmod +x "$MONITOR_SCRIPT"

nohup bash "$MONITOR_SCRIPT" "$RUN_DIR/config" </dev/null &>/dev/null &
MONITOR_PID=$!

echo ""
echo "Launched $TOTAL_WORKERS workers in background."
echo "  Queue:      $QUEUE_FILE"
echo "  PIDs:       $PID_FILE"
echo "  Logs:       $LOG_DIR/"
echo "  Monitor PID: $MONITOR_PID"
echo ""
echo "To track progress:"
echo "  tail -f $LOG_DIR/worker_*.log"
echo "  cat $QUEUE_FILE | wc -l   # remaining jobs"
