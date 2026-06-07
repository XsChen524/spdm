#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# ManiMamba Ablation Experiment Runner (v4)
# ============================================================
#
# Usage:
#   bash scripts/ablation/run_all_ablation.sh                 # GPU 0, 1 slot/GPU, groups sequential
#   bash scripts/ablation/run_all_ablation.sh --para 2        # GPU 0, 2 slots/GPU, groups sequential
#   bash scripts/ablation/run_all_ablation.sh 0 2             # GPUs 0,2, 1 slot/GPU
#   bash scripts/ablation/run_all_ablation.sh --para 2 0 2    # GPUs 0,2, 2 slots each
#   bash scripts/ablation/run_all_ablation.sh --para 3 0 1 2 3  # GPUs 0-3, 3 slots each
#
#   GROUP_PARA=2 bash scripts/ablation/run_all_ablation.sh --para 2 0 2
#     → 2 groups concurrent, each round-robins across GPUs 0,2, 2 slots/GPU
#
# The script detaches immediately.  All output is appended to:
#   log/ablation/manimamba/all.log          (overall)
#   log/ablation/manimamba/<group>.log      (per group)
#
# NOTE: Old v2 ablation groups (A/B/C/D/S/M/T/N/R) are commented out below.
#       They require the v2 code with --ablation flag.
#       V4 groups use the current code with --ablation support.

PARA=1
GPUS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --para) PARA="$2"; shift 2 ;;
        *)      GPUS+=("$1"); shift ;;
    esac
done

if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0)
fi

mkdir -p log/ablation/manimamba

LOG_FILE="log/ablation/manimamba/all.log"

nohup bash -c '
set -u
SCRIPT_DIR="'"${SCRIPT_DIR}"'"
source "${SCRIPT_DIR}/ablation_common.sh"

export GPU_LIST="'"${GPUS[*]}"'"
export PARA_PER_GPU="'"${PARA}"'"
export _GPU_COUNTER_FILE="/tmp/_manimamba_gpu_counter_$$"
echo 0 > "$_GPU_COUNTER_FILE"

GROUP_PARA=1

sem="/tmp/_manimamba_group_sem_$$"
mkfifo "$sem"
exec 10<>"$sem"
rm -f "$sem"
for ((i = 0; i < GROUP_PARA; i++)); do
    echo >&10
done

run_variant() {
    local group="$1" script="$2" variant="$3"
    (
        read -r _ <&10
        echo "[$(date "+%H:%M:%S")] START ${group}/${variant} (GPUs: ${GPU_LIST})"
        bash "$script" "$variant" > "log/ablation/manimamba/${group}_${variant}.log" 2>&1
        local ec=$?
        if [[ $ec -eq 0 ]]; then
            echo "[$(date "+%H:%M:%S")] DONE  ${group}/${variant} ✓"
        else
            echo "[$(date "+%H:%M:%S")] FAIL  ${group}/${variant} (exit $ec)"
        fi
        echo >&10
    ) &
}

# ============================================================
# Variant Selection — comment out lines to skip individual variants
# ============================================================

# # --- v4 Baseline (current code, no --ablation flag needed) ---
run_variant baseline     scripts/ablation/run_baseline.sh V3

# # --- v4 Ablation: tanh + alpha (alpha=1.0, BC via alpha*tanh(geo_x_proj)) ---
# run_variant V4_TanhAlpha  scripts/ablation/V4_TanhAlpha/run_V4_TanhAlpha.sh TA

# # --- v4 Ablation: w/o B+C (disable geometry B/C injection) ---
# run_variant V4_NoBC         scripts/ablation/V4_NoBC/run_V4_NoBC.sh NB

# --- v4 Ablation: w/ dt (add dt modulation on top of BC-direct baseline) ---
# run_variant V4_WDt          scripts/ablation/V4_WDt/run_V4_WDt.sh WD

# --- v4 Ablation: linear interpolation (replace sparse scatter) ---
# run_variant V4_LinearInterp scripts/ablation/V4_LinearInterp/run_V4_LinearInterp.sh LI

# --- v4 Ablation: Geodesic Smoothness Regularization (λ=0.01) ---
# run_variant V4_GeoSmoothReg scripts/ablation/V4_GeoSmoothReg/run_V4_GeoSmoothReg.sh GR

# ============================================================
# OLD v2 ablation groups (require v2 code) — commented out
# ============================================================

# # --- v1/v2 Baselines (require v2 code with --ablation support) ---
# run_variant baseline     scripts/ablation/run_baseline.sh V1
# run_variant baseline     scripts/ablation/run_baseline.sh V2

# # --- A: Path ablation (A1=no path A, A2=random geometry) ---
# run_variant A_path       scripts/ablation/A_path/run_A.sh A1
# run_variant A_path       scripts/ablation/A_path/run_A.sh A2

# # --- B: Injection strategy (B1=output gate, B2=over-modulate, B3=unbounded) ---
# run_variant B_injection  scripts/ablation/B_injection/run_B.sh B1
# run_variant B_injection  scripts/ablation/B_injection/run_B.sh B2
# run_variant B_injection  scripts/ablation/B_injection/run_B.sh B3

# # --- C: SPD mean computation (C1=arithmetic mean) ---
# run_variant C_spd        scripts/ablation/C_spd/run_C.sh C1

# # --- D: GeoMamba encoder (D1=MLP encoder, D3=no residual) ---
# run_variant D_geomamba   scripts/ablation/D_geomamba/run_D.sh D1
# run_variant D_geomamba   scripts/ablation/D_geomamba/run_D.sh D3

# # --- F: Low-rank covariance, ECL only (F1=rank0, F2_N=rank N) ---
# run_variant F_lowrank    scripts/ablation/F_lowrank/run_F.sh F1
# run_variant F_lowrank    scripts/ablation/F_lowrank/run_F.sh F2_4
# run_variant F_lowrank    scripts/ablation/F_lowrank/run_F.sh F2_8
# run_variant F_lowrank    scripts/ablation/F_lowrank/run_F.sh F2_32

# # --- S: Scaling function (S2=x10 tanh, S4=x10 direct, S5=softplus) ---
# run_variant S_scale      scripts/ablation/S_scale/run_S.sh S2
# run_variant S_scale      scripts/ablation/S_scale/run_S.sh S4
# run_variant S_scale      scripts/ablation/S_scale/run_S.sh S5

# # --- M: Injection point (M3=broadcast modulate, M4=dt light B) ---
# run_variant M_inject     scripts/ablation/M_inject/run_M.sh M3
# run_variant M_inject     scripts/ablation/M_inject/run_M.sh M4

# # --- T: Temporal alignment (T1=explicit align, T2=sparse align) ---
# run_variant T_align      scripts/ablation/T_align/run_T.sh T1
# run_variant T_align      scripts/ablation/T_align/run_T.sh T2

# # --- N: Condition number (N1=condnum-aware dt) ---
# run_variant N_condnum    scripts/ablation/N_condnum/run_N.sh N1

# # --- R: Regularization (R1=tangent reg, R2=geodesic smooth) ---
# run_variant R_regularize scripts/ablation/R_regularize/run_R.sh R1
# run_variant R_regularize scripts/ablation/R_regularize/run_R.sh R2

# ============================================================

wait
echo "[$(date "+%H:%M:%S")] All groups finished"
rm -f "${_GPU_COUNTER_FILE:-}"
    python output/ablation/update_ablation_v4.py
' >> "$LOG_FILE" 2>&1 &

DISOWN_PID=$!
disown $DISOWN_PID 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Launched ablation runner as PID $DISOWN_PID"
echo "  GPUs: ${GPUS[*]}  slots/GPU: ${PARA}  groups concurrent: ${GROUP_PARA:-1}"
echo "  Log:  tail -f ${LOG_FILE}"
