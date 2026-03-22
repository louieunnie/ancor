#!/usr/bin/env bash
# 논문 experiment section용 ablation: α, β, λ_comp, τ 스윕 + 각 조합 best 모델 저장
# 총 13 run: alpha(3) + beta(3) + loss_ratio(4) + joint_temp(3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_FILE="${LOG_FILE:-logs/paper_ablation_save_$(date +%F_%H-%M-%S).log}"
RESUME_SWEEP="${RESUME_SWEEP:-1}"

echo "[START] paper ablation with save" | tee "${LOG_FILE}"
echo "[CFG] LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

SWEEP_PHASE=alpha \
SWEEP_ALPHA="0.3 0.5 0.7" \
FIX_BETA=0.3 \
SAVE_EACH=1 \
RESUME_SWEEP="${RESUME_SWEEP}" \
LOG_FILE="${LOG_FILE}" \
bash run_paper_hyperparam_sweep.sh

SWEEP_PHASE=beta \
SWEEP_BETA="0.1 0.3 0.5" \
FIX_ALPHA=0.5 \
SAVE_EACH=1 \
RESUME_SWEEP="${RESUME_SWEEP}" \
LOG_FILE="${LOG_FILE}" \
bash run_paper_hyperparam_sweep.sh

SWEEP_PHASE=loss_ratio \
SWEEP_LOSS_EXCL_W="0.3 0.6 1.0 2.0" \
LOSS_INFO_NCE_W=1.0 \
SAVE_EACH=1 \
RESUME_SWEEP="${RESUME_SWEEP}" \
LOG_FILE="${LOG_FILE}" \
bash run_paper_hyperparam_sweep.sh

SWEEP_PHASE=joint_temp \
SWEEP_JOINT_TEMP="0.5 0.8 1.0" \
SAVE_EACH=1 \
RESUME_SWEEP="${RESUME_SWEEP}" \
LOG_FILE="${LOG_FILE}" \
bash run_paper_hyperparam_sweep.sh

echo "[DONE] paper ablation" | tee -a "${LOG_FILE}"
