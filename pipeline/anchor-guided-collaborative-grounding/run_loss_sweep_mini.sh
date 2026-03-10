#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/anc/bin/python}"
DATASET="${DATASET:-gmner}"
EPOCHS="${EPOCHS:-6}"
MODEL_NAME="${MODEL_NAME:-roberta-large}"
SEED="${SEED:-45}"
SAVE_BEST="${SAVE_BEST:-0}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/loss_sweep_mini_$(date +%F_%H-%M-%S).log}"

mkdir -p "${SCRIPT_DIR}/logs"
echo "[INFO] Mini sweep log file: ${LOG_FILE}"

already_ran() {
  local name="$1"
  if [[ ! -f "${LOG_FILE}" ]]; then
    return 1
  fi
  grep -qE "^\\[RUN\\] ${name} \\|" "${LOG_FILE}"
}

run_exp() {
  local name="$1"
  local cons_w="$2"
  local cons_alpha="$3"

  local ce_w="1.0"
  local excl_w="0.5"
  local margin_w="1.0"
  local cons_sim_thr="0.50"
  local excl_margin="0.3"
  local rank_margin="0.5"
  local rank_topk="5"

  if already_ran "${name}"; then
    echo "[SKIP] ${name} (already exists in log)" | tee -a "${LOG_FILE}"
    return 0
  fi

  echo "" | tee -a "${LOG_FILE}"
  echo "============================================================" | tee -a "${LOG_FILE}"
  echo "[RUN] ${name} | $(date '+%F %T')" | tee -a "${LOG_FILE}"
  echo "  seed=${SEED} dataset=${DATASET} epochs=${EPOCHS}" | tee -a "${LOG_FILE}"
  echo "  w_ce=${ce_w} w_cons=${cons_w} w_excl=${excl_w} w_margin=${margin_w}" | tee -a "${LOG_FILE}"
  echo "  cons_alpha=${cons_alpha} cons_sim_thr=${cons_sim_thr} excl_margin=${excl_margin} rank_margin=${rank_margin} rank_topk=${rank_topk}" | tee -a "${LOG_FILE}"
  echo "============================================================" | tee -a "${LOG_FILE}"

  DATASET="${DATASET}" \
  MODEL_NAME="${MODEL_NAME}" \
  SEED="${SEED}" \
  SAVE_BEST="${SAVE_BEST}" \
  EPOCHS="${EPOCHS}" \
  LOSS_CE_W="${ce_w}" \
  LOSS_CONS_W="${cons_w}" \
  LOSS_EXCL_W="${excl_w}" \
  LOSS_MARGIN_W="${margin_w}" \
  EXCL_MARGIN="${excl_margin}" \
  RANK_MARGIN="${rank_margin}" \
  RANK_TOPK="${rank_topk}" \
  CONS_ALPHA="${cons_alpha}" \
  CONS_SIM_THR="${cons_sim_thr}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  ./run_train.sh 2>&1 | tee -a "${LOG_FILE}"
}

# Two targeted checks around the current best setting.
run_exp "consw035_t050_e03_a065" "0.35" "0.65"
run_exp "consw03_t050_e03_a070" "0.30" "0.70"

echo "[DONE] Mini loss sweep finished. log=${LOG_FILE}" | tee -a "${LOG_FILE}"
