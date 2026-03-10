#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Usage:
#   DATASET=gmner EPOCHS=6 ./run_loss_sweep.sh
#
# Optional:
#   MODEL_NAME=roberta-large
#   PYTHON_BIN=python3

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-gmner}"
EPOCHS="${EPOCHS:-6}"
MODEL_NAME="${MODEL_NAME:-roberta-large}"
SEED="${SEED:-45}"
SAVE_BEST="${SAVE_BEST:-0}"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/loss_sweep_latest.log}"
echo "[INFO] Unified log file: ${LOG_FILE}"

already_ran() {
  local name="$1"
  if [[ ! -f "${LOG_FILE}" ]]; then
    return 1
  fi
  # Skip if this run name already exists in the unified log.
  grep -qE "^\\[RUN\\] ${name} \\|" "${LOG_FILE}"
}

run_exp() {
  local name="$1"
  local ce_w="$2"
  local cons_w="$3"
  local excl_w="$4"
  local margin_w="$5"
  local excl_margin="$6"
  local rank_margin="$7"
  local rank_topk="$8"
  local cons_alpha="$9"
  local cons_sim_thr="${10}"

  if already_ran "${name}"; then
    echo "[SKIP] ${name} (already exists in log)" | tee -a "${LOG_FILE}"
    return 0
  fi

  echo ""
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
  ./run_train.sh 2>&1 | tee -a "${LOG_FILE}"
}

# Focused sweep only: keep promising regions, drop collapsed settings.

# Consistency weight (exclude 0.1; it was unstable/collapsed)
run_exp "consw02_t050_e03_a065" 1.0 0.2 0.5 1.0 0.3 0.5 5 0.65 0.50
run_exp "consw03_t050_e03_a065" 1.0 0.3 0.5 1.0 0.3 0.5 5 0.65 0.50

# Consistency threshold (exclude 0.55; too strict in prior runs)
run_exp "thr045_e03_a065" 1.0 0.2 0.5 1.0 0.3 0.5 5 0.65 0.45
run_exp "thr050_e03_a065" 1.0 0.2 0.5 1.0 0.3 0.5 5 0.65 0.50

# Exclusion margin (focus on activating-yet-stable range)
run_exp "thr050_e01_a065" 1.0 0.2 0.5 1.0 0.1 0.5 5 0.65 0.50

echo "[DONE] Loss sweep finished. log=${LOG_FILE}" | tee -a "${LOG_FILE}"
