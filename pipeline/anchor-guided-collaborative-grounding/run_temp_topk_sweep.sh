#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs runs

# ----------------------------
# Fixed baseline (keep stable)
# ----------------------------
MODEL_NAME="${MODEL_NAME:-xlm-roberta-large}"
TRAIN_JSON="${TRAIN_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/train.jsonl}"
DEV_JSON="${DEV_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/dev.jsonl}"
TEST_JSON="${TEST_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/test.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/anc/bin/python}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-6}"
LR="${LR:-5e-6}"
SEED="${SEED:-45}"

LOSS_EXCL_W="${LOSS_EXCL_W:-0.6}"
LOSS_INFO_NCE_W="${LOSS_INFO_NCE_W:-1.0}"
INFO_NCE_TAU="${INFO_NCE_TAU:-0.05}"
EXCL_MARGIN="${EXCL_MARGIN:-0.3}"
EXCL_MODE="${EXCL_MODE:-sim_prob}"
EXCL_VARIANT="${EXCL_VARIANT:-pairwise}"
EXCL_PAIR_ALPHA="${EXCL_PAIR_ALPHA:-0.5}"
EXCL_PAIR_BETA="${EXCL_PAIR_BETA:-0.3}"
EXCL_PAIR_TYPE_MODE="${EXCL_PAIR_TYPE_MODE:-neutral}"
EXCL_PAIR_TYPE_FACTOR="${EXCL_PAIR_TYPE_FACTOR:-1.5}"
EXCL_PAIR_NAME_THR="${EXCL_PAIR_NAME_THR:-0.5}"
EXCL_PAIR_REGION_THR="${EXCL_PAIR_REGION_THR:-0.5}"
EXCL_PAIR_EASY_SCALE="${EXCL_PAIR_EASY_SCALE:-0.5}"
EXCL_PAIR_HARD_SCALE="${EXCL_PAIR_HARD_SCALE:-1.5}"

# ----------------------------
# Sweep axes (only 2)
# ----------------------------
TEMPS_STR="${SWEEP_TEMPERATURES:-0.6 0.8 1.0}"
TOPKS_STR="${SWEEP_TOPK_NEGS:-8 16 32 64 96 128}"
read -r -a TEMPS <<< "${TEMPS_STR}"
read -r -a TOPKS <<< "${TOPKS_STR}"

LOG_FILE="${LOG_FILE:-logs/sweep_temp_topk_$(date +%F_%H-%M-%S).log}"
SAVE_EACH="${SAVE_EACH:-0}"      # 1: save best model for each combo
RESUME_SWEEP="${RESUME_SWEEP:-1}" # 1: skip already finished combos

echo "[START] temp-topk sweep" | tee -a "${LOG_FILE}"
echo "[CFG] MODEL_NAME=${MODEL_NAME} LR=${LR} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} SEED=${SEED}" | tee -a "${LOG_FILE}"
echo "[CFG] SWEEP_TEMPERATURES=${TEMPS_STR}" | tee -a "${LOG_FILE}"
echo "[CFG] SWEEP_TOPK_NEGS=${TOPKS_STR}" | tee -a "${LOG_FILE}"
echo "[CFG] LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

has_done_combo() {
  local temp="$1"
  local topk="$2"
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "0"
    return
  fi
  if /workspace/conda-envs/anc/bin/python - "${LOG_FILE}" "${temp}" "${topk}" <<'PY'
import sys
log_path, temp, topk = sys.argv[1], sys.argv[2], sys.argv[3]
needle = f"[DONE-RUN] temp={temp} topk={topk}"
ok = False
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.strip() == needle:
            ok = True
            break
sys.exit(0 if ok else 1)
PY
  then
    echo "1"
  else
    echo "0"
  fi
}

for temp in "${TEMPS[@]}"; do
  for topk in "${TOPKS[@]}"; do
    if [[ "${RESUME_SWEEP}" == "1" && "$(has_done_combo "${temp}" "${topk}")" == "1" ]]; then
      echo "[SKIP] temp=${temp} topk=${topk} (already done)" | tee -a "${LOG_FILE}"
      continue
    fi

    save_best_this=0
    save_path_this=""
    if [[ "${SAVE_EACH}" == "1" ]]; then
      save_best_this=1
      safe_temp="${temp//./p}"
      save_path_this="${SCRIPT_DIR}/runs/best_${MODEL_NAME//\//_}_temp${safe_temp}_topk${topk}.pt"
    fi

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "[RUN] temp=${temp} topk=${topk}" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    MODEL_NAME="${MODEL_NAME}" \
    TRAIN_JSON="${TRAIN_JSON}" \
    DEV_JSON="${DEV_JSON}" \
    TEST_JSON="${TEST_JSON}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
    LR="${LR}" \
    TEMPERATURE="${temp}" \
    SEED="${SEED}" \
    LOSS_EXCL_W="${LOSS_EXCL_W}" \
    LOSS_INFO_NCE_W="${LOSS_INFO_NCE_W}" \
    INFO_NCE_TAU="${INFO_NCE_TAU}" \
    INFO_NCE_TOPK_NEG="${topk}" \
    EXCL_MARGIN="${EXCL_MARGIN}" \
    EXCL_MODE="${EXCL_MODE}" \
    EXCL_VARIANT="${EXCL_VARIANT}" \
    EXCL_PAIR_ALPHA="${EXCL_PAIR_ALPHA}" \
    EXCL_PAIR_BETA="${EXCL_PAIR_BETA}" \
    EXCL_PAIR_TYPE_MODE="${EXCL_PAIR_TYPE_MODE}" \
    EXCL_PAIR_TYPE_FACTOR="${EXCL_PAIR_TYPE_FACTOR}" \
    EXCL_PAIR_NAME_THR="${EXCL_PAIR_NAME_THR}" \
    EXCL_PAIR_REGION_THR="${EXCL_PAIR_REGION_THR}" \
    EXCL_PAIR_EASY_SCALE="${EXCL_PAIR_EASY_SCALE}" \
    EXCL_PAIR_HARD_SCALE="${EXCL_PAIR_HARD_SCALE}" \
    SAVE_BEST="${save_best_this}" \
    SAVE_PATH="${save_path_this}" \
    bash run_train_infonce_excl.sh 2>&1 | tee -a "${LOG_FILE}"

    echo "[DONE-RUN] temp=${temp} topk=${topk}" | tee -a "${LOG_FILE}"
  done
done

echo "[DONE] sweep finished: ${LOG_FILE}" | tee -a "${LOG_FILE}"
