#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs runs
LOG_FILE="logs/save_roberta_bert_from_repro_$(date +%F_%H-%M-%S).log"
echo "[START] ${LOG_FILE}" | tee -a "${LOG_FILE}"

TRAIN_JSON="${TRAIN_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/train.jsonl}"
DEV_JSON="${DEV_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/dev.jsonl}"
TEST_JSON="${TEST_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/test.jsonl}"

run_one() {
  local name="$1"
  local model_name="$2"
  local lr="$3"
  local save_path="$4"

  echo "============================================================" | tee -a "${LOG_FILE}"
  echo "[RUN] ${name}" | tee -a "${LOG_FILE}"
  echo "[RUN] model=${model_name} lr=${lr} save_path=${save_path}" | tee -a "${LOG_FILE}"
  echo "[RUN] train=${TRAIN_JSON}" | tee -a "${LOG_FILE}"
  echo "[RUN] dev=${DEV_JSON}" | tee -a "${LOG_FILE}"
  echo "[RUN] test=${TEST_JSON}" | tee -a "${LOG_FILE}"
  echo "============================================================" | tee -a "${LOG_FILE}"

  MODEL_NAME="${model_name}" \
  TRAIN_JSON="${TRAIN_JSON}" \
  DEV_JSON="${DEV_JSON}" \
  TEST_JSON="${TEST_JSON}" \
  SAVE_BEST=1 \
  SAVE_PATH="${save_path}" \
  BATCH_SIZE=4 \
  EPOCHS=6 \
  LR="${lr}" \
  TEMPERATURE=0.8 \
  SEED=45 \
  LOSS_EXCL_W=0.6 \
  LOSS_INFO_NCE_W=1.0 \
  INFO_NCE_TAU=0.05 \
  INFO_NCE_TOPK_NEG=32 \
  EXCL_MARGIN=0.3 \
  EXCL_MODE=sim_prob \
  EXCL_VARIANT=pairwise \
  EXCL_PAIR_ALPHA=0.5 \
  EXCL_PAIR_BETA=0.3 \
  EXCL_PAIR_TYPE_MODE=neutral \
  EXCL_PAIR_TYPE_FACTOR=1.5 \
  EXCL_PAIR_NAME_THR=0.5 \
  EXCL_PAIR_REGION_THR=0.5 \
  EXCL_PAIR_EASY_SCALE=0.5 \
  EXCL_PAIR_HARD_SCALE=1.5 \
  PYTHON_BIN=/workspace/conda-envs/anc/bin/python \
  bash run_train_infonce_excl.sh 2>&1 | tee -a "${LOG_FILE}"
}

run_one \
  "roberta_repro_save_best" \
  "roberta-large" \
  "5e-6" \
  "${SCRIPT_DIR}/runs/best_roberta_repro_seed45.pt"

run_one \
  "bert_repro_lr8e-6_save_best" \
  "bert-base-uncased" \
  "8e-6" \
  "${SCRIPT_DIR}/runs/best_bert_repro_lr8e-6_seed45.pt"

echo "[DONE] ${LOG_FILE}" | tee -a "${LOG_FILE}"
