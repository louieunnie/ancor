#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_FILE="${LOG_FILE:-logs/sweep_bertbase_lr_from_roberta_$(date +%F_%H-%M-%S).log}"
mkdir -p logs

echo "[START] ${LOG_FILE}" | tee -a "${LOG_FILE}"

for lr in 3e-6 5e-6 8e-6 1e-5; do
  echo "===== [RUN] LR=${lr} =====" | tee -a "${LOG_FILE}"
  MODEL_NAME=bert-base-uncased \
  SAVE_BEST=0 \
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
done

echo "[DONE] ${LOG_FILE}" | tee -a "${LOG_FILE}"
