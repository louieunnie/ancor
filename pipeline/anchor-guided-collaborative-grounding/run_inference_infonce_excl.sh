#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/anc/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL_NAME="${MODEL_NAME:-roberta-large}"
TEMPERATURE="${TEMPERATURE:-0.8}"
REGION_DIM="${REGION_DIM:-2048}"
JOINT_INFER_MODE="${JOINT_INFER_MODE:-none}" # none | greedy_conflict

INFER_JSON="${INFER_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/pred_gmner_bert/8114_vk_input.jsonl}"
NPZ_DIR="${NPZ_DIR:-/workspace/root/data2/twitter_images/all}"
IMG_DIR="${IMG_DIR:-/workspace/IJCAI2019_data/all_images}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/runs/best_bertbase_lr8e-6.pt}"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/runs/inference_infonce_excl_$(date +%F_%H-%M-%S)_8114_result.jsonl}"
SAVE_ALL_ENTITIES_PATH="${SAVE_ALL_ENTITIES_PATH:-}"

if [[ ! -f "${INFER_JSON}" ]]; then
  echo "[ERROR] INFER_JSON file not found: ${INFER_JSON}"
  exit 1
fi
if [[ ! -d "${NPZ_DIR}" ]]; then
  echo "[ERROR] NPZ_DIR directory not found: ${NPZ_DIR}"
  exit 1
fi
if [[ ! -d "${IMG_DIR}" ]]; then
  echo "[ERROR] IMG_DIR directory not found: ${IMG_DIR}"
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH file not found: ${MODEL_PATH}"
  exit 1
fi

mkdir -p "$(dirname "${SAVE_PATH}")"

echo "[INFO] Start InfoNCE+Excl inference"
echo "[INFO] INFER_JSON=${INFER_JSON}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] SAVE_PATH=${SAVE_PATH}"
if [[ -n "${SAVE_ALL_ENTITIES_PATH}" ]]; then
  echo "[INFO] SAVE_ALL_ENTITIES_PATH=${SAVE_ALL_ENTITIES_PATH}"
fi
echo "[INFO] MODEL_NAME=${MODEL_NAME} BATCH_SIZE=${BATCH_SIZE} REGION_DIM=${REGION_DIM}"
echo "[INFO] JOINT_INFER_MODE=${JOINT_INFER_MODE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/main_inference_infonce_excl.py" \
  --infer-json "${INFER_JSON}" \
  --npz-dir "${NPZ_DIR}" \
  --img-dir "${IMG_DIR}" \
  --model-path "${MODEL_PATH}" \
  --save-path "${SAVE_PATH}" \
  --save-all-entities-path "${SAVE_ALL_ENTITIES_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --model-name "${MODEL_NAME}" \
  --temperature "${TEMPERATURE}" \
  --region-dim "${REGION_DIM}" \
  --joint-infer-mode "${JOINT_INFER_MODE}"

