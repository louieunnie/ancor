#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-gmner}"
GOLD_DATA_ROOT="${GOLD_DATA_ROOT:-/workspace/ancor-private/pipeline/pipeline/anchor-guided-collaborative-grounding/input_data/gold_data}"
MODEL_NAME="${MODEL_NAME:-roberta-large}"
USE_XML_CLIP_REGIONS="${USE_XML_CLIP_REGIONS:-0}"
USE_CLIP_REGION_ENCODER="${USE_CLIP_REGION_ENCODER:-0}"
XML_DIR="${XML_DIR:-}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
CLIP_DEVICE="${CLIP_DEVICE:-cpu}"

NPZ_DIR="${NPZ_DIR:-/workspace/root/data2/twitter_images/all}"
IMG_DIR="${IMG_DIR:-/workspace/IJCAI2019_data/all_images}"

TRAIN_JSON="${TRAIN_JSON:-/workspace/ancor/pipeline/anchor-guided-collaborative-grounding/input_data/vk_v2_gpt41mini/gmner_gold/train.jsonl}"
DEV_JSON="${DEV_JSON:-/workspace/ancor/pipeline/anchor-guided-collaborative-grounding/input_data/vk_v2_gpt41mini/dev.jsonl}"
TEST_JSON="${TEST_JSON:-/workspace/ancor/pipeline/anchor-guided-collaborative-grounding/input_data/vk_v2_gpt41mini/gmner_gold/test.jsonl}"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/best_model.pt}"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/runs/inference_results.jsonl}"

if [[ ! -f "${TRAIN_JSON}" || ! -f "${DEV_JSON}" || ! -f "${TEST_JSON}" ]]; then
  echo "[ERROR] train/dev/test jsonl paths are invalid."
  exit 1
fi

if [[ ! -d "${NPZ_DIR}" || ! -d "${IMG_DIR}" ]]; then
  echo "[ERROR] NPZ_DIR or IMG_DIR directory does not exist."
  exit 1
fi

if [[ "${USE_XML_CLIP_REGIONS}" == "1" ]]; then
  if [[ -z "${XML_DIR}" || ! -d "${XML_DIR}" ]]; then
    echo "[ERROR] XML_DIR is required when USE_XML_CLIP_REGIONS=1."
    exit 1
  fi
fi

mkdir -p "$(dirname "${SAVE_PATH}")"

cat > "${SCRIPT_DIR}/config.py" <<EOF
train_json = r"${TRAIN_JSON}"
dev_json = r"${DEV_JSON}"
test_json = r"${TEST_JSON}"
npz_dir = r"${NPZ_DIR}"
img_dir = r"${IMG_DIR}"
model_path = r"${MODEL_PATH}"
save_path = r"${SAVE_PATH}"
model_name = r"${MODEL_NAME}"
use_xml_clip_regions = ${USE_XML_CLIP_REGIONS}
use_clip_region_encoder = ${USE_CLIP_REGION_ENCODER}
xml_dir = r"${XML_DIR}"
clip_model_name = r"${CLIP_MODEL_NAME}"
clip_device = r"${CLIP_DEVICE}"
EOF

echo "[INFO] Wrote config.py (InfoNCE+Excl slim trainer)"
echo "[INFO] Start training: ${PYTHON_BIN} main_train_infonce_excl.py"
"${PYTHON_BIN}" main_train_infonce_excl.py

