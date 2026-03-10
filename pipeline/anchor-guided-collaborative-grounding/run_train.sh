#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-gmner}"  # gmner | fmnerg
GOLD_DATA_ROOT="${GOLD_DATA_ROOT:-/workspace/ancor-private/pipeline/pipeline/anchor-guided-collaborative-grounding/input_data/gold_data}"
MODEL_NAME="${MODEL_NAME:-roberta-large}"
USE_XML_CLIP_REGIONS="${USE_XML_CLIP_REGIONS:-0}"
XML_DIR="${XML_DIR:-}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
CLIP_DEVICE="${CLIP_DEVICE:-cpu}"

# Fixed default paths (can still be overridden)
NPZ_DIR="${NPZ_DIR:-/workspace/root/data2/twitter_images/all}"
IMG_DIR="${IMG_DIR:-/workspace/IJCAI2019_data/all_images}"

resolve_split_path() {
  local split="$1"
  local ds_dir="${GOLD_DATA_ROOT}/${DATASET}"
  local p=""
  if [[ "${DATASET}" == "gmner" ]]; then
    p="${ds_dir}/output_${split}.jsonl"
  else
    # fmnerg: prefer third_output_* files (contains visual knowledge), fallback to grounding_*.
    if [[ -f "${ds_dir}/third_output_fmnerg_${split}.jsonl" ]]; then
      p="${ds_dir}/third_output_fmnerg_${split}.jsonl"
    else
      p="${ds_dir}/grounding_${split}_fmnerg_.jsonl"
    fi
  fi
  echo "${p}"
}

# Default train/dev/test based on DATASET, override allowed by env
TRAIN_JSON="${TRAIN_JSON:-$(resolve_split_path train)}"
DEV_JSON="${DEV_JSON:-$(resolve_split_path dev)}"
TEST_JSON="${TEST_JSON:-$(resolve_split_path test)}"

# Optional shared outputs
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/best_model.pt}"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/runs/inference_results.jsonl}"

if [[ -z "${NPZ_DIR}" || -z "${IMG_DIR}" ]]; then
  echo "[ERROR] NPZ_DIR and IMG_DIR are required."
  echo "Example:"
  echo "  NPZ_DIR=/path/to/npz IMG_DIR=/path/to/images ./run_train.sh"
  exit 1
fi

if [[ ! -f "${TRAIN_JSON}" || ! -f "${DEV_JSON}" || ! -f "${TEST_JSON}" ]]; then
  echo "[ERROR] train/dev/test jsonl paths are invalid."
  echo "  TRAIN_JSON=${TRAIN_JSON}"
  echo "  DEV_JSON=${DEV_JSON}"
  echo "  TEST_JSON=${TEST_JSON}"
  exit 1
fi

if [[ ! -d "${NPZ_DIR}" || ! -d "${IMG_DIR}" ]]; then
  echo "[ERROR] NPZ_DIR or IMG_DIR directory does not exist."
  echo "  NPZ_DIR=${NPZ_DIR}"
  echo "  IMG_DIR=${IMG_DIR}"
  exit 1
fi

if [[ "${USE_XML_CLIP_REGIONS}" == "1" ]]; then
  if [[ -z "${XML_DIR}" || ! -d "${XML_DIR}" ]]; then
    echo "[ERROR] XML_DIR is required when USE_XML_CLIP_REGIONS=1."
    echo "  XML_DIR=${XML_DIR}"
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
xml_dir = r"${XML_DIR}"
clip_model_name = r"${CLIP_MODEL_NAME}"
clip_device = r"${CLIP_DEVICE}"
EOF

echo "[INFO] Wrote config.py"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] TRAIN_JSON=${TRAIN_JSON}"
echo "[INFO] DEV_JSON=${DEV_JSON}"
echo "[INFO] TEST_JSON=${TEST_JSON}"
echo "[INFO] IMG_DIR=${IMG_DIR}"
echo "[INFO] NPZ_DIR=${NPZ_DIR}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] USE_XML_CLIP_REGIONS=${USE_XML_CLIP_REGIONS}"
if [[ "${USE_XML_CLIP_REGIONS}" == "1" ]]; then
  echo "[INFO] XML_DIR=${XML_DIR}"
  echo "[INFO] CLIP_MODEL_NAME=${CLIP_MODEL_NAME}"
  echo "[INFO] CLIP_DEVICE=${CLIP_DEVICE}"
fi
echo "[INFO] Start training: ${PYTHON_BIN} main_train.py"
"${PYTHON_BIN}" main_train.py
