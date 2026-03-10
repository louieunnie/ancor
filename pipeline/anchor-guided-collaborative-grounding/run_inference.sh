#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DATASET="${DATASET:-gmner}"   # gmner | fmnerg
SPLIT="${SPLIT:-test}"        # train | dev | test
GOLD_DATA_ROOT="${GOLD_DATA_ROOT:-/workspace/ancor-private/pipeline/pipeline/anchor-guided-collaborative-grounding/input_data/gold_data}"
MODEL_NAME="${MODEL_NAME:-roberta-large}"

resolve_split_path() {
  local split="$1"
  local ds_dir="${GOLD_DATA_ROOT}/${DATASET}"
  local p=""
  if [[ "${DATASET}" == "gmner" ]]; then
    p="${ds_dir}/output_${split}.jsonl"
  else
    if [[ -f "${ds_dir}/third_output_fmnerg_${split}.jsonl" ]]; then
      p="${ds_dir}/third_output_fmnerg_${split}.jsonl"
    else
      p="${ds_dir}/grounding_${split}_fmnerg_.jsonl"
    fi
  fi
  echo "${p}"
}

# Required inputs
INFER_JSON="${INFER_JSON:-$(resolve_split_path "${SPLIT}")}"
NPZ_DIR="${NPZ_DIR:-/workspace/root/data2/twitter_images/all}"
IMG_DIR="${IMG_DIR:-/workspace/IJCAI2019_data/all_images}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/best_model.pt}"

# Output file
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/runs/inference_$(date +%F_%H-%M-%S).jsonl}"

if [[ -z "${INFER_JSON}" || -z "${NPZ_DIR}" || -z "${IMG_DIR}" ]]; then
  echo "[ERROR] INFER_JSON, NPZ_DIR, IMG_DIR are required."
  echo "Example:"
  echo "  INFER_JSON=/path/to/test.jsonl NPZ_DIR=/path/to/npz IMG_DIR=/path/to/images ./run_inference.sh"
  exit 1
fi

if [[ ! -f "${INFER_JSON}" ]]; then
  echo "[ERROR] INFER_JSON file not found: ${INFER_JSON}"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH file not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -d "${NPZ_DIR}" || ! -d "${IMG_DIR}" ]]; then
  echo "[ERROR] NPZ_DIR or IMG_DIR directory does not exist."
  echo "  NPZ_DIR=${NPZ_DIR}"
  echo "  IMG_DIR=${IMG_DIR}"
  exit 1
fi

mkdir -p "$(dirname "${SAVE_PATH}")"
echo "[INFO] DATASET=${DATASET} SPLIT=${SPLIT}"
echo "[INFO] INFER_JSON=${INFER_JSON}"
echo "[INFO] IMG_DIR=${IMG_DIR}"
echo "[INFO] NPZ_DIR=${NPZ_DIR}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"

INFER_JSON="${INFER_JSON}" NPZ_DIR="${NPZ_DIR}" IMG_DIR="${IMG_DIR}" MODEL_PATH="${MODEL_PATH}" SAVE_PATH="${SAVE_PATH}" BATCH_SIZE="${BATCH_SIZE}" MODEL_NAME="${MODEL_NAME}" \
"${PYTHON_BIN}" - <<'PY'
import os
import torch
from torch.utils.data import DataLoader
from src.collator import collate_fn_infer
from src.dataset import InferenceDataset
from src.model import GroundingModel
from src.evaluate import inference_and_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_json = os.environ["INFER_JSON"]
npz_dir = os.environ["NPZ_DIR"]
img_dir = os.environ["IMG_DIR"]
model_path = os.environ["MODEL_PATH"]
save_path = os.environ["SAVE_PATH"]
batch_size = int(os.environ.get("BATCH_SIZE", "4"))
model_name = os.environ.get("MODEL_NAME", "roberta-large")

dataset = InferenceDataset(infer_json, npz_dir, img_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)

model = GroundingModel(temperature=0.8, model_name=model_name).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"[INFO] Loaded model from {model_path}")
inference_and_save(model, device, loader, save_path=save_path)
print(f"[INFO] Inference complete. Results saved to {save_path}")
PY
