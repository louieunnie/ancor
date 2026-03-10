#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/s1_both_e05_h15_nt50_rt50_seed45.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing env file: ${ENV_FILE}"
  exit 1
fi

TS="$(date +%F_%H-%M-%S)"
LOG_FILE="${PROJECT_DIR}/logs/reproduce_s1_both_e05_h15_nt50_rt50_seed45_${TS}.log"
TEST_OUT="${PROJECT_DIR}/runs/test_reproduce_s1_both_e05_h15_nt50_rt50_seed45_${TS}.jsonl"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"

set -a
source "${ENV_FILE}"
set +a

export TEST_SAVE_PATH="${TEST_OUT}"

echo "[INFO] Start clean reproduction run"
echo "[INFO] ENV_FILE=${ENV_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] TEST_SAVE_PATH=${TEST_SAVE_PATH}"

"${PROJECT_DIR}/run_train.sh" 2>&1 | tee "${LOG_FILE}"
