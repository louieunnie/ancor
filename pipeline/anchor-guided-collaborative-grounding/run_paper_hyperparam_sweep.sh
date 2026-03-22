#!/usr/bin/env bash
# Paper-oriented hyperparameter sweeps for InfoNCE+Excl (main_train_infonce_excl.py).
#
# Mapping (논문 기호 ↔ 코드):
#   λ_ij = α(1-sim_entity) + β(1-sim_region) + γ(1-sim_type)
#     → EXCL_PAIR_ALPHA, EXCL_PAIR_BETA
#     → γ: 기본은 implicit γ = max(0, 1-α-β). EXCL_PAIR_GAMMA 를 주면 (α,β,γ) 양수 합으로 정규화.
#   L = λ_comp L_comp + λ_nce L_nce
#     → LOSS_EXCL_W (=λ_comp), LOSS_INFO_NCE_W (=λ_nce). 보통 λ_nce=1 고정 후 λ_comp 만 스윕.
#   Joint softmax temperature τ (P ∝ exp(s/τ))
#     → TEMPERATURE
#   InfoNCE temperature (contrastive)
#     → INFO_NCE_TAU
#
# 사용법:
#   SWEEP_PHASE=alpha bash run_paper_hyperparam_sweep.sh
#   SWEEP_PHASE=all   bash run_paper_hyperparam_sweep.sh   # 순차로 전부 (시간 매우 김)
#
# tmux:
#   tmux new -s paper_hp 'cd ... && SWEEP_PHASE=alpha bash run_paper_hyperparam_sweep.sh'
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs runs

# ---------- 공통 baseline (필요 시 덮어쓰기) ----------
MODEL_NAME="${MODEL_NAME:-xlm-roberta-large}"
TRAIN_JSON="${TRAIN_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/train.jsonl}"
DEV_JSON="${DEV_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/dev.jsonl}"
TEST_JSON="${TEST_JSON:-${SCRIPT_DIR}/input_data/vk_v2_gpt41mini/gmner_gold/test.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/anc/bin/python}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-6}"
LR="${LR:-5e-6}"
SEED="${SEED:-45}"
SAVE_BEST="${SAVE_BEST:-0}"
SAVE_EACH="${SAVE_EACH:-0}"

TEMPERATURE="${TEMPERATURE:-0.8}"
INFO_NCE_TAU="${INFO_NCE_TAU:-0.05}"
INFO_NCE_TOPK_NEG="${INFO_NCE_TOPK_NEG:-32}"
LOSS_EXCL_W="${LOSS_EXCL_W:-0.6}"
LOSS_INFO_NCE_W="${LOSS_INFO_NCE_W:-1.0}"

EXCL_MARGIN="${EXCL_MARGIN:-0.3}"
EXCL_MODE="${EXCL_MODE:-sim_prob}"
EXCL_VARIANT="${EXCL_VARIANT:-pairwise}"
EXCL_PAIR_TYPE_MODE="${EXCL_PAIR_TYPE_MODE:-neutral}"
EXCL_PAIR_TYPE_FACTOR="${EXCL_PAIR_TYPE_FACTOR:-1.5}"
EXCL_PAIR_NAME_THR="${EXCL_PAIR_NAME_THR:-0.5}"
EXCL_PAIR_REGION_THR="${EXCL_PAIR_REGION_THR:-0.5}"
EXCL_PAIR_EASY_SCALE="${EXCL_PAIR_EASY_SCALE:-0.5}"
EXCL_PAIR_HARD_SCALE="${EXCL_PAIR_HARD_SCALE:-1.5}"

SWEEP_PHASE="${SWEEP_PHASE:-alpha}"
LOG_FILE="${LOG_FILE:-logs/paper_hp_${SWEEP_PHASE}_$(date +%F_%H-%M-%S).log}"
RESUME_SWEEP="${RESUME_SWEEP:-1}"

has_done() {
  local tag="$1"
  if [[ ! -f "${LOG_FILE}" ]]; then
    return 1
  fi
  grep -Fxq "[DONE-RUN] ${tag}" "${LOG_FILE}" 2>/dev/null
}

run_train_tag() {
  local tag="$1"
  shift
  if [[ "${RESUME_SWEEP}" == "1" ]] && has_done "${tag}"; then
    echo "[SKIP] ${tag}" | tee -a "${LOG_FILE}"
    return 0
  fi

  local save_best_this="${SAVE_BEST}"
  local save_path_this=""
  if [[ "${SAVE_EACH}" == "1" ]]; then
    save_best_this=1
    safe_tag="${tag//[^a-zA-Z0-9._-]/_}"
    save_path_this="${SCRIPT_DIR}/runs/paper_hp_${safe_tag}.pt"
  fi

  echo "============================================================" | tee -a "${LOG_FILE}"
  echo "[RUN] ${tag}" | tee -a "${LOG_FILE}"
  echo "============================================================" | tee -a "${LOG_FILE}"

  env \
    MODEL_NAME="${MODEL_NAME}" \
    TRAIN_JSON="${TRAIN_JSON}" \
    DEV_JSON="${DEV_JSON}" \
    TEST_JSON="${TEST_JSON}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
    LR="${LR}" \
    SEED="${SEED}" \
    TEMPERATURE="${TEMPERATURE}" \
    INFO_NCE_TAU="${INFO_NCE_TAU}" \
    INFO_NCE_TOPK_NEG="${INFO_NCE_TOPK_NEG}" \
    LOSS_EXCL_W="${LOSS_EXCL_W}" \
    LOSS_INFO_NCE_W="${LOSS_INFO_NCE_W}" \
    EXCL_MARGIN="${EXCL_MARGIN}" \
    EXCL_MODE="${EXCL_MODE}" \
    EXCL_VARIANT="${EXCL_VARIANT}" \
    EXCL_PAIR_TYPE_MODE="${EXCL_PAIR_TYPE_MODE}" \
    EXCL_PAIR_TYPE_FACTOR="${EXCL_PAIR_TYPE_FACTOR}" \
    EXCL_PAIR_NAME_THR="${EXCL_PAIR_NAME_THR}" \
    EXCL_PAIR_REGION_THR="${EXCL_PAIR_REGION_THR}" \
    EXCL_PAIR_EASY_SCALE="${EXCL_PAIR_EASY_SCALE}" \
    EXCL_PAIR_HARD_SCALE="${EXCL_PAIR_HARD_SCALE}" \
    SAVE_BEST="${save_best_this}" \
    SAVE_PATH="${save_path_this}" \
    "$@" \
    bash run_train_infonce_excl.sh 2>&1 | tee -a "${LOG_FILE}"

  echo "[DONE-RUN] ${tag}" | tee -a "${LOG_FILE}"
}

echo "[START] phase=${SWEEP_PHASE} log=${LOG_FILE}" | tee -a "${LOG_FILE}"

phase_alpha() {
  # β 고정 → α 스윕 (γ = 1-α-β, implicit; EXCL_PAIR_GAMMA 미설정)
  local bet="${FIX_BETA:-0.3}"
  local vals="${SWEEP_ALPHA:-0.2 0.35 0.5 0.65}"
  read -r -a ARR <<< "${vals}"
  for a in "${ARR[@]}"; do
    run_train_tag "alpha_sweep_a${a}_b${bet}" \
      EXCL_PAIR_ALPHA="${a}" EXCL_PAIR_BETA="${bet}"
  done
}

phase_beta() {
  local alf="${FIX_ALPHA:-0.5}"
  local vals="${SWEEP_BETA:-0.1 0.2 0.3 0.4}"
  read -r -a ARR <<< "${vals}"
  for b in "${ARR[@]}"; do
    run_train_tag "beta_sweep_a${alf}_b${b}" \
      EXCL_PAIR_ALPHA="${alf}" EXCL_PAIR_BETA="${b}"
  done
}

phase_gamma() {
  # α,β,γ "비율"을 주면 코드에서 합으로 정규화 → γ 스윕 (논문 Figure용)
  local a="${FIX_ALPHA_RAW:-0.5}"
  local b="${FIX_BETA_RAW:-0.3}"
  local vals="${SWEEP_GAMMA_RAW:-0.1 0.2 0.3 0.5 0.8}"
  read -r -a ARR <<< "${vals}"
  for g in "${ARR[@]}"; do
    run_train_tag "gamma_sweep_raw_a${a}_b${b}_g${g}" \
      EXCL_PAIR_ALPHA="${a}" EXCL_PAIR_BETA="${b}" EXCL_PAIR_GAMMA="${g}"
  done
}

phase_loss_ratio() {
  # λ_nce=1 고정, λ_comp (LOSS_EXCL_W) 스윕 → 비율 해석 쉬움
  local vals="${SWEEP_LOSS_EXCL_W:-0.1 0.3 0.6 1.0 2.0 5.0}"
  read -r -a ARR <<< "${vals}"
  for w in "${ARR[@]}"; do
    run_train_tag "loss_excl_w_${w}_info1" \
      LOSS_EXCL_W="${w}" LOSS_INFO_NCE_W=1.0
  done
}

phase_joint_temp() {
  # Joint competition temperature τ (논문 예시: 0.01~1; 0.01은 매우 sharp 해서 불안정할 수 있음)
  local vals="${SWEEP_JOINT_TEMP:-0.01 0.05 0.1 0.5 1.0}"
  read -r -a ARR <<< "${vals}"
  for t in "${ARR[@]}"; do
    run_train_tag "joint_temp_${t}" \
      TEMPERATURE="${t}"
  done
}

phase_nce_tau() {
  local vals="${SWEEP_NCE_TAU:-0.01 0.03 0.05 0.07 0.1}"
  read -r -a ARR <<< "${vals}"
  for tau in "${ARR[@]}"; do
    run_train_tag "nce_tau_${tau}" \
      INFO_NCE_TAU="${tau}"
  done
}

case "${SWEEP_PHASE}" in
  alpha) phase_alpha ;;
  beta) phase_beta ;;
  gamma) phase_gamma ;;
  loss_ratio) phase_loss_ratio ;;
  joint_temp) phase_joint_temp ;;
  nce_tau) phase_nce_tau ;;
  all)
    phase_alpha
    phase_beta
    phase_gamma
    phase_loss_ratio
    phase_joint_temp
    phase_nce_tau
    ;;
  *)
    echo "Unknown SWEEP_PHASE=${SWEEP_PHASE}. Use: alpha|beta|gamma|loss_ratio|joint_temp|nce_tau|all"
    exit 1
    ;;
esac

echo "[DONE] phase=${SWEEP_PHASE} log=${LOG_FILE}" | tee -a "${LOG_FILE}"
