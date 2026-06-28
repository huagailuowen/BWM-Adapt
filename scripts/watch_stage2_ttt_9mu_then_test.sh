#!/usr/bin/env bash
set -u

REPO_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/boundless-world-model"
CY_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy"
HOLD_SESSION="${HOLD_SESSION:-hold_gpu23}"
TRAIN_SESSION="${TRAIN_SESSION:-bwm_stage2_ttt_9mu_10k_4gpu}"

CONFIG_PATH="${CONFIG_PATH:-configs/train/train_push_box_9mu_medium_c_stage2_ttt_10k_4gpu.yaml}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/data/libero_push_box_friction_9mu_450}"
TRAIN_METADATA_PATH="${TRAIN_METADATA_PATH:-data/push_box_bwm_friction_9mu_450/train.jsonl}"
TEST_METADATA_PATH="${TEST_METADATA_PATH:-data/push_box_bwm_friction_9mu_450/test.jsonl}"
STAGE1_CKPT="${STAGE1_CKPT:-outputs/push_box_medium_c_stage1_10k_4gpu/step-10000.safetensors}"
STAGE2_CKPT="${STAGE2_CKPT:-outputs/push_box_9mu_medium_c_stage2_ttt_10k_4gpu/step-10000.safetensors}"

RUN_NAME="${RUN_NAME:-push_box_9mu_medium_c_stage2_ttt_10k_4gpu_test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${RUN_NAME}}"
SAMPLE_INDICES="${SAMPLE_INDICES:-215,29,70,388,151,61,105,88,178,410,98,72,233,335,134,246,3,399}"
SUPPORT_COUNT="${SUPPORT_COUNT:-2}"
TTT_ADAPT_SCOPE="${TTT_ADAPT_SCOPE:-context}"
TTT_ADAPTER_LR="${TTT_ADAPTER_LR:-0.001}"
TTT_ADAPTER_GRAD_CLIP="${TTT_ADAPTER_GRAD_CLIP:-0.1}"
TTT_ADAPTER_REG_WEIGHT="${TTT_ADAPTER_REG_WEIGHT:-0.0001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${REPO_DIR}/logs" "${CY_DIR}/logs"
POST_LOG="${REPO_DIR}/logs/${RUN_NAME}_posttrain_$(date +%Y%m%d_%H%M%S).log"

start_holders_if_missing() {
  if tmux has-session -t "${HOLD_SESSION}" 2>/dev/null; then
    echo "[guard] holder session ${HOLD_SESSION} already exists"
    return 0
  fi
  echo "[guard] starting GPU2/GPU3 holders"
  tmux new-session -d -s "${HOLD_SESSION}" -n gpu2 \
    "cd '${CY_DIR}' && source '${REPO_DIR}/.venv/bin/activate' && python -u scripts/hold_gpu0_full.py 2 2>&1 | tee logs/hold_gpu2.log"
  tmux new-window -t "${HOLD_SESSION}:" -n gpu3 \
    "cd '${CY_DIR}' && source '${REPO_DIR}/.venv/bin/activate' && python -u scripts/hold_gpu0_full.py 3 2>&1 | tee logs/hold_gpu3.log"
}

on_exit() {
  status=$?
  echo "[guard] post-train test command exited with status ${status}"
  start_holders_if_missing
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

cd "${REPO_DIR}" || exit 1
source .venv/bin/activate

echo "[watch] log: ${POST_LOG}"
echo "[watch] waiting for training tmux session: ${TRAIN_SESSION}"
while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 300
done

start_holders_if_missing

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "[watch] final checkpoint missing: ${STAGE2_CKPT}" | tee -a "${POST_LOG}"
  echo "[watch] training did not complete cleanly enough for post-train testing; holders are running." | tee -a "${POST_LOG}"
  exit 2
fi

export CUDA_VISIBLE_DEVICES
echo "[test] using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${POST_LOG}"
echo "[test] samples: ${SAMPLE_INDICES}" | tee -a "${POST_LOG}"
echo "[test] ttt_adapt_scope: ${TTT_ADAPT_SCOPE}" | tee -a "${POST_LOG}"
echo "[test] output root: ${OUTPUT_ROOT}" | tee -a "${POST_LOG}"

set +e
{
  echo "[test] stage1 baseline inference"
  python -u scripts/infer.py \
    --config "${CONFIG_PATH}" \
    --dataset_metadata_path "${TEST_METADATA_PATH}" \
    --ckpt_path "${STAGE1_CKPT}" \
    --output_path "${OUTPUT_ROOT}/stage1_baseline" \
    --sample_indices "${SAMPLE_INDICES}"
  baseline_status=$?
  echo "[test] stage1 baseline status=${baseline_status}"
  if [[ "${baseline_status}" -ne 0 ]]; then
    exit "${baseline_status}"
  fi

  echo "[test] stage2 TTT inference and three-way comparison"
  python -u scripts/infer_stage2_ttt.py \
    --config "${CONFIG_PATH}" \
    --dataset_metadata_path "${TEST_METADATA_PATH}" \
    --support_metadata_path "${TRAIN_METADATA_PATH}" \
    --stage2_ckpt_path "${STAGE2_CKPT}" \
    --output_path "${OUTPUT_ROOT}/stage2_ttt" \
    --comparison_output_path "${OUTPUT_ROOT}/comparison_videos" \
    --baseline_pred_dir "${OUTPUT_ROOT}/stage1_baseline" \
    --sample_indices "${SAMPLE_INDICES}" \
    --support_count "${SUPPORT_COUNT}" \
    --ttt_adapt_scope "${TTT_ADAPT_SCOPE}" \
    --ttt_adapter_lr "${TTT_ADAPTER_LR}" \
    --ttt_adapter_grad_clip "${TTT_ADAPTER_GRAD_CLIP}" \
    --ttt_adapter_reg_weight "${TTT_ADAPTER_REG_WEIGHT}" \
    --skip_existing
  ttt_status=$?
  echo "[test] stage2 TTT status=${ttt_status}"
  if [[ "${ttt_status}" -ne 0 ]]; then
    exit "${ttt_status}"
  fi

  echo "[test] detailed support/query comparison videos"
  python -u scripts/make_ttt_support_comparison.py \
    --results-path "${OUTPUT_ROOT}/results.jsonl" \
    --train-metadata-path "${TRAIN_METADATA_PATH}" \
    --test-metadata-path "${TEST_METADATA_PATH}" \
    --dataset-base-path "${DATASET_BASE_PATH}" \
    --baseline-pred-dir "${OUTPUT_ROOT}/stage1_baseline" \
    --stage2-pred-dir "${OUTPUT_ROOT}/stage2_ttt" \
    --output-dir "${OUTPUT_ROOT}/comparison_videos_with_support" \
    --contact-sheet-path "${OUTPUT_ROOT}/contact_sheets/overview_18_support_gt_query_gt_stage1_stage2ttt_midframe.png"
  detailed_status=$?
  echo "[test] detailed comparison status=${detailed_status}"
  exit "${detailed_status}"
} 2>&1 | tee -a "${POST_LOG}"
status=${PIPESTATUS[0]}
set -e

exit "${status}"
