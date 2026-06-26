#!/usr/bin/env bash
set -u

REPO_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/boundless-world-model"
CY_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy"
HOLD_SESSION="hold_gpu23"
CONFIG_PATH="${CONFIG_PATH:-configs/train/train_push_box_medium_c_stage2_ttt_10k_2gpu.yaml}"
SAMPLE_INDICES="${SAMPLE_INDICES:-34,2,40,10,20,52,99,80,70,90}"
RUN_NAME="${RUN_NAME:-push_box_medium_c_stage2_ttt_10k_test_inner5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${RUN_NAME}}"
STAGE2_CKPT="${STAGE2_CKPT:-outputs/push_box_medium_c_stage2_ttt_10k_2gpu/step-10000.safetensors}"
BASELINE_DIR="${BASELINE_DIR:-outputs/push_box_medium_c_stage1_10k_4gpu_test_more/finetuned}"

mkdir -p "${REPO_DIR}/logs" "${CY_DIR}/logs"
TEST_LOG="${REPO_DIR}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

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
  echo "[guard] test command exited with status ${status}"
  start_holders_if_missing
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

start_holders_if_missing
cd "${REPO_DIR}" || exit 1
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[guard] launching stage2 TTT test on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[guard] samples: ${SAMPLE_INDICES}"
echo "[guard] log: ${TEST_LOG}"
set +e
python -u scripts/infer_stage2_ttt.py \
  --config "${CONFIG_PATH}" \
  --dataset_metadata_path "data/push_box_bwm_calibrated_v2_100pairs/test.jsonl" \
  --support_metadata_path "data/push_box_bwm_calibrated_v2_100pairs/train.jsonl" \
  --stage2_ckpt_path "${STAGE2_CKPT}" \
  --output_path "${OUTPUT_ROOT}/stage2_ttt" \
  --comparison_output_path "${OUTPUT_ROOT}/comparison_videos" \
  --baseline_pred_dir "${BASELINE_DIR}" \
  --sample_indices "${SAMPLE_INDICES}" \
  --support_count 2 \
  --skip_existing \
  2>&1 | tee "${TEST_LOG}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
