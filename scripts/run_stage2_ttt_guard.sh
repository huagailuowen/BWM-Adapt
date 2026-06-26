#!/usr/bin/env bash
set -u

REPO_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/boundless-world-model"
CY_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy"
HOLD_SESSION="hold_gpu23"
CONFIG_PATH="configs/train/train_push_box_medium_c_stage2_ttt.yaml"

mkdir -p "${REPO_DIR}/logs" "${CY_DIR}/logs"
TRAIN_LOG="${REPO_DIR}/logs/stage2_ttt_$(date +%Y%m%d_%H%M%S).log"

start_holders() {
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

stop_holders() {
  if tmux has-session -t "${HOLD_SESSION}" 2>/dev/null; then
    echo "[guard] stopping GPU2/GPU3 holders"
    tmux kill-session -t "${HOLD_SESSION}"
  fi
}

on_exit() {
  status=$?
  echo "[guard] training command exited with status ${status}"
  start_holders
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

stop_holders
cd "${REPO_DIR}" || exit 1
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "[guard] launching stage2 TTT training"
echo "[guard] log: ${TRAIN_LOG}"
set +e
accelerate launch --multi_gpu --num_processes 4 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
  scripts/train_stage2_ttt.py \
  --config "${CONFIG_PATH}" 2>&1 | tee "${TRAIN_LOG}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
