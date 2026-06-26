#!/usr/bin/env bash
set -u

REPO_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/boundless-world-model"
CY_DIR="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy"
HOLD_SESSION="hold_gpu23"
CONFIG_PATH="configs/train/train_push_box_medium_c_stage2_ttt_10k_2gpu.yaml"

mkdir -p "${REPO_DIR}/logs" "${CY_DIR}/logs"
TRAIN_LOG="${REPO_DIR}/logs/stage2_ttt_10k_2gpu_$(date +%Y%m%d_%H%M%S).log"

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
  echo "[guard] training command exited with status ${status}"
  start_holders_if_missing
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

start_holders_if_missing
cd "${REPO_DIR}" || exit 1
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"

echo "[guard] launching 10k stage2 TTT training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[guard] log: ${TRAIN_LOG}"
set +e
accelerate launch --multi_gpu --num_processes 2 --num_cpu_threads_per_process 4 --main_process_port 29601 --mixed_precision bf16 \
  scripts/train_stage2_ttt.py \
  --config "${CONFIG_PATH}" 2>&1 | tee "${TRAIN_LOG}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
