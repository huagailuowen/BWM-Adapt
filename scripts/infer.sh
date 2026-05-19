#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_CONFIG="${LOCAL_CONFIG:-${SCRIPT_DIR}/local.sh}"

if [[ -f "${LOCAL_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG}"
fi

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-configs/infer/robotwin_ti2v_720p.yaml}"
MODEL_PATH="${MODEL_PATH:-/path/to/Wan2.2-TI2V-5B}"
CKPT_PATH="${CKPT_PATH:-ckpt/5_15_chunk_57_720p_2k_200ood/step-12000/step-12000.safetensors}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-/path/to/RoboTwin2.0_lerobot}"
METADATA_PATH="${METADATA_PATH:-/path/to/episodes_val_720p_test0.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-/path/to/RoboTwin2.0_lerobot/stat.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/robotwin_infer}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/infer.py \
  --config "${CONFIG_PATH}" \
  --model_path "${MODEL_PATH}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --metadata_path "${METADATA_PATH}" \
  --action_stat_path "${ACTION_STAT_PATH}" \
  --output_path "${OUTPUT_DIR}" \
  --max_samples "${MAX_SAMPLES}"
