#!/bin/bash

# Copy this file to scripts/local.sh and fill in machine-local paths.
# scripts/local.sh is ignored by git.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-configs/infer/robotwin_ti2v_720p.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Wan2.2-TI2V-5B}"
CKPT_PATH="${CKPT_PATH:-ckpt/5_15_chunk_57_720p_2k_200ood/step-12000/step-12000.safetensors}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-/path/to/RoboTwin2.0_lerobot}"
METADATA_PATH="${METADATA_PATH:-/path/to/RoboTwin2.0_lerobot/episodes_val_720p_test0.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-/path/to/RoboTwin2.0_lerobot/stat.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/robotwin_infer}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
