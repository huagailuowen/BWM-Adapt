#!/bin/bash
#SBATCH --job-name=mb-rnd-dino-rk12
#SBATCH --partition=yejin
#SBATCH --qos=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
CACHE_ROOT=/tmp/${USER}/bwm_shared
WAN_SOURCE=${ROOT}/models/Wan2.2-TI2V-5B
WAN_LOCAL=${CACHE_ROOT}/Wan2.2-TI2V-5B
DINO_SOURCE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/checkpoints/dinov2-base
DINO_LOCAL=${CACHE_ROOT}/dinov2-base
BLM_SOURCE=${ROOT}/ckpt/BLM/step-12000.safetensors
BLM_LOCAL=${CACHE_ROOT}/BLM-step-12000.safetensors
CONFIG_TEMPLATE=${ROOT}/configs/train/train_mass_balance_workspace_random_30ratio_train20_mainview_dinov2_random_k1k2_total8_4envpergpu_2gpu_4500.yaml
RUN_CONFIG=${ROOT}/tmp/run_configs/mass_balance_workspace_random_dinov2_random_k1k2_${SLURM_JOB_ID}.yaml

cd "${ROOT}"
mkdir -p "${CACHE_ROOT}" "${ROOT}/logs/methods" "${ROOT}/tmp/run_configs"

(
    flock 9
    if [[ ! -s "${WAN_LOCAL}/diffusion_pytorch_model-00003-of-00003.safetensors" || ! -s "${WAN_LOCAL}/Wan2.2_VAE.pth" ]]; then
        mkdir -p "${WAN_LOCAL}"
        cp -aL "${WAN_SOURCE}/." "${WAN_LOCAL}/"
    fi
    if [[ ! -s "${DINO_LOCAL}/config.json" ]]; then
        mkdir -p "${DINO_LOCAL}"
        cp -aL "${DINO_SOURCE}/." "${DINO_LOCAL}/"
    fi
    if [[ ! -s "${BLM_LOCAL}" ]]; then
        cp -aL "${BLM_SOURCE}" "${BLM_LOCAL}"
    fi
) 9>"${CACHE_ROOT}/staging.lock"

sed \
    -e "s|models/Wan2.2-TI2V-5B|${WAN_LOCAL}|" \
    -e "s|/afs/ir/users/c/y/cyzhou05/TTT-Physics/checkpoints/dinov2-base|${DINO_LOCAL}|" \
    -e "s|ckpt/BLM/step-12000.safetensors|${BLM_LOCAL}|" \
    -e "s|seed_20260902/checkpoints|seed_20260902_job_${SLURM_JOB_ID}/checkpoints|" \
    "${CONFIG_TEMPLATE}" > "${RUN_CONFIG}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
.venv/bin/accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/methods/train_dinov2_event80.py \
    --config "${RUN_CONFIG}"
