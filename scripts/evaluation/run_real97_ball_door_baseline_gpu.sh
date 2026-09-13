#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
TASK=${1:?door or ball required}
METHOD=${2:?standard or dino required}
CONFIG=configs/evaluation/real97_ball_door_baselines_20260912_v1.json
RUN="$ROOT/outputs/infer_real97_${TASK}_${METHOD}_step5500_reference_job${SLURM_JOB_ID}"
mkdir -p "$RUN" logs/slurm
exec > >(tee -a "$RUN/inference.log") 2>&1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BWM_REQUIRE_CUDA=1 BWM_EXPECTED_WORLD_SIZE=1
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/infer_real97_ball_door_baselines.py \
  --task "$TASK" --method "$METHOD" --config "$CONFIG" --output "$RUN"
