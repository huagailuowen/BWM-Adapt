#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --job-name=ev80-zlr-ablate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6%2
#SBATCH --requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%A_%a.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%A_%a.err

set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

exec .venv/bin/python scripts/methods/run_event80_inner_schedule_ablation.py \
  --config configs/evaluation/inference/event80/ours_step7272_inner_schedule_ablation.yaml \
  --variant-index "${SLURM_ARRAY_TASK_ID}"
