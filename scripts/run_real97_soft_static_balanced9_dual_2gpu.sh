#!/usr/bin/env bash
#SBATCH --job-name=ours-soft-static-balanced9-dualz
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
.venv/bin/python scripts/prepare_real97_soft_static_balanced9_dual.py
export BWM_GROUPED_TRAIN_ENTRYPOINT=scripts/train_real97_soft_static_balanced9_dual.py
exec bash scripts/run_real97_pair.sh \
    configs/train/train_real97_soft_static_balanced9_dual_c32_lr003_2gpu_20260913_v1.yaml
