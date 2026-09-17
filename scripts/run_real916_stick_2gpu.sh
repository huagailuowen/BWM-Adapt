#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
METHOD=${1:?ours or standard}
case "$METHOD" in ours|standard) ;; *) exit 2 ;; esac
export BWM_STICK916_METHOD="$METHOD"
export BWM_GROUPED_TRAIN_ENTRYPOINT=scripts/train_real916_stick.py
export PYTHONUNBUFFERED=1
nvidia-smi
.venv/bin/python scripts/prepare_real916_stick.py
# Shared launcher stages immutable data/Wan/BLM to node-local storage and
# derives the 23h30 deadline from the actual Slurm allocation StartTime.
exec bash scripts/run_real97_pair.sh "configs/train/train_real916_stick_${METHOD}_lift15_3x8_2gpu_v1.yaml"

