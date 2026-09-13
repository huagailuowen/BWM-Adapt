#!/usr/bin/env bash
#SBATCH --job-name=ours-soft-static-1frame
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
python3 scripts/prepare_real97_soft_static_1frame.py \
    --contract configs/train/real97_soft_eef_static_1frame_data_contract_20260911_v1.json
export BWM_GROUPED_TRAIN_ENTRYPOINT=scripts/train_real97_soft_static_1frame_grouped_context.py
# This existing launcher supports one independent two-GPU worker and computes
# the save deadline from Slurm StartTime + 84600, including preparation/loading.
exec bash scripts/run_real97_pair.sh \
    configs/train/train_real97_soft_eef_static_1frame_c32_2gpu_20260911_v1.yaml
