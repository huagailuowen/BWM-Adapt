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
export BWM_STICK916_BASELINE=${1:?Expected dino or ttt}
ENTRY=scripts/methods/train_real916_stick_baselines.py
case "$BWM_STICK916_BASELINE" in
  dino)
    export BWM_DINO_TRAIN_ENTRYPOINT="$ENTRY"
    exec bash scripts/run_real97_dinov2_2gpu.sh configs/train/train_real916_stick_dinov2_k1k2_3x8_2gpu_v1.yaml
    ;;
  ttt)
    export BWM_TTT_TRAIN_ENTRYPOINT="$ENTRY"
    exec bash scripts/run_real97_ttt_2gpu.sh configs/train/train_real916_stick_ttt_3x2x4_2gpu_v1.yaml
    ;;
  *) printf 'Unknown baseline: %s\n' "$BWM_STICK916_BASELINE" >&2; exit 2 ;;
esac
