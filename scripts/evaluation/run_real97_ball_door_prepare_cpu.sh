#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/real97-ball-door-prepare-%j.out
#SBATCH --error=logs/real97-ball-door-prepare-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/prepare_real97_ball_door_reference_cpu.py \
  --task "${1:?task required}" \
  --config configs/evaluation/real97_ball_door_ours_reference_20260911_v1.json \
  --output "outputs/evaluation_real97_ball_door_ours_20260911/${1}_prep_job${SLURM_JOB_ID}"
