#!/usr/bin/env bash
#SBATCH --job-name=real97-soft-track-v3
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/real97-soft-track-v3-%j.out
#SBATCH --error=logs/real97-soft-track-v3-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/calibrate_real97_tracking.py \
  --inventory outputs/evaluation_real97_dataset_audit_20260909/episodes.jsonl \
  --calibration configs/evaluation/real97_object_calibration_v3.json \
  --output "outputs/evaluation_real97_dataset_audit_20260909/soft_calibration_v3_job${SLURM_JOB_ID}" \
  --tasks soft --workers 4

