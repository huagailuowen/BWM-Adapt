#!/usr/bin/env bash
#SBATCH --job-name=stick-fine-contact
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/stick-fine-contact-%j.out
#SBATCH --error=logs/stick-fine-contact-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Requires a CPU compute allocation}"
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/refine_stick_corner_contact_cpu.py \
  --workers 4 --output "outputs/evaluation_stick_fine_contact_job${SLURM_JOB_ID}"
