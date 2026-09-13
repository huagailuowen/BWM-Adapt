#!/usr/bin/env bash
#SBATCH --job-name=real97-dino-color
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/real97-dino-color-%j.out
#SBATCH --error=logs/real97-dino-color-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/filter_real97_open_vocab_cpu.py \
  --config configs/evaluation/real97_soft_open_vocab_color_filter.json \
  --output "outputs/evaluation_real97_dataset_audit_20260909/dino_color_job${SLURM_JOB_ID}"
