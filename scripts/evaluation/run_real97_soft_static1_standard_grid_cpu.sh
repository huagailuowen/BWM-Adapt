#!/usr/bin/env bash
#SBATCH --job-name=soft-static1-standard-grid
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/soft-static1-standard-grid-%j.out
#SBATCH --error=logs/soft-static1-standard-grid-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/regrid_real97_soft_static1_with_standard.py \
  --ours outputs/infer_real97_soft_ours_static1_step4364_job115499 \
  --standard outputs/infer_real97_soft_standard_step5500_static_start_job114497
