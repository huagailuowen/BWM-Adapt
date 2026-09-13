#!/usr/bin/env bash
#SBATCH --job-name=stick-visible-contact
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/stick-visible-contact-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/stick-visible-contact-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-real97-eval-20260909/bin/python scripts/evaluation/audit_stick_visible_contact_cpu.py \
  --workers 4 --output "outputs/evaluation_stick_visible_contact_job${SLURM_JOB_ID}"
