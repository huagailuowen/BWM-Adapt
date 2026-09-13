#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/real97-reference-prepare-%j.out
#SBATCH --error=logs/real97-reference-prepare-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/prepare_real97_reference_trial_cpu.py \
  --task "${1:?task required}" --output "outputs/evaluation_real97_reference_ttt_v3_20260910/${1}_prep_job${SLURM_JOB_ID}"
