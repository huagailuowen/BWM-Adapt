#!/usr/bin/env bash
#SBATCH --job-name=stick-action-precision
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stick-action-precision-%j.out
#SBATCH --error=logs/stick-action-precision-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_stick_action_precision_cpu.py \
  --config configs/evaluation/real97_stick_action_precision_20260913_v1.json \
  --workers 8 --output "outputs/eval_real97_stick_action_precision_job${SLURM_JOB_ID}"
