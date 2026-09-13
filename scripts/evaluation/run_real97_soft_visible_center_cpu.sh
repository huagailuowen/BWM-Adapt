#!/usr/bin/env bash
#SBATCH --job-name=real97-soft-visible-center
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/real97-soft-visible-center-%j.out
#SBATCH --error=logs/real97-soft-visible-center-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/replay_real97_soft_visible_center_cpu.py \
  --config configs/evaluation/real97_soft_dino_visible_center.json \
  --source-run "${1:?Source run required}" "${@:2}"
