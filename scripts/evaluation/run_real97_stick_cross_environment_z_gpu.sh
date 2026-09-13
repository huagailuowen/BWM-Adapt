#!/usr/bin/env bash
#SBATCH --job-name=stick-Z-swap
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/stick-Z-swap-%j.out
#SBATCH --error=logs/stick-Z-swap-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?compute allocation required}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/infer_real97_stick_cross_environment_z.py \
  --config configs/evaluation/real97_stick_cross_environment_z_v1.json \
  --output "outputs/infer_real97_stick_cross_environment_Z_job${SLURM_JOB_ID}"
