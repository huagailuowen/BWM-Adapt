#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/real97-soft-static-start-%j.out
#SBATCH --error=logs/real97-soft-static-start-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?compute allocation required}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
nvidia-smi
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/infer_real97_standard_reference.py \
  --task soft --episode-start-history-padding \
  --config configs/evaluation/real97_standard_reference_20260911.json \
  --output "outputs/infer_real97_soft_standard_step5500_static_start_job${SLURM_JOB_ID}"
