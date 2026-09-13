#!/usr/bin/env bash
#SBATCH --job-name=soft-ours-static1-4364
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/soft-ours-static1-4364-%j.out
#SBATCH --error=logs/soft-ours-static1-4364-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?compute allocation required}"
OUTPUT=${1:-outputs/infer_real97_soft_ours_static1_step4364_job${SLURM_JOB_ID}}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/infer_real97_soft_ours_static1.py \
  --config configs/evaluation/real97_soft_ours_static1_step4364_20260912_v1.json \
  --output "$OUTPUT"
