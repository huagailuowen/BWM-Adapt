#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=real97-ttt-score-lpips
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/real97-ttt-score-lpips-%j.out
#SBATCH --error=logs/real97-ttt-score-lpips-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute node required}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/score_real97_ttt_reference.py --config configs/evaluation/real97_ttt_reference_scores_20260914_v1.json --mode lpips
