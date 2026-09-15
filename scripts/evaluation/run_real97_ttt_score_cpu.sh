#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=real97-ttt-score-cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/real97-ttt-score-cpu-%j.out
#SBATCH --error=logs/real97-ttt-score-cpu-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute node required}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_ttt_reference.py --config configs/evaluation/real97_ttt_reference_scores_20260914_v1.json --mode cpu --workers 12
