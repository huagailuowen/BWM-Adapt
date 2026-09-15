#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=soft-family9-compare
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/soft-family9-compare-%A_%a.out
#SBATCH --error=logs/soft-family9-compare-%A_%a.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
test -n "$SLURM_JOB_ID"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
if [[ "$1" == aggregate ]]; then
 exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real97_soft_family_mean_comparison.py \
  --config configs/evaluation/real97_soft_family_mean_comparison_20260914_v1.json --aggregate-only
else
 exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real97_soft_family_mean_comparison.py \
  --config configs/evaluation/real97_soft_family_mean_comparison_20260914_v1.json --shard "$SLURM_ARRAY_TASK_ID"
fi
