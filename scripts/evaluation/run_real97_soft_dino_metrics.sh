#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
case "${1:?track, image or aggregate}" in
 track)
  export CUDA_VISIBLE_DEVICES=""
  exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_soft_dino_matched.py track --shard "${SLURM_ARRAY_TASK_ID:?}"
  ;;
 image)
  exec .venv/bin/python scripts/evaluation/score_real97_soft_dino_matched.py image
  ;;
 aggregate)
  export CUDA_VISIBLE_DEVICES=""
  exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_soft_dino_matched.py aggregate
  ;;
 *) exit 2 ;;
esac

