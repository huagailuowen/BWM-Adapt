#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "${1:?mode}" in
 soft-track)
 export CUDA_VISIBLE_DEVICES=""
 exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_soft_ttt_matched.py track --shard "$SLURM_ARRAY_TASK_ID" ;;
 soft-aggregate)
 export CUDA_VISIBLE_DEVICES=""
 exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_soft_ttt_matched.py aggregate ;;
 door-cpu)
 export CUDA_VISIBLE_DEVICES=""
 exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_door_dual6000.py --config configs/evaluation/real97_door_dual6000_scores_20260917_v1.json --mode cpu --workers 8 ;;
 door-lpips)
 exec .venv/bin/python scripts/evaluation/score_real97_door_dual6000.py --config configs/evaluation/real97_door_dual6000_scores_20260917_v1.json --mode lpips ;;
 *) exit 2 ;;
esac
