#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/soft-static9-score-%A_%a.out
#SBATCH --error=logs/soft-static9-score-%A_%a.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:?mode}" in
  track)
    export CUDA_VISIBLE_DEVICES=""
    exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_soft_static9_standard_lora.py track --shard "$SLURM_ARRAY_TASK_ID" ;;
  images)
    .venv/bin/python scripts/check_real97_cuda.py --expected 1
    exec .venv/bin/python scripts/evaluation/score_real97_soft_static9_standard_lora.py images ;;
  aggregate)
    export CUDA_VISIBLE_DEVICES=""
    exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real97_soft_static9_standard_lora.py aggregate ;;
  *) exit 2 ;;
esac
