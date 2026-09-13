#!/usr/bin/env bash
#SBATCH --job-name=soft-static-metric
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/soft-static-metric-%A_%a.out
#SBATCH --error=logs/soft-static-metric-%A_%a.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?CPU compute allocation required}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == aggregate ]]; then
    exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real97_soft_static_objectcentric_cpu.py \
      --config configs/evaluation/real97_soft_static_objectcentric_v1.json \
      --output "outputs/eval_real97_soft_static_objectcentric_job${2:?source array job required}" --aggregate-only
elif [[ "${1:-}" == resume ]]; then
    exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real97_soft_static_objectcentric_cpu.py \
      --config configs/evaluation/real97_soft_static_objectcentric_v1.json \
      --output "outputs/eval_real97_soft_static_objectcentric_job${2:?original array job required}" \
      --shard "$SLURM_ARRAY_TASK_ID" --resume
else
    exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real97_soft_static_objectcentric_cpu.py \
      --config configs/evaluation/real97_soft_static_objectcentric_v1.json \
      --output "outputs/eval_real97_soft_static_objectcentric_job${SLURM_ARRAY_JOB_ID}" --shard "$SLURM_ARRAY_TASK_ID" --resume
fi
