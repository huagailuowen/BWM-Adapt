#!/usr/bin/env bash
#SBATCH --job-name=real97-soft-dino-video
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/real97-soft-dino-video-%A_%a.out
#SBATCH --error=logs/real97-soft-dino-video-%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "${1:-track}" == "aggregate" ]]; then
  exec .venv-real97-eval-20260909/bin/python scripts/evaluation/track_real97_soft_dino_cpu.py \
    --config configs/evaluation/real97_soft_dino_fullvideo.json --run "${2:?Run directory required}" --aggregate-only
else
  exec .venv-real97-eval-20260909/bin/python scripts/evaluation/track_real97_soft_dino_cpu.py \
    --config configs/evaluation/real97_soft_dino_fullvideo.json \
    --run "outputs/evaluation_real97_dataset_audit_20260909/dino_fullvideo_job${SLURM_ARRAY_JOB_ID}" \
    --shard "${SLURM_ARRAY_TASK_ID}"
fi
