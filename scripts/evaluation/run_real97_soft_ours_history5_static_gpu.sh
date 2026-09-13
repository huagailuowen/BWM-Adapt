#!/usr/bin/env bash
#SBATCH --job-name=soft-ours-static5300
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/soft-ours-static5300-%j.out
#SBATCH --error=logs/soft-ours-static5300-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?compute allocation required}"
OUTPUT=${1:-outputs/infer_real97_soft_ours_history5_static_step5300_job${SLURM_JOB_ID}}
printf '[inference] job=%s output=%s; reuse completed contexts and variants\n' "$SLURM_JOB_ID" "$OUTPUT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/infer_real97_soft_ours_history5_static.py \
  --config configs/evaluation/real97_soft_ours_history5_static_step5300_v1.json \
  --output "$OUTPUT"
