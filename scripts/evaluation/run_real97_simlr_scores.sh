#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/simlr-scores-%A_%a.out
#SBATCH --error=logs/simlr-scores-%A_%a.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE=${1:?scoring task required}
CPU=.venv-real97-eval-20260909/bin/python
if [[ "$MODE" != lpips ]]; then export CUDA_VISIBLE_DEVICES=""; fi
case "$MODE" in
  door|ball)
    exec "$CPU" "scripts/evaluation/score_real97_${MODE}_simlr6500.py" \
      --config "configs/evaluation/real97_${MODE}_simlr_scores_20260918_v1.json" --mode cpu --workers 8 ;;
  soft)
    exec "$CPU" scripts/evaluation/score_real97_soft_simlr.py \
      --config configs/evaluation/real97_soft_simlr_scores_20260918_v1.json --shard "$SLURM_ARRAY_TASK_ID" ;;
  soft-aggregate)
    exec "$CPU" scripts/evaluation/score_real97_soft_simlr.py \
      --config configs/evaluation/real97_soft_simlr_scores_20260918_v1.json --aggregate-only ;;
  stick)
    "$CPU" scripts/evaluation/score_real916_stick_final.py \
      --config configs/evaluation/real97_stick_simlr_scores_20260918_v1.json --mode cpu --workers 8
    exec "$CPU" scripts/evaluation/score_real916_stick_visible_tail.py \
      --config configs/evaluation/real97_stick_simlr_tail_20260918_v1.json ;;
  lpips)
    .venv/bin/python scripts/check_real97_cuda.py --expected 1
    for task in door ball; do
      .venv/bin/python "scripts/evaluation/score_real97_${task}_simlr6500.py" \
        --config "configs/evaluation/real97_${task}_simlr_scores_20260918_v1.json" --mode lpips
    done
    .venv/bin/python scripts/evaluation/score_real916_stick_final.py \
      --config configs/evaluation/real97_stick_simlr_scores_20260918_v1.json --mode lpips
    exec .venv/bin/python scripts/evaluation/summarize_real97_simlr.py --soft-lpips ;;
  summarize)
    exec .venv/bin/python scripts/evaluation/summarize_real97_simlr.py ;;
  *) exit 2 ;;
esac
