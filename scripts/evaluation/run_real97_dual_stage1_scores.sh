#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dual-stage1-scores-%j.out
#SBATCH --error=logs/dual-stage1-scores-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE=${1:?cpu or lpips}
TASK=${2:?ball or door}
case "$MODE" in
 cpu) export CUDA_VISIBLE_DEVICES=""; PYTHON=.venv-real97-eval-20260909/bin/python ;;
 lpips) PYTHON=.venv/bin/python ;;
 *) exit 2 ;;
esac
exec "$PYTHON" scripts/evaluation/score_real97_dual_stage1.py \
 --config "configs/evaluation/real97_${TASK}_dual6500_stage1_scores_20260918_v1.json" --mode "$MODE" --workers 8
