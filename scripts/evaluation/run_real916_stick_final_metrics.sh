#!/usr/bin/env bash
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
MODE=${1:?cpu or lpips}
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$MODE" in
 cpu) export CUDA_VISIBLE_DEVICES=""; PYTHON=.venv-real97-eval-20260909/bin/python ;;
 lpips) .venv/bin/python scripts/check_real97_cuda.py --expected 1; PYTHON=.venv/bin/python ;;
 *) exit 2 ;;
esac
exec "$PYTHON" scripts/evaluation/score_real916_stick_final.py \
 --config configs/evaluation/real916_stick_final_metrics_20260917_v1.json --mode "$MODE" --workers 8
