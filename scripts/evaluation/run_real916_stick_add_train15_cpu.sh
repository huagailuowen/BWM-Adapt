#!/usr/bin/env bash
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:?prepare or score}" in
 prepare) exec .venv/bin/python scripts/evaluation/prepare_real916_stick_add_train15.py --config configs/evaluation/real916_stick_test45_train30_20260917_v1.json ;;
 score) export CUDA_VISIBLE_DEVICES=""; exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real916_stick_test45_train30.py --config configs/evaluation/real916_stick_test45_train30_20260917_v1.json ;;
 *) exit 2 ;;
esac
