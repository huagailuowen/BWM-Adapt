#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stick916-visible-tail-%j.out
#SBATCH --error=logs/stick916-visible-tail-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real916_stick_visible_tail.py --config configs/evaluation/real916_stick_visible_tail_20260917_v2.json
