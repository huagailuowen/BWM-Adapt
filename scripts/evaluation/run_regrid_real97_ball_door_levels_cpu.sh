#!/usr/bin/env bash
#SBATCH --job-name=ball-door-grid-levels
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/ball-door-grid-levels-%j.out
#SBATCH --error=logs/ball-door-grid-levels-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/regrid_real97_ball_door_levels_cpu.py
