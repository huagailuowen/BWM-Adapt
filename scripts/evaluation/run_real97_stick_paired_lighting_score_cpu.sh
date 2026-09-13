#!/usr/bin/env bash
#SBATCH --job-name=stick-object-paired
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stick-object-paired-%j.out
#SBATCH --error=logs/stick-object-paired-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv-real97-eval-20260909/bin/python" scripts/evaluation/score_real97_stick_paired_lighting_cpu.py \
  --workers 8 --output "outputs/eval_real97_stick_ours_standard_light_pair_job${SLURM_JOB_ID}"
