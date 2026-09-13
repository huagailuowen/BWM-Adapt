#!/usr/bin/env bash
#SBATCH --job-name=stick-near-table-z
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/stick-near-table-z-%j.out
#SBATCH --error=logs/stick-near-table-z-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?compute allocation required}"
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
OUTPUT="outputs/infer_real97_stick_near_table_unbounded_job${SLURM_JOB_ID}"
mkdir -p "$OUTPUT"
exec 8>"$OUTPUT/.inference.lock"
flock -n 8
exec "$ROOT/.venv/bin/python" scripts/evaluation/infer_real97_stick_near_table.py \
  --config configs/evaluation/real97_stick_near_table_unbounded_20260912_v1.json \
  --output "$OUTPUT"
