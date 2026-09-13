#!/usr/bin/env bash
#SBATCH --job-name=stick-light-ours-std
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/stick-light-ours-std-%j.out
#SBATCH --error=logs/stick-light-ours-std-%j.err

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
PAIR="outputs/infer_real97_stick_light_aug_pair_job${SLURM_JOB_ID}"
mkdir -p "$PAIR"
exec 8>"$PAIR/.pair.lock"
flock -n 8
status=0
for method in standard ours; do
  printf '[method_start] %s\n' "$method"
  if "$ROOT/.venv/bin/python" scripts/evaluation/infer_real97_stick_paired_light_augmentation.py \
      --config configs/evaluation/real97_stick_paired_light_augmentation_20260912_v1.json \
      --method "$method" --pair-root "$PAIR"; then
    printf '[method_complete] %s\n' "$method"
  else
    code=$?
    printf '[method_failed] %s exit=%s; preserving completed files\n' "$method" "$code" >&2
    status=1
  fi
done
exit "$status"
