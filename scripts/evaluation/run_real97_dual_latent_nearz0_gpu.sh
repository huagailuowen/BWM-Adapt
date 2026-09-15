#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/real97-dual-nearz0-infer-%j.out
#SBATCH --error=logs/real97-dual-nearz0-infer-%j.err

set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
: "${SLURM_JOB_ID:?Compute allocation required}"
TASK="${1:?door or ball required}"
case "$TASK" in door|ball) ;; *) exit 2 ;; esac
CONFIG=configs/evaluation/real97_ball_door_dual_latent_nearz0_20260913_v1.json
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$ROOT/.venv/bin/python" scripts/check_real97_cuda.py --expected 1
exec "$ROOT/.venv/bin/python" scripts/evaluation/infer_real97_dual_latent_nearz0.py \
  --config "$CONFIG" --task "$TASK" \
  --output "outputs/infer_real97_${TASK}_dual_z0_noise05_step5500_job${SLURM_JOB_ID}"
