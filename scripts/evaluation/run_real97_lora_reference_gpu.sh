#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --job-name=real97-lora-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/real97-lora-reference-%j.out
#SBATCH --error=logs/real97-lora-reference-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
TASK="${1:?door or ball required}"
case "$TASK" in door|ball) ;; *) exit 2 ;; esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_real97_cuda.py --expected 1
exec .venv/bin/python scripts/evaluation/infer_real97_lora_reference.py --config configs/evaluation/real97_ball_door_lora_reference_20260914_v1.json --task "$TASK" --output "outputs/infer_real97_${TASK}_lora_tta_standard5500_reference_job${SLURM_JOB_ID}"
