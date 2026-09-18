#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/dual6500-cluster-shared-noise-%j.out
#SBATCH --error=logs/dual6500-cluster-shared-noise-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
TASK=${1:?door or ball}
case "$TASK" in door|ball) ;; *) exit 2 ;; esac
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUTPUT="outputs/infer_real97_${TASK}_dual6500_cluster_shared_noise05_20260917_v1"
mkdir -p "$OUTPUT"
exec 8>"$OUTPUT/.shared_noise_run.lock"
flock -n 8
.venv/bin/python scripts/check_real97_cuda.py --expected 1
CONFIG="configs/evaluation/real97_${TASK}_dual6500_cluster_shared_noise05_20260917_v1.json"
.venv/bin/python scripts/evaluation/prepare_real97_dual6500_cluster_reference.py --config "$CONFIG" --task "$TASK"
RESOLVED="outputs/evaluation_real97_${TASK}_dual6500_cluster_shared_noise05_20260917_v1/resolved_config.json"
exec .venv/bin/python scripts/evaluation/infer_real97_dual_latent_reference.py \
  --config "$RESOLVED" --task "$TASK" --output "$OUTPUT"
