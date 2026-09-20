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
#SBATCH --output=logs/real97-simlr-%j.out
#SBATCH --error=logs/real97-simlr-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?Compute allocation required}"
TASK=${1:?door, ball, stick, or soft}
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/check_real97_cuda.py --expected 1
CONFIG="configs/evaluation/real97_${TASK}_ours_simlr_mean_20260918_v1.json"
case "$TASK" in
  door|ball)
    .venv/bin/python scripts/evaluation/prepare_real97_dual6500_cluster_reference.py --config "$CONFIG" --task "$TASK"
    exec .venv/bin/python scripts/evaluation/infer_real97_dual_latent_reference.py \
      --config "outputs/evaluation_real97_${TASK}_dual6500_simlr_cluster_20260918_v1/resolved_config.json" \
      --task "$TASK" --output "outputs/infer_real97_${TASK}_dual6500_simlr_cluster_20260918_v1"
    ;;
  soft)
    exec .venv/bin/python scripts/evaluation/infer_real97_soft_family_mean_static.py \
      --config "$CONFIG" --output outputs/infer_real97_soft_5500_simlr_family_20260918_v1
    ;;
  stick)
    exec .venv/bin/python scripts/evaluation/infer_real916_stick_simlr_globalmean.py \
      --config "$CONFIG" --method ours
    ;;
  *) exit 2 ;;
esac
