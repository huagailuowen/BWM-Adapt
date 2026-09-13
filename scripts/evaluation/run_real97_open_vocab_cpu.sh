#!/usr/bin/env bash
#SBATCH --job-name=real97-open-vocab-probe
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/real97-open-vocab-probe-%j.out
#SBATCH --error=logs/real97-open-vocab-probe-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?CPU compute allocation required}"
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/probe_real97_open_vocab_cpu.py \
  --config configs/evaluation/real97_soft_open_vocab_probe.json \
  --output "outputs/evaluation_real97_dataset_audit_20260909/open_vocab_probe_job${SLURM_JOB_ID}"

