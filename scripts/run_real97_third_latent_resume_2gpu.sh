#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
TASK=${1:?door or ball}
case "$TASK" in door|ball) ;; *) exit 2 ;; esac
CONFIG="configs/train/train_real97_${TASK}_third_latent_resume6500_600steps_6x10_2gpu_20260918_v1.yaml"
RUN="$ROOT/outputs/real97_${TASK}_third_latent_resume6500_600steps_6x10_20260918_v1_job${SLURM_JOB_ID}"
mkdir "$RUN"
exec > >(tee "$RUN/train.log") 2>&1
nvidia-smi
.venv/bin/python scripts/check_real97_cuda.py --expected 2
START=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* StartTime=\([^ ]*\).*/\1/p')
export BWM_WALLCLOCK_CHECKPOINT_AT=$(( $(date -d "$START" +%s) + 84600 ))
export BWM_REQUIRE_CUDA=1 BWM_EXPECTED_WORLD_SIZE=2
export BWM_LOCAL_CHECKPOINT_ROOT="/tmp/${USER}/bwm_grouped_checkpoints/${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 NCCL_DEBUG=WARN
export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/scripts/evaluation${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/prepare_real97_third_latent_resume.py --config "$CONFIG" --output "$RUN"
exec .venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="$SLURM_JOB_ID" \
  scripts/train_real97_third_latent_resume.py --config "$RUN/runtime.yaml" --find_unused_parameters
