#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb-rnd30-pool45
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
CONFIG=$ROOT/configs/train/train_mass_balance_workspace_random_30ratio_train20_noleak_standard_pooled_wm_2gpu_24h.yaml
CACHE_ROOT=/tmp/$USER/bwm_shared_cache
MODEL_SOURCE=$ROOT/models/Wan2.2-TI2V-5B
MODEL_LOCAL=$CACHE_ROOT/Wan2.2-TI2V-5B
CKPT_SOURCE=$ROOT/ckpt/BLM/step-12000.safetensors
CKPT_LOCAL=$CACHE_ROOT/ckpt/BLM/step-12000.safetensors
RUN_CONFIG=$ROOT/tmp/run_configs/mass_balance_workspace_random_30ratio_train20_pooled_$SLURM_JOB_ID.yaml

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
mkdir -p "$CACHE_ROOT" "$(dirname "$CKPT_LOCAL")" tmp/run_configs logs/methods outputs

nvidia-smi -L || true
"$ROOT/.venv/bin/python" - <<'PY'
import torch

count = torch.cuda.device_count()
print(f"[gpu-check] torch={torch.__version__} cuda={torch.version.cuda} devices={count}")
if count != 2:
    raise SystemExit(f"Expected 2 allocated GPUs, but PyTorch detected {count}.")
for index in range(count):
    with torch.cuda.device(index):
        value = torch.ones(1, device=f"cuda:{index}")
        if value.item() != 1.0:
            raise SystemExit(f"GPU {index} failed the allocation check.")
torch.cuda.synchronize()
PY

(
  flock 9
  if [[ ! -f "$MODEL_LOCAL/.copy_complete" && ! -f "$MODEL_LOCAL/.bwm_stage_complete" ]]; then
    mkdir -p "$MODEL_LOCAL"
    rsync -aL "$MODEL_SOURCE/" "$MODEL_LOCAL/"
    touch "$MODEL_LOCAL/.copy_complete"
  fi
) 9>"$CACHE_ROOT/Wan2.2-TI2V-5B.lock"

(
  flock 8
  if [[ ! -f "$CKPT_LOCAL.copy_complete" || ! -s "$CKPT_LOCAL" ]]; then
    rsync -a "$CKPT_SOURCE" "$CKPT_LOCAL"
    touch "$CKPT_LOCAL.copy_complete"
  fi
) 8>"$CACHE_ROOT/ckpt.lock"

sed \
  -e "s|models/Wan2.2-TI2V-5B|$MODEL_LOCAL|g" \
  -e "s|ckpt/BLM/step-12000.safetensors|$CKPT_LOCAL|g" \
  -e "s|__JOB_ID__|$SLURM_JOB_ID|g" \
  "$CONFIG" > "$RUN_CONFIG"

PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "[start] train_envs=20 heldout_envs=10 local_batch=32 ranks=2 global_batch=64 max_updates=4500 main_view_only=true"
exec "$ROOT/.venv/bin/python" -m accelerate.commands.launch \
  --num_machines 1 \
  --num_processes 2 \
  --num_cpu_threads_per_process 16 \
  --main_process_port "$PORT" \
  --mixed_precision bf16 \
  "$ROOT/scripts/train.py" \
  --config "$RUN_CONFIG"
