#!/usr/bin/env bash
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
test "${SLURM_JOB_PARTITION:-}" = yejin-interactive
CACHE=/tmp/$USER/bwm_shared
mkdir -p "$CACHE" logs/methods
(
  flock 9
  if [[ ! -s "$CACHE/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors" || ! -s "$CACHE/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ]]; then
    mkdir -p "$CACHE/Wan2.2-TI2V-5B"
    rsync -aL --partial models/Wan2.2-TI2V-5B/ "$CACHE/Wan2.2-TI2V-5B/"
  fi
  if [[ ! -s "$CACHE/BLM-step-12000.safetensors" ]]; then
    rsync -aL ckpt/BLM/step-12000.safetensors "$CACHE/BLM-step-12000.safetensors.partial"
    mv "$CACHE/BLM-step-12000.safetensors.partial" "$CACHE/BLM-step-12000.safetensors"
  fi
) 9>"$CACHE/staging.lock"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
timeout 12m .venv/bin/python scripts/methods/probe_ttt_full6_pure_checkpoint.py --mode baseline --updates 1 --environments 1 > "logs/methods/ttt-full6-probe-${SLURM_JOB_ID}-baseline.log" 2>&1
printf '[baseline exit] %s\n' "$?"
timeout 40m .venv/bin/python scripts/methods/probe_ttt_full6_pure_checkpoint.py --mode pure --updates 2 --environments 4 > "logs/methods/ttt-full6-probe-${SLURM_JOB_ID}-pure.log" 2>&1
printf '[pure exit] %s\n' "$?"
