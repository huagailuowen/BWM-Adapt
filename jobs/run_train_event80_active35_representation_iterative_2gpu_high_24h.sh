#!/usr/bin/env bash
#SBATCH --job-name=event80-repr-iter
#SBATCH --partition=yejin
#SBATCH --qos=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
case "${1:-}" in
  c4) TEMPLATE=configs/train/train_push_box_event80_iterative_c4_2gpu_24h.yaml ;;
  c8) TEMPLATE=configs/train/train_push_box_event80_iterative_c8_2gpu_24h.yaml ;;
  c64) TEMPLATE=configs/train/train_push_box_event80_iterative_c64_2gpu_24h.yaml ;;
  c128) TEMPLATE=configs/train/train_push_box_event80_iterative_c128_2gpu_24h.yaml ;;
  direct3072) TEMPLATE=configs/train/train_push_box_event80_iterative_direct_token3072_2gpu_24h.yaml ;;
  c32_100_100) TEMPLATE=configs/train/train_push_box_event80_iterative_c32_100_100_2gpu_24h.yaml ;;
  c32_400_400) TEMPLATE=configs/train/train_push_box_event80_iterative_c32_400_400_2gpu_24h.yaml ;;
  *) printf 'usage: %s c4|c8|c64|c128|direct3072|c32_100_100|c32_400_400\n' "$0" >&2; exit 2 ;;
esac

RUN_CONFIG="tmp/run_configs/event80_active35_${1}_iterative_${SLURM_JOB_ID}.yaml"
CACHE_ROOT="/tmp/${USER}/bwm_shared"
WAN_LOCAL="${CACHE_ROOT}/Wan2.2-TI2V-5B"
BLM_SOURCE="${ROOT}/ckpt/BLM/step-12000.safetensors"
BLM_LOCAL="${CACHE_ROOT}/BLM-step-12000.safetensors"
mkdir -p "$CACHE_ROOT" tmp/run_configs logs/methods
# Use the trainer's safe optimizer-boundary checkpoint before the wall-time limit.
export BWM_WALLCLOCK_CHECKPOINT_AT=$(( $(date +%s) + 23 * 3600 ))
(
  flock 9
  complete=true
  for file in diffusion_pytorch_model.safetensors.index.json \
    diffusion_pytorch_model-00001-of-00003.safetensors \
    diffusion_pytorch_model-00002-of-00003.safetensors \
    diffusion_pytorch_model-00003-of-00003.safetensors Wan2.2_VAE.pth; do
    if [[ ! -s "${WAN_LOCAL}/${file}" ]]; then complete=false; fi
  done
  if [[ "$complete" != true ]]; then
    printf '[cache] staging Wan in %s\n' "$WAN_LOCAL"
    mkdir -p "$WAN_LOCAL"
    rsync -aL --partial "${ROOT}/models/Wan2.2-TI2V-5B/" "${WAN_LOCAL}/"
  else
    printf '[cache] reusing %s\n' "$WAN_LOCAL"
  fi
  if [[ ! -s "$BLM_LOCAL" ]] || [[ "$(stat -c %s "$BLM_LOCAL")" != "$(stat -Lc %s "$BLM_SOURCE")" ]]; then
    rsync -aL --partial "$BLM_SOURCE" "${BLM_LOCAL}.partial"
    mv "${BLM_LOCAL}.partial" "$BLM_LOCAL"
  fi
) 9>"${CACHE_ROOT}/staging.lock"

# Prepare a job-specific immutable config and record the exact active pool.
# Read only the pure ordering function, avoiding a second model/torch import.
.venv/bin/python - "$TEMPLATE" "$RUN_CONFIG" "$WAN_LOCAL" "$BLM_LOCAL" <<'PY'
import ast
import json
import os
from pathlib import Path
import sys
import yaml

template, destination, wan, blm = sys.argv[1:]
config = yaml.safe_load(Path(template).read_text())
group = config['grouped_context_stage1']
manifest_path = Path('configs/methods/event80/manifests/ours_88823_step7272_active35.yaml')
manifest = yaml.safe_load(manifest_path.read_text())
tree = ast.parse(Path('scripts/train_stage1_grouped_context.py').read_text())
ordering = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_nested_uniform_group_order')
namespace = {}
exec(compile(ast.Module(body=[ordering], type_ignores=[]), '<curriculum-order>', 'exec'), namespace)
order = namespace[ordering.name](80, group['grouped_context_curriculum_initial_groups'], group['grouped_context_curriculum_total_groups'])
expected = manifest['selection']['active_environment_ids']
if sorted(order) != sorted(expected):
    raise RuntimeError(f'Active pool differs from reference 88823: {order}')
if group['grouped_context_curriculum_variant'] != 'default' or group['grouped_context_curriculum_joint_steps'] != 0:
    raise RuntimeError('This job requires the original alternating curriculum, not joint training.')
config['model']['model_paths'] = wan
config['output']['ckpt_path'] = blm
output = Path(config['output']['output_path'])
output = output.parent.with_name(output.parent.name + '_job_' + os.environ['SLURM_JOB_ID']) / output.name
output.mkdir(parents=True, exist_ok=False)
config['output']['output_path'] = str(output)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False))
(output / 'launch_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
(output / 'active_pool_manifest.json').write_text(json.dumps({
    'reference_manifest': str(manifest_path),
    'active_environment_ids': expected,
    'curriculum_group_order': order,
    'excluded_environment_ids': sorted(set(range(80)) - set(order)),
    'sampling': group['grouped_context_sampling_mode'],
    'world_size': 2,
    'clips_per_rank': 16,
    'effective_clips_per_update': 32,
}, indent=2) + '\n')
print(f'[launch] config={destination} output={output}', flush=True)
print(f'[launch] active35 curriculum order={order}', flush=True)
PY

PORT=$(.venv/bin/python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('', 0))
    print(sock.getsockname()[1])
PY
)
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
nvidia-smi -L
exec .venv/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --num_processes 2 \
  --num_cpu_threads_per_process 8 --main_process_port "$PORT" \
  --mixed_precision bf16 \
  scripts/train_stage1_grouped_context.py --config "$RUN_CONFIG" --find_unused_parameters
