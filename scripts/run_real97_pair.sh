#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:4
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err

set -euo pipefail
REPO=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$REPO"
CONFIG_A=${1:?first experiment config required}
CONFIG_B=${2:-}
EXPECTED_GPUS=2
if [[ -n "$CONFIG_B" ]]; then EXPECTED_GPUS=4; fi
CACHE=/tmp/${USER}/bwm_shared_cache
mkdir -p "$CACHE" logs/slurm tmp/run_configs

# Use allocation start, not model-load completion or the first optimizer step.
JOB_INFO=$(scontrol show job "$SLURM_JOB_ID" -o)
START_TEXT=${JOB_INFO#* StartTime=}
START_TEXT=${START_TEXT%% *}
START_EPOCH=$(date -d "$START_TEXT" +%s)
export BWM_WALLCLOCK_CHECKPOINT_AT=$((START_EPOCH + 84600))
printf '[allocation] job=%s start=%s paired_checkpoint_epoch=%s\n' "$SLURM_JOB_ID" "$START_TEXT" "$BWM_WALLCLOCK_CHECKPOINT_AT"

# Run before any expensive checkpoint/data copies. A bad device must not turn
# this GPU allocation into hours of silent CPU training.
.venv/bin/python scripts/check_real97_cuda.py --expected "$EXPECTED_GPUS"
export BWM_REQUIRE_CUDA=1
export BWM_EXPECTED_WORLD_SIZE=2

# Reusable immutable base weights; do not remove other runs' caches.
(
  flock 9
  if [[ ! -f "$CACHE/Wan2.2-TI2V-5B/.copy_complete" ]]; then
    mkdir -p "$CACHE/Wan2.2-TI2V-5B"
    rsync -a models/Wan2.2-TI2V-5B/ "$CACHE/Wan2.2-TI2V-5B/"
    touch "$CACHE/Wan2.2-TI2V-5B/.copy_complete"
  fi
) 9>"$CACHE/Wan2.2-TI2V-5B.lock"
(
  flock 8
  if [[ ! -s "$CACHE/ckpt/BLM/step-12000.safetensors" || ! -f "$CACHE/ckpt/BLM/step-12000.safetensors.copy_complete" ]]; then
    mkdir -p "$CACHE/ckpt/BLM"
    rsync -a ckpt/BLM/step-12000.safetensors "$CACHE/ckpt/BLM/"
    touch "$CACHE/ckpt/BLM/step-12000.safetensors.copy_complete"
  fi
) 8>"$CACHE/ckpt.lock"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}"
if [[ ${#GPU_IDS[@]} -ne "$EXPECTED_GPUS" ]]; then
  printf 'Expected %s allocated GPU identifiers, got %s\n' "$EXPECTED_GPUS" "${GPU_IDS[*]}" >&2
  exit 2
fi

run_experiment() (
  set -euo pipefail
  local slot=$1 config=$2 devices=$3
  local run_root="$REPO/outputs/real97_$(basename "$config" .yaml)_job${SLURM_JOB_ID}"
  mkdir "$run_root"
  exec >"$run_root/train.log" 2>&1
  local runtime="$run_root/runtime.yaml"
  export CUDA_VISIBLE_DEVICES="$devices"
  .venv/bin/python scripts/check_real97_cuda.py --expected 2
  .venv/bin/python - "$config" "$runtime" "$run_root" "$CACHE" <<'PY'
import hashlib, json, os, pathlib, shutil, subprocess, sys, fcntl
import yaml
config_path, runtime, output, cache = map(pathlib.Path, sys.argv[1:])
cfg = yaml.safe_load(config_path.read_text())
flat = {k: v for section in cfg.values() if isinstance(section, dict) for k, v in section.items()}
source = pathlib.Path(flat['dataset_base_path'])
manifest = pathlib.Path(flat['dataset_metadata_path']).parent
summary = json.loads((manifest / 'manifest_summary.json').read_text())
tag = summary['task'] + '_' + summary['version'] + '_' + summary['train_manifest_sha256'][:12]
local = cache / 'datasets_real' / tag
local.parent.mkdir(parents=True, exist_ok=True)
with (local.parent / (tag + '.lock')).open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (local / '.copy_complete').exists():
        local.mkdir(exist_ok=True)
        subprocess.run(['rsync', '-a', '--files-from=' + str(manifest / 'stage_files.txt'),
                        str(source) + '/', str(local) + '/'], check=True)
        (local / '.copy_complete').touch()
frozen = output / 'input_manifest'
shutil.copytree(manifest, frozen)
shutil.copy2(config_path, output / 'submitted_config.yaml')
flat.update(dataset_base_path=str(local), dataset_metadata_path=str(frozen / 'train.jsonl'),
            action_stat_path=str(frozen / 'action_stats.json'), output_path=str(output),
            model_paths=str(cache / 'Wan2.2-TI2V-5B'),
            ckpt_path=str(cache / 'ckpt/BLM/step-12000.safetensors'))
runtime.write_text(yaml.safe_dump({'training': flat}, sort_keys=False))
(output / 'allocation.json').write_text(json.dumps({
    'job_id': os.environ['SLURM_JOB_ID'], 'wallclock_checkpoint_epoch': os.environ['BWM_WALLCLOCK_CHECKPOINT_AT'],
    'original_dataset': str(source), 'local_dataset': str(local), 'original_config': str(config_path),
}, indent=2))
print('[prepared]', runtime, 'action=', flat['action_type'], 'local_dataset=', local, flush=True)
PY
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=16 NCCL_DEBUG=WARN
  export BWM_LOCAL_CHECKPOINT_ROOT="/tmp/${USER}/bwm_grouped_checkpoints/${SLURM_JOB_ID}/${slot}"
  printf '[training] slot=%s visible_gpus=%s output=%s\n' "$slot" "$devices" "$run_root"
  exec .venv/bin/python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="${SLURM_JOB_ID}_${slot}" \
    scripts/train_stage1_grouped_context.py --config "$runtime" --find_unused_parameters
)

# Separate process groups/rendezvous/output. A child failure does not kill its peer.
run_experiment A "$CONFIG_A" "${GPU_IDS[0]},${GPU_IDS[1]}" &
PID_A=$!
if [[ -z "$CONFIG_B" ]]; then
  printf '[single] pid=%s; one independent 2-GPU trainer\n' "$PID_A"
  wait "$PID_A"
  exit $?
fi
run_experiment B "$CONFIG_B" "${GPU_IDS[2]},${GPU_IDS[3]}" &
PID_B=$!
printf '[pair] pid_A=%s pid_B=%s; independent 2-GPU trainers\n' "$PID_A" "$PID_B"
set +e
wait "$PID_A"; STATUS_A=$?
printf '[pair] A exit=%s; B is not cancelled\n' "$STATUS_A"
wait "$PID_B"; STATUS_B=$?
printf '[pair] B exit=%s\n' "$STATUS_B"
[[ "$STATUS_A" -eq 0 && "$STATUS_B" -eq 0 ]]
