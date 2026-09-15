#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err

set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
CONFIG=${1:?DINO training config required}
CACHE=/tmp/${USER}/bwm_shared_cache
RUN="$ROOT/outputs/real97_$(basename "$CONFIG" .yaml)_job${SLURM_JOB_ID}"
mkdir -p "$CACHE" logs/slurm
mkdir "$RUN"
exec > >(tee -a "$RUN/train.log") 2>&1

JOB_INFO=$(scontrol show job "$SLURM_JOB_ID" -o)
START_TEXT=${JOB_INFO#* StartTime=}
START_TEXT=${START_TEXT%% *}
START_EPOCH=$(date -d "$START_TEXT" +%s)
export BWM_DINO_DEADLINE_EPOCH=$((START_EPOCH + 84600))
printf '[allocation] job=%s start=%s checkpoint_epoch=%s output=%s\n' \
  "$SLURM_JOB_ID" "$START_TEXT" "$BWM_DINO_DEADLINE_EPOCH" "$RUN"
nvidia-smi
.venv/bin/python scripts/check_real97_cuda.py --expected 2
.venv/bin/python scripts/prepare_real97_soft_static_balanced9_baselines.py
export BWM_REQUIRE_CUDA=1 BWM_EXPECTED_WORLD_SIZE=2

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

.venv/bin/python - "$CONFIG" "$RUN" "$CACHE" <<'PY'
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

config_path, output, cache = map(Path, sys.argv[1:])
cfg = yaml.safe_load(config_path.read_text())
flat = {key: value for section in cfg.values() if isinstance(section, dict)
        for key, value in section.items()}
source = Path(flat["dataset_base_path"])
manifest = Path(flat["dataset_metadata_path"]).parent
summary = json.loads((manifest / "manifest_summary.json").read_text())
identity = summary.get("source_dataset_cache_identity", summary)
tag = identity["task"] + "_" + identity["version"] + "_" + identity["train_manifest_sha256"][:12]
local = cache / "datasets_real" / tag
local.parent.mkdir(parents=True, exist_ok=True)
with (local.parent / (tag + ".lock")).open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (local / ".copy_complete").exists():
        local.mkdir(exist_ok=True)
        subprocess.run(["rsync", "-a", "--files-from=" + str(manifest / "stage_files.txt"),
                        str(source) + "/", str(local) + "/"], check=True)
        (local / ".copy_complete").touch()
dino_source = Path(flat["dinov2_model_path"])
dino_local = cache / "dinov2-base"
with (cache / "dinov2-base.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (dino_local / ".copy_complete").exists():
        dino_local.mkdir(exist_ok=True)
        subprocess.run(["rsync", "-a", str(dino_source) + "/", str(dino_local) + "/"], check=True)
        (dino_local / ".copy_complete").touch()
frozen = output / "input_manifest"
shutil.copytree(manifest, frozen)
shutil.copy2(config_path, output / "submitted_config.yaml")
active_path = frozen / "dinov2_active_environments.yaml"
active_path.write_text(yaml.safe_dump({
    "selection": {"active_environment_ids": sorted(summary["environment_ids"].values())},
    "environment_names": summary["environment_ids"], "source": "train_only_manifest",
}, sort_keys=False))
flat.update(
    dataset_base_path=str(local), dataset_metadata_path=str(frozen / "train.jsonl"),
    action_stat_path=str(frozen / "action_stats.json"), output_path=str(output),
    model_paths=str(cache / "Wan2.2-TI2V-5B"),
    ckpt_path=str(cache / "ckpt/BLM/step-12000.safetensors"),
    dinov2_model_path=str(dino_local), dinov2_active_environment_manifest=str(active_path),
)
(output / "runtime.yaml").write_text(yaml.safe_dump({"training": flat}, sort_keys=False))
(output / "allocation.json").write_text(json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"],
    "wallclock_checkpoint_epoch": int(os.environ["BWM_DINO_DEADLINE_EPOCH"]),
    "original_dataset": str(source), "local_dataset": str(local),
    "original_dino": str(dino_source), "local_dino": str(dino_local),
    "original_config": str(config_path), "train_manifest_sha256": summary["train_manifest_sha256"],
}, indent=2))
print("[prepared]", output / "runtime.yaml", "action=", flat["action_type"], flush=True)
PY

export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BWM_LOCAL_CHECKPOINT_ROOT="/tmp/${USER}/bwm_grouped_checkpoints/${SLURM_JOB_ID}/dino"
PORT=$((20000 + SLURM_JOB_ID % 10000))
exec .venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint="localhost:$PORT" \
  --rdzv_id="${SLURM_JOB_ID}_dino" \
  scripts/methods/train_real97_soft_balanced9_dinov2.py --config "$RUN/runtime.yaml" --find_unused_parameters
