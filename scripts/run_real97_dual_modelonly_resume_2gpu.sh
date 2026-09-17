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
TASK=${1:?door or ball}
case "$TASK" in door|ball) ;; *) exit 2 ;; esac
CONFIG="configs/train/train_real97_${TASK}_dual_modelonly_resume5500_500steps_6x10_2gpu_20260916_v1.yaml"
CACHE="/tmp/${USER}/bwm_shared_cache"
RUN="$ROOT/outputs/real97_${TASK}_dual_modelonly_resume5500_500steps_6x10_20260916_v1_job${SLURM_JOB_ID}"
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
.venv/bin/python - "$CONFIG" "$RUN" "$CACHE" <<'PY'
import fcntl, hashlib, json, os, pathlib, shutil, subprocess, sys
import yaml
config_path, output, cache = map(pathlib.Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text())
flat = config["training"]
checkpoint = pathlib.Path(flat["ckpt_path"])
context = pathlib.Path(flat["grouped_context_resume_context_table"])
marker = checkpoint.parent / ("." + checkpoint.name + ".complete")
if not marker.is_file() or not checkpoint.is_file() or not context.is_file():
    raise RuntimeError("Refuse to resume an incomplete model/Z pair")
manifest = pathlib.Path(flat["dataset_metadata_path"]).parent
summary = json.loads((manifest / "manifest_summary.json").read_text())
identity = summary.get("source_dataset_cache_identity", summary)
tag = identity["task"] + "_" + identity["version"] + "_" + identity["train_manifest_sha256"][:12]
local_data = cache / "datasets_real" / tag
wan = cache / "Wan2.2-TI2V-5B"
ckpt_tag = hashlib.sha256(str(checkpoint).encode()).hexdigest()[:16]
local_checkpoint = cache / "resume_checkpoints" / ckpt_tag / checkpoint.name
cache.mkdir(parents=True, exist_ok=True)

def stage_dir(source, target, lock_path, files=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / ".copy_complete").is_file():
            target.mkdir(exist_ok=True)
            command = ["rsync", "-a"]
            if files is not None:
                command += ["--files-from=" + str(files)]
            subprocess.run(command + [str(source) + "/", str(target) + "/"], check=True)
            (target / ".copy_complete").touch()

stage_dir(pathlib.Path("models/Wan2.2-TI2V-5B"), wan, cache / "Wan2.2-TI2V-5B.lock")
local_data.parent.mkdir(parents=True, exist_ok=True)
stage_dir(pathlib.Path(flat["dataset_base_path"]), local_data,
          local_data.parent / (tag + ".lock"), manifest / "stage_files.txt")
local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
with (local_checkpoint.parent / ".stage.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    complete = local_checkpoint.parent / ".copy_complete"
    if not complete.is_file() or not local_checkpoint.is_file():
        temporary = local_checkpoint.with_suffix(".partial")
        subprocess.run(["rsync", "-a", str(checkpoint), str(temporary)], check=True)
        os.replace(temporary, local_checkpoint)
        complete.touch()
frozen = output / "input_manifest"
shutil.copytree(manifest, frozen)
shutil.copy2(config_path, output / "submitted_config.yaml")
# The source pair is read-only. Only a small provenance Z copy lives in the new run.
shutil.copy2(context, frozen / "resume_step5500.context_table.json")
flat.update(dataset_base_path=str(local_data), dataset_metadata_path=str(frozen / "train.jsonl"),
            action_stat_path=str(frozen / "action_stats.json"),
            ckpt_path=str(local_checkpoint),
            grouped_context_resume_context_table=str(frozen / "resume_step5500.context_table.json"),
            output_path=str(output), model_paths=str(wan))
(output / "runtime.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
(output / "resume_provenance.json").write_text(json.dumps({
    "source_model": str(checkpoint), "source_context_table": str(context),
    "source_step": 5500, "target_step": 6000, "additional_updates": 500,
    "optimizer_state": "fresh AdamW; source model and context weights restored",
    "frozen_contexts": True, "batch_per_rank_virtual_groups_x_episodes": [6,10],
    "world_size": 2, "keep_last_ordinary_pairs": 1,
    "wallclock_checkpoint_epoch": os.environ["BWM_WALLCLOCK_CHECKPOINT_AT"],
}, indent=2))
print("[prepared]", output / "runtime.yaml", flush=True)
PY
exec .venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="$SLURM_JOB_ID" \
  scripts/train_real97_dual_modelonly_resume.py --config "$RUN/runtime.yaml" --find_unused_parameters

