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
CONFIG="configs/train/train_real97_${TASK}_dual_joint_resume6000_500steps_c005_6x10_2gpu_20260917_v1.yaml"
RUN="$ROOT/outputs/real97_${TASK}_dual_joint_resume6000_500steps_c005_6x10_20260917_v1_job${SLURM_JOB_ID}"
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
.venv/bin/python - "$CONFIG" "$RUN" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
import yaml
from scripts.evaluation.infer_real97_standard_reference import copy_cached
config_path,output=map(Path,sys.argv[1:])
config=yaml.safe_load(config_path.read_text());flat=config['training']
checkpoint=Path(flat['ckpt_path']);context=Path(flat['grouped_context_resume_context_table'])
marker=checkpoint.parent/('.'+checkpoint.name+'.complete')
if not all(p.is_file() for p in (checkpoint,context,marker)):
    raise RuntimeError('Missing complete step6000 model/context pair')
if checkpoint.stem!='step-6000' or context.name!='step-6000.context_table.json':
    raise RuntimeError('Expected exact step6000 bundle')
manifest=Path(flat['dataset_metadata_path']).parent
cache=Path('/tmp')/os.environ['USER']/'bwm_shared_cache';cache.mkdir(parents=True,exist_ok=True)
wan=cache/'Wan2.2-TI2V-5B'
copy_cached(Path('models/Wan2.2-TI2V-5B').resolve(),wan,directory=True)
summary=json.loads((manifest/'manifest_summary.json').read_text())
identity=summary.get('source_dataset_cache_identity',summary)
tag=identity['task']+'_'+identity['version']+'_'+identity['train_manifest_sha256'][:12]
local_data=cache/'datasets_real'/tag
copy_cached(Path(flat['dataset_base_path']),local_data,directory=True,files=manifest/'stage_files.txt')
stat=checkpoint.stat();tag=hashlib.sha256(f'{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:16]
local_checkpoint=cache/'resume_checkpoints'/tag/checkpoint.name
copy_cached(checkpoint,local_checkpoint)
frozen=output/'input_manifest';shutil.copytree(manifest,frozen)
shutil.copy2(config_path,output/'submitted_config.yaml')
shutil.copy2(context,frozen/'resume_step6000.context_table.json')
flat.update(dataset_base_path=str(local_data),dataset_metadata_path=str(frozen/'train.jsonl'),
            action_stat_path=str(frozen/'action_stats.json'),ckpt_path=str(local_checkpoint),
            grouped_context_resume_context_table=str(frozen/'resume_step6000.context_table.json'),
            model_paths=str(wan),output_path=str(output))
(output/'runtime.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(output/'resume_provenance.json').write_text(json.dumps(dict(
    source_model=str(checkpoint),source_context_table=str(context),source_step=6000,target_step=6500,
    additional_updates=500,mode='joint_model_and_context',model_lr=1e-5,context_lr=.005,
    optimizer_state='fresh AdamW; model/context weights restored',world_size=2,
    batch_per_rank_virtual_groups_x_episodes=[6,10],keep_last_ordinary_pairs=1,
    context_clamp=[flat['grouped_context_clamp_min'],flat['grouped_context_clamp_max']],
    source_outputs_modified=False),indent=2)+'\n')
print('[prepared]',output/'runtime.yaml',flush=True)
PY
exec .venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="$SLURM_JOB_ID" \
  scripts/train_real97_dual_joint_resume.py --config "$RUN/runtime.yaml" --find_unused_parameters
