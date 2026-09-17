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
CONFIG=configs/train/train_real97_soft_static9_standard_5x6_2gpu_20260917_v1.yaml
RUN="$ROOT/outputs/real97_soft_static9_standard_5x6_20260917_v1_job${SLURM_JOB_ID}"
mkdir "$RUN"
exec > >(tee "$RUN/train.log") 2>&1
nvidia-smi
.venv/bin/python scripts/check_real97_cuda.py --expected 2
START=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* StartTime=\([^ ]*\).*/\1/p')
export BWM_STANDARD_DEADLINE_EPOCH=$(( $(date -d "$START" +%s) + 84600 ))
export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/scripts/evaluation" PYTHONUNBUFFERED=1
export BWM_REQUIRE_CUDA=1 BWM_EXPECTED_WORLD_SIZE=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=16 NCCL_DEBUG=WARN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python - "$CONFIG" "$RUN" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
import yaml
from scripts.evaluation.infer_real97_standard_reference import copy_cached
config_path,output=map(Path,sys.argv[1:]);config=yaml.safe_load(config_path.read_text());flat=config['training']
manifest=Path(flat['dataset_metadata_path']).parent
cache=Path('/tmp')/os.environ['USER']/'bwm_shared_cache';cache.mkdir(parents=True,exist_ok=True)
wan=cache/'Wan2.2-TI2V-5B';base=cache/'ckpt/BLM/step-12000.safetensors'
copy_cached(Path('models/Wan2.2-TI2V-5B').resolve(),wan,directory=True)
copy_cached(Path('ckpt/BLM/step-12000.safetensors').resolve(),base)
summary=json.loads((manifest/'manifest_summary.json').read_text())
identity=summary.get('source_dataset_cache_identity',summary)
tag=identity['task']+'_'+identity['version']+'_'+identity['train_manifest_sha256'][:12]
local=cache/'datasets_real'/tag
copy_cached(Path(flat['dataset_base_path']),local,directory=True,files=manifest/'stage_files.txt')
rows=[json.loads(line) for line in (manifest/'train.jsonl').read_text().splitlines() if line.strip()]
expected={'soft-1l','soft-2l','soft-4l','soft-1r','soft-7r','soft-5r','soft-2m','soft-6m','soft-8'}
if {r['environment'] for r in rows}!=expected or any(r['dataset_split']!='train' for r in rows):
    raise RuntimeError('Unexpected nine-environment train pool')
frozen=output/'input_manifest';shutil.copytree(manifest,frozen)
shutil.copy2(config_path,output/'submitted_config.yaml')
flat.update(dataset_base_path=str(local),dataset_metadata_path=str(frozen/'train.jsonl'),
            action_stat_path=str(frozen/'action_stats.json'),ckpt_path=str(base),model_paths=str(wan),output_path=str(output))
(output/'runtime.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(output/'allocation.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
 deadline_epoch=int(os.environ['BWM_STANDARD_DEADLINE_EPOCH']),environments=sorted(expected),
 manifest_sha256=hashlib.sha256((frozen/'train.jsonl').read_bytes()).hexdigest(),
 observed_frames=1,predicted_frames=32,per_rank_batch=[5,6],world_size=2,
 source_checkpoint='ckpt/BLM/step-12000.safetensors',method='standard_pooled'),indent=2)+'\n')
PY
exec .venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="$SLURM_JOB_ID" \
 scripts/methods/train_real97_soft_static9_standard.py --config "$RUN/runtime.yaml"
