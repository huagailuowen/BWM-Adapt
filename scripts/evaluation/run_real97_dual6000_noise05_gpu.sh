#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/dual6000-noise05-%j.out
#SBATCH --error=logs/dual6000-noise05-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
TASK=${1:?door or ball}
case "$TASK" in door|ball) ;; *) exit 2 ;; esac
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/check_real97_cuda.py --expected 1
CONFIG="configs/evaluation/real97_${TASK}_dual6000_noise05_20260917_v1.json"
.venv/bin/python - "$CONFIG" "$TASK" <<'PY'
import json,sys,fcntl,os,shutil
from pathlib import Path
from scripts.evaluation.prepare_real97_dual_latent_reference import prepare
p=Path(sys.argv[1]);task=sys.argv[2];config=json.loads(p.read_text());setting=config['tasks'][task]
run=Path(setting['training_run'])
for name in ('step-6000.safetensors','.step-6000.safetensors.complete','step-6000.context_table.json'):
 if not (run/name).is_file():raise RuntimeError('Missing complete final model/table: '+name)
prepared=Path(setting['prepared']);prepared.parent.mkdir(parents=True,exist_ok=True)
with (prepared.parent/'.prepare.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);prepare(p,task)
 planpath=prepared/'plan.json';plan=json.loads(planpath.read_text())
 plan.update(stage2_initial=setting['initial_context'],known_environment_prior=True,initialization_noise_range=[-.5,.5])
 temporary=planpath.with_suffix('.json.partial');temporary.write_text(json.dumps(plan,indent=2)+'\n');temporary.replace(planpath)
 # Identical model, z0, query seeds and rows: reuse only completed Stage1 variants.
 # GT/Stage2/comparisons are never linked or copied from the other experiment.
 source=Path(config['stage1_reuse_source']);output=Path(config['output'])
 done=json.loads((source/'inference_complete.json').read_text())
 if (done['model_step'],done['table_step'])!=(6000,6000):raise RuntimeError('Stage1 source uses another checkpoint')
 for sub in ('completed_variants','extensions/completed_variants'):
  for marker in (source/sub).glob('*_stage1.json'):
   record=json.loads(marker.read_text());key=record['key']
   raw=Path('raw/stage1') if sub=='completed_variants' else Path('extensions/raw/stage1')
   video=source/raw/(key+'.mp4');target=output/raw/video.name;target.parent.mkdir(parents=True,exist_ok=True)
   if not video.is_file() or video.stat().st_size==0:raise RuntimeError('Incomplete Stage1 source')
   if not target.exists():
    tmp=target.with_suffix('.mp4.partial')
    if tmp.exists():tmp.unlink()
    try:os.link(video,tmp)
    except OSError:shutil.copy2(video,tmp)
    tmp.replace(target)
   dest=output/sub/marker.name;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():shutil.copy2(marker,dest)
PY
exec .venv/bin/python scripts/evaluation/infer_real97_dual_latent_reference.py \
 --config "$CONFIG" --task "$TASK" --output "outputs/infer_real97_${TASK}_dual6000_noise05_20260917_v1"
