#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/real97-door-dual6000-cluster-%j.out
#SBATCH --error=logs/real97-door-dual6000-cluster-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
: "${SLURM_JOB_ID:?Compute allocation required}"
CONFIG=configs/evaluation/real97_door_dual6000_cluster_mean_20260916_v1.json
export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/scripts/evaluation${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/check_real97_cuda.py --expected 1
.venv/bin/python - "$CONFIG" <<'PY'
import json,sys,fcntl
from pathlib import Path
from scripts.evaluation.prepare_real97_dual_latent_reference import prepare
config_path=Path(sys.argv[1]);config=json.loads(config_path.read_text());setting=config['tasks']['door']
run=Path(setting['training_run']);step=setting['model_step']
for name in (f'step-{step}.safetensors', f'.step-{step}.safetensors.complete', f'step-{step}.context_table.json'):
    if not (run/name).is_file():
        raise RuntimeError('Required completed model/table bundle missing: '+str(run/name))
prepared=Path(setting['prepared']);prepared.parent.mkdir(parents=True,exist_ok=True)
with (prepared.parent/'.prepare.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    prepare(config_path,'door')
    plan_path=prepared/'plan.json';plan=json.loads(plan_path.read_text())
    plan.update(stage2_initial=setting['initial_context'],known_family_prior=True,
                initialization_groups=setting['initialization_groups'],
                environment_initialization_group=setting['environment_initialization_group'],
                excluded_from_group_means=setting['excluded_from_group_means'])
    temp=plan_path.with_suffix('.json.partial');temp.write_text(json.dumps(plan,indent=2)+'\n');temp.replace(plan_path)
PY
exec .venv/bin/python scripts/evaluation/infer_real97_dual_latent_reference.py \
  --config "$CONFIG" --task door \
  --output outputs/infer_real97_door_dual6000_cluster_mean_after117981_v1
