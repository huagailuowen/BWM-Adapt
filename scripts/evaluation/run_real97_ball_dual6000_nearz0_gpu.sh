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
#SBATCH --output=logs/ball-dual6000-infer-%j.out
#SBATCH --error=logs/ball-dual6000-infer-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/check_real97_cuda.py --expected 1
CONFIG=configs/evaluation/real97_ball_dual6000_nearz0_20260917_v1.json
.venv/bin/python - "$CONFIG" <<'PY'
import sys,json,fcntl
from pathlib import Path
from scripts.evaluation.prepare_real97_dual_latent_reference import prepare
p=Path(sys.argv[1]);s=json.loads(p.read_text())['tasks']['ball'];run=Path(s['training_run'])
for name in ['step-6000.safetensors','.step-6000.safetensors.complete','step-6000.context_table.json']:
 if not (run/name).is_file():raise RuntimeError('Missing final bundle '+name)
folder=Path(s['prepared']).parent;folder.mkdir(parents=True,exist_ok=True)
with (folder/'.prepare.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);prepare(p,'ball')
 planpath=Path(s['prepared'])/'plan.json';plan=json.loads(planpath.read_text());plan['stage2_initial']=s['initial_context']
 tmp=planpath.with_suffix('.partial');tmp.write_text(json.dumps(plan,indent=2));tmp.replace(planpath)
PY
exec .venv/bin/python scripts/evaluation/infer_real97_dual_latent_reference.py --config "$CONFIG" --task ball --output outputs/infer_real97_ball_dual6000_nearz0_20260917_v1
