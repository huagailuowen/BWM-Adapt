#!/usr/bin/env bash
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
method=${1:?dino or ttt}
mode=${2:?infer cpu or lpips}
config="configs/evaluation/real916_stick_${method}_final_auto_20260919_v1.json"
case "$mode" in
  infer) exec .venv/bin/python -u scripts/evaluation/infer_real916_stick_baseline_final.py --config "$config" ;;
  cpu) exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real916_stick_baseline_final.py --config "$config" --mode cpu ;;
  lpips) exec .venv/bin/python -u scripts/evaluation/score_real916_stick_baseline_final.py --config "$config" --mode lpips ;;
  *) exit 2 ;;
esac
