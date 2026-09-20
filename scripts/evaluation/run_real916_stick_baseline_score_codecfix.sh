#!/usr/bin/env bash
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
method=${1:?dino or ttt}
mode=${2:?cpu or lpips}
config="configs/evaluation/real916_stick_${method}_final_codecfix_20260919_v1.json"
if [[ "$mode" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    exec .venv-real97-eval-20260909/bin/python -u scripts/evaluation/score_real916_stick_baseline_final.py --config "$config" --mode cpu
fi
exec .venv/bin/python -u scripts/evaluation/score_real916_stick_baseline_final.py --config "$config" --mode lpips
