#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/stick916-lora-%j.out
#SBATCH --error=logs/stick916-lora-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONFIG=configs/evaluation/real916_stick_lora_standard4559_20260918_v1.json
case "${1:?infer, cpu or lpips}" in
 infer)
 .venv/bin/python scripts/check_real97_cuda.py --expected 1
 exec .venv/bin/python scripts/evaluation/infer_real916_stick_lora.py --config "$CONFIG" ;;
 cpu)
 export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
 exec .venv-real97-eval-20260909/bin/python scripts/evaluation/score_real916_stick_lora.py --config "$CONFIG" --mode cpu ;;
 lpips)
 .venv/bin/python scripts/check_real97_cuda.py --expected 1
 exec .venv/bin/python scripts/evaluation/score_real916_stick_lora.py --config "$CONFIG" --mode lpips ;;
 *) exit 2 ;;
esac
