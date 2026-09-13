#!/usr/bin/env bash
#SBATCH --job-name=door-4env-supports
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/door-4env-supports-%j.out
#SBATCH --error=logs/door-4env-supports-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
CONFIG=configs/evaluation/real97_door_12u_3u4d_4d_supports_near_table_20260911_v3.json
PREPARED=outputs/evaluation_real97_support_overrides_20260911_v1/job114799/door_prep
OUTPUT=outputs/infer_real97_door_ours_reference_job114799
[[ -d "$OUTPUT" ]] || exit 2
exec 8>"$OUTPUT/.support_revision.lock"
flock -n 8 || exit 2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python3 scripts/evaluation/prepare_real97_support_override.py prepare \
    --config "$CONFIG" --task door --job-id 114799 --reuse-output-job-id
"$ROOT/.venv/bin/python" scripts/evaluation/infer_real97_unbounded_context.py \
    --config "$CONFIG" --task door --prepared "$PREPARED" --output "$OUTPUT"
python3 scripts/evaluation/prepare_real97_support_override.py regrid \
    --config "$CONFIG" --task door --job-id 114799 --reuse-output-job-id
