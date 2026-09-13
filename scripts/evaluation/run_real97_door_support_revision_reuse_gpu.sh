#!/usr/bin/env bash
#SBATCH --job-name=door-support25-reuse
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G|141G|180G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/door-support25-reuse-%j.out
#SBATCH --error=logs/door-support25-reuse-%j.err
set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
CONFIG=configs/evaluation/real97_door_support_overrides_20260911_v2.json
PREPARED=outputs/evaluation_real97_support_overrides_20260911_v1/job114799/door_prep
OUTPUT=outputs/infer_real97_door_ours_reference_job114799
[[ -d "$OUTPUT" ]] || { printf 'Expected existing output: %s\n' "$OUTPUT" >&2; exit 2; }
exec 8>"$OUTPUT/.support_revision.lock"
flock -n 8 || { printf 'Another support revision owns this output\n' >&2; exit 2; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python3 scripts/evaluation/prepare_real97_support_override.py prepare \
    --config "$CONFIG" --task door --job-id 114799 --reuse-output-job-id

# Explicit output path keeps the existing directory without spoofing Slurm's
# allocation ID. The reference inference entrypoint handles node-local staging.
"$ROOT/.venv/bin/python" scripts/evaluation/infer_real97_ball_door_reference.py \
    --prepared "$PREPARED" --output "$OUTPUT"

python3 scripts/evaluation/prepare_real97_support_override.py regrid \
    --config "$CONFIG" --task door --job-id 114799 --reuse-output-job-id
