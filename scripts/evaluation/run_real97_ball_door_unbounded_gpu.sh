#!/usr/bin/env bash
#SBATCH --job-name=door-ball-unbounded-Z
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
#SBATCH --output=logs/door-ball-unbounded-%j.out
#SBATCH --error=logs/door-ball-unbounded-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
CONFIG=configs/evaluation/real97_ball_door_unbounded_context_20260911_v1.json
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

run_task() (
    set -euo pipefail
    task="$1"
    output_id="$SLURM_JOB_ID"
    reuse=()
    if [[ "$task" == door ]]; then output_id=114799; reuse=(--reuse-output-job-id); fi
    prepared="outputs/evaluation_real97_support_overrides_20260911_v1/job${output_id}/${task}_prep"
    output="outputs/infer_real97_${task}_ours_reference_job${output_id}"
    mkdir -p "$output"
    exec 8>"$output/.support_revision.lock"
    flock -n 8 || return 2
    python3 scripts/evaluation/prepare_real97_support_override.py prepare \
        --config "$CONFIG" --task "$task" --job-id "$output_id" "${reuse[@]}" || return "$?"
    "$ROOT/.venv/bin/python" scripts/evaluation/infer_real97_unbounded_context.py \
        --config "$CONFIG" --task "$task" --prepared "$prepared" --output "$output" || return "$?"
    python3 scripts/evaluation/prepare_real97_support_override.py regrid \
        --config "$CONFIG" --task "$task" --job-id "$output_id" "${reuse[@]}" || return "$?"
)

status=0
for task in door ball; do
    if run_task "$task"; then
        printf '[unbounded] completed %s\n' "$task"
    else
        printf '[unbounded] failed %s; the other task will still be attempted\n' "$task" >&2
        status=1
    fi
done
exit "$status"
