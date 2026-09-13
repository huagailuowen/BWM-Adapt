#!/usr/bin/env bash
#SBATCH --job-name=real97-support-revision
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
#SBATCH --output=logs/real97-support-revision-%j.out
#SBATCH --error=logs/real97-support-revision-%j.err
set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
CONFIG=configs/evaluation/real97_ball_door_support_overrides_20260911_v1.json
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# One low-priority GPU, sequential tasks. A failed task does not prevent the
# other task from being attempted; all source experiments remain read-only.
run_task() {
    local task="$1"
    local prepared="outputs/evaluation_real97_support_overrides_20260911_v1/job${SLURM_JOB_ID}/${task}_prep"
    local output="outputs/infer_real97_${task}_ours_reference_job${SLURM_JOB_ID}"
    if [[ -f "$output/support_override_complete.json" ]]; then
        printf '[override] already completed: %s\n' "$task"
        return 0
    fi
    python3 scripts/evaluation/prepare_real97_support_override.py prepare \
        --config "$CONFIG" --task "$task" --job-id "$SLURM_JOB_ID" || return "$?"
    # Reuse the existing GPU/cache/inference implementation, not a new model
    # path. Prepopulated Stage1 and unaffected Stage2 completion markers skip
    # those predictions; changed environments have no cached adapted context.
    bash scripts/evaluation/run_real97_ball_door_infer_gpu.sh "$task" "$prepared" || return "$?"
    python3 scripts/evaluation/prepare_real97_support_override.py regrid \
        --config "$CONFIG" --task "$task" --job-id "$SLURM_JOB_ID" || return "$?"
}

status=0
for task in door ball; do
    if run_task "$task"; then
        printf '[override] task complete: %s\n' "$task"
    else
        code=$?
        printf '[override] task failed: %s exit=%s\n' "$task" "$code" >&2
        status=1
    fi
done
exit "$status"
