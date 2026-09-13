#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin
#SBATCH --gres=gpu:4
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/%x-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
if [[ $# != 2 ]]; then
  printf 'Expected exactly two independent experiment configs.\n' >&2
  exit 2
fi
start=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* StartTime=\([^ ]*\).*/\1/p')
start_epoch=$(date -d "$start" +%s)
export BWM_STANDARD_DEADLINE_EPOCH=$((start_epoch + 23 * 3600 + 30 * 60))
gpu_list="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
IFS=, read -r -a gpu_ids <<< "$gpu_list"
if [[ ${#gpu_ids[@]} != 4 ]]; then
  printf 'Expected four allocated GPU identifiers, got: %s\n' "$gpu_list" >&2
  exit 2
fi
printf '[pair] job=%s first=%s second=%s save_deadline=%s\n' "$SLURM_JOB_ID" "$1" "$2" "$BWM_STANDARD_DEADLINE_EPOCH"
# Each worker probes only its own two cards. Do not fail both experiments
# because a different pair has a broken card.
CUDA_VISIBLE_DEVICES="${gpu_ids[0]},${gpu_ids[1]}" \
  BWM_STANDARD_PORT="$((20000 + SLURM_JOB_ID % 10000))" \
  bash "$ROOT/scripts/run_real97_standard_worker.sh" "$1" \
  >"$ROOT/logs/standard-pair-${SLURM_JOB_ID}-first.log" 2>&1 &
first_pid=$!
CUDA_VISIBLE_DEVICES="${gpu_ids[2]},${gpu_ids[3]}" \
  BWM_STANDARD_PORT="$((30000 + SLURM_JOB_ID % 10000))" \
  bash "$ROOT/scripts/run_real97_standard_worker.sh" "$2" \
  >"$ROOT/logs/standard-pair-${SLURM_JOB_ID}-second.log" 2>&1 &
second_pid=$!
first_status=0
second_status=0
wait "$first_pid" || first_status=$?
wait "$second_pid" || second_status=$?
printf '[pair] first_exit=%s second_exit=%s\n' "$first_status" "$second_status"
if (( first_status != 0 || second_status != 0 )); then exit 1; fi
