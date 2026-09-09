#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=kqv-formal-metric
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export PYTHONPATH="$ROOT"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

LIGHT=configs/evaluation/action_tasks/ttt_kqv/lightswitch_physicalpress_step2200_all4env_support8_query15_v1.yaml
MASS_BALANCE=configs/evaluation/action_tasks/ttt_kqv/mass_balance_workspace_random30_step1673_id5_ood5_k1_dense15_v1.yaml
for config in "$LIGHT" "$MASS_BALANCE"; do
  test -s "$config" || { echo "[fatal] missing $config" >&2; exit 3; }
  .venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py \
    --config "$config"
  .venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py \
    --config "$config" --lpips --lpips-net alex --lpips-device cuda \
    --lpips-batch-size 8
done

echo "[done] formal TTT-KQV metrics complete"
