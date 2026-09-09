#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb-pool45-near
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
CONFIG=configs/evaluation/action_tasks/standard_pooled_wm/mass_balance_workspace_random30_step4500_id5_ood5_k1_nearest_unbalanced_support_dense15_v1.yaml
CKPT_SOURCE=outputs/method_benchmarks/mass_balance_workspace_random_30ratio_train20_noleak/standard_pooled_wm/job_112174/checkpoints/step-4500.safetensors
META=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_dense15_v1/protocol/combined_train_test.jsonl
DATASET=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_workspace_random_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
OUT=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/standard_pooled_wm/step_4500/seed_20260902
for required in "$CONFIG" "$CKPT_SOURCE" "$META"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done

CACHE=/tmp/$USER/bwm_shared_cache
MODEL=$CACHE/Wan2.2-TI2V-5B
LOCAL_CKPT=$CACHE/trained_ckpts/mass-balance-random30-pooled-job112174-step4500.safetensors
mkdir -p "$MODEL" "$(dirname "$LOCAL_CKPT")" logs/methods
(
  flock 9
  if [[ ! -f "$MODEL/.copy_complete" && ! -f "$MODEL/.bwm_stage_complete" ]]; then
    rsync -a models/Wan2.2-TI2V-5B/ "$MODEL/"
    touch "$MODEL/.copy_complete"
  fi
) 9>"$CACHE/Wan2.2-TI2V-5B.lock"
(
  flock 8
  if [[ ! -s "$LOCAL_CKPT" ]]; then
    partial="$LOCAL_CKPT.partial.$SLURM_JOB_ID"
    rm -f "$partial"
    rsync -a "$CKPT_SOURCE" "$partial"
    mv -f "$partial" "$LOCAL_CKPT"
  fi
) 8>"$CACHE/mass-balance-random30-pooled-step4500.lock"

nvidia-smi -L
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
torch.empty(1, device="cuda")
print(f"[gpu-check] {torch.cuda.get_device_name(0)}", flush=True)
PY

.venv/bin/python scripts/evaluation/infer_standard_pooled_transfer.py \
  --config "$CONFIG" --model-paths "$MODEL" --checkpoint "$LOCAL_CKPT"
.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$META" --dataset-root "$DATASET" \
  --transfer-plan "$OUT/transfer/transfer_plan.json" \
  --prediction-root "$OUT/transfer/raw" \
  --output-dir "$OUT/grids_support_plus_queries" \
  --width 224 --height 224 --fps 10 --quality 6 --columns 5 \
  --support-size 0 --prediction-label "Pooled WM query"
printf '%s\n' "$CKPT_SOURCE" > "$OUT/source_checkpoint.txt"
touch "$OUT/inference.complete"
echo "[done] output=$OUT"
