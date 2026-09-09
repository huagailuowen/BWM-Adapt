#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb-dino45-near
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
TRAIN_CONFIG=configs/train/train_mass_balance_workspace_random_30ratio_train20_mainview_dinov2_random_k1k2_total8_4envpergpu_2gpu_4500.yaml
METRIC_CONFIG=configs/evaluation/action_tasks/dinov2/mass_balance_workspace_random30_step4500_id5_ood5_k1_nearest_unbalanced_support_dense15_v1.yaml
CKPT_SOURCE=outputs/method_benchmarks/mass_balance_workspace_random_30ratio_train20/dinov2_concat_mlp_random_k1k2_total8_4envpergpu/seed_20260902_job_112770/checkpoints/step-4500.safetensors
MANIFEST=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/ttt_kqv/step_1673/seed_20260902/input_support_query_manifest.json
META=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_dense15_v1/protocol/combined_train_test.jsonl
DATASET=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_workspace_random_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
OUT=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/dinov2_concat_mlp_pool1/step_4500/seed_20260902
for required in "$TRAIN_CONFIG" "$METRIC_CONFIG" "$CKPT_SOURCE" "$MANIFEST" "$META"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done
mkdir -p "$OUT/flat" logs/methods

mapfile -t index_lists < <(
  .venv/bin/python - "$MANIFEST" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(str(index) for row in rows for index in row["support_indices"]))
print(",".join(str(index) for row in rows for index in row["query_indices"]))
PY
)
SUPPORTS=${index_lists[0]}
QUERIES=${index_lists[1]}

CACHE=/tmp/$USER/bwm_shared_cache
MODEL=$CACHE/Wan2.2-TI2V-5B
DINO=$CACHE/dinov2-base
BLM=$CACHE/ckpt/BLM/step-12000.safetensors
LOCAL_CKPT=$CACHE/trained_ckpts/mass-balance-random30-dino-job112770-step4500.safetensors
EXTRACTED_WAN=$CACHE/extracted/mass-balance-random30-dino-job112770-step4500-wan.safetensors
mkdir -p "$MODEL" "$DINO" "$(dirname "$BLM")" "$(dirname "$LOCAL_CKPT")" "$(dirname "$EXTRACTED_WAN")"
(
  flock 9
  if [[ ! -f "$MODEL/.copy_complete" && ! -f "$MODEL/.bwm_stage_complete" ]]; then
    rsync -a models/Wan2.2-TI2V-5B/ "$MODEL/"
    touch "$MODEL/.copy_complete"
  fi
) 9>"$CACHE/Wan2.2-TI2V-5B.lock"
(
  flock 8
  if [[ ! -s "$DINO/config.json" ]]; then
    rsync -a /afs/ir/users/c/y/cyzhou05/TTT-Physics/checkpoints/dinov2-base/ "$DINO/"
  fi
) 8>"$CACHE/dinov2-base.lock"
(
  flock 7
  if [[ ! -s "$BLM" ]]; then rsync -a ckpt/BLM/step-12000.safetensors "$BLM"; fi
  if [[ ! -s "$LOCAL_CKPT" ]]; then
    partial="$LOCAL_CKPT.partial.$SLURM_JOB_ID"
    rm -f "$partial"
    rsync -a "$CKPT_SOURCE" "$partial"
    mv -f "$partial" "$LOCAL_CKPT"
  fi
) 7>"$CACHE/mass-balance-random30-dino-step4500.lock"

nvidia-smi -L
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
torch.empty(1, device="cuda")
print(f"[gpu-check] {torch.cuda.get_device_name(0)}", flush=True)
PY

.venv/bin/python scripts/methods/infer_dinov2_event80.py \
  --config "$TRAIN_CONFIG" \
  --dataset_metadata_path "$META" \
  --model_paths "$MODEL" \
  --ckpt_path "$BLM" \
  --dinov2_model_path "$DINO" \
  --dinov2_checkpoint_path "$LOCAL_CKPT" \
  --wan_checkpoint_output "$EXTRACTED_WAN" \
  --support_indices "$SUPPORTS" \
  --sample_indices "$QUERIES" \
  --expected_supports_per_environment 1 \
  --output_path "$OUT/flat" \
  --num_inference_steps 25 --cfg_scale 1 --fps 10 --quality 6 \
  --seed 20260902 --skip_existing

.venv/bin/python scripts/evaluation/organize_grouped_transfer_outputs.py \
  --manifest "$MANIFEST" --flat-root "$OUT/flat" --output-root "$OUT" \
  --method-slug dinov2_concat_mlp_pool1
cp -f "$MANIFEST" "$OUT/support_query_manifest.json"
.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$META" --dataset-root "$DATASET" \
  --transfer-plan "$OUT/transfer/transfer_plan.json" \
  --prediction-root "$OUT/transfer/raw" \
  --output-dir "$OUT/grids_support_plus_queries" \
  --width 224 --height 224 --fps 10 --quality 6 --columns 5 \
  --support-size 1 --prediction-label "DINOv2 K=1 query"
.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py \
  --config "$METRIC_CONFIG"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py \
  --config "$METRIC_CONFIG" --lpips --lpips-net alex --lpips-device cuda \
  --lpips-batch-size 8
printf '%s\n' "$CKPT_SOURCE" > "$OUT/source_checkpoint.txt"
printf '%s\n' "$MANIFEST" > "$OUT/support_protocol.txt"
touch "$OUT/inference.complete"
echo "[done] output=$OUT"
