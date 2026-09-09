#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=light-dino-pool-eval
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

TRAIN_CONFIG=configs/train/train_lightswitch_physicalpress_maincam_active12_dinov2_k2q6_4envpergpu_2gpu_24h.yaml
CKPT_SOURCE=outputs/method_benchmarks/lightswitch_physicalpress33_maincam/dinov2_concat_mlp_k2q6_4envpergpu/seed_20260801_job_112771/checkpoints/step-4500.safetensors
BASE_MANIFEST=results/lightswitch/physicalpress33_all4env_support8_query15_v1/protocol/support_query_manifest.json
PAIR_MANIFEST=results/lightswitch/physicalpress33_all4env_support8_query15_v1/protocol/support_query_manifest_red1_blue1_matched_action.json
META=data/lightswitch_randominitial_absolute_eef_physicalpress33jitter11to22_maincam_group20_20260827/physical_press_event_group20_train.jsonl
DATASET=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/robomme_light_switch_independent_controls_random8_fixed_close_buttons_no_pause_random_initial_absolute_eef_200eps_hai-machine_lerobot
BENCHMARK=results/lightswitch/physicalpress33_all4env_support8_query15_v1

for required in "$TRAIN_CONFIG" "$CKPT_SOURCE" "$BASE_MANIFEST" "$META"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done
mkdir -p logs/methods "$(dirname "$PAIR_MANIFEST")"

# Use an action-matched red/blue pair so button identity is not confounded by
# action magnitude. All 15 disjoint queries remain identical to the K=8 run.
.venv/bin/python - "$BASE_MANIFEST" "$META" "$PAIR_MANIFEST" <<'PY'
import json
import os
from pathlib import Path
import statistics
import sys

base_path, metadata_path, output_path = map(Path, sys.argv[1:])
rows = json.loads(base_path.read_text(encoding="utf-8"))
metadata = [
    json.loads(line)
    for line in metadata_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
selected_rows = []
for row in rows:
    supports = [int(index) for index in row["support_indices"]]
    by_color = {"blue": {}, "red": {}}
    for index in supports:
        item = metadata[index]
        color = str(item["button_color"])
        action_id = int(item["action_id"])
        by_color[color].setdefault(action_id, []).append(index)
    common_actions = sorted(set(by_color["blue"]) & set(by_color["red"]))
    if not common_actions:
        raise RuntimeError(
            f"No action-matched red/blue support pair for {row['causal_class']}"
        )
    median_action = statistics.median(
        int(metadata[index]["action_id"]) for index in supports
    )
    action_id = min(
        common_actions,
        key=lambda value: (abs(value - median_action), value),
    )
    pair = [by_color["blue"][action_id][0], by_color["red"][action_id][0]]
    selected = dict(row)
    selected["support_indices"] = pair
    selected["support_episode_indices"] = [
        int(metadata[index]["episode_index"]) for index in pair
    ]
    selected["support_selection"] = {
        "rule": "one_blue_one_red_with_matched_action_nearest_support_median",
        "matched_action_id": action_id,
        "button_order": ["blue", "red"],
        "candidate_support_indices": supports,
    }
    selected_rows.append(selected)

temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(selected_rows, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output_path)
for row in selected_rows:
    print(
        f"[pair] env={row['causal_class']} supports={row['support_indices']} "
        f"action={row['support_selection']['matched_action_id']}",
        flush=True,
    )
PY

CACHE=/tmp/$USER/bwm_shared_cache
MODEL=$CACHE/Wan2.2-TI2V-5B
DINO=$CACHE/dinov2-base
BLM=$CACHE/ckpt/BLM/step-12000.safetensors
LOCAL_CKPT=$CACHE/trained_ckpts/lightswitch-dino-concat-mlp-job112771-step4500.safetensors
EXTRACTED_WAN=$CACHE/extracted/lightswitch-dino-concat-mlp-job112771-step4500-wan.safetensors
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
  if [[ ! -s "$LOCAL_CKPT" ]]; then rsync -a "$CKPT_SOURCE" "$LOCAL_CKPT"; fi
) 7>"$CACHE/lightswitch-dino-step4500.lock"

nvidia-smi -L
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
torch.empty(1, device="cuda")
print(f"[gpu-check] {torch.cuda.get_device_name(0)}", flush=True)
PY

run_variant() {
  local method_slug=$1
  local manifest=$2
  local support_size=$3
  local eval_config=$4
  local prediction_label=$5
  local out="$BENCHMARK/methods/$method_slug/step_4500/seed_20260828"
  local flat="$out/flat"
  local supports queries
  mkdir -p "$flat"
  mapfile -t index_lists < <(
    .venv/bin/python - "$manifest" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(str(index) for row in rows for index in row["support_indices"]))
print(",".join(str(index) for row in rows for index in row["query_indices"]))
PY
  )
  supports=${index_lists[0]}
  queries=${index_lists[1]}

  .venv/bin/python scripts/methods/infer_dinov2_event80.py \
    --config "$TRAIN_CONFIG" \
    --dataset_metadata_path "$META" \
    --model_paths "$MODEL" \
    --ckpt_path "$BLM" \
    --dinov2_model_path "$DINO" \
    --dinov2_checkpoint_path "$LOCAL_CKPT" \
    --wan_checkpoint_output "$EXTRACTED_WAN" \
    --support_indices "$supports" \
    --sample_indices "$queries" \
    --expected_supports_per_environment "$support_size" \
    --output_path "$flat" \
    --num_inference_steps 25 --cfg_scale 1 --fps 10 --quality 6 \
    --seed 20260828 --skip_existing

  .venv/bin/python scripts/evaluation/organize_grouped_transfer_outputs.py \
    --manifest "$manifest" --flat-root "$flat" --output-root "$out" \
    --method-slug "$method_slug"
  cp -f "$manifest" "$out/support_query_manifest.json"
  .venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
    --metadata-path "$META" --dataset-root "$DATASET" \
    --transfer-plan "$out/transfer/transfer_plan.json" \
    --prediction-root "$out/transfer/raw" \
    --output-dir "$out/grids_support_plus_queries" \
    --width 224 --height 224 --fps 10 --quality 6 --columns 5 \
    --support-size "$support_size" --prediction-label "$prediction_label"
  .venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py \
    --config "$eval_config"
  .venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py \
    --config "$eval_config" --lpips --lpips-net alex --lpips-device cuda
  printf '%s\n' "$CKPT_SOURCE" > "$out/source_checkpoint.txt"
  printf '%s\n' "$manifest" > "$out/support_protocol.txt"
  touch "$out/inference.complete"
  echo "[variant-done] method=$method_slug output=$out"
}

run_variant \
  dinov2_concat_mlp_pool8 \
  "$BASE_MANIFEST" \
  8 \
  configs/evaluation/action_tasks/dinov2/lightswitch_physicalpress_step4500_pool8_all4env_query15_v1.yaml \
  "DINOv2 pool-8 query"

run_variant \
  dinov2_concat_mlp_pool2_red_blue_matched_action \
  "$PAIR_MANIFEST" \
  2 \
  configs/evaluation/action_tasks/dinov2/lightswitch_physicalpress_step4500_pool2_red_blue_all4env_query15_v1.yaml \
  "DINOv2 red+blue pool-2 query"

echo "[done] both Light DINO pooling protocols completed"
