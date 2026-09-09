#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb30fp-k2-bracket
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err

set -euo pipefail

REPO=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$REPO"
source /afs/ir/users/c/y/cyzhou05/TTT-Physics/envs/activate_bwm.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"

TRAIN_OUT=outputs/mass_balance_fixed_pose_30ratio_train20_mainview_4env8chunk_c32_oldrandom_stage1_4300_110422
CONFIG=configs/train/train_mass_balance_fixed_pose_30ratio_train20_mainview_4env8chunk_stride2_curriculum_c32_old_random_stage1_4300.yaml
META=data/mass_balance_fixed_pose_30ratio_train20_mainview_stride2_41f_20260902/train.jsonl
BASE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_fixed_pose_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
CKPT=$TRAIN_OUT/step-3900.safetensors
TABLE=$TRAIN_OUT/step-3900.context_table.json
K1_ROOT=results/mass_balance/fixed_pose_30ratio_noleak_nearest_unbalanced_center_support_dense15_v1/methods/ours/step_3900/seed_20260723
K1_CANDIDATES=$K1_ROOT/action_evaluation/candidate_outcomes.jsonl

ROOT=results/mass_balance/fixed_pose_30ratio_noleak_k2_bracket_dense15_model_predictions_v1/methods/ours/step_3900/seed_20260723
TRANSFER=$ROOT/transfer
K2_META=$ROOT/k2_support_metadata.jsonl
SHARED=/tmp/$USER/bwm_shared_cache
MODEL=$SHARED/Wan2.2-TI2V-5B
LOCAL_CKPT=$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.safetensors
LOCAL_TABLE=$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.context_table.json

mkdir -p "$ROOT" "$ROOT/support_stage2" "$ROOT/internal_comparisons_unused" \
  "$TRANSFER/raw" "$TRANSFER/grids" "$SHARED/trained_ckpts" logs/slurm

for required in "$CKPT" "$TABLE" "$META" "$K1_CANDIDATES"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done

(
  flock 9
  if [[ ! -f "$MODEL/.copy_complete" && ! -f "$MODEL/.bwm_stage_complete" ]]; then
    mkdir -p "$MODEL"
    rsync -a models/Wan2.2-TI2V-5B/ "$MODEL/"
    touch "$MODEL/.copy_complete"
  else
    echo "[cache] reuse $MODEL"
  fi
) 9>"$SHARED/Wan2.2-TI2V-5B.lock"

(
  flock 8
  if [[ ! -s "$LOCAL_CKPT" || ! -s "$LOCAL_TABLE" ]]; then
    rsync -a "$CKPT" "$LOCAL_CKPT"
    rsync -a "$TABLE" "$LOCAL_TABLE"
  else
    echo "[cache] reuse $LOCAL_CKPT"
  fi
) 8>"$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.lock"

.venv/bin/python - "$META" "$K1_CANDIDATES" "$ROOT" "$K2_META" <<'PY'
import json
import sys
from pathlib import Path

meta_path, candidate_path, root_path, support_meta_path = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
candidates = [
    json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()
]
ratio_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 19]
sources = []
targets_by_source = []
support_rows = []
selection = []
support_mapping = {}

for ratio_index in ratio_indices:
    options = []
    for candidate in candidates:
        indices = [int(value) for value in candidate.get("sample_indices", [])]
        if len(indices) != 1:
            continue
        index = indices[0]
        row = rows[index]
        if int(row["ratio_index"]) != ratio_index or int(row["window_index"]) != 1:
            continue
        options.append(
            {
                "index": index,
                "action_id": str(candidate["action_id"]),
                "order": float(candidate["action_order"]),
                "gt_tilt": float(candidate["ground_truth_value"]),
            }
        )
    options.sort(key=lambda item: item["order"])
    if len(options) != 15:
        raise ValueError(f"ratio_index={ratio_index}: expected 15 actions, got {len(options)}")

    balanced = [item for item in options if abs(item["gt_tilt"]) <= 3.0]
    if not balanced:
        raise ValueError(f"ratio_index={ratio_index}: no balanced action")
    low = min(item["order"] for item in balanced)
    high = max(item["order"] for item in balanced)
    left = [item for item in options if item["order"] < low]
    right = [item for item in options if item["order"] > high]
    if left and right:
        pair = [max(left, key=lambda item: item["order"]), min(right, key=lambda item: item["order"])]
        rule = "nearest_unbalanced_left_and_right"
    elif right:
        pair = [max(balanced, key=lambda item: item["order"]), min(right, key=lambda item: item["order"])]
        rule = "right_edge_balanced_and_nearest_right"
    elif left:
        pair = [max(left, key=lambda item: item["order"]), min(balanced, key=lambda item: item["order"])]
        rule = "nearest_left_and_left_edge_balanced"
    else:
        raise ValueError(f"ratio_index={ratio_index}: no boundary support pair")

    source = pair[0]["index"]
    support_indices = [item["index"] for item in pair]
    target_indices = [item["index"] for item in options]
    sources.append(source)
    support_mapping[str(source)] = support_indices
    targets_by_source.append(f"{source}:" + ",".join(map(str, target_indices)))
    support_rows.extend(rows[index] for index in support_indices)
    selection.append(
        {
            "ratio_index": ratio_index,
            "mass_ratio": float(rows[source]["right_to_left_mass_ratio"]),
            "selection_rule": rule,
            "support_indices": support_indices,
            "support_action_ids": [item["action_id"] for item in pair],
            "support_action_orders": [item["order"] for item in pair],
            "support_gt_tilts_deg": [item["gt_tilt"] for item in pair],
            "prediction_targets": target_indices,
        }
    )

ratio02 = next(item for item in selection if item["ratio_index"] == 2)
if ratio02["support_action_ids"] != ["1", "2"]:
    raise ValueError(f"ratio=0.2 must use actions 1 and 2, got {ratio02['support_action_ids']}")

with support_meta_path.open("w") as handle:
    for row in support_rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
(root_path / "source_indices.txt").write_text(",".join(map(str, sources)) + "\n")
(root_path / "targets_by_source.txt").write_text(";".join(targets_by_source) + "\n")
(root_path / "support_mapping.json").write_text(json.dumps(support_mapping, indent=2, sort_keys=True) + "\n")
(root_path / "support_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
PY

SOURCES=$(tr -d '\n' < "$ROOT/source_indices.txt")
TARGETS=$(tr -d '\n' < "$ROOT/targets_by_source.txt")

echo "[run] K=2 sources=$SOURCES"
.venv/bin/python scripts/infer_stage2_ttt.py \
  --config "$CONFIG" --model_paths "$MODEL" --ckpt_path "$LOCAL_CKPT" \
  --dataset_metadata_path "$META" --dataset_base_path "$BASE" \
  --stage2_ckpt_path "$LOCAL_CKPT" --support_metadata_path "$K2_META" \
  --support_count 2 --ttt_support_gradient_accumulation \
  --ttt_adapt_scope context --ttt_context_fp32 \
  --ttt_initial_context_table_path "$LOCAL_TABLE" --ttt_context_table_path "$LOCAL_TABLE" \
  --ttt_context_trajectory_path "$ROOT/context_trajectory.jsonl" \
  --ttt_context_pca_output_path "$ROOT/context_trajectory_pca.svg" \
  --resume_context_trajectory --skip_existing \
  --sample_indices "$SOURCES" --output_path "$ROOT/support_stage2" \
  --comparison_output_path "$ROOT/internal_comparisons_unused" \
  --ttt_transfer_targets_by_source "$TARGETS" \
  --ttt_transfer_output_path "$TRANSFER/raw" \
  --ttt_transfer_plan_output_path "$TRANSFER/transfer_plan.json" \
  --ttt_transfer_skip_existing \
  --stage2_inner_steps 40 --stage2_inner_lr 3.0 \
  --stage2_inner_lr_schedule "3.0:10,1.5:10,0.5:10,0.15:10" \
  --stage2_inner_grad_clip 1.0 --stage2_context_reg_weight 0.001 \
  --stage2_context_clamp_min 0.0 --stage2_context_clamp_max 1.0 \
  --num_inference_steps 25 --cfg_scale 1 --fps 10 --quality 6 --seed 20260723

.venv/bin/python - "$TRANSFER/transfer_plan.json" "$ROOT/support_mapping.json" "$META" <<'PY'
import json
import sys
from pathlib import Path

plan_path, mapping_path, metadata_path = map(Path, sys.argv[1:])
plan = json.loads(plan_path.read_text())
mapping = json.loads(mapping_path.read_text())
metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
for environment in plan:
    source = str(int(environment["source_index"]))
    supports = [int(value) for value in mapping[source]]
    environment["support_indices"] = supports
    environment["support_sample_ids"] = [metadata[index].get("sample_id") for index in supports]
temporary = plan_path.with_suffix(".json.partial")
temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
temporary.replace(plan_path)
PY

cat > "$ROOT/action_evaluation.yaml" <<YAML
version: 1
task: mass_balance
method_name: ours_k2_bracket_model_predictions
metadata_jsonl: $META
dataset_root: $BASE
output_dir: $ROOT/action_evaluation
transfer_plans:
  - {path: $TRANSFER/transfer_plan.json, domain: id}
action_field: action_id
action_order_field: sampled_support_offset_m
selection_strategy: boundary_crossing
use_model_predictions_for_support: true
environment_report_fields: [ratio_index, mass_ratio]
action_report_fields: [action_id, support_bin_index, sampled_support_offset_m]
minimum_actions: 15
extractor: {main_view_width: 224, fps: 10.0}
outcome: {terminal_window: 5}
targets:
  - {id: balanced, kind: scalar_interval, min: -3.0, max: 3.0}
YAML

.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py --config "$ROOT/action_evaluation.yaml"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py --config "$ROOT/action_evaluation.yaml"

cat > "$ROOT/inference_protocol.json" <<EOF
{
  "method": "ours_k2_bracket_model_predictions",
  "checkpoint_step": 3900,
  "support_size": 2,
  "support_rule": "nearest actions outside both sides of the GT-balanced interval, with an in-range boundary fallback",
  "ratio_0.2_support_actions": [1, 2],
  "support_ground_truth_usage": "inner-loop optimization only",
  "action_selection_source": "model rollout for all 15 actions, including both support actions",
  "inner_steps": 40,
  "inner_lr_schedule": "3.0:10,1.5:10,0.5:10,0.15:10",
  "context_dtype": "float32",
  "model_loads_for_adaptation_and_rollout": 1
}
EOF
touch "$ROOT/complete"
echo "[done] $ROOT"
