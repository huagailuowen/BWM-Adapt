#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb30fp-ood-k2
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
TRAIN_META=data/mass_balance_fixed_pose_30ratio_train20_mainview_stride2_41f_20260902/train.jsonl
TEST_META=data/mass_balance_fixed_pose_30ratio_train20_mainview_stride2_41f_20260902/test.jsonl
BASE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_fixed_pose_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
CKPT=$TRAIN_OUT/step-3900.safetensors
TABLE=$TRAIN_OUT/step-3900.context_table.json

ROOT=results/mass_balance/fixed_pose_30ratio_noleak_test10_ood_k2_bracket_dense15_model_predictions_v1/methods/ours/step_3900/seed_20260723
TRANSFER=$ROOT/transfer
K2_META=$ROOT/k2_support_metadata.jsonl
SHARED=/tmp/$USER/bwm_shared_cache
MODEL=$SHARED/Wan2.2-TI2V-5B
LOCAL_CKPT=$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.safetensors
LOCAL_TABLE=$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.context_table.json

mkdir -p "$ROOT" "$ROOT/support_stage2" "$ROOT/internal_comparisons_unused" \
  "$TRANSFER/raw" "$TRANSFER/grids" "$SHARED/trained_ckpts" logs/slurm

for required in "$CKPT" "$TABLE" "$TRAIN_META" "$TEST_META"; do
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

.venv/bin/python - "$TRAIN_META" "$TEST_META" "$ROOT" "$K2_META" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

train_path, test_path, root_path, support_meta_path = map(Path, sys.argv[1:])
train_rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
rows = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
by_ratio = defaultdict(list)
for index, row in enumerate(rows):
    if int(row["window_index"]) != 1:
        continue
    by_ratio[float(row["right_to_left_mass_ratio"])].append(
        {
            "index": index,
            "action_id": int(row["action_id"]),
            "order": float(row["sampled_support_offset_m"]),
            "gt_tilt": float(row["metrics"]["hold_final_beam_tilt_deg"]),
        }
    )

if len(by_ratio) != 10:
    raise ValueError(f"Expected 10 OOD ratios, got {len(by_ratio)}")

sources = []
targets_by_source = []
support_rows = []
selection = []
support_mapping = {}
for ratio, options in sorted(by_ratio.items()):
    options.sort(key=lambda item: item["order"])
    if len(options) != 15 or len({item["action_id"] for item in options}) != 15:
        raise ValueError(f"ratio={ratio}: expected 15 unique actions")
    balanced = [item for item in options if abs(item["gt_tilt"]) <= 3.0]
    if not balanced:
        raise ValueError(f"ratio={ratio}: no GT-balanced action")
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
        raise ValueError(f"ratio={ratio}: no boundary support pair")

    source = pair[0]["index"]
    supports = [item["index"] for item in pair]
    targets = [item["index"] for item in options]
    sources.append(source)
    support_mapping[str(source)] = supports
    targets_by_source.append(f"{source}:" + ",".join(map(str, targets)))
    support_rows.extend(rows[index] for index in supports)
    selection.append(
        {
            "domain": "ood",
            "mass_ratio": ratio,
            "selection_rule": rule,
            "support_indices": supports,
            "support_action_ids": [item["action_id"] for item in pair],
            "support_action_orders": [item["order"] for item in pair],
            "support_gt_tilts_deg": [item["gt_tilt"] for item in pair],
            "prediction_targets": targets,
        }
    )

active_values = sorted({float(row["friction_mu"]) for row in train_rows})
with support_meta_path.open("w") as handle:
    for row in support_rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
(root_path / "source_indices.txt").write_text(",".join(map(str, sources)) + "\n")
(root_path / "targets_by_source.txt").write_text(";".join(targets_by_source) + "\n")
(root_path / "active_values.txt").write_text(",".join(map(str, active_values)) + "\n")
(root_path / "support_mapping.json").write_text(json.dumps(support_mapping, indent=2, sort_keys=True) + "\n")
(root_path / "support_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
PY

SOURCES=$(tr -d '\n' < "$ROOT/source_indices.txt")
TARGETS=$(tr -d '\n' < "$ROOT/targets_by_source.txt")
ACTIVE=$(tr -d '\n' < "$ROOT/active_values.txt")

echo "[run] OOD K=2 sources=$SOURCES"
.venv/bin/python scripts/infer_stage2_ttt.py \
  --config "$CONFIG" --model_paths "$MODEL" --ckpt_path "$LOCAL_CKPT" \
  --dataset_metadata_path "$TEST_META" --dataset_base_path "$BASE" \
  --stage2_ckpt_path "$LOCAL_CKPT" --support_metadata_path "$K2_META" \
  --support_count 2 --ttt_support_gradient_accumulation \
  --ttt_adapt_scope context --ttt_context_fp32 \
  --ttt_initial_context_table_path "$LOCAL_TABLE" --ttt_context_table_path "$LOCAL_TABLE" \
  --ttt_context_active_values "$ACTIVE" \
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

.venv/bin/python - "$TRANSFER/transfer_plan.json" "$ROOT/support_mapping.json" "$TEST_META" <<'PY'
import copy
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
plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

metrics_plan = copy.deepcopy(plan)
for environment in metrics_plan:
    supports = set(map(int, environment["support_indices"]))
    environment["target_indices"] = [
        int(index) for index in environment["target_indices"] if int(index) not in supports
    ]
metrics_path = plan_path.with_name("transfer_plan_disjoint_metrics.json")
metrics_path.write_text(json.dumps(metrics_plan, indent=2, sort_keys=True) + "\n")
PY

.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$TEST_META" --dataset-root "$BASE" \
  --transfer-plan "$TRANSFER/transfer_plan_disjoint_metrics.json" \
  --prediction-root "$TRANSFER/raw" --output-dir "$TRANSFER/grids" \
  --width 224 --height 224 --fps 10 --quality 6 --columns 5 --support-size 2 \
  --prediction-label "Ours K2 OOD"

cat > "$ROOT/action_evaluation.yaml" <<YAML
version: 1
task: mass_balance
method_name: ours_k2_bracket_model_predictions
metadata_jsonl: $TEST_META
dataset_root: $BASE
output_dir: $ROOT/action_evaluation
transfer_plans:
  - {path: $TRANSFER/transfer_plan.json, domain: ood}
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

cat > "$ROOT/video_metrics.yaml" <<YAML
version: 1
task: mass_balance
method_name: ours_k2_bracket_model_predictions
metadata_jsonl: $TEST_META
dataset_root: $BASE
output_dir: $ROOT/action_evaluation
transfer_plans:
  - {path: $TRANSFER/transfer_plan_disjoint_metrics.json, domain: ood}
action_field: action_id
action_order_field: sampled_support_offset_m
environment_report_fields: [ratio_index, mass_ratio]
action_report_fields: [action_id, support_bin_index, sampled_support_offset_m]
minimum_actions: 13
extractor: {main_view_width: 224, fps: 10.0}
outcome: {terminal_window: 5}
targets:
  - {id: balanced, kind: scalar_interval, min: -3.0, max: 3.0}
YAML

.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py --config "$ROOT/action_evaluation.yaml"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py --config "$ROOT/video_metrics.yaml"

cat > "$ROOT/inference_protocol.json" <<EOF
{
  "domain": "ood",
  "environment_count": 10,
  "method": "ours_k2_bracket_model_predictions",
  "checkpoint_step": 3900,
  "support_size": 2,
  "support_ground_truth_usage": "inner-loop optimization only",
  "action_selection_source": "model rollout for all 15 actions, including supports",
  "video_metric_queries_per_environment": 13,
  "inner_steps": 40,
  "inner_lr_schedule": "3.0:10,1.5:10,0.5:10,0.15:10",
  "context_dtype": "float32",
  "model_loads_for_adaptation_and_rollout": 1
}
EOF
touch "$ROOT/complete"
echo "[done] $ROOT"
