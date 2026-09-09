#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb30fp-idood-near
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
COMBINED_META=results/mass_balance/fixed_pose_30ratio_noleak_id5_ood5_k1_dense15_v1/protocol/combined_train_test.jsonl
BASE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_fixed_pose_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
CKPT=$TRAIN_OUT/step-3900.safetensors
TABLE=$TRAIN_OUT/step-3900.context_table.json

ROOT=results/mass_balance/fixed_pose_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/ours/step_3900/seed_20260723
PROTOCOL=$ROOT/protocol
TRANSFER=$ROOT/transfer
SHARED=/tmp/$USER/bwm_shared_cache
MODEL=$SHARED/Wan2.2-TI2V-5B
LOCAL_CKPT=$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.safetensors
LOCAL_TABLE=$SHARED/trained_ckpts/mass-balance-fixed30-noleak-step3900.context_table.json

mkdir -p "$ROOT/support_stage2" "$ROOT/internal_comparisons_unused" \
  "$PROTOCOL" "$TRANSFER/raw" "$TRANSFER/grids/id" "$TRANSFER/grids/ood" \
  "$SHARED/trained_ckpts" logs/slurm

for required in "$CKPT" "$TABLE" "$TRAIN_META" "$COMBINED_META"; do
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

.venv/bin/python - "$TRAIN_META" "$COMBINED_META" "$PROTOCOL" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

train_path, combined_path, protocol_root = map(Path, sys.argv[1:])
train_rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
rows = [json.loads(line) for line in combined_path.read_text().splitlines() if line.strip()]
selection = {
    0: "id",
    5: "id",
    10: "id",
    14: "id",
    19: "id",
    20: "ood",
    22: "ood",
    24: "ood",
    27: "ood",
    29: "ood",
}

by_ratio = defaultdict(list)
for index, row in enumerate(rows):
    ratio_index = int(row["ratio_index"])
    if ratio_index not in selection or int(row["window_index"]) != 1:
        continue
    by_ratio[ratio_index].append(
        {
            "index": index,
            "action_id": int(row["action_id"]),
            "support_bin": int(row["support_bin_index"]),
            "order": float(row["sampled_support_offset_m"]),
            "gt_tilt": float(row["metrics"]["hold_final_beam_tilt_deg"]),
        }
    )

sources = []
id_sources = []
ood_sources = []
target_mappings = []
source_domains = {}
selection_rows = []
for ratio_index, domain in selection.items():
    options = sorted(by_ratio[ratio_index], key=lambda item: item["support_bin"])
    if len(options) != 15 or len({item["action_id"] for item in options}) != 15:
        raise ValueError(f"ratio_index={ratio_index}: expected 15 unique actions")
    balanced = [item for item in options if abs(item["gt_tilt"]) <= 3.0]
    unbalanced = [item for item in options if abs(item["gt_tilt"]) > 3.0]
    if not balanced or not unbalanced:
        raise ValueError(f"ratio_index={ratio_index}: missing balanced or unbalanced candidates")
    balanced_orders = [item["order"] for item in balanced]
    support = min(
        unbalanced,
        key=lambda item: (
            min(abs(item["order"] - order) for order in balanced_orders),
            abs(item["gt_tilt"]),
            abs(item["order"]),
            item["support_bin"],
        ),
    )
    source = support["index"]
    targets = [item["index"] for item in options if item["index"] != source]
    row = rows[source]
    sources.append(source)
    (id_sources if domain == "id" else ood_sources).append(source)
    source_domains[str(source)] = domain
    target_mappings.append(f"{source}:" + ",".join(map(str, targets)))
    selection_rows.append(
        {
            "domain": domain,
            "ratio_index": ratio_index,
            "mass_ratio": float(row["right_to_left_mass_ratio"]),
            "source_index": source,
            "source_sample_id": row.get("sample_id"),
            "source_action_id": support["action_id"],
            "source_support_bin_index": support["support_bin"],
            "source_action_order_m": support["order"],
            "source_gt_terminal_tilt_deg": support["gt_tilt"],
            "query_indices": targets,
            "query_count": len(targets),
        }
    )

active = sorted({float(row["friction_mu"]) for row in train_rows})
(protocol_root / "source_indices.txt").write_text(",".join(map(str, sources)) + "\n")
(protocol_root / "id_source_indices.txt").write_text(",".join(map(str, id_sources)) + "\n")
(protocol_root / "ood_source_indices.txt").write_text(",".join(map(str, ood_sources)) + "\n")
(protocol_root / "target_mapping.txt").write_text(";".join(target_mappings) + "\n")
(protocol_root / "active_ratios.txt").write_text(",".join(map(str, active)) + "\n")
(protocol_root / "source_domains.json").write_text(
    json.dumps(source_domains, indent=2, sort_keys=True) + "\n"
)
(protocol_root / "evaluation_selection.json").write_text(
    json.dumps(selection_rows, indent=2, sort_keys=True) + "\n"
)
(protocol_root / "protocol.json").write_text(
    json.dumps(
        {
            "version": 1,
            "task": "mass_balance",
            "dataset": "fixed_pose_30ratio_noleak",
            "method": "ours",
            "checkpoint_step": 3900,
            "environment_selection": {
                "id_ratio_indices": [0, 5, 10, 14, 19],
                "ood_ratio_indices": [20, 22, 24, 27, 29],
            },
            "support_size": 1,
            "support_policy": "GT-nearest unbalanced action to the balanced interval",
            "query_policy": "all 14 remaining actions from the same environment and window",
            "support_window_raw_frames": [40, 120],
            "support_window_stride": 2,
            "stage2": {
                "initial_context": "active training-table mean",
                "context_dtype": "float32",
                "steps": 40,
                "learning_rate_schedule": "3.0:10,1.5:10,0.5:10,0.15:10",
                "gradient_clip": 1.0,
                "regularization_weight": 0.001,
                "clamp": [0.0, 1.0],
            },
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

SOURCES=$(tr -d '\n' < "$PROTOCOL/source_indices.txt")
ID_SOURCES=$(tr -d '\n' < "$PROTOCOL/id_source_indices.txt")
OOD_SOURCES=$(tr -d '\n' < "$PROTOCOL/ood_source_indices.txt")
TARGETS=$(tr -d '\n' < "$PROTOCOL/target_mapping.txt")
ACTIVE=$(tr -d '\n' < "$PROTOCOL/active_ratios.txt")

echo "[run] fixed-pose no-leak ID5/OOD5 K1 nearest-unbalanced sources=$SOURCES"
.venv/bin/python scripts/infer_stage2_ttt.py \
  --config "$CONFIG" --model_paths "$MODEL" --ckpt_path "$LOCAL_CKPT" \
  --dataset_metadata_path "$COMBINED_META" --dataset_base_path "$BASE" \
  --stage2_ckpt_path "$LOCAL_CKPT" --support_metadata_path "$COMBINED_META" \
  --ttt_support_same_as_query \
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

.venv/bin/python - "$TRANSFER/transfer_plan.json" "$PROTOCOL/source_domains.json" "$COMBINED_META" <<'PY'
import json
import sys
from pathlib import Path

plan_path, domains_path, metadata_path = map(Path, sys.argv[1:])
plan = json.loads(plan_path.read_text())
domains = json.loads(domains_path.read_text())
metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
split = {"id": [], "ood": []}
for environment in plan:
    source = int(environment["source_index"])
    domain = domains[str(source)]
    environment["domain"] = domain
    environment.setdefault("support_indices", [source])
    environment.setdefault("support_sample_ids", [metadata[source].get("sample_id")])
    split[domain].append(environment)
plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
for domain, environments in split.items():
    plan_path.with_name(f"transfer_plan_{domain}.json").write_text(
        json.dumps(environments, indent=2, sort_keys=True) + "\n"
    )
PY

.venv/bin/python scripts/analyze_context_trajectory_active_pca.py \
  --table-path "$LOCAL_TABLE" --trajectory-path "$ROOT/context_trajectory.jsonl" \
  --active-frictions "$ACTIVE" --id-indices "$ID_SOURCES" --ood-indices "$OOD_SOURCES" \
  --output-svg "$ROOT/active_mass_ratio_context_trajectory.svg" \
  --output-csv "$ROOT/context_direction_summary.csv" --parameter-label mass_ratio \
  --title "Fixed-pose no-leak mass balance: ID/OOD nearest-unbalanced K1"

for DOMAIN in id ood; do
  .venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
    --metadata-path "$COMBINED_META" --dataset-root "$BASE" \
    --transfer-plan "$TRANSFER/transfer_plan_${DOMAIN}.json" \
    --prediction-root "$TRANSFER/raw" --output-dir "$TRANSFER/grids/${DOMAIN}" \
    --width 224 --height 224 --fps 10 --quality 6 --columns 5 --support-size 1 \
    --prediction-label "Ours K1 ${DOMAIN^^}"
done

cat > "$ROOT/action_evaluation.yaml" <<YAML
version: 1
task: mass_balance
method_name: ours
metadata_jsonl: $COMBINED_META
dataset_root: $BASE
output_dir: $ROOT/action_evaluation
transfer_plans:
  - {path: $TRANSFER/transfer_plan_id.json, domain: id}
  - {path: $TRANSFER/transfer_plan_ood.json, domain: ood}
action_field: action_id
action_order_field: sampled_support_offset_m
selection_strategy: boundary_crossing
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
method_name: ours
metadata_jsonl: $COMBINED_META
dataset_root: $BASE
output_dir: $ROOT/action_evaluation
transfer_plans:
  - {path: $TRANSFER/transfer_plan_id.json, domain: id}
  - {path: $TRANSFER/transfer_plan_ood.json, domain: ood}
action_field: action_id
action_order_field: sampled_support_offset_m
environment_report_fields: [ratio_index, mass_ratio]
action_report_fields: [action_id, support_bin_index, sampled_support_offset_m]
minimum_actions: 14
extractor: {main_view_width: 224, fps: 10.0}
outcome: {terminal_window: 5}
targets:
  - {id: balanced, kind: scalar_interval, min: -3.0, max: 3.0}
YAML

.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py --config "$ROOT/action_evaluation.yaml"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py --config "$ROOT/video_metrics.yaml"

find "$TRANSFER/grids" -name '*.mp4' -printf '%p\n' | sort > "$ROOT/grid_videos.txt"
find "$TRANSFER/raw" -name '*.mp4' -printf '%p\n' | sort > "$ROOT/raw_videos.txt"
touch "$ROOT/complete"
echo "[done] $ROOT"
