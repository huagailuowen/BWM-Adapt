#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mbalfp-near-unbal
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err

set -euo pipefail

REPO=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$REPO"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

TRAIN_JOB=91441
TRAIN_OUT=outputs/mass_balance_fixed_pose_c32_oldrandom_stride2_stage1_4300_${TRAIN_JOB}
CONFIG=configs/train/train_mass_balance_fixed_pose_20ratio_stride2_curriculum_c32_old_random_stage1_4300.yaml
META=data/mass_balance_fixed_pose_20ratio_stride2_41f_20260722/train.jsonl
BASE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_fixed_pose_20ratio_15support_300eps_direct_approach_absolute_eef_lerobot_2026-07-22_hai-machine
CKPT=${TRAIN_OUT}/step-4300.safetensors
TABLE=${TRAIN_OUT}/step-4300.context_table.json
REFERENCE_CANDIDATES=results/mass_balance/fixed_pose_dense15_boundary_104381/methods/ours/action_evaluation/candidate_outcomes.jsonl

OUT_DIR=outputs/infer_mass_balance_fixed_pose_91441_stage1_stage2_nearest_unbalanced_center_support_dense15_${SLURM_JOB_ID}
STAGE1_RAW=${OUT_DIR}/stage1_raw
STAGE2_RAW=${OUT_DIR}/stage2_raw
SUPPORT_COMPARE=${OUT_DIR}/gt_stage1_stage2
TRANSFER_ROOT=${OUT_DIR}/same_ratio_other_supports
LOCAL_ROOT=/tmp/${USER}/mass_balance_fixed_pose_91441_eval_${SLURM_JOB_ID}
SHARED_ROOT=/tmp/${USER}/bwm_shared_cache
LOCAL_MODEL=${SHARED_ROOT}/Wan2.2-TI2V-5B
MODEL_LOCK=${SHARED_ROOT}/Wan2.2-TI2V-5B.lock

mkdir -p "$OUT_DIR" "$STAGE1_RAW" "$STAGE2_RAW" "$SUPPORT_COMPARE" \
  "$TRANSFER_ROOT/raw" "$TRANSFER_ROOT/grids" "$LOCAL_ROOT" "$SHARED_ROOT" logs/slurm

if [ ! -f "$CKPT" ] || [ ! -f "$TABLE" ]; then
  echo "Missing paired final checkpoint/table: $CKPT $TABLE" >&2
  exit 3
fi

LOCAL_CKPT=${LOCAL_ROOT}/$(basename "$CKPT")
LOCAL_TABLE=${LOCAL_ROOT}/$(basename "$TABLE")
rsync -a "$CKPT" "$LOCAL_CKPT"
rsync -a "$TABLE" "$LOCAL_TABLE"

.venv/bin/python - <<PY
import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("$META").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
candidates = [
    json.loads(line)
    for line in Path("$REFERENCE_CANDIDATES").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ratio_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 19]
window_index = 1
sources = []
mappings = []
selection_rows = []

for ratio_index in ratio_indices:
    ratio_candidates = []
    for candidate in candidates:
        sample_indices = [int(value) for value in candidate.get("sample_indices", [])]
        if len(sample_indices) != 1:
            continue
        sample_index = sample_indices[0]
        row = rows[sample_index]
        if (
            int(row["ratio_index"]) == ratio_index
            and int(row["window_index"]) == window_index
            and candidate.get("ground_truth_value") is not None
        ):
            ratio_candidates.append(candidate)
    if len(ratio_candidates) != 15:
        raise ValueError(
            f"Expected 15 GT-scored actions for ratio_index={ratio_index}, "
            f"got {len(ratio_candidates)}"
        )
    balanced_candidates = [
        item
        for item in ratio_candidates
        if abs(float(item["ground_truth_value"])) <= 3.0
    ]
    unbalanced_candidates = [
        item
        for item in ratio_candidates
        if abs(float(item["ground_truth_value"])) > 3.0
    ]
    if not balanced_candidates:
        raise ValueError(
            f"No GT-balanced action inside [-3, 3] deg for ratio_index={ratio_index}"
        )
    if not unbalanced_candidates:
        raise ValueError(
            f"No GT-unbalanced action outside [-3, 3] deg for ratio_index={ratio_index}"
        )
    balanced_orders = [float(item["action_order"]) for item in balanced_candidates]
    support_candidate = min(
        unbalanced_candidates,
        key=lambda item: (
            min(
                abs(float(item["action_order"]) - balanced_order)
                for balanced_order in balanced_orders
            ),
            abs(float(item["action_order"])),
            abs(float(item["ground_truth_value"])),
            float(item["action_order"]),
            str(item["action_id"]),
        ),
    )
    source = int(support_candidate["sample_indices"][0])
    targets = sorted(
        [
            int(item["sample_indices"][0])
            for item in ratio_candidates
            if int(item["sample_indices"][0]) != source
        ],
        key=lambda index: int(rows[index]["support_bin_index"]),
    )
    if len(targets) != 14:
        raise ValueError(
            f"Expected 14 query actions for ratio_index={ratio_index}, got {targets}"
        )
    sources.append(source)
    mappings.append(f"{source}:" + ",".join(str(index) for index in targets))
    row = rows[source]
    selection_rows.append(
        {
            "ratio_index": ratio_index,
            "mass_ratio": float(row["right_to_left_mass_ratio"]),
            "source_index": source,
            "source_episode": int(row["episode_index"]),
            "source_support_bin": int(row["support_bin_index"]),
            "source_action_id": str(support_candidate["action_id"]),
            "source_action_order_m": float(support_candidate["action_order"]),
            "source_gt_terminal_tilt_deg": float(support_candidate["ground_truth_value"]),
            "source_window": [int(row["start_frame"]), int(row["end_frame"])],
            "target_indices": targets,
            "target_support_bins": [int(rows[index]["support_bin_index"]) for index in targets],
        }
    )

active = sorted({float(row["friction_mu"]) for row in rows})
Path("$OUT_DIR/source_indices.txt").write_text(
    ",".join(str(index) for index in sources) + "\n",
    encoding="utf-8",
)
Path("$OUT_DIR/target_mapping.txt").write_text(
    ";".join(mappings) + "\n",
    encoding="utf-8",
)
Path("$OUT_DIR/active_ratios.txt").write_text(
    ",".join(f"{value:.10g}" for value in active) + "\n",
    encoding="utf-8",
)
Path("$OUT_DIR/source_support_bins.txt").write_text(
    ",".join(str(item["source_support_bin"]) for item in selection_rows) + "\n",
    encoding="utf-8",
)
Path("$OUT_DIR/evaluation_selection.json").write_text(
    json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with Path("$OUT_DIR/support_selection.tsv").open("w", encoding="utf-8") as handle:
    handle.write(
        "sample_index\tratio_index\tright_to_left_mass_ratio\t"
        "episode_index\tsupport_bin_index\twindow\n"
    )
    for item in selection_rows:
        handle.write(
            f"{item['source_index']}\t{item['ratio_index']}\t{item['mass_ratio']}\t"
            f"{item['source_episode']}\t{item['source_support_bin']}\t40-120\n"
        )
PY

SOURCES=$(tr -d '\n' < "$OUT_DIR/source_indices.txt")
TARGETS_BY_SOURCE=$(tr -d '\n' < "$OUT_DIR/target_mapping.txt")
ACTIVE=$(tr -d '\n' < "$OUT_DIR/active_ratios.txt")
SOURCE_SUPPORT_BINS=$(tr -d '\n' < "$OUT_DIR/source_support_bins.txt")

cat > "$OUT_DIR/evaluation_manifest.txt" <<EOF
training_job=$TRAIN_JOB
checkpoint=$CKPT
context_table=$TABLE
config=$CONFIG
source_indices=$SOURCES
support_selection=oracle_gt_nearest_unbalanced_to_beam_center_tilt_outside_3deg
source_support_bins=$SOURCE_SUPPORT_BINS
source_window=40-120
stage2_initial_context=active_table_mean
stage2_schedule=3.0:10,1.5:10,0.5:10,0.15:10
stage2_context_dtype=fp32
transfer_support_bins=0,1,2,3,4,5,6,8,9,10,11,12,13,14
transfer_window=40-120
video_fps=10
EOF

(
  flock 9
  if [ ! -f "${LOCAL_MODEL}/.copy_complete" ]; then
    echo "[stage] populate shared model cache ${LOCAL_MODEL}"
    mkdir -p "$LOCAL_MODEL"
    rsync -a models/Wan2.2-TI2V-5B/ "$LOCAL_MODEL/"
    touch "${LOCAL_MODEL}/.copy_complete"
  else
    echo "[stage] reuse shared model cache ${LOCAL_MODEL}"
  fi
) 9>"$MODEL_LOCK"

echo "[start] checkpoint=$CKPT table=$TABLE sources=$SOURCES output=$OUT_DIR"

.venv/bin/python scripts/infer.py \
  --config "$CONFIG" \
  --model_paths "$LOCAL_MODEL" \
  --ckpt_path "$LOCAL_CKPT" \
  --physical_context_table_path "$LOCAL_TABLE" \
  --output_path "$STAGE1_RAW" \
  --sample_indices "$SOURCES" \
  --num_inference_steps 25 \
  --cfg_scale 1 \
  --fps 10 \
  --quality 6 \
  --seed 20260723

.venv/bin/python scripts/infer_stage2_ttt.py \
  --config "$CONFIG" \
  --model_paths "$LOCAL_MODEL" \
  --ckpt_path "$LOCAL_CKPT" \
  --stage2_ckpt_path "$LOCAL_CKPT" \
  --support_metadata_path "$META" \
  --ttt_support_same_as_query \
  --ttt_adapt_scope context \
  --ttt_context_fp32 \
  --ttt_initial_context_table_path "$LOCAL_TABLE" \
  --ttt_context_table_path "$LOCAL_TABLE" \
  --ttt_context_trajectory_path "$OUT_DIR/context_trajectory.jsonl" \
  --ttt_context_pca_output_path "$OUT_DIR/full_table_context_trajectory.svg" \
  --sample_indices "$SOURCES" \
  --output_path "$STAGE2_RAW" \
  --comparison_output_path "$OUT_DIR/internal_comparisons_unused" \
  --stage2_inner_steps 40 \
  --stage2_inner_lr 3.0 \
  --stage2_inner_lr_schedule "3.0:10,1.5:10,0.5:10,0.15:10" \
  --stage2_inner_grad_clip 1.0 \
  --stage2_context_reg_weight 0.001 \
  --stage2_context_clamp_min 0.0 \
  --stage2_context_clamp_max 1.0 \
  --num_inference_steps 25 \
  --cfg_scale 1 \
  --fps 10 \
  --quality 6 \
  --seed 20260723

.venv/bin/python scripts/make_gt_multi_pred_comparison.py \
  --metadata-path "$META" \
  --dataset-base-path "$BASE" \
  --pred "Stage1_oracle_ratio_Z=$STAGE1_RAW" \
  --pred "Stage2_mean_init_TTT=$STAGE2_RAW" \
  --output-dir "$SUPPORT_COMPARE" \
  --indices "$SOURCES" \
  --width 224 \
  --height 224 \
  --fps 10 \
  --quality 6 \
  --output-suffix "_gt_stage1_stage2"

.venv/bin/python scripts/analyze_context_trajectory_active_pca.py \
  --table-path "$LOCAL_TABLE" \
  --trajectory-path "$OUT_DIR/context_trajectory.jsonl" \
  --active-frictions "$ACTIVE" \
  --id-indices "$SOURCES" \
  --ood-indices "" \
  --output-svg "$OUT_DIR/active_mass_ratio_context_trajectory.svg" \
  --output-csv "$OUT_DIR/context_direction_summary.csv" \
  --parameter-label mass_ratio \
  --title "Fixed-pose mass balance C32: training-time Z table and inference-time trajectories"

.venv/bin/python scripts/infer_context_transfer_grid.py \
  --config "$CONFIG" \
  --model_paths "$LOCAL_MODEL" \
  --ckpt_path "$LOCAL_CKPT" \
  --trajectory_path "$OUT_DIR/context_trajectory.jsonl" \
  --source_indices "$SOURCES" \
  --target_indices_by_source "$TARGETS_BY_SOURCE" \
  --targets_per_source 14 \
  --parameter_label mass_ratio \
  --condition_fields "right_to_left_mass_ratio,support_bin_index" \
  --raw_output_path "$TRANSFER_ROOT/raw" \
  --grid_output_path "$TRANSFER_ROOT/grids" \
  --plan_output_path "$TRANSFER_ROOT/transfer_plan.json" \
  --num_inference_steps 25 \
  --cfg_scale 1 \
  --fps 10 \
  --quality 6 \
  --seed 20260723

for path in "$TRANSFER_ROOT"/grids/*_2x14_gt_transfer.mp4; do
  [ -e "$path" ] || continue
  base=$(basename "$path")
  mv "$path" "$TRANSFER_ROOT/grids/ID_${base/_mu/_ratio}"
done

find "$SUPPORT_COMPARE" -maxdepth 1 -name '*.mp4' -printf '%p\n' \
  | sort > "$OUT_DIR/support_videos.txt"
find "$TRANSFER_ROOT/grids" -maxdepth 1 -name '*.mp4' -printf '%p\n' \
  | sort > "$OUT_DIR/transfer_grid_videos.txt"
find "$TRANSFER_ROOT/raw" -name '*.mp4' -printf '%p\n' \
  | sort > "$OUT_DIR/transfer_raw_videos.txt"

echo "[summary] supports=$(wc -l < "$OUT_DIR/support_videos.txt") \
grids=$(wc -l < "$OUT_DIR/transfer_grid_videos.txt") \
transfers=$(wc -l < "$OUT_DIR/transfer_raw_videos.txt") output=$OUT_DIR"

ACTION_CONFIG=${OUT_DIR}/action_evaluation_dense15_boundary.yaml
ACTION_RESULT=results/mass_balance/fixed_pose_nearest_unbalanced_center_support_dense15_${SLURM_JOB_ID}/methods/ours/action_evaluation
cat > "$ACTION_CONFIG" <<YAML
version: 1
task: mass_balance
method_name: ours
metadata_jsonl: $META
dataset_root: $BASE
output_dir: $ACTION_RESULT
transfer_plans:
  - {path: $TRANSFER_ROOT/transfer_plan.json, domain: id}
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
.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py --config "$ACTION_CONFIG"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py --config "$ACTION_CONFIG"
echo "[evaluation] action=$ACTION_RESULT video_metrics=$(dirname "$ACTION_RESULT")/video_metrics domain=id"
