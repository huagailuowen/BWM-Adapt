#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=ls-redblue2-ours
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail

REPO=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$REPO"
source /afs/ir/users/c/y/cyzhou05/TTT-Physics/envs/activate_bwm.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

MODEL_STEP=3100
CONFIG=configs/train/train_lightswitch_physicalpress33jitter11to22_maincam_group20_c32_3wave1400_actions8_stage1_4500.yaml
TRAIN_ROOT=outputs/lightswitch_physicalpress33jitter11to22_maincam_group20_c32_3wave1400_actions8_high2gpu_107549
CKPT=$TRAIN_ROOT/step-$MODEL_STEP.safetensors
TABLE=$TRAIN_ROOT/step-$MODEL_STEP.context_table.json
META=data/lightswitch_randominitial_absolute_eef_physicalpress33jitter11to22_maincam_group20_20260827/physical_press_event_group20_train.jsonl
BASE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/robomme_light_switch_independent_controls_random8_fixed_close_buttons_no_pause_random_initial_absolute_eef_200eps_hai-machine_lerobot
BENCHMARK=results/lightswitch/physicalpress33_all4env_support8_query15_v1
PAIR_MANIFEST=$BENCHMARK/protocol/support_query_manifest_red1_blue1_matched_action.json
ROOT=$BENCHMARK/methods/ours_red1_blue1_matched_action/step_$MODEL_STEP/seed_20260828
SUPPORT_META=$ROOT/protocol/support_red1_blue1_metadata.jsonl
SOURCE_FILE=$ROOT/protocol/source_indices.txt
TARGET_FILE=$ROOT/protocol/target_indices_by_source.txt
SOURCE_RAW=$ROOT/adapted_sources_raw
TRANSFER=$ROOT/transfer
EVAL_CONFIG=configs/evaluation/action_tasks/lightswitch_physicalpress_step3100_red1blue1_all4env_query15_v1.yaml
ACTIVE=4,8,12,19,3,7,13,18

for required in "$CKPT" "$TABLE" "$META" "$PAIR_MANIFEST"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done
mkdir -p "$ROOT/protocol" "$SOURCE_RAW" "$TRANSFER/raw" "$TRANSFER/grids_support_plus_queries" logs/methods

.venv/bin/python - "$PAIR_MANIFEST" "$META" "$SUPPORT_META" "$SOURCE_FILE" "$TARGET_FILE" <<'PY'
import json
import os
from pathlib import Path
import sys

manifest_path, metadata_path, support_path, source_path, target_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
support_indices = [int(index) for row in manifest for index in row["support_indices"]]
if len(support_indices) != 2 * len(manifest):
    raise RuntimeError("Every Light environment must provide exactly one red and one blue support.")
support_rows = [metadata[index] for index in support_indices]
for row, indices in zip(manifest, (support_indices[i:i + 2] for i in range(0, len(support_indices), 2))):
    colors = {str(metadata[index]["button_color"]) for index in indices}
    if colors != {"red", "blue"}:
        raise RuntimeError(f"Invalid support colors for {row.get('causal_class')}: {colors}")

def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)

atomic_write(support_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in support_rows))
atomic_write(source_path, ",".join(str(int(row["source_index"])) for row in manifest) + "\n")
atomic_write(
    target_path,
    ";".join(
        f"{int(row['source_index'])}:" + ",".join(str(int(index)) for index in row["query_indices"])
        for row in manifest
    ) + "\n",
)
atomic_write(Path(str(support_path) + ".manifest.json"), json.dumps(manifest, indent=2) + "\n")
print(f"[protocol] environments={len(manifest)} supports={len(support_rows)} queries={sum(len(row['query_indices']) for row in manifest)}")
PY

SOURCES=$(cat "$SOURCE_FILE")
TARGETS=$(cat "$TARGET_FILE")
SHARED=/tmp/$USER/bwm_shared_cache
MODEL=$SHARED/Wan2.2-TI2V-5B
LOCAL_CKPT=$SHARED/trained_ckpts/lightswitch-physicalpress-step-$MODEL_STEP.safetensors
LOCAL_TABLE=$SHARED/trained_ckpts/lightswitch-physicalpress-step-$MODEL_STEP.context_table.json
mkdir -p "$MODEL" "$SHARED/trained_ckpts"
(
  flock 9
  if [[ ! -f "$MODEL/.copy_complete" && ! -f "$MODEL/.bwm_stage_complete" ]]; then
    rsync -a models/Wan2.2-TI2V-5B/ "$MODEL/"
    touch "$MODEL/.copy_complete"
  fi
) 9>"$SHARED/Wan2.2-TI2V-5B.lock"
(
  flock 8
  if [[ ! -s "$LOCAL_CKPT" ]]; then rsync -a "$CKPT" "$LOCAL_CKPT"; fi
  if [[ ! -s "$LOCAL_TABLE" ]]; then rsync -a "$TABLE" "$LOCAL_TABLE"; fi
) 8>"$SHARED/trained_ckpts/lightswitch-physicalpress-step-$MODEL_STEP.lock"

nvidia-smi -L
if [[ ! -e "$ROOT/inference.complete" ]]; then
  .venv/bin/python scripts/infer_stage2_ttt.py \
    --config "$CONFIG" --model_paths "$MODEL" --ckpt_path "$LOCAL_CKPT" \
    --dataset_metadata_path "$META" --dataset_base_path "$BASE" \
    --stage2_ckpt_path "$LOCAL_CKPT" --support_metadata_path "$SUPPORT_META" \
    --support_count 2 --ttt_support_gradient_accumulation \
    --ttt_adapt_scope context --ttt_context_fp32 \
    --ttt_initial_context_table_path "$LOCAL_TABLE" --ttt_context_table_path "$LOCAL_TABLE" \
    --ttt_context_active_values "$ACTIVE" \
    --ttt_context_trajectory_path "$ROOT/context_trajectory.jsonl" \
    --ttt_context_pca_output_path "$ROOT/context_trajectory_pca.svg" \
    --resume_context_trajectory --skip_existing --stage2_fixed_timestep_index 500 \
    --frame_stride 1 --sample_indices "$SOURCES" --output_path "$SOURCE_RAW" \
    --comparison_output_path "$ROOT/internal_comparisons_unused" \
    --ttt_transfer_targets_by_source "$TARGETS" \
    --ttt_transfer_output_path "$TRANSFER/raw" \
    --ttt_transfer_plan_output_path "$TRANSFER/transfer_plan.json" \
    --ttt_transfer_skip_existing \
    --stage2_inner_steps 80 --stage2_inner_lr 3.0 \
    --stage2_inner_lr_schedule "3.0:20,1.5:20,0.5:20,0.15:20" \
    --stage2_inner_grad_clip 1.0 --stage2_context_reg_weight 0.001 \
    --stage2_context_clamp_min 0.0 --stage2_context_clamp_max 1.0 \
    --num_inference_steps 25 --cfg_scale 1 --fps 30 --quality 6 --seed 20260828
  touch "$ROOT/inference.complete"
fi

.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$META" --dataset-root "$BASE" \
  --transfer-plan "$TRANSFER/transfer_plan.json" --prediction-root "$TRANSFER/raw" \
  --output-dir "$TRANSFER/grids_support_plus_queries" \
  --width 224 --height 224 --fps 30 --quality 6 --columns 5 \
  --support-size 2 --prediction-label "Ours red+blue support query"
.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py --config "$EVAL_CONFIG"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py \
  --config "$EVAL_CONFIG" --lpips --lpips-net alex --lpips-device cuda
printf '%s\n' "$CKPT" > "$ROOT/source_checkpoint.txt"
printf '%s\n' "$PAIR_MANIFEST" > "$ROOT/support_protocol.txt"
echo "[done] Ours Light red+blue support root=$ROOT"
