#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=grav-ours4300-eval
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
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
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

CONFIG=configs/train/train_gravity80_full61_curriculum_c32_old_random_stage1_16300.yaml
TRAIN_ROOT=outputs/gravity80_full61_c32_oldrandom_resume89152_step3837_to4300_112001
CKPT=$TRAIN_ROOT/step-4300.safetensors
TABLE=$TRAIN_ROOT/step-4300.context_table.json
META=data/gravity_bwm_full61_20260710/train.jsonl
BASE=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/gravity/change_gravity_lerobot_v21
ACTIVE_MANIFEST=configs/methods/gravity/manifests/ours_89152_step3837_active25.yaml
REFERENCE_PLAN=results/gravity/gravity80_uniform5id5ood_strict_v1/methods/ours/step_3837/seed_20260712/transfer/transfer_plan.json
ROOT=results/gravity/gravity80_uniform5id5ood_strict_v1/methods/ours/step_4300/seed_20260712
TRANSFER=$ROOT/transfer

for required in "$CKPT" "$TABLE" "$META" "$ACTIVE_MANIFEST" "$REFERENCE_PLAN"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 1; }
done

mapfile -t PLAN_VALUES < <(.venv/bin/python - "$REFERENCE_PLAN" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(str(int(row["source_index"])) for row in rows))
print(";".join(
    f'{int(row["source_index"])}:' + ",".join(str(int(value)) for value in row["target_indices"])
    for row in rows
))
PY
)
SOURCES=${PLAN_VALUES[0]}
TARGETS=${PLAN_VALUES[1]}

ACTIVE_VALUES=$(.venv/bin/python - "$META" "$ACTIVE_MANIFEST" <<'PY'
import json
import sys
import yaml

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
manifest = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
active_ids = [int(value) for value in manifest["selection"]["active_environment_ids"]]
value_by_id = {}
for row in rows:
    value_by_id.setdefault(int(row["gravity_index"]), float(row["friction_mu"]))
missing = [value for value in active_ids if value not in value_by_id]
if missing:
    raise ValueError(f"Active gravity ids absent from metadata: {missing}")
print(",".join(str(value_by_id[value]) for value in active_ids))
PY
)

CACHE=/tmp/$USER/bwm_shared_cache
MODEL=$CACHE/Wan2.2-TI2V-5B
LOCAL_CKPT=$CACHE/trained_ckpts/gravity-ours-step-4300.safetensors
LOCAL_TABLE=$CACHE/trained_ckpts/gravity-ours-step-4300.context_table.json
mkdir -p "$CACHE" "$CACHE/trained_ckpts" "$ROOT/support_stage2" "$TRANSFER/raw" "$TRANSFER/grids"

(
  flock 9
  if [[ ! -f "$MODEL/.copy_complete" && ! -f "$MODEL/.bwm_stage_complete" ]]; then
    mkdir -p "$MODEL"
    rsync -a models/Wan2.2-TI2V-5B/ "$MODEL/"
    touch "$MODEL/.copy_complete"
  fi
) 9>"$CACHE/Wan2.2-TI2V-5B.lock"

(
  flock 8
  if [[ ! -f "${LOCAL_CKPT}.copy_complete" || ! -s "$LOCAL_CKPT" || ! -s "$LOCAL_TABLE" ]]; then
    rsync -a "$CKPT" "$LOCAL_CKPT"
    rsync -a "$TABLE" "$LOCAL_TABLE"
    touch "${LOCAL_CKPT}.copy_complete"
  else
    echo "[cache] reuse $LOCAL_CKPT"
  fi
) 8>"$CACHE/trained_ckpts/gravity-ours-step-4300.lock"

echo "[run] sources=$SOURCES"
.venv/bin/python scripts/infer_stage2_ttt.py \
  --config "$CONFIG" --model_paths "$MODEL" --ckpt_path "$LOCAL_CKPT" \
  --dataset_metadata_path "$META" --dataset_base_path "$BASE" \
  --stage2_ckpt_path "$LOCAL_CKPT" --support_metadata_path "$META" \
  --ttt_support_same_as_query --ttt_adapt_scope context --ttt_context_fp32 \
  --ttt_initial_context_table_path "$LOCAL_TABLE" \
  --ttt_context_table_path "$LOCAL_TABLE" \
  --ttt_context_active_values "$ACTIVE_VALUES" \
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
  --num_inference_steps 25 --cfg_scale 1 --fps 20 --quality 6 --seed 20260712

.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$META" --dataset-root "$BASE" \
  --transfer-plan "$TRANSFER/transfer_plan.json" --prediction-root "$TRANSFER/raw" \
  --output-dir "$TRANSFER/grids" --width 224 --height 224 --fps 20 --quality 6 \
  --columns 5 --support-size 1 --prediction-label "Ours step-4300"

cat > "$ROOT/inference_protocol.json" <<EOF
{
  "method": "ours_random_c32",
  "checkpoint_step": 4300,
  "environment_protocol": "uniform_5id_5ood_k1",
  "support_sources": "$SOURCES",
  "inner_steps": 40,
  "inner_lr_schedule": "3.0:10,1.5:10,0.5:10,0.15:10",
  "context_dtype": "float32",
  "context_clamp": [0.0, 1.0],
  "context_regularization": 0.001,
  "model_loads_for_stage2_and_transfer": 1
}
EOF
touch "$ROOT/complete"
echo "[done] $ROOT"
