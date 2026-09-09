#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=light-kqv2200-eval
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/methods/%x-%j.err

set -euo pipefail

ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
TRAIN_CONFIG=configs/train/train_lightswitch_physicalpress_maincam_active12_ttt_kqv_prequential8_4envpergpu_2gpu_24h.yaml
CKPT_SOURCE=outputs/method_benchmarks/lightswitch_physicalpress33_maincam/ttt_kqv_prequential8_4envpergpu/seed_20260801_job_111531/checkpoints/step-2200.safetensors
MANIFEST=results/lightswitch/physicalpress33_all4env_support8_query15_v1/protocol/support_query_manifest.json
META=data/lightswitch_randominitial_absolute_eef_physicalpress33jitter11to22_maincam_group20_20260827/physical_press_event_group20_train.jsonl
DATASET=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/robomme-lightSwitch/robomme_light_switch_independent_controls_random8_fixed_close_buttons_no_pause_random_initial_absolute_eef_200eps_hai-machine_lerobot
OUT=results/lightswitch/physicalpress33_all4env_support8_query15_v1/methods/ttt_kqv/step_2200/seed_20260801
for required in "$TRAIN_CONFIG" "$CKPT_SOURCE" "$MANIFEST" "$META"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done

nvidia-smi -L
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
torch.empty(1, device="cuda")
print(f"[gpu-check] {torch.cuda.get_device_name(0)}", flush=True)
PY

mapfile -t INDEX_LISTS < <(
  .venv/bin/python - "$MANIFEST" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
print(",".join(str(x) for row in rows for x in row["support_indices"]))
print(",".join(str(x) for row in rows for x in row["query_indices"]))
PY
)
SUPPORTS=${INDEX_LISTS[0]}
QUERIES=${INDEX_LISTS[1]}

CACHE=/tmp/$USER/bwm_shared_cache
MODEL=$CACHE/Wan2.2-TI2V-5B
BLM=$CACHE/ckpt/BLM/step-12000.safetensors
LOCAL_CKPT=$CACHE/trained_ckpts/lightswitch-ttt-kqv-step2200.safetensors
mkdir -p "$MODEL" "$(dirname "$BLM")" "$(dirname "$LOCAL_CKPT")" "$OUT/flat"
(
  flock 9
  if [[ ! -f "$MODEL/.copy_complete" && ! -f "$MODEL/.bwm_stage_complete" ]]; then
    rsync -a models/Wan2.2-TI2V-5B/ "$MODEL/"
    touch "$MODEL/.copy_complete"
  fi
) 9>"$CACHE/Wan2.2-TI2V-5B.lock"
(
  flock 8
  if [[ ! -s "$BLM" ]]; then rsync -a ckpt/BLM/step-12000.safetensors "$BLM"; fi
) 8>"$CACHE/ckpt.lock"
(
  flock 7
  if [[ ! -s "$LOCAL_CKPT" ]]; then rsync -a "$CKPT_SOURCE" "$LOCAL_CKPT"; fi
) 7>"$CACHE/trained_ckpts/lightswitch-ttt-kqv-step2200.lock"

.venv/bin/python scripts/methods/infer_ttt_kqv_grouped.py \
  --config "$TRAIN_CONFIG" \
  --dataset_metadata_path "$META" \
  --model_paths "$MODEL" \
  --ckpt_path "$BLM" \
  --ttt_checkpoint_path "$LOCAL_CKPT" \
  --support_indices "$SUPPORTS" \
  --sample_indices "$QUERIES" \
  --expected_supports_per_environment 8 \
  --output_path "$OUT/flat" \
  --num_inference_steps 25 --cfg_scale 1 --fps 10 --quality 6 \
  --seed 20260801 --skip_existing

.venv/bin/python scripts/evaluation/organize_grouped_transfer_outputs.py \
  --manifest "$MANIFEST" --flat-root "$OUT/flat" --output-root "$OUT" \
  --method-slug ttt_kqv
.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$META" --dataset-root "$DATASET" \
  --transfer-plan "$OUT/transfer/transfer_plan.json" \
  --prediction-root "$OUT/transfer/raw" \
  --output-dir "$OUT/grids_support_plus_queries" \
  --width 224 --height 224 --fps 10 --quality 6 --columns 5 \
  --support-size 8 --prediction-label "TTT-KQV query"
printf '%s\n' "$CKPT_SOURCE" > "$OUT/source_checkpoint.txt"
echo "[done] output=$OUT"
