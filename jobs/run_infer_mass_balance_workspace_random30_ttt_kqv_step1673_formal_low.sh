#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb-kqv1673-eval
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
TRAIN_CONFIG=configs/train/train_mass_balance_workspace_random_30ratio_train20_mainview_ttt_kqv_prequential8_4envpergpu_2gpu_24h.yaml
CKPT_SOURCE=outputs/method_benchmarks/mass_balance_workspace_random_30ratio_train20/ttt_kqv_prequential8_4envpergpu/seed_20260902_job_111579/checkpoints/step-1673.safetensors
PROTOCOL_ROOT=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_dense15_v1/protocol
META=$PROTOCOL_ROOT/combined_train_test.jsonl
DATASET=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_workspace_random_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
OUT=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_dense15_v1/methods/ttt_kqv/step_1673/seed_20260902
MANIFEST=$OUT/input_support_query_manifest.json
for required in "$TRAIN_CONFIG" "$CKPT_SOURCE" "$META" "$PROTOCOL_ROOT/transfer_plan_id.json" "$PROTOCOL_ROOT/transfer_plan_ood.json"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done
mkdir -p "$OUT/flat"

.venv/bin/python - "$PROTOCOL_ROOT" "$MANIFEST" <<'PY'
import json, pathlib, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = []
for domain in ("id", "ood"):
    for row in json.load(open(root / f"transfer_plan_{domain}.json")):
        rows.append({**row, "domain": domain, "support_indices": [row["source_index"]], "query_indices": row["target_indices"]})
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(rows, indent=2) + "\n")
temporary.replace(output)
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

nvidia-smi -L
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
torch.empty(1, device="cuda")
print(f"[gpu-check] {torch.cuda.get_device_name(0)}", flush=True)
PY

CACHE=/tmp/$USER/bwm_shared_cache
MODEL=$CACHE/Wan2.2-TI2V-5B
BLM=$CACHE/ckpt/BLM/step-12000.safetensors
LOCAL_CKPT=$CACHE/trained_ckpts/mass-balance-ttt-kqv-step1673.safetensors
mkdir -p "$MODEL" "$(dirname "$BLM")" "$(dirname "$LOCAL_CKPT")"
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
) 7>"$CACHE/trained_ckpts/mass-balance-ttt-kqv-step1673.lock"

.venv/bin/python scripts/methods/infer_ttt_kqv_grouped.py \
  --config "$TRAIN_CONFIG" \
  --dataset_metadata_path "$META" \
  --model_paths "$MODEL" \
  --ckpt_path "$BLM" \
  --ttt_checkpoint_path "$LOCAL_CKPT" \
  --ttt_environment_key ratio_index \
  --support_indices "$SUPPORTS" \
  --sample_indices "$QUERIES" \
  --expected_supports_per_environment 1 \
  --output_path "$OUT/flat" \
  --num_inference_steps 25 --cfg_scale 1 --fps 10 --quality 6 \
  --seed 20260902 --skip_existing

.venv/bin/python scripts/evaluation/organize_grouped_transfer_outputs.py \
  --manifest "$MANIFEST" --flat-root "$OUT/flat" --output-root "$OUT" \
  --method-slug ttt_kqv
.venv/bin/python scripts/evaluation/compose_context_transfer_support_grids.py \
  --metadata-path "$META" --dataset-root "$DATASET" \
  --transfer-plan "$OUT/transfer/transfer_plan.json" \
  --prediction-root "$OUT/transfer/raw" \
  --output-dir "$OUT/grids_support_plus_queries" \
  --width 224 --height 224 --fps 10 --quality 6 --columns 5 \
  --support-size 1 --prediction-label "TTT-KQV query"
printf '%s\n' "$CKPT_SOURCE" > "$OUT/source_checkpoint.txt"
echo "[done] output=$OUT"
