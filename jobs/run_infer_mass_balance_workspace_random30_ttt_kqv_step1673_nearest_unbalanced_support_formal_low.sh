#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=mb-kqv1673-near
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
METRIC_CONFIG=configs/evaluation/action_tasks/ttt_kqv/mass_balance_workspace_random30_step1673_id5_ood5_k1_nearest_unbalanced_support_dense15_v1.yaml
CKPT_SOURCE=outputs/method_benchmarks/mass_balance_workspace_random_30ratio_train20/ttt_kqv_prequential8_4envpergpu/seed_20260902_job_111579/checkpoints/step-1673.safetensors
PROTOCOL_ROOT=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_dense15_v1/protocol
META=$PROTOCOL_ROOT/combined_train_test.jsonl
DATASET=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/mass_balance/libero_mass_balance_workspace_random_30ratio_15support_450eps_combined_train20_test10_nominal_z_absolute_eef_lerobot_2026-09-02_hai-machine
REFERENCE_CANDIDATES=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_dense15_v1/methods/ours/step_4300/seed_20260902/action_evaluation/candidate_outcomes.jsonl
OUT=results/mass_balance/workspace_random_30ratio_noleak_id5_ood5_k1_nearest_unbalanced_support_dense15_v1/methods/ttt_kqv/step_1673/seed_20260902
MANIFEST=$OUT/input_support_query_manifest.json
for required in "$TRAIN_CONFIG" "$METRIC_CONFIG" "$CKPT_SOURCE" "$META" "$REFERENCE_CANDIDATES" "$PROTOCOL_ROOT/transfer_plan_id.json" "$PROTOCOL_ROOT/transfer_plan_ood.json"; do
  test -s "$required" || { echo "[fatal] missing $required" >&2; exit 3; }
done
mkdir -p "$OUT/flat"

.venv/bin/python - "$PROTOCOL_ROOT" "$META" "$REFERENCE_CANDIDATES" "$MANIFEST" "$OUT" <<'PY'
import json
import pathlib
import sys

protocol_root = pathlib.Path(sys.argv[1])
metadata_path = pathlib.Path(sys.argv[2])
candidate_path = pathlib.Path(sys.argv[3])
output_path = pathlib.Path(sys.argv[4])
output_root = pathlib.Path(sys.argv[5])
metadata = [
    json.loads(line)
    for line in metadata_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
candidates = [
    json.loads(line)
    for line in candidate_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

plans = []
selections = []
for domain in ("id", "ood"):
    for base_plan in json.loads(
        (protocol_root / f"transfer_plan_{domain}.json").read_text(encoding="utf-8")
    ):
        ratio_index = int(base_plan["source_ratio_index"])
        ratio_candidates = []
        for candidate in candidates:
            sample_indices = [int(value) for value in candidate.get("sample_indices", [])]
            if len(sample_indices) != 1 or candidate.get("ground_truth_value") is None:
                continue
            sample_index = sample_indices[0]
            row = metadata[sample_index]
            if int(row["ratio_index"]) == ratio_index and int(row["window_index"]) == 1:
                ratio_candidates.append(candidate)
        if len(ratio_candidates) != 15:
            raise ValueError(
                f"Expected 15 GT-scored actions for ratio_index={ratio_index}, "
                f"got {len(ratio_candidates)}"
            )
        balanced = [
            item for item in ratio_candidates
            if abs(float(item["ground_truth_value"])) <= 3.0
        ]
        unbalanced = [
            item for item in ratio_candidates
            if abs(float(item["ground_truth_value"])) > 3.0
        ]
        if not balanced or not unbalanced:
            raise ValueError(
                f"Need both balanced and unbalanced actions for ratio_index={ratio_index}"
            )
        balanced_orders = [float(item["action_order"]) for item in balanced]
        support_candidate = min(
            unbalanced,
            key=lambda item: (
                min(
                    abs(float(item["action_order"]) - order)
                    for order in balanced_orders
                ),
                abs(float(item["action_order"])),
                abs(float(item["ground_truth_value"])),
                float(item["action_order"]),
                str(item["action_id"]),
            ),
        )
        source_index = int(support_candidate["sample_indices"][0])
        source_row = metadata[source_index]
        target_indices = sorted(
            (
                int(item["sample_indices"][0])
                for item in ratio_candidates
                if int(item["sample_indices"][0]) != source_index
            ),
            key=lambda index: int(metadata[index]["support_bin_index"]),
        )
        if len(target_indices) != 14:
            raise ValueError(
                f"Expected 14 disjoint queries for ratio_index={ratio_index}, "
                f"got {len(target_indices)}"
            )
        source_sample_id = str(source_row.get("sample_id", f"sample{source_index:04d}"))
        target_sample_ids = [
            str(metadata[index].get("sample_id", f"sample{index:04d}"))
            for index in target_indices
        ]
        plan = {
            **base_plan,
            "domain": domain,
            "protocol_source_index": source_index,
            "source_index": source_index,
            "source_sample_id": source_sample_id,
            "source_episode_index": int(source_row["episode_index"]),
            "source_support_bin_index": int(source_row["support_bin_index"]),
            "source_sampled_support_offset_m": float(source_row["sampled_support_offset_m"]),
            "source_gt_terminal_tilt_deg": float(support_candidate["ground_truth_value"]),
            "support_selection": "gt_nearest_unbalanced_to_balanced_action_set",
            "support_indices": [source_index],
            "query_indices": target_indices,
            "target_indices": target_indices,
            "target_sample_ids": target_sample_ids,
        }
        plans.append(plan)
        selections.append({
            "domain": domain,
            "ratio_index": ratio_index,
            "mass_ratio": float(source_row["right_to_left_mass_ratio"]),
            "source_index": source_index,
            "source_sample_id": source_sample_id,
            "source_support_bin_index": int(source_row["support_bin_index"]),
            "source_sampled_support_offset_m": float(source_row["sampled_support_offset_m"]),
            "source_gt_terminal_tilt_deg": float(support_candidate["ground_truth_value"]),
            "balanced_support_bins": sorted(
                int(metadata[int(item["sample_indices"][0])]["support_bin_index"])
                for item in balanced
            ),
            "query_indices": target_indices,
        })

output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")
temporary.replace(output_path)
(output_root / "support_selection.json").write_text(
    json.dumps(selections, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(output_root / "evaluation_manifest.txt").write_text(
    "dataset_protocol=workspace_random_30ratio_train20_test10_noleak_mainview\n"
    "domains=5_id_5_ood\n"
    "support_size=1\n"
    "queries_per_environment=14\n"
    "support_selection=gt_nearest_unbalanced_to_balanced_action_set\n"
    "balanced_threshold_deg=[-3,3]\n"
    "support_window=40-120_stride2\n"
    "query_window=40-120_stride2\n"
    "candidate_gt_source=existing_dataset_ground_truth_only\n",
    encoding="utf-8",
)
PY

mapfile -t INDEX_LISTS < <(
  .venv/bin/python - "$MANIFEST" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1]))
print(",".join(str(index) for row in rows for index in row["support_indices"]))
print(",".join(str(index) for row in rows for index in row["query_indices"]))
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

.venv/bin/python scripts/evaluation/evaluate_sim_action_selection.py \
  --config "$METRIC_CONFIG"
.venv/bin/python scripts/evaluation/evaluate_sim_transfer_metrics.py \
  --config "$METRIC_CONFIG" \
  --lpips --lpips-net alex --lpips-device cuda --lpips-batch-size 8

printf '%s\n' "$CKPT_SOURCE" > "$OUT/source_checkpoint.txt"
echo "[done] output=$OUT"
