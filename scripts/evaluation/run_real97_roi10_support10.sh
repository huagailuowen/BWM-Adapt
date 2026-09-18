#!/usr/bin/env bash
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/scripts/evaluation:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
task="$1"
config="configs/evaluation/real97_${task}_dual6500_roi10_support10_20260918_v1.json"
if [[ "${2:-infer}" == "prepare" ]]; then
    exec .venv/bin/python -u scripts/evaluation/prepare_real97_roi10_support10.py --config "$config" --task "$task"
fi
prepared="outputs/evaluation_real97_${task}_dual6500_roi10_support10_20260918_v1/${task}_prep"
output="outputs/infer_real97_${task}_dual6500_roi10_support10_20260918_v1"
.venv/bin/python -u scripts/evaluation/infer_real97_unbounded_context.py \
    --config "$config" --task "$task" --prepared "$prepared" --output "$output"
exec .venv/bin/python -u scripts/evaluation/render_real97_final_support_reconstruction.py --config "$config" --task "$task"
