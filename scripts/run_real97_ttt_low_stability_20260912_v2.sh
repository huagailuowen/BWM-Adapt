#!/usr/bin/env bash
#SBATCH --job-name=real97-ttt-low-stability-v2
#SBATCH --partition=yejin-lo
#SBATCH --account=yejin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/real97-ttt-low-stability-v2-%j.out
#SBATCH --error=logs/slurm/real97-ttt-low-stability-v2-%j.err
set -Eeuo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export BWM_TTT_TRAIN_ENTRYPOINT=scripts/methods/benchmark_real97_ttt.py
BENCH="$ROOT/outputs/benchmark_real97_ttt_stability_v2_job${SLURM_JOB_ID}"
mkdir -p "$BENCH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$BENCH/gpu_hardware.csv"
timeout 7150s nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 2 > "$BENCH/gpu_samples.csv" 2> "$BENCH/gpu_samples.err" &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT
for task in door ball; do
  for variant in reference_repeat stream_gpu_only stream_selective48; do
    stem="real97_ttt_stability_${task}_${variant}_20260912_v2"
    config="configs/benchmarks/${stem}.yaml"
    run="$ROOT/outputs/real97_${stem}_job${SLURM_JOB_ID}"
    BWM_TTT_BENCHMARK_EXPECTED_JSON=$(.venv/bin/python - "$config" "$variant" <<'PY'
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))['training']
variant = sys.argv[2]
expected = dict(ttt_environments_per_rank=3, ttt_streams_per_environment=2,
                ttt_sequence_length=4, ttt_protocol='oneminute_write_then_predict')
if variant == 'reference_repeat':
    expected.update(ttt_max_updates=1, ttt_saved_tensor_policy='legacy',
                    ttt_saved_tensor_cpu_offload=True, ttt_backward_per_stream=False,
                    ttt_sync_per_update=False)
else:
    expected.update(ttt_max_updates=10, ttt_backward_per_stream=True, ttt_sync_per_update=True)
    if variant == 'stream_gpu_only':
        expected.update(ttt_saved_tensor_policy='legacy', ttt_saved_tensor_cpu_offload=False)
    else:
        expected.update(ttt_saved_tensor_policy='selective', ttt_saved_tensor_cpu_offload=True,
                        ttt_gpu_saved_tensor_budget_gib=48.0)
for key, value in expected.items():
    assert cfg.get(key) == value, (sys.argv[1], key, cfg.get(key), value)
print(json.dumps(expected))
PY
)
    export BWM_TTT_BENCHMARK_EXPECTED_JSON
    printf '[benchmark_start] task=%s variant=%s expected=%s\n' "$task" "$variant" "$BWM_TTT_BENCHMARK_EXPECTED_JSON"
    set +e
    timeout --signal=TERM --kill-after=45s 25m bash scripts/run_real97_ttt_2gpu.sh "$config"
    code=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$task" "$variant" "$code" "$run" >> "$BENCH/runs.tsv"
    printf '[benchmark_end] task=%s variant=%s exit=%s\n' "$task" "$variant" "$code"
  done
done
