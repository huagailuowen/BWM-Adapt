#!/usr/bin/env bash
#SBATCH --job-name=real97-ttt-low-soak-v3
#SBATCH --partition=yejin-lo
#SBATCH --account=yejin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/real97-ttt-low-soak-v3-%j.out
#SBATCH --error=logs/slurm/real97-ttt-low-soak-v3-%j.err
set -Eeuo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export BWM_TTT_TRAIN_ENTRYPOINT=scripts/methods/benchmark_real97_ttt.py
export BWM_TTT_BENCHMARK_EXPECTED_JSON='{"ttt_environments_per_rank":3,"ttt_streams_per_environment":2,"ttt_sequence_length":4,"ttt_protocol":"oneminute_write_then_predict","ttt_max_updates":120,"ttt_saved_tensor_policy":"legacy","ttt_saved_tensor_cpu_offload":false,"ttt_backward_per_stream":true,"ttt_sync_per_update":true}'
BENCH="$ROOT/outputs/benchmark_real97_ttt_soak_v3_job${SLURM_JOB_ID}"
mkdir -p "$BENCH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$BENCH/gpu_hardware.csv"
timeout 10750s nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 2 > "$BENCH/gpu_samples.csv" 2> "$BENCH/gpu_samples.err" &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT
for task in door ball; do
  stem="real97_ttt_soak_${task}_stream_gpu_only_20260912_v3"
  config="configs/benchmarks/${stem}.yaml"
  .venv/bin/python - "$config" <<'PY'
import json, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))['training']
for key, value in json.loads(os.environ['BWM_TTT_BENCHMARK_EXPECTED_JSON']).items():
    assert cfg.get(key) == value, (key, cfg.get(key), value)
assert cfg['stage1_warmup_steps'] == 100
assert cfg['video_light_augmentation_probability'] == 0.7
assert cfg['spatial_loss_mode'] == 'none'
PY
  run="$ROOT/outputs/real97_${stem}_job${SLURM_JOB_ID}"
  printf '[soak_start] task=%s config=%s\n' "$task" "$config"
  set +e
  timeout --signal=TERM --kill-after=45s 80m bash scripts/run_real97_ttt_2gpu.sh "$config"
  code=$?
  set -e
  printf '%s\tstream_gpu_only\t%s\t%s\n' "$task" "$code" "$run" >> "$BENCH/runs.tsv"
  printf '[soak_end] task=%s exit=%s\n' "$task" "$code"
done
