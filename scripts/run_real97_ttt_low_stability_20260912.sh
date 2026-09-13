#!/usr/bin/env bash
#SBATCH --job-name=real97-ttt-low-stability
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
#SBATCH --output=logs/slurm/real97-ttt-low-stability-%j.out
#SBATCH --error=logs/slurm/real97-ttt-low-stability-%j.err
set -Eeuo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export BWM_TTT_TRAIN_ENTRYPOINT=scripts/methods/benchmark_real97_ttt.py
BENCH="$ROOT/outputs/benchmark_real97_ttt_stability_job${SLURM_JOB_ID}"
mkdir -p "$BENCH"
timeout 7150s nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 2 > "$BENCH/gpu_samples.csv" 2> "$BENCH/gpu_samples.err" &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT
for task in door ball; do
  for variant in reference_repeat stream_gpu_only stream_selective48; do
    stem="real97_ttt_stability_${task}_${variant}_20260912_v1"
    config="configs/benchmarks/${stem}.yaml"
    run="$ROOT/outputs/real97_${stem}_job${SLURM_JOB_ID}"
    printf '[benchmark_start] task=%s variant=%s config=%s\n' "$task" "$variant" "$config"
    set +e
    timeout --signal=TERM --kill-after=45s 25m bash scripts/run_real97_ttt_2gpu.sh "$config"
    code=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$task" "$variant" "$code" "$run" >> "$BENCH/runs.tsv"
    printf '[benchmark_end] task=%s variant=%s exit=%s\n' "$task" "$variant" "$code"
  done
done
