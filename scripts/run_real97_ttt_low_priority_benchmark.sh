#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --gres=gpu:2
#SBATCH --constraint=141G|180G
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
BENCH="$ROOT/outputs/benchmark_real97_ttt_performance_job${SLURM_JOB_ID}"
mkdir "$BENCH"
export BWM_TTT_TRAIN_ENTRYPOINT=scripts/methods/benchmark_real97_ttt.py
export PYTHONUNBUFFERED=1
# Finite profiling of this allocation only; not a training-status polling loop.
timeout 7150s nvidia-smi \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader -l 1 > "$BENCH/gpu_samples.csv" 2> "$BENCH/gpu_samples.err" &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT
for task in door ball; do
  for variant in reference selective_offload selective_stream_backward; do
    config="configs/benchmarks/real97_ttt_perf_${task}_${variant}_20260912_v1.yaml"
    child="$ROOT/outputs/real97_$(basename "$config" .yaml)_job${SLURM_JOB_ID}"
    printf '[benchmark_start] task=%s variant=%s run=%s\n' "$task" "$variant" "$child"
    set +e
    timeout --signal=TERM --kill-after=45s 25m bash scripts/run_real97_ttt_2gpu.sh "$config"
    status=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$task" "$variant" "$status" "$child" >> "$BENCH/runs.tsv"
    printf '[benchmark_end] task=%s variant=%s status=%s\n' "$task" "$variant" "$status"
  done
done
.venv/bin/python - "$BENCH" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); result=[]
for line in (root/'runs.tsv').read_text().splitlines():
    task,variant,status,path=line.split('\t')
    marker=Path(path)/'benchmark_complete.json'
    item={'task':task,'variant':variant,'exit_code':int(status),'run':path,'completed':marker.exists()}
    if marker.exists():
        run=json.loads(marker.read_text())
        item['records']=run['records']
    result.append(item)
(root/'summary.json').write_text(json.dumps({'runs':result,'gpu_samples':str(root/'gpu_samples.csv'),
    'note':'Same-node sequential probes. One complete optimizer update per variant; not production training.'},indent=2)+'\n')
PY
