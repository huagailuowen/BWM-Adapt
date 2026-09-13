#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --job-name=real97-cpu-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/%x-%j.err
set -euo pipefail
if [[ -z "$SLURM_JOB_ID" ]]; then
  printf 'A Slurm compute allocation is required.\n' >&2
  exit 2
fi
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export OPENCV_FFMPEG_CAPTURE_OPTIONS='threads;1'
PYTHON="$ROOT/.venv-real97-eval-20260909/bin/python"
PREVIOUS="$ROOT/outputs/evaluation_real97_dataset_audit_20260909/tracking_v1"
if [[ $# -gt 0 ]]; then PREVIOUS="$1"; fi
RUN="$ROOT/outputs/evaluation_real97_dataset_audit_20260909/tracking_cpu_job$SLURM_JOB_ID"
mkdir "$RUN"
"$PYTHON" - "$PREVIOUS" "$RUN" <<'PY'
import json, pathlib, sys
previous, output = map(pathlib.Path, sys.argv[1:])
planned = [json.loads(line) for line in (previous/'measurement_manifest.jsonl').read_text().splitlines() if line.strip()]
finished = []
source = previous/'episode_outcomes.jsonl'
if source.exists():
    lines = source.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            finished.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
            (output/'incomplete_previous_tail.txt').write_text(line)
key = lambda row: (row['task'], row['environment'], row['episode_index'])
done = {key(row) for row in finished}
pending = [row for row in planned if key(row) not in done]
for name, rows in [('completed_before_move.jsonl', finished), ('pending.jsonl', pending), ('measurement_manifest.jsonl', planned)]:
    with (output/name).open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row)+'\n')
print(f'[resume] completed={len(finished)} pending={len(pending)} planned={len(planned)}', flush=True)
PY
if [[ -s "$RUN/pending.jsonl" ]]; then
  "$PYTHON" scripts/evaluation/audit_real97_tracking.py \
    --inventory "$RUN/pending.jsonl" \
    --output "$RUN/remaining" \
    --workers 2 --other-per-env-split 0
fi
"$PYTHON" - "$RUN" <<'PY'
import json, pathlib, runpy, sys
output = pathlib.Path(sys.argv[1])
rows = []
for path in [output/'completed_before_move.jsonl', output/'remaining/episode_outcomes.jsonl']:
    if path.exists():
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
keys = [(r['task'],r['environment'],r['episode_index']) for r in rows]
if len(keys) != len(set(keys)):
    raise RuntimeError('Duplicate completed episode keys.')
planned = [json.loads(line) for line in (output/'measurement_manifest.jsonl').read_text().splitlines() if line.strip()]
expected = {(r['task'],r['environment'],r['episode_index']) for r in planned}
if set(keys) != expected:
    raise RuntimeError('Incomplete audit; refusing to publish a complete summary.')
with (output/'episode_outcomes.jsonl').open('x') as handle:
    for row in rows:
        handle.write(json.dumps(row)+'\n')
namespace = runpy.run_path('scripts/evaluation/audit_real97_tracking.py', run_name='real97_audit_summary')
namespace['summarize'](rows, output)
print('[complete]', output, 'episodes=', len(rows), flush=True)
PY
