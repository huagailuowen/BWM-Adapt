#!/usr/bin/env bash
#SBATCH --job-name=stick-Z-preview
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/stick-Z-preview-%j.out
#SBATCH --error=logs/stick-Z-preview-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?CPU compute allocation required}"
.venv-real97-eval-20260909/bin/python - <<'PY'
import cv2
import json
from pathlib import Path
root = Path("outputs/infer_real97_stick_cross_environment_Z_job114572")
meta = json.loads((root / "inference_complete.json").read_text())
preview = root / "previews"
preview.mkdir(exist_ok=True)
rows = []
for name in meta["grids"]:
    path = Path(name)
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if count != 41:
        raise RuntimeError(f"Incorrect grid frame count: {path}: {count}")
    snapshots = {}
    for label, index in (("middle", 20), ("last", 40)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Unreadable grid: {path} frame {index}")
        destination = preview / (path.stem + f"_{label}.png")
        if not cv2.imwrite(str(destination), frame):
            raise RuntimeError(str(destination))
        snapshots[label] = str(destination)
    cap.release()
    rows.append(dict(grid=str(path), frames=count, fps=fps, shape=list(frame.shape), previews=snapshots))
(root / "preview_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows))
PY
