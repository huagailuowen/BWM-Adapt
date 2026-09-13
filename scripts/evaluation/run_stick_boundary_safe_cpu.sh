#!/usr/bin/env bash
#SBATCH --job-name=stick-boundary-v8
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/stick-boundary-v8-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/stick-boundary-v8-%j.err
set -euo pipefail
ROOT=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
exec "$ROOT/.venv-real97-eval-20260909/bin/python" -u scripts/evaluation/reevaluate_stick_boundary_safe_cpu.py \
  --config configs/evaluation/stick_terminal_contact_boundary_v8.json \
  --output "outputs/stick_terminal_contact_boundary_job${SLURM_JOB_ID}"
