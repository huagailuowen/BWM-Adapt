#!/usr/bin/env bash
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stick-contact-decision-%j.out
#SBATCH --error=logs/stick-contact-decision-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
: "${SLURM_JOB_ID:?compute allocation required}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/reevaluate_stick_contact_decision_cpu.py \
  --output "outputs/stick_terminal_contact_decision_job${SLURM_JOB_ID}"
