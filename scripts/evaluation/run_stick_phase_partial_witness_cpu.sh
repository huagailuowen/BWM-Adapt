#!/bin/bash
#SBATCH --job-name=stick-partial-witness
#SBATCH --account=yejin
#SBATCH --partition=yejin-lo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/stick-partial-witness-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/stick-partial-witness-%j.err
set -euo pipefail
cd /hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export STICK_PHASE_FLOW_CONFIG="$PWD/configs/evaluation/stick_phase_interval_partial_witness_v1.json"
exec .venv-real97-eval-20260909/bin/python scripts/evaluation/audit_stick_phase_interval_flow_cpu.py
