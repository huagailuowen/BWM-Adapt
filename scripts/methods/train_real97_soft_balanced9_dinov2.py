#!/usr/bin/env python3
"""Opt-in Soft static-start DINO; reuse the existing K=1/2 concat-MLP trainer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.methods import train_real97_dinov2 as runner
from scripts.train_real97_soft_static_1frame_grouped_context import build_static_dataset

if __name__ == '__main__':
    runner.build_dataset = build_static_dataset
    runner.main()
