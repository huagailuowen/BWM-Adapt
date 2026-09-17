#!/usr/bin/env python3
"""Single-frame pooled Standard on the frozen nine-environment legal-start pool."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.methods import train_real97_standard_pooled as pooled
from scripts.train_real97_soft_static_1frame_grouped_context import build_static_dataset

original_parser = pooled.wan_parser


def parser_with_explicit_warmup():
    parser = original_parser()
    if '--stage1_warmup_steps' not in parser._option_string_actions:
        parser.add_argument('--stage1_warmup_steps', type=int, default=100)
    return parser


if __name__ == '__main__':
    pooled.build_dataset = build_static_dataset
    pooled.wan_parser = parser_with_explicit_warmup
    pooled.main()
