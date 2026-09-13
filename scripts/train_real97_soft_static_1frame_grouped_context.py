#!/usr/bin/env python3
"""Opt-in single-frame Ours with canonical target EEF and curated legal starts."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_stage1_grouped_context as grouped
from wan_video_action.data.history_prefix import CanonicalTargetEEF

_legacy_build_dataset = grouped.build_dataset


def build_static_dataset(args, runtime_config):
    actual = (int(args.num_frames), int(args.num_history_frames), int(args.frame_stride),
              args.action_type, int(args.action_dim), int(args.height), int(args.width))
    if actual != (33, 1, 3, 'eef_target', 14, 160, 320):
        raise ValueError(f'Invalid single-frame static Soft contract: {actual}')
    if str(getattr(args, 'spatial_loss_mode', 'none')) != 'none':
        raise ValueError('Static Soft uses uniform flow loss, not ROI')
    if args.task not in ('sft', 'sft:train'):
        raise ValueError('Static Soft requires flow-matching SFT')
    dataset = _legacy_build_dataset(args, runtime_config)
    for row in dataset.data:
        if row.get('dataset_split') != 'train' or row.get('action_semantics') != 'eef_target':
            raise ValueError('Only approved train episodes with recorded target EEF are allowed')
        if (row.get('sampling_kind') != 'stationary_valid' or
                row.get('stationary_start_annotation_version') != 'soft-pull-stationary-start-v3-pre8-post3-8px'):
            raise ValueError('Do not substitute an unrestricted window manifest')
        start = int(row['start_frame'])
        if row.get('native_frame_indices') != [start + 3*k for k in range(33)]:
            raise ValueError('Video/action native indices must be start,...,start+96')
    with open(args.action_stat_path) as handle:
        stats = json.load(handle)['eef_target']
    dataset.special_operator_map['action'] = CanonicalTargetEEF(args, stats)
    print('[static_soft] annotated_starts_only; condition=1 future=32 stride=3; '
          'canonical_target_eef=8+6zeros; legacy_single_frame_flow_loss', flush=True)
    return dataset


if __name__ == '__main__':
    # No history-prefix dataset or loss replacement: retain the one-frame loss.
    grouped.build_dataset = build_static_dataset
    grouped.main()
