#!/usr/bin/env python3
"""Align one-frame Ours and repeated-history Standard on native timestamps."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from real97_trial_common import require_compute, read_jsonl, write_json
from infer_real97_reference_trial import labeled, write_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--standard', type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    ours, standard = args.ours.resolve(), args.standard.resolve()
    completed = json.loads((ours / 'inference_complete.json').read_text())
    if completed['observed_frames'] != 1 or completed['raw_frames'] != 33:
        raise RuntimeError('Expected completed single-frame Ours inference')
    plan = json.loads((ours / 'plan.json').read_text())
    queries = read_jsonl(ours / 'query.jsonl')
    output = ours / 'grids_with_standard'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ours_indices = list(range(29))
    standard_indices = list(range(4, 33))
    fps = 20 / 3
    records = []
    for env in plan['environments']:
        columns, selections = [], []
        for index in env['query_indices']:
            row = queries[index]
            key = f"q{index:04d}_{env['environment']}_{row['dataset_split']}_ep{row['episode_index']:06d}"
            own = json.loads((ours / 'completed' / (key + '.json')).read_text())
            base = json.loads((standard / 'completed' / (key + '.json')).read_text())
            if own['row'] != row:
                raise RuntimeError(f'Ours query provenance mismatch: {key}')
            for field in ('environment', 'episode_index', 'source_episode_index', 'dataset_split',
                          'video', 'action', 'action_semantics', 'total_frames'):
                if row.get(field) != base['row'].get(field):
                    raise RuntimeError(f'Cross-method query identity mismatch: {key}, {field}')
            if row['num_history_frames'] != 1 or base['row']['num_history_frames'] != 5:
                raise RuntimeError('Unexpected conditioning-frame counts')
            if base['row']['observed_native_frame_indices'] != [0] * 5:
                raise RuntimeError('Standard is not the repeated frame-zero experiment')
            native = [row['native_frame_indices'][frame] for frame in ours_indices]
            base_native = [base['row']['native_frame_indices'][frame] for frame in standard_indices]
            if native != base_native:
                raise RuntimeError(f'Native-frame alignment mismatch: {key}')
            paths = {
                'gt': own['paths']['gt'], 'standard': base['paths']['standard'],
                'stage1': own['paths']['stage1'], 'stage2': own['paths']['stage2'],
            }
            videos = {kind: imageio.mimread(path) for kind, path in paths.items()}
            if any(len(frames) != 33 for frames in videos.values()):
                raise RuntimeError(f'Unexpected raw video length: {key}')
            shapes = {np.asarray(frame).shape for frames in videos.values() for frame in frames}
            if len(shapes) != 1:
                raise RuntimeError(f'Video image geometries differ: {key}, {shapes}')
            labels = {'gt': 'GT', 'standard': 'Standard 5500',
                      'stage1': f"Ours Stage1 {completed['model_step']}",
                      'stage2': f"Ours Stage2 {completed['model_step']}"}
            column = []
            for own_index, base_index in zip(ours_indices, standard_indices):
                column.append(np.vstack([
                    labeled(np.asarray(videos[kind][base_index if kind == 'standard' else own_index]),
                            f"{labels[kind]} | {row['dataset_split']} ep{row['episode_index']}")
                    for kind in ('gt', 'standard', 'stage1', 'stage2')
                ]))
            columns.append(column)
            selections.append({'query_index': index, 'query': key, 'split': row['dataset_split'],
                               'episode_index': row['episode_index'], 'native_frame_indices': native,
                               'source_videos': paths,
                               'valid_evaluation_frames': [i for i in row['evaluation_frame_indices'] if i < 29]})
        destination = output / (env['environment'] + '_gt_standard_stage1_stage2_train_test.mp4')
        write_video(destination, (np.hstack([column[frame] for column in columns]) for frame in range(29)), fps)
        records.append({'environment': env['environment'], 'video': str(destination), 'columns': selections})
        write_json(output / 'progress.json', {'environments': records})
        print('[grid_done] ' + str(destination), flush=True)
    write_json(output / 'alignment_manifest.json', {
        'ours': str(ours), 'standard': str(standard), 'rows': ['GT', 'Standard', 'Ours Stage1', 'Ours Stage2'],
        'ours_raw_frame_indices': ours_indices, 'standard_raw_frame_indices': standard_indices,
        'logical_native_frame_indices': list(range(0, 85, 3)), 'display_frames': 29, 'fps': fps,
        'end_padding_retained_if_present': True, 'original_outputs_modified': False,
        'environments': records,
    })
    (output / 'README.md').write_text(
        '# Time-aligned four-row grids\n\n'
        'Rows: GT, Standard step5500, Ours Stage1, Ours Stage2.\n'
        'Columns retain the original per-environment train/test query order.\n\n'
        'Standard has five repeated frame-zero observations; Ours has one.\n'
        'Use Standard raw frames 4..32 and Ours raw frames 0..28. Both refer to\n'
        'native frames 0,3,...,84, with the same end padding when necessary.\n'
        'The final four Ours predictions (native 87..96) have no Standard counterpart\n'
        'and remain available in the original raw videos and three-row grids.\n'
        'This is a visual comparison, not an approved tracking metric.\n'
    )
    write_json(output / 'complete.json', {'environments': len(records),
                                        'queries': sum(len(row['columns']) for row in records),
                                        'rows': 4, 'frames': 29})


if __name__ == '__main__':
    main()
