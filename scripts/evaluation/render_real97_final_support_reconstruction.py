#!/usr/bin/env python3
"""Reconstruct each support using the FINAL shared Z; never adapt again here."""
import argparse
import copy
import json
from pathlib import Path
import sys

from real97_trial_common import ROOT, read_jsonl, require_compute, write_json
from infer_real97_reference_trial import stage, write_video, labeled
from prepare_real97_support_override import render_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', required=True)
    own = parser.parse_args()
    require_compute()
    config = json.loads(own.config.read_text())
    prepared = ROOT / config['tasks'][own.task]['prepared']
    output = ROOT / config['output']
    dest = output / 'support_reconstruction'
    dest.mkdir(parents=True, exist_ok=True)
    plan, runtime, checkpoint = stage(prepared, dest)
    import torch
    import numpy as np
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one allocated GPU is required')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from wan_video_action.utils import save_video, set_global_seed
    sys.argv = ['infer_stage2_ttt', '--config', str(runtime), '--stage2_ckpt_path', str(checkpoint),
                '--support_metadata_path', str(prepared / 'support.jsonl')]
    args = ttt.parse_args()
    args.dataset_metadata_path = str(prepared / 'support.jsonl')
    args.num_frames = int(plan['setting']['num_frames'])
    args.frame_stride = int(plan['setting']['frame_stride'])
    args.height = int(plan['setting']['height'])
    args.width = int(plan['setting']['width'])
    dataset = build_infer_dataset(args)
    rows = read_jsonl(prepared / 'support.jsonl')
    pipe = build_pipeline(args)
    ttt._freeze_pipe(pipe)
    fps = float(plan['frames_per_second'])
    results = []
    for env in plan['environments']:
        name = env['environment']
        state = json.loads((output / 'contexts' / (name + '.json')).read_text())
        if state['support_indices'] != env['support_indices']:
            raise RuntimeError(f'Final Z was optimized on different supports: {name}')
        z = torch.tensor(state['context'], device=pipe.device, dtype=torch.float32)
        columns = []
        for index in sorted(env['support_indices'], key=lambda i: rows[i]['action_level']):
            row = rows[index]
            key = f"s{index:04d}_{name}_L{row['action_level']:02d}_ep{row['episode_index']:06d}"
            paths = {kind: dest / 'raw' / kind / (key + '.mp4') for kind in ['gt', 'stage2']}
            comparison = dest / 'comparisons' / (key + '.mp4')
            marker = dest / 'completed' / (key + '.json')
            seed = int(plan['protocol']['seed']) + 200000 + index
            if not marker.is_file():
                sample = dataset[index]
                if tuple(sample['video'].shape) != (1, 3, args.num_frames, args.height, args.width):
                    raise RuntimeError(f'Support video shape mismatch: {sample["video"].shape}')
                if tuple(np.shape(sample['action'])) != (1, args.num_frames, 14):
                    raise RuntimeError(f'Support action shape mismatch: {np.shape(sample["action"])}')
                paths['gt'].parent.mkdir(parents=True, exist_ok=True)
                paths['stage2'].parent.mkdir(parents=True, exist_ok=True)
                save_video(sample['video'], output_path=str(paths['gt']), fps=fps, quality=8)
                rollout = prepare_sample_for_rollout(copy.copy(sample), index, pipe, args)
                rollout.update(physical_context=z, output_path=str(paths['stage2']))
                old_seed = args.seed
                args.seed = seed
                set_global_seed(seed)
                try:
                    _run_autoregressive(pipe, rollout, args)
                finally:
                    args.seed = old_seed
                videos = {kind: imageio.mimread(str(path)) for kind, path in paths.items()}
                if any(len(v) != int(row['length']) for v in videos.values()):
                    raise RuntimeError('Support reconstruction frame count mismatch')
                for kind in paths:
                    write_video(paths[kind], videos[kind], fps)
                frames = [np.vstack([labeled(videos[kind][f],
                    f"{kind} | SUPPORT FIT | {name} L{row['action_level']} | final shared Z")
                    for kind in ['gt', 'stage2']]) for f in range(int(row['length']))]
                write_video(comparison, frames, fps)
                write_json(marker, dict(index=index, row=row, seed=seed,
                    paths={k: str(v) for k, v in paths.items()}, final_context=str(output / 'contexts' / (name + '.json')),
                    adapted_again=False, heldout=False, conditioning='first frame and recorded actions only'))
                del sample, rollout, videos, frames
                torch.cuda.empty_cache()
            columns.append(comparison)
            results.append(dict(environment=name, level=row['action_level'], support_index=index, comparison=str(comparison)))
            print('[support_reconstruction_done] ' + key, flush=True)
        render_grid(columns, dest / 'grids' / (name + '_support_levels01-10_gt_stage2.mp4'))
    # Keep held-out query and train-query grids separate, ordered by action level.
    query_rows = read_jsonl(prepared / 'query.jsonl')
    for env in plan['environments']:
        for split in ['train', 'test']:
            indices = sorted([i for i in env['query_indices'] if query_rows[i]['dataset_split'] == split],
                             key=lambda i: (query_rows[i]['action_level'], i))
            comparisons = []
            for i in indices:
                r = query_rows[i]
                key = f"q{i:04d}_{env['environment']}_{split}_L{r['action_level']:02d}_ep{r['episode_index']:06d}"
                comparisons.append(output / 'comparisons' / (key + '.mp4'))
            if comparisons:
                render_grid(comparisons, output / 'grids' / split / (env['environment'] + '_gt_stage2_levels.mp4'))
    write_json(dest / 'complete.json', dict(supports=len(results), environments=len(plan['environments']),
        support_fit_not_generalization=True, per_clip_adaptation=False, results=results))


if __name__ == '__main__':
    main()
