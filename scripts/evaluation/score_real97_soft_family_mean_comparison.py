#!/usr/bin/env python3
"""Opt-in family/global initialization comparison with unchanged tracking rules."""
from collections import defaultdict
import copy
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import score_real97_soft_balanced9_static_matched as base

ROOT, read, write = base.ROOT, base.read, base.write
original_process, original_aggregate = base.process, base.aggregate
METHODS = ('standard', 'stage1', 'global_stage2', 'family_stage2')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def process(index, row, config, detect, tracking, filtering, cohort):
    key = base.legacy.key_for(index, row)
    output = ROOT / config['output']
    destination = output / 'cases' / key
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / config['source_inference']
    old = ROOT / config['original_metrics']
    current = read(source / 'completed' / (key + '.json'))
    standard = read(ROOT / config['standard_inference'] / 'completed' / (key + '.json'))
    old_config = read(old / f"shard{index % config['shards']:02d}" / 'config.json')
    if (old_config['detector'] != read(ROOT / config['detector_config'])
            or old_config['tracking'] != tracking or old_config['filtering'] != filtering):
        raise RuntimeError('Reference tracker configuration changed; do not mix metric versions')
    common_row = copy.deepcopy(row)
    common_row.update(length=29, native_frame_indices=row['native_frame_indices'][:29],
                      evaluation_frame_indices=[i for i in row['evaluation_frame_indices'] if i <= 28])
    provenance = []
    for kind in ('gt', 'standard', 'stage1'):
        cached = read(old / 'cases' / key / (kind + '_tracks.json'))
        previous = cached['signature']
        raw = standard['paths']['standard'] if kind == 'standard' else current['paths'][kind]
        expected = dict(row=common_row, method=kind, tracking=tracking, filtering=filtering, source=raw)
        if any(previous[field] != expected[field] for field in ('row', 'method', 'tracking', 'filtering')):
            raise RuntimeError('Reference tracks do not match the query/protocol')
        old_hash, new_hash = digest(previous['source']), digest(raw)
        reused = old_hash == new_hash
        path = destination / (kind + '_tracks.json')
        if reused and not path.exists():
            write(path, dict(signature=expected, tracks=cached['tracks']))
        provenance.append(dict(method=kind, reused=reused, original_sha256=old_hash,
                               current_sha256=new_hash, source=previous['source']))
    write(destination / 'reference_cache_reuse.json', provenance)
    original_process(index, row, config, detect, tracking, filtering, cohort)


def mean(values):
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def masked_score(gt, pred, row, config, indices):
    masked = copy.deepcopy(row)
    masked['evaluation_frame_indices'] = list(indices)
    result = base.legacy.score(gt, pred, masked, config)
    # Last common valid point is not a substitute for the real last future frame.
    end = row['evaluation_frame_indices'][-1] if row['evaluation_frame_indices'] else None
    if end not in indices:
        result['center_fde_px'] = None
    return result


def aggregate(rows, config):
    original_aggregate(rows, config)
    from infer_real97_reference_trial import write_video, labeled
    import imageio.v2 as imageio

    output = ROOT / config['output']
    old_output = ROOT / config['original_metrics']
    old_source = ROOT / config['original_inference']
    source = ROOT / config['source_inference']
    cohort = {entry['index']: entry for entry in read(
        ROOT / config['standard_inference'] / 'matched_cohort.json')['cases']}
    init = read(source / 'group_initialization.json')
    results = []
    for index, row in enumerate(rows):
        key = base.legacy.key_for(index, row)
        destination = output / 'cases' / key
        new_summary = read(destination / 'summary.json')
        old_summary = read(old_output / 'cases' / key / 'summary.json')
        if old_summary['row'] != row or new_summary['row'] != row:
            raise RuntimeError('Initialization comparison requires identical query rows')
        references = {kind: read(destination / (kind + '_tracks.json'))
                      for kind in ('gt', 'standard', 'stage1', 'stage2')}
        old_prediction = read(old_output / 'cases' / key / 'stage2_tracks.json')
        row_common = references['gt']['signature']['row']
        if (old_prediction['signature']['row'] != row_common
                or old_prediction['signature']['tracking'] != references['gt']['signature']['tracking']
                or old_prediction['signature']['filtering'] != references['gt']['signature']['filtering']):
            raise RuntimeError('Old Stage2 cache does not use the same window/tracker')
        tracks = {
            'gt': references['gt']['tracks'], 'standard': references['standard']['tracks'],
            'stage1': references['stage1']['tracks'], 'family_stage2': references['stage2']['tracks'],
            'global_stage2': old_prediction['tracks'],
        }
        wanted = row_common['evaluation_frame_indices']
        good = lambda kind, i: tracks[kind][i]['measurement_valid'] and tracks[kind][i]['center'] is not None
        pair_indices = [i for i in wanted if all(good(kind, i) for kind in ('gt', 'global_stage2', 'family_stage2'))]
        shared_indices = [i for i in wanted if all(good(kind, i) for kind in ('gt',) + METHODS)]
        natural = {kind: base.legacy.score(tracks['gt'], tracks[kind], row_common, config) for kind in METHODS}
        common = {kind: masked_score(tracks['gt'], tracks[kind], row_common, config, shared_indices) for kind in METHODS}
        paired = {kind: masked_score(tracks['gt'], tracks[kind], row_common, config, pair_indices)
                  for kind in ('global_stage2', 'family_stage2')}
        raw = read(source / 'completed' / (key + '.json'))
        old_raw = read(old_source / 'completed' / (key + '.json'))
        std_raw = read(ROOT / config['standard_inference'] / 'completed' / (key + '.json'))
        if digest(old_raw['paths']['stage2']) != digest(old_prediction['signature']['source']):
            raise RuntimeError('Old Stage2 video no longer matches the cached tracker source')
        frames = {
            'gt': base.decode(raw['paths']['gt'])[:29],
            'standard': base.decode(std_raw['paths']['standard'])[4:],
            'stage1': base.decode(raw['paths']['stage1'])[:29],
            'global_stage2': base.decode(old_raw['paths']['stage2'])[:29],
            'family_stage2': base.decode(raw['paths']['stage2'])[:29],
        }
        images = dict(standard=new_summary['image_metrics']['standard'],
                      stage1=new_summary['image_metrics']['stage1'],
                      family_stage2=new_summary['image_metrics']['stage2'],
                      global_stage2=base.image_score(frames['gt'], frames['global_stage2'], wanted))
        global_ade, family_ade = (paired[kind]['center_ade_px'] for kind in ('global_stage2', 'family_stage2'))
        result = dict(index=index, key=key, environment=row['environment'], split=row['dataset_split'],
                      family=init['environment_to_group'][row['environment']], cohort=cohort[index],
                      requested_frames=len(wanted), pair_common_frames=len(pair_indices),
                      all_methods_common_frames=len(shared_indices), metrics=natural,
                      common_mask_metrics=common, paired_stage2_metrics=paired, image_metrics=images,
                      family_minus_global_ADE_px=(family_ade-global_ade) if family_ade is not None and global_ade is not None else None,
                      family_minus_global_FDE_px=(paired['family_stage2']['center_fde_px']-paired['global_stage2']['center_fde_px'])
                        if all(paired[k]['center_fde_px'] is not None for k in paired) else None,
                      formal_metric_approved=False)
        write(destination / 'initialization_comparison.json', result)
        results.append(result)
        kinds = ('gt',) + METHODS
        video = [
            np.vstack([labeled(cv2.resize(cv2.cvtColor(frames[kind][i], cv2.COLOR_BGR2RGB), (320, 160)),
                               f"{kind} | {row['dataset_split']} ep{row['episode_index']} f{row_common['native_frame_indices'][i]}")
                       for kind in kinds])
            for i in range(29)
        ]
        write_video(output / 'comparisons_5way' / (key + '.mp4'), video, config['playback_fps'])
        panels = [
            np.hstack([base.legacy.annotated(frames[kind][i], tracks[kind][i], kind) for i in (0, 9, 18, 28)])
            for kind in kinds
        ]
        cv2.imwrite(str(destination / 'initialization_contact_sheet.jpg'), np.vstack(panels))

    groups = defaultdict(list)
    for item in results:
        groups['all'].append(item)
        groups['split/' + item['split']].append(item)
        groups['environment/' + item['environment'] + '/' + item['split']].append(item)
        groups['family/' + item['family'] + '/' + item['split']].append(item)
        label = 'both_env_seen' if item['cohort']['standard_environment_seen'] else 'ours_ID_standard_OOD'
        groups[label + '/' + item['split']].append(item)
    statistics = {}
    for name, items in groups.items():
        wanted = sum(item['requested_frames'] for item in items)
        methods = {}
        for kind in METHODS:
            methods[kind] = dict(
                queries=len(items),
                natural_ADE_px=mean([item['metrics'][kind]['center_ade_px'] for item in items]),
                common_ADE_px=mean([item['common_mask_metrics'][kind]['center_ade_px'] for item in items]),
                final_FDE_px=mean([item['metrics'][kind]['center_fde_px'] for item in items]),
                final_valid_queries=sum(item['metrics'][kind]['center_fde_px'] is not None for item in items),
                paired_coverage=sum(item['metrics'][kind]['paired_frames'] for item in items)/wanted if wanted else None,
                psnr_db=mean([item['image_metrics'][kind]['psnr_db'] for item in items]),
                ssim=mean([item['image_metrics'][kind]['ssim'] for item in items]),
            )
        changes = [item['family_minus_global_ADE_px'] for item in items if item['family_minus_global_ADE_px'] is not None]
        statistics[name] = dict(
            methods=methods, requested_frames=wanted,
            all_methods_common_coverage=sum(item['all_methods_common_frames'] for item in items)/wanted if wanted else None,
            paired_stage2_common_coverage=sum(item['pair_common_frames'] for item in items)/wanted if wanted else None,
            paired_stage2_ADE_px={kind: mean([item['paired_stage2_metrics'][kind]['center_ade_px'] for item in items])
                                 for kind in ('global_stage2', 'family_stage2')},
            paired_stage2_FDE_px={kind: mean([item['paired_stage2_metrics'][kind]['center_fde_px'] for item in items])
                                 for kind in ('global_stage2', 'family_stage2')},
            family_minus_global_ADE_px=mean(changes),
            family_better_queries=sum(value < -1e-6 for value in changes),
            family_worse_queries=sum(value > 1e-6 for value in changes),
            tied_queries=sum(abs(value) <= 1e-6 for value in changes),
        )
    latent = {}
    for env in init['environment_to_group']:
        old = read(old_source / 'contexts' / (env + '.json'))
        new = read(source / 'contexts' / (env + '.json'))
        latent[env] = dict(family=init['environment_to_group'][env],
                          global_update_L2=float(np.linalg.norm(np.asarray(old['context'])-old['initial'])),
                          family_update_L2=float(np.linalg.norm(np.asarray(new['context'])-new['initial'])),
                          final_Z_difference_L2=float(np.linalg.norm(np.asarray(new['context'])-old['context'])))
    summary = dict(
        completed_queries=len(results), groups=statistics, latent_updates=latent,
        common_future='native frames 3,6,...,84; exclude observed/padded frames',
        original_metrics=config['original_metrics'], new_inference=config['source_inference'],
        known_family_prior=True, formal_metric_approved=False, manual_review_required=True,
        source_model_and_table_steps=5500,
        caveat='Standard training environment set/history contract differs; the two Stage2 variants share checkpoint, table, supports, queries, seeds and update schedule.',
    )
    write(output / 'initialization_comparison_summary.json', summary)
    write(output / 'initialization_comparison_per_query.json', results)

    report = ['# Soft family-mean versus global-mean initialization', '',
              'Same checkpoint/table5500, supports, queries and 40-step update schedule. 1R belongs to R.',
              'Object-center units: native 512x256 crop pixels. Common future: native3..84.',
              'ADE table uses only frames valid for GT and all four methods. FDE uses the actual final eligible frame.',
              'Tracking is provisional; missing centers are not zero error. Manual overlays still require inspection.', '']
    for split in ('train', 'test'):
        report += [f'## {split}', '', '| Method | common ADE | final FDE | coverage | PSNR | SSIM |',
                   '|---|---:|---:|---:|---:|---:|']
        for kind, values in statistics['split/' + split]['methods'].items():
            fmt = lambda value: 'NA' if value is None else f'{value:.4f}'
            report.append('| ' + kind + ' | ' + ' | '.join(fmt(values[key]) for key in
                          ('common_ADE_px', 'final_FDE_px', 'paired_coverage', 'psnr_db', 'ssim')) + ' |')
        info = statistics['split/' + split]
        report += ['', f"Pair-common Stage2 ADE: {info['paired_stage2_ADE_px']}.",
                   f"Family better/worse/tied queries: {info['family_better_queries']}/{info['family_worse_queries']}/{info['tied_queries']}.", '']
    report += ['## Per-environment test comparison', '',
               '| Environment | Global ADE | Family ADE | Difference (family-global) |',
               '|---|---:|---:|---:|']
    for env in sorted({row['environment'] for row in rows}):
        values = statistics['environment/' + env + '/test']
        pair = values['paired_stage2_ADE_px']
        report.append(f"| {env} | {pair['global_stage2']} | {pair['family_stage2']} | {values['family_minus_global_ADE_px']} |")
        for split in ('train', 'test', 'all'):
            selected = [item for item in results if item['environment'] == env and (split == 'all' or item['split'] == split)]
            videos = [imageio.mimread(str(output / 'comparisons_5way' / (item['key'] + '.mp4'))) for item in selected]
            write_video(output / 'grids_5way' / f'{env}_{split}.mp4',
                        (np.hstack([video[i] for video in videos]) for i in range(29)), config['playback_fps'])
    (output / 'initialization_comparison_report.md').write_text('\n'.join(report) + '\n')
    page = ['<!doctype html><html><meta charset="utf-8"><body><h1>Soft initialization comparison</h1>',
            '<p>Rows: GT / Standard / Stage1 / global Stage2 / family Stage2. Provisional tracker overlays.</p>']
    for item in results:
        key = html.escape(item['key'])
        page += [f'<h2>{key}</h2><a href="comparisons_5way/{key}.mp4">Five-way video</a>',
                 f'<p><img loading="lazy" width="1536" src="cases/{key}/initialization_contact_sheet.jpg"></p>']
    (output / 'initialization_comparison_index.html').write_text('\n'.join(page + ['</body></html>']))
    write(output / 'initialization_comparison_complete.json', dict(queries=len(results), formal_metric_approved=False))
    print('[initialization_comparison]', json.dumps(statistics['split/test']), flush=True)


if __name__ == '__main__':
    base.process = process
    base.aggregate = aggregate
    base.main()
