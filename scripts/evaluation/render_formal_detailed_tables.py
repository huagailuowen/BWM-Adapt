#!/usr/bin/env python3
"""Render detailed companions from existing formal scores, without rerunning metrics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

import build_event80_formal_ablation_table as ablation
import build_sim_all_methods_table as simulation


ROOT = Path(__file__).resolve().parents[2]
ALIASES = {
    'standard': ('standard_pooled_wm',),
    'lora': ('lora_tta',),
    'dino': ('dinov2_',),
    'ttt_kqv': ('ttt_kqv',),
    'ours': ('ours',),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_fde(task, method, cache):
    value = task['values'][method]
    if task['object_metric'] != 'Centroid ADE' or value.get('object') is None:
        return None, None
    if task['id'] == 'event80':
        path = ROOT / task['source']
        if path not in cache:
            cache[path] = list(csv.DictReader(path.open()))
        names = {'standard': 'standard_pooled_wm', 'lora': 'lora_tta',
                 'dino': 'dinov2_amortized_context', 'ttt_kqv': 'ttt_kqv', 'ours': 'ours'}
        rows = [r for r in cache[path] if r['method'] == names[method]]
        if len(rows) != 1 or not math.isclose(float(rows[0]['centroid_ade_px']), value['object'], rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"Formal ADE/source mismatch: {task['id']}/{method}")
        return float(rows[0]['centroid_fde_px']), str(path.relative_to(ROOT))

    recorded = task.get('metric_provenance', {}).get(method)
    if recorded:
        candidates = [ROOT / recorded / 'video_metrics/object_centric/object_summary.json']
    else:
        roots = [ROOT / task['source']]
        if task['id'] == 'mass_collision':
            roots.append(ROOT / 'results/mass_collision/noleak_grid_id5_ood5_k1_balanced_visible_or_min_action_v4')
        candidates = []
        for root in roots:
            for pattern in ('methods/*/*/seed_*/video_metrics/object_centric/object_summary.json',
                            'methods/*/train_job_*/*/seed_*/video_metrics/object_centric/object_summary.json'):
                candidates.extend(root.glob(pattern))
        candidates = [p for p in candidates if
                      p.parts[p.parts.index('methods') + 1].startswith(ALIASES[method])]
    matches = []
    for path in candidates:
        if not path.is_file():
            continue
        if path not in cache:
            cache[path] = json.loads(path.read_text())
        rows = cache[path].get('aggregation', {}).get('summary', [])
        if not rows or any(r.get('centroid_ade_px') is None or r.get('centroid_fde_px') is None for r in rows):
            continue
        ade = sum(r['centroid_ade_px'] for r in rows) / len(rows)
        if math.isclose(ade, value['object'], rel_tol=1e-7, abs_tol=1e-7):
            fde = sum(r['centroid_fde_px'] for r in rows) / len(rows)
            matches.append((fde, str(path.relative_to(ROOT))))
    if len(matches) != 1:
        raise ValueError(f"Expected one matching formal source for {task['id']}/{method}, found {matches}")
    return matches[0]


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_markdown(path, title, rows):
    headers = ['Method'] + [m[1] for m in ablation.METRICS]
    lines = [title, '',
             '| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(
            [row['label']] + [ablation.formatted(row, metric) for metric in ablation.METRICS]
        ) + ' |')
    lines += ['', 'Values copied from the current formal scoreboard; no rollout or metric recomputation.',
              'ADE and FDE are main-view object-centroid errors in pixels. Both are lower-is-better.', '']
    path.write_text('\n'.join(lines))


def render_ablation_companion(destination, stem_name, title, rows):
    stem = destination / stem_name
    ablation.render(title, rows, stem)
    order = [metric[0] for metric in ablation.METRICS]
    write_csv(stem.with_suffix('.csv'), [
        {'method': row['method'], 'label': row['label'],
         'checkpoint_step': row.get('checkpoint_step'),
         'source_scoreboard': row.get('source_scoreboard'),
         **{key: row.get(key) for key in order}}
        for row in rows
    ])
    write_ablation_markdown(stem.with_suffix('.md'), '# ' + title, rows)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Render on a Slurm compute node, not the login node.')
    os.chdir(ROOT)
    config = yaml.safe_load(Path('configs/evaluation/sim_all_methods_main_table_v1.yaml').read_text())
    output = ROOT / config['output_dir']
    cache, records, sources = {}, [], []
    for task in config['tasks']:
        for method in config['methods']:
            key = method['id']
            value = task['values'][key]
            fde, source = resolve_fde(task, key, cache)
            value['centroid_fde_px'] = fde
            centroid = task['object_metric'] == 'Centroid ADE'
            records.append({
                'task_id': task['id'], 'task': task['label'], 'method_id': key, 'method': method['label'],
                'psnr': value.get('psnr'), 'ssim': value.get('ssim'), 'lpips': value.get('lpips'),
                'centroid_ade_px': value.get('object') if centroid else None,
                'centroid_fde_px': fde,
                'object_metric_name': task['object_metric'], 'object_metric_unit': task['object_unit'],
                'object_metric_value': value.get('object'), 'action_success': value.get('action_success'),
                'fde_source': source,
            })
            if source:
                sources.append({'task': task['id'], 'method': key, 'source': source,
                                'sha256': digest(ROOT / source), 'centroid_fde_px': fde})
    # Resolve all provenance before overwriting any detailed artifacts.
    simulation.render_detailed(config, output)
    write_csv(output / 'sim_all_methods_main_table_detailed.csv', records)
    (output / 'sim_all_methods_main_table_detailed_sources.json').write_text(json.dumps({
        'render_job_id': os.environ['SLURM_JOB_ID'], 'metrics_recomputed': False,
        'aggregation': 'Same arithmetic mean of formal domain summaries as recorded ADE.',
        'sources': sources,
    }, indent=2) + '\n')

    spec = yaml.safe_load(Path('configs/evaluation/event80_formal_ablation.yaml').read_text())
    destination = ROOT / spec['output_dir']
    rows = json.loads((destination / 'scoreboard.json').read_text())
    order = ['psnr_multiview', 'ssim_multiview', 'lpips_multiview',
             'centroid_ade_px', 'centroid_fde_px', 'action_success_all']
    definitions = {m[0]: m for m in ablation.METRICS}
    ablation.METRICS = [definitions[key] for key in order]
    detailed_stem = 'event80_formal_ablation_table_detailed'
    render_ablation_companion(destination, detailed_stem,
                              spec['title'] + ': Detailed Metrics', rows)
    rows_without_c4 = [row for row in rows if row['method'] != 'ours_context_dim_4']
    without_c4_stem = 'event80_formal_ablation_table_without_c4'
    render_ablation_companion(destination, without_c4_stem,
                              spec['title'] + ': Detailed Metrics (without C=4)', rows_without_c4)
    print(json.dumps({'simulation_rows': len(records), 'ablation_rows': len(rows),
                      'simulation': str(output / 'sim_all_methods_main_table_detailed.svg'),
                      'ablation': str((destination / detailed_stem).with_suffix('.svg')),
                      'ablation_without_c4': str((destination / without_c4_stem).with_suffix('.svg'))}), flush=True)


if __name__ == '__main__':
    main()
