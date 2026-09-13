#!/usr/bin/env python3
"""Summarize completed finite TTT benchmarks, not a training monitor."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def probe_comparison(reference, candidate):
    per_parameter = []
    for name, x in reference['clipped_gradient_probes'].items():
        y = candidate['clipped_gradient_probes'][name]
        xx = sum(value * value for value in x)
        yy = sum(value * value for value in y)
        difference = sum((a - b) ** 2 for a, b in zip(x, y))
        per_parameter.append({
            'parameter': name, 'sampled_elements': len(x),
            'cosine': sum(a * b for a, b in zip(x, y)) / math.sqrt(xx * yy) if xx and yy else None,
            'relative_l2': math.sqrt(difference / xx) if xx else None,
            'max_absolute_error': max(abs(a - b) for a, b in zip(x, y)),
        })
    reference_norm = reference['preclip_gradient_norm']
    return {
        'loss_difference': candidate['loss'] - reference['loss'],
        'max_position_loss_difference': max(abs(a - b) for a, b in zip(
            candidate['position_losses'], reference['position_losses'])),
        'preclip_gradient_norm_relative_difference': (
            candidate['preclip_gradient_norm'] / reference_norm - 1 if reference_norm else None),
        'gradient_scope': 'global norm plus sampled elements, not a full elementwise comparison',
        'per_parameter': per_parameter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_directory', type=Path)
    parser.add_argument('--timezone', default='America/Los_Angeles')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    samples = []
    with (args.benchmark_directory / 'gpu_samples.csv').open() as handle:
        for row in csv.reader(handle):
            if len(row) != 7:
                continue
            try:
                stamp = datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(
                    tzinfo=ZoneInfo(args.timezone)).timestamp()
                samples.append((stamp, row[1].strip(), float(row[3].strip().split()[0]),
                                float(row[5].strip().split()[0]) / 1024,
                                float(row[6].strip().split()[0])))
            except ValueError:
                continue
    runs = {}
    failures = []
    with (args.benchmark_directory / 'runs.tsv').open() as handle:
        entries = list(csv.reader(handle, delimiter='\t'))
    for task, variant, exit_code, directory in entries:
        completion = Path(directory) / 'benchmark_complete.json'
        if int(exit_code) or not completion.is_file():
            failures.append({'task': task, 'variant': variant, 'exit_code': int(exit_code),
                             'completion_exists': completion.is_file()})
            continue
        payload = json.loads(completion.read_text())
        records = payload['records']
        ranks = [rank for record in records for rank in record['ranks']]
        intervals = [(min(r['started_epoch'] for r in record['ranks']),
                      max(r['finished_epoch'] for r in record['ranks'])) for record in records]
        selected = [row for row in samples if any(start <= row[0] <= end for start, end in intervals)]
        gpu_statistics = []
        for gpu in sorted({row[1] for row in selected}):
            values = [row for row in selected if row[1] == gpu]
            gpu_statistics.append({
                'gpu': gpu, 'samples': len(values),
                'utilization_mean_percent': statistics.mean(row[2] for row in values),
                'utilization_median_percent': statistics.median(row[2] for row in values),
                'peak_memory_gib': max(row[3] for row in values),
                'mean_power_watts': statistics.mean(row[4] for row in values),
            })
        finite = all(math.isfinite(value) for rank in ranks for value in (
            rank['loss'], rank['preclip_gradient_norm'], *rank['position_losses'],
            *(value for values in rank['clipped_gradient_probes'].values() for value in values)))
        seconds = [record['max_rank_seconds'] for record in records]
        effective = {key: payload['config'].get(key) for key in (
            'ttt_max_updates', 'ttt_saved_tensor_policy', 'ttt_saved_tensor_cpu_offload',
            'ttt_gpu_saved_tensor_budget_gib', 'ttt_backward_per_stream', 'ttt_sync_per_update',
            'ttt_environments_per_rank', 'ttt_sequence_length', 'ttt_streams_per_environment',
            'ttt_protocol', 'video_light_augmentation_enabled', 'video_light_augmentation_probability',
            'dataset_metadata_path', 'spatial_loss_mode', 'learning_rate', 'ttt_slow_learning_rate',
            'stage1_warmup_steps', 'height', 'width', 'num_frames', 'frame_stride', 'action_type')}
        result = {
            'directory': directory, 'updates': len(records), 'effective_config': effective,
            'seconds': seconds, 'steady_mean_seconds': statistics.mean(seconds[1:] or seconds),
            'losses': [record['ranks'][0]['loss'] for record in records],
            'gradient_norms': [record['ranks'][0]['preclip_gradient_norm'] for record in records],
            'peak_gpu_allocated_gib': max(rank['cuda_peak_allocated_bytes'] for rank in ranks) / 2**30,
            'gpu_allocated_gib_by_step': [max(rank['cuda_peak_allocated_bytes'] for rank in record['ranks']) / 2**30 for record in records],
            'peak_process_cpu_rss_gib': max(rank['process_peak_rss_kib'] for rank in ranks) / 2**20,
            'finite': finite, 'gpu_statistics': gpu_statistics,
            'requested_updates_completed': len(records) == payload['config']['ttt_max_updates'],
        }
        runs[(task, variant)] = (result, records)
    comparisons = {}
    for (task, variant), (result, records) in runs.items():
        reference = runs.get((task, 'reference_repeat'))
        if reference and variant != 'reference_repeat':
            result['first_step_comparison'] = probe_comparison(reference[1][0]['ranks'][0], records[0]['ranks'][0])
            result['first_step_speedup'] = reference[1][0]['max_rank_seconds'] / records[0]['max_rank_seconds']
        other = runs.get((task, 'stream_selective48'))
        if variant == 'stream_gpu_only' and other:
            pairs = zip(records, other[1])
            comparisons[task] = [{'step': a['step'], **probe_comparison(a['ranks'][0], b['ranks'][0])}
                                 for a, b in pairs]
    summary = {
        'benchmark_directory': str(args.benchmark_directory), 'failures': failures,
        'runs': {f'{task}/{variant}': result for (task, variant), (result, _) in runs.items()},
        'gpu_vs_selective_comparisons': comparisons,
        'caveats': ['Same-node first-step controls; exclude model-loading time.',
                    'GPU utilization is activity, not achieved FLOPS.',
                    'Gradient checks sample elements and compare global norms; no bitwise/full-gradient claim.',
                    'Short-run stability does not prove 5500-step convergence.'],
    }
    destination = args.benchmark_directory / 'analysis_summary.json'
    temporary = destination.with_suffix('.json.partial')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(destination)
    for name, result in summary['runs'].items():
        print(json.dumps({'run': name, 'updates': result['updates'], 'steady_seconds': result['steady_mean_seconds'],
                          'first_step_speedup': result.get('first_step_speedup'),
                          'gpu_gib': result['peak_gpu_allocated_gib'], 'cpu_gib': result['peak_process_cpu_rss_gib'],
                          'finite': result['finite'], 'gpu_statistics': result['gpu_statistics'],
                          'first_step_loss_difference': result.get('first_step_comparison', {}).get('loss_difference')}))
    print(json.dumps({'summary': str(destination), 'failures': failures}))
    if args.require_complete:
        expected = {(task, variant) for task in ('door', 'ball') for variant in (
            'reference_repeat', 'stream_gpu_only', 'stream_selective48')}
        if failures or set(runs) != expected or any(
            not result['finite'] or not result['requested_updates_completed'] for result, _ in runs.values()):
            raise SystemExit('Benchmark is incomplete or contains failures.')


if __name__ == '__main__':
    main()
