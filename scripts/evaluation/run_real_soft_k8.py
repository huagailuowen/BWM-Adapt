#!/usr/bin/env python3
"""Real Soft K=8: preserve K=4 queries/supports, add train-only windows."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

import infer_real97_soft_balanced9_dual_static1 as baseline
import infer_real97_soft_family_mean_static as runner
import run_real_support_number as common


ROOT = common.ROOT
DEST = common.DEST / 'soft/k8'
SOURCE = ROOT / 'outputs/infer_real97_soft_5500_simlr_family_20260918_v1'
CONFIG = ROOT / 'configs/evaluation/real97_soft_ours_simlr_mean_20260918_v1.json'


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))


def track_reader(config):
    import cv2

    root = Path(config['tracking_cache'])
    if not root.is_absolute():
        root = ROOT / root
    cache = {}
    measured = {}
    detector = None
    opened_video = None
    capture = None

    def center(env, episode, frame, local_data, video):
        nonlocal detector, opened_video, capture
        key = (env, episode)
        if key not in cache:
            path = root / f'{env}_ep{episode:06d}' / 'tracks.jsonl'
            cache[key] = {int(item['frame']): item.get('center') for item in rows(path)} if path.exists() else {}
        found = cache[key].get(frame)
        if found is not None:
            return found
        measurement = (env, episode, frame)
        if measurement in measured:
            return measured[measurement]
        if detector is None:
            detector = baseline.SilverDetector()
        path = local_data / video
        if path != opened_video:
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(str(path))
            opened_video = path
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot decode {path} frame {frame}')
        item = detector(image)
        measured[measurement] = item.get('center') if isinstance(item, dict) else None
        return measured[measurement]

    return center


def measure(source, start, center, local_data):
    env, episode = source['environment'], int(source['episode_index'])
    stride = int(source.get('frame_stride', 3))
    samples = []
    for i in (0, 8, 16, 24, 32):
        frame = start + i * stride
        if frame >= int(source['total_frames']):
            continue
        xy = center(env, episode, frame, local_data, source['video'][0])
        if xy is not None:
            samples.append((frame, float(xy[0])))
    if len(samples) < 2:
        return {'net_dx': 0.0, 'excursion': 0.0, 'frames': [frame for frame, _ in samples]}
    xs = [x for _, x in samples]
    return {'net_dx': xs[-1] - xs[0], 'excursion': max(xs) - min(xs),
            'frames': [frame for frame, _ in samples]}


def ranked_candidate(source, starts, center, local_data, minimum_dx, minimum_excursion):
    choices = []
    for start in starts:
        result = measure(source, start, center, local_data)
        good = abs(result['net_dx']) >= minimum_dx and result['excursion'] >= minimum_excursion
        # Prefer informative, early windows; preserve the legal static frame stride.
        rank = (int(good), min(abs(result['net_dx']), 80),
                min(result['excursion'], 100), -start)
        choices.append((rank, start, result))
    return max(choices)


def extend(plan, queries, supports, physical, config, runtime):
    source_queries = rows(SOURCE / 'query.jsonl')
    source_supports = rows(SOURCE / 'support.jsonl')
    if queries != source_queries or supports != source_supports:
        raise RuntimeError('The K=4 query/support cohort changed; refusing to extend it')
    original_plan = common.read(SOURCE / 'plan.json')
    if [(e['environment'], e['support_indices'], e['query_indices']) for e in plan['environments']] != [
            (e['environment'], e['support_indices'], e['query_indices']) for e in original_plan['environments']]:
        raise RuntimeError('The K=4 environment indices changed')

    local_data = Path(yaml.safe_load(runtime.read_text())['inference']['dataset_base_path'])
    center = track_reader(config)
    forbidden = {(row['environment'], int(row['episode_index'])) for row in queries
                 if row['dataset_split'] == 'train'}
    sources = {}
    for row in physical:
        if row['dataset_split'] != 'train':
            continue
        key = (row['environment'], int(row['episode_index']))
        if key not in sources or int(row['start_frame']) < int(sources[key]['start_frame']):
            sources[key] = row

    selections = []
    for env in plan['environments']:
        name = env['environment']
        old = [supports[i] for i in env['support_indices']]
        used = {(int(row['episode_index']), int(row['start_frame'])) for row in old}
        episodes = {episode for episode, _ in used}
        candidates = []
        available = sorted(((episode, source) for (candidate_env, episode), source in sources.items()
                            if candidate_env == name and (name, episode) not in forbidden
                            and episode not in episodes), key=lambda item: item[0])
        count = min(6, len(available))
        indices = ([0] if count == 1 else
                   [round(i * (len(available) - 1) / (count - 1)) for i in range(count)])
        for index in indices:
            episode, source = available[index]
            start = int(source['start_frame'])
            rank, _, result = ranked_candidate(
                source, [start], center, local_data,
                config['support_minimum_net_dx_px'], config['support_minimum_excursion_px'])
            candidates.append((rank, episode, start, source, result))
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        chosen = candidates[:4]
        if len(chosen) + len(old) < 8 and name != 'soft-5r':
            raise RuntimeError(f'Fewer than eight disjoint train episodes for {name}')

        if len(chosen) + len(old) < 8:
            # soft-5r has six train episodes after the two fixed train queries.
            # Its remaining slots use different legal windows of train episodes.
            shifted = []
            for (candidate_env, episode), source in sources.items():
                if candidate_env != name or (name, episode) in forbidden:
                    continue
                stride = int(source.get('frame_stride', 3))
                max_start = int(source['total_frames']) - 1 - 32 * stride
                first = int(source['start_frame'])
                starts = [s for s in (first + 3 * stride, first + 6 * stride,
                                       first + 9 * stride)
                          if s <= max_start and (episode, s) not in used]
                if not starts and max_start >= first + stride:
                    starts = [first + stride * ((max_start - first) // stride)]
                if not starts:
                    continue
                rank, start, result = ranked_candidate(
                    source, starts, center, local_data,
                    config['support_minimum_net_dx_px'],
                    config['support_minimum_excursion_px'])
                shifted.append((rank, episode, start, source, result))
            shifted.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            selected_episodes = {item[1] for item in chosen}
            for item in shifted:
                if len(old) + len(chosen) == 8:
                    break
                if item[1] in selected_episodes:
                    continue
                chosen.append(item)
                selected_episodes.add(item[1])

        if len(old) + len(chosen) != 8:
            raise RuntimeError(f'Unable to select eight legal train-only chunks for {name}')
        for _, episode, start, source, result in chosen:
            if (name, episode) in forbidden or (episode, start) in used:
                raise RuntimeError(f'Illegal or duplicate Soft support: {name}/{episode}/{start}')
            row = baseline.static_window(source, 'support', start=start)
            row['sample_id'] = f"{row['sample_id']}__k8_start{start:04d}"
            row['support_direction'] = 'right' if result['net_dx'] >= 0 else 'left'
            row['support_dx_px'] = result['net_dx']
            row['support_excursion_px'] = result['excursion']
            row['support_measurement_frames'] = result['frames']
            supports.append(row)
            env['support_indices'].append(len(supports) - 1)
            used.add((episode, start))
        if supports[env['support_indices'][0]:env['support_indices'][0] + 4] != old:
            raise RuntimeError(f'Original four supports changed for {name}')
        selected = [supports[i] for i in env['support_indices']]
        if any(row['dataset_split'] != 'train' or
               (name, int(row['episode_index'])) in forbidden for row in selected):
            raise RuntimeError(f'Support/query leakage in {name}')
        if len({(int(row['episode_index']), int(row['start_frame'])) for row in selected}) != 8:
            raise RuntimeError(f'Duplicate support window in {name}')
        if len({int(row['episode_index']) for row in selected}) != (6 if name == 'soft-5r' else 8):
            raise RuntimeError(f'Unexpected distinct-episode count in {name}')
        env['direction_counts'] = {direction: sum(row.get('support_direction') == direction
                                                  for row in selected)
                                   for direction in ('left', 'right')}
        selections.append({'environment': name, 'support_indices': env['support_indices'],
                           'episode_start': [(int(row['episode_index']), int(row['start_frame']))
                                             for row in selected],
                           'new_support_quality': [
                               {'episode': item[1], 'start': item[2],
                                'net_dx_px': item[4]['net_dx'],
                                'excursion_px': item[4]['excursion']} for item in chosen]})
    plan['support_count'] = len(supports)
    plan['support_policy'] = ('Original K4 prefix plus four train-only chunks; '
                              'soft-5r permits different windows of the same train episode')
    return plan, supports, selections


def prepare_k8(original_prepare, config, output):
    base = output / '_k4_preparation'
    base_config = copy.deepcopy(config)
    base_config['output'] = str(base)
    plan, runtime, checkpoint, queries, supports = original_prepare(base_config, base)
    for key in ('model_step', 'table_step', 'snapshot', 'preferred_final_step_available'):
        if key in base_config:
            config[key] = base_config[key]
    if len(supports) != 36 or len(queries) != 36:
        raise RuntimeError('Unexpected K=4 preparation cohort')
    if queries != rows(SOURCE / 'query.jsonl'):
        raise RuntimeError('Fixed Soft queries changed')
    if (output / 'support.jsonl').exists():
        supports = rows(output / 'support.jsonl')
        plan = common.read(output / 'plan.json')
        if len(supports) != 72 or any(len(e['support_indices']) != 8 for e in plan['environments']):
            raise RuntimeError('Existing K=8 preparation is incomplete')
        if any(supports[e['support_indices'][i]] != rows(SOURCE / 'support.jsonl')[4*j+i]
               for j, e in enumerate(plan['environments']) for i in range(4)):
            raise RuntimeError('Existing K=8 support prefix changed')
    else:
        physical = rows(base / 'input_manifest/physical_train.jsonl')
        plan, supports, selections = extend(plan, queries, supports, physical, config, runtime)
        write_rows(output / 'support.jsonl', supports)
        common.write(output / 'plan.json', plan)
        common.write(output / 'support_k8_protocol.json', {
            'K': 8, 'source_K4': str(SOURCE), 'train_only': True,
            'query_unchanged': True, 'support_loss_reduction': 'mean',
            'soft_5r_duplicate_episode_different_window_permitted': True,
            'selection': selections})

    write_rows(output / 'query.jsonl', queries)
    manifest = output / 'input_manifest'
    shutil.copytree(base / 'input_manifest', manifest, dirs_exist_ok=True,
                    ignore=lambda _directory, names: {'experiment.json'} & set(names))
    shutil.copy2(base / 'group_initialization.json', output / 'group_initialization.json')
    base_stage1 = base / 'raw/stage1'
    if base_stage1.is_dir():
        target = output / 'raw/stage1'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(base_stage1.resolve(), target_is_directory=True)
    settings = yaml.safe_load(runtime.read_text())
    settings['inference']['dataset_metadata_path'] = str(output / 'query.jsonl')
    settings['inference']['output_path'] = str(output / 'raw/stage2')
    settings['inference']['ttt_support_loss_reduction'] = 'mean'
    target_runtime = output / 'runtime.yaml'
    target_runtime.write_text(yaml.safe_dump(settings, sort_keys=False))
    return plan, target_runtime, checkpoint, queries, supports


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Soft K=8 inference must run on a compute allocation')
    os.chdir(ROOT)
    DEST.mkdir(parents=True, exist_ok=True)
    config = common.read(CONFIG)
    config['output'] = str(DEST)
    config['source_inference'] = str(SOURCE)
    config['support_number_experiment'] = {'K': 8, 'source_K4': str(SOURCE),
                                            'support_loss_reduction': 'mean'}
    path = DEST / 'config.json'
    if path.exists() and common.read(path) != config:
        raise RuntimeError('Existing Soft K=8 configuration differs')
    common.write(path, config)
    if not (DEST / 'inference_complete.json').exists():
        original_prepare = runner.prepare
        runner.prepare = lambda cfg, out: prepare_k8(original_prepare, cfg, out)
        sys.argv = [str(ROOT / 'scripts/evaluation/infer_real97_soft_family_mean_static.py'),
                    '--config', str(path), '--output', str(DEST)]
        runner.main()
    if not (DEST / 'metric_pipeline_complete.json').exists():
        common.score('soft', DEST)


if __name__ == '__main__':
    main()
