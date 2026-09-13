#!/usr/bin/env python3
"""Prepare isolated support overrides and reuse already completed predictions."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def publish_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def publish_jsonl(path, rows):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    os.replace(temporary, path)


def copy_file(source, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def relocate(value, source, destination):
    if isinstance(value, str):
        return str(destination) + value[len(str(source)):] if value.startswith(str(source) + '/') else value
    if isinstance(value, list):
        return [relocate(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item, source, destination) for key, item in value.items()}
    return value


def query_key(index, row):
    return (f"q{index:04d}_{row['environment']}_{row['dataset_split']}_"
            f"L{int(row['action_level']):02d}_ep{int(row['episode_index']):06d}")


def locations(task, job_id):
    prepared = ROOT / 'outputs/evaluation_real97_support_overrides_20260911_v1' / f'job{job_id}' / f'{task}_prep'
    # Match the unchanged reference launcher's output naming exactly.
    output = ROOT / f'outputs/infer_real97_{task}_ours_reference_job{job_id}'
    return prepared, output


def invalidate_stage2(output, environments):
    """Explicitly replace predictions only; never touch GT, Stage1 or weights."""
    def belongs(stem):
        return any(stem == name or stem.startswith(name + '_') or ('_' + name + '_') in stem
                   for name in environments)
    removed = []
    folders = ['contexts', 'raw/stage2', 'completed', 'comparisons', 'grids',
               'completed_variants', 'extensions/raw/stage2', 'extensions/completed',
               'extensions/comparisons', 'extensions/grids', 'extensions/completed_variants',
               'initializations']
    for folder in folders:
        for path in (output / folder).rglob('*'):
            if not path.is_file() or not belongs(path.stem):
                continue
            if folder.endswith('completed_variants') and '_stage2' not in path.stem:
                continue
            path.unlink()
            removed.append(str(path.relative_to(output)))
    for name in ['progress.json', 'inference_complete.json', 'support_override_complete.json',
                 'training_inference_Z_pca.svg', 'unbounded_inference_complete.json',
                 'initialization_distances.json', 'initialization_context_trajectory.svg']:
        path = output / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def prepare(config_path, task, job_id):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    setting = config['tasks'][task]
    source_prepared = (ROOT / setting['source_prepared']).resolve()
    source_output = (ROOT / setting['source_inference']).resolve()
    prepared, output = locations(task, job_id)
    if output.resolve() == source_output or prepared.resolve() == source_prepared:
        raise ValueError('Refusing to modify the source experiment')
    ready = output / 'support_override_prepared.json'
    if ready.exists():
        previous = read_json(ready)
        if previous['config_sha256'] == fingerprint:
            print(f'[override] resume {task}: {output}', flush=True)
            return
        if setting.get('reset_existing_stage2', False):
            changed = set(setting['support_levels'])
            if not setting.get('partial_stage2_replacement', False):
                changed |= set(previous['changed_environments'])
            removed = invalidate_stage2(output, changed)
            publish_json(output / 'stage2_replacement_record.json', {
                'previous_config_sha256': previous['config_sha256'],
                'new_config_sha256': fingerprint, 'removed_stage2_artifacts': removed,
                'gt_and_stage1_preserved': True,
            })
        elif not config.get('allow_reconfigure_before_stage2', False):
            raise ValueError('Support configuration changed for an existing rerun')
        affected = set(setting['support_levels'])
        if not setting.get('partial_stage2_replacement', False):
            affected |= set(previous['changed_environments'])
        for name in affected:
            if (output / 'contexts' / (name + '.json')).exists():
                raise ValueError(f'Existing adapted context requires explicit invalidation: {name}')
            for folder in ['raw/stage2', 'completed', 'extensions/raw/stage2', 'extensions/completed']:
                for path in (output / folder).glob('*'):
                    if path.name.startswith(name + '_') or ('_' + name + '_') in path.name:
                        raise ValueError(f'Existing Stage2 result requires explicit invalidation: {path}')
        print(f'[override] revise unstarted Stage2 in existing directory: {output}', flush=True)

    preserve_plan = setting.get('partial_stage2_replacement', False) and (prepared / 'plan.json').exists()
    plan = read_json((prepared if preserve_plan else source_prepared) / 'plan.json')
    experiment = copy.deepcopy(plan['experiment']) if preserve_plan else read_json(source_prepared / 'experiment.json')
    support = read_jsonl((prepared if preserve_plan else source_prepared) / 'support.jsonl')
    queries = read_jsonl(source_prepared / 'query.jsonl')
    templates = read_jsonl(source_prepared / 'extension_action_templates.jsonl')
    requested = setting['support_levels']
    query_episodes = {(row['environment'], int(row['episode_index'])) for row in queries}
    source_support_count = len(support)
    selection = []
    found = set()
    fallback_cache = {}

    def additional_train_candidates(environment, level):
        manifest_path = setting.get('fallback_training_manifest')
        if not manifest_path:
            return []
        if 'windows' not in fallback_cache:
            fallback_cache['windows'] = read_jsonl(ROOT / manifest_path)
            fallback_cache['events'] = {
                (row['environment'].removesuffix('_lerobot'), int(row['episode_index'])): row
                for row in read_jsonl(ROOT / setting['fallback_chunk_events'])
            }
        metadata_path = Path(plan['source_dataset']) / (environment + '_lerobot') / 'meta/episodes.jsonl'
        episodes = read_jsonl(metadata_path)
        eligible = {int(row['episode_index']): row for row in episodes
                    if row.get('split') == 'train' and int(row.get('level', -1)) == level
                    and (environment, int(row['episode_index'])) not in query_episodes}
        by_episode = {}
        for row in fallback_cache['windows']:
            if (row['environment'] == environment and int(row['episode_index']) in eligible
                    and row['dataset_split'] == 'train' and row['sampling_kind'] == 'precise'):
                by_episode.setdefault(int(row['episode_index']), []).append(row)
        candidates = []
        for episode in sorted(by_episode):
            windows = sorted(by_episode[episode], key=lambda row: int(row['start_frame']))
            row = copy.deepcopy(windows[(len(windows)-1)//2])
            annotation = fallback_cache['events'][(environment, episode)]
            if annotation['split'] != 'train' or annotation['source'] != 'target_action':
                raise ValueError('Fallback Support must use frozen train target-action events')
            if int(annotation['num_frames']) != int(row['total_frames']):
                raise ValueError('Fallback Support event/video length mismatch')
            indices = [min(int(row['start_frame']) + k*int(row['frame_stride']), int(row['end_frame']))
                       for k in range(int(row['length']))]
            row.update(action_level=level, native_frame_indices=indices,
                       evaluation_frame_indices=[k for k in range(1, len(indices)) if indices[k] > indices[k-1]],
                       source_event_annotation=annotation['events'],
                       support_candidate_source='frozen_training_manifest_same_level_query_disjoint')
            candidates.append((None, row))
        return candidates

    # Append instead of compacting: unchanged environments retain support
    # indices, so their cached contexts keep exactly the same provenance.
    for environment in plan['environments']:
        name = environment['environment']
        if name not in requested:
            continue
        found.add(name)
        indices = []
        for level in requested[name]:
            candidates = [
                (index, support[index]) for index in environment['support_indices']
                if int(support[index]['action_level']) == level
            ]
            candidates.extend((None, row) for row in templates
                              if row['environment'] == name and int(row['action_level']) == level)
            candidates = [(index, row) for index, row in candidates
                          if row['dataset_split'] == 'train'
                          and (name, int(row['episode_index'])) not in query_episodes]
            if not candidates:
                candidates = additional_train_candidates(name, level)
            if not candidates:
                raise ValueError(f'No query-disjoint train support: {name}, level {level}')
            index, row = candidates[0]
            if index is None:
                row = copy.deepcopy(row)
                row.pop('template_index', None)
                row['support_selection_reason'] = 'explicit_user_level_override'
                index = len(support)
                support.append(row)
            indices.append(index)
            selection.append({
                'environment': name, 'level': level, 'support_index': index,
                'episode_index': row['episode_index'], 'start_frame': row['start_frame'],
                'end_frame': row['end_frame'], 'dataset_split': row['dataset_split'],
                'reused_original_support': index < source_support_count,
            })
        environment.update(
            support_indices=indices, requested_support_levels=requested[name],
            actual_support_levels=requested[name], missing_support_levels=[],
            support_exception='explicit_user_override',
        )
    if found != set(requested):
        raise ValueError(f'Unknown requested environments: {set(requested) - found}')

    prepared.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    reference = prepared / 'reference'
    if not reference.exists():
        reference.symlink_to(source_prepared / 'reference', target_is_directory=True)
    for filename in ['query.jsonl', 'extension_action_templates.jsonl']:
        copy_file(source_prepared / filename, prepared / filename)
    publish_jsonl(prepared / 'support.jsonl', support)
    experiment.setdefault(task, {}).setdefault('explicit_support_levels_by_environment', {}).update(requested)
    plan.update(prepared=str(prepared), experiment=experiment, support_count=len(support))
    initial_policy = setting.get('initial_context', 'mean_training_table')
    plan['stage2_initial'] = initial_policy
    if config.get('disable_context_clamp', False):
        plan['protocol']['ttt']['ttt_disable_context_clamp'] = True
        plan['effective_context_bounds'] = None
        plan['context_bounds_override_entrypoint'] = 'scripts/evaluation/infer_real97_unbounded_context.py'
    plan['active_support_count'] = sum(len(env['support_indices']) for env in plan['environments'])
    plan['support_manifest_retains_unused_original_rows'] = True
    plan['support_override'] = {
        'source_prepared': str(source_prepared), 'source_inference': str(source_output),
        'requested_levels': requested, 'selected': selection,
        'initial_context': initial_policy, 'reuse_source_stage1': True,
    }
    plan['protocol'][task].setdefault('explicit_support_levels_by_environment', {}).update(requested)
    publish_json(prepared / 'plan.json', plan)
    publish_json(prepared / 'experiment.json', experiment)
    with (source_prepared / 'stage_files.txt').open() as handle:
        stage_files = {line.strip() for line in handle if line.strip()}
    for row in support:
        stage_files.add(row['action'])
        stage_files.update(row['video'])
    (prepared / 'stage_files.txt').write_text('\n'.join(sorted(stage_files)) + '\n')

    changed_keys = {query_key(index, row) for index, row in enumerate(queries)
                    if row['environment'] in requested}
    changed_variant_keys = {key + '_stage2' for key in changed_keys}

    def affected_extension(stem):
        return any(stem.startswith(name + '_anchorq') for name in requested)

    def affected_grid(stem):
        return any(stem.startswith(name + '_') for name in requested)

    def reuse_folder(relative, reject):
        source = source_output / relative
        if not source.exists():
            return
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json', '.mp4') or reject(path.stem):
                continue
            destination = output / relative / path.relative_to(source)
            if path.suffix == '.json':
                if not destination.exists():
                    publish_json(destination, relocate(read_json(path), source_output, output))
            else:
                copy_file(path, destination)

    reuse_folder('raw/gt', lambda stem: False)
    reuse_folder('raw/stage1', lambda stem: False)
    reuse_folder('raw/stage2', lambda stem: stem in changed_keys)
    reuse_folder('completed_variants', lambda stem: stem in changed_variant_keys)
    reuse_folder('completed', lambda stem: stem in changed_keys)
    reuse_folder('comparisons', lambda stem: stem in changed_keys)
    reuse_folder('contexts', lambda stem: stem in requested)
    reuse_folder('grids', lambda stem: affected_grid(stem) or stem == 'level_order')
    reuse_folder('extensions/raw/stage1', lambda stem: False)
    reuse_folder('extensions/raw/stage2', affected_extension)
    reuse_folder('extensions/completed_variants', lambda stem: affected_extension(stem) and stem.endswith('_stage2'))
    reuse_folder('extensions/completed', affected_extension)
    reuse_folder('extensions/comparisons', affected_extension)
    reuse_folder('extensions/grids', affected_grid)

    record = {
        'config_sha256': fingerprint, 'task': task, 'prepared': str(prepared),
        'output': str(output), 'source_inference': str(source_output),
        'selection': selection, 'changed_environments': list(requested),
        'stage2_factual_queries_to_rerun': len(changed_keys),
        'original_query_count': len(queries),
        'context_policy': 'only_unchanged_environments_reuse_cached_contexts',
        'source_outputs_modified': False,
    }
    publish_json(ready, record)
    print(json.dumps(record, indent=2), flush=True)


def render_grid(inputs, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.partial.mp4')
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-filter_threads', '1', '-filter_complex_threads', '1']
    for path in inputs:
        command += ['-threads', '1', '-i', str(path)]
    if len(inputs) > 1:
        streams = ''.join(f'[{i}:v]' for i in range(len(inputs)))
        command += ['-filter_complex', f'{streams}hstack=inputs={len(inputs)}:shortest=1[v]', '-map', '[v]']
    else:
        command += ['-map', '0:v:0']
    command += ['-an', '-r', '20', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '4', str(temporary)]
    subprocess.run(command, check=True)
    os.replace(temporary, destination)


def regrid(config_path, task, job_id):
    config = read_json(config_path)
    requested = config['tasks'][task]['support_levels']
    prepared, output = locations(task, job_id)
    queries = read_jsonl(prepared / 'query.jsonl')
    groups = {}
    for index, row in enumerate(queries):
        groups.setdefault((row['environment'], row['dataset_split']), []).append((index, row))
    order = []
    for (environment, split), rows in sorted(groups.items()):
        rows.sort(key=lambda item: (int(item[1]['action_level']), int(item[1]['episode_index']), item[0]))
        folder = output / 'grids' if split == 'test' else output / 'grids/train'
        destination = folder / f'{environment}_gt_stage1_stage2_{split}_levels.mp4'
        if environment in requested:
            inputs = [output / 'comparisons' / (query_key(index, row) + '.mp4') for index, row in rows]
            render_grid(inputs, destination)
        order.append({
            'environment': environment, 'split': split, 'video': str(destination),
            'columns': [{'level': row['action_level'], 'episode_index': row['episode_index'],
                         'query_index': index} for index, row in rows],
        })
    # The unchanged reference engine may produce its original mixed grids.
    # Keep those as an archive rather than allowing them to obscure test grids.
    for path in (output / 'grids').glob('*_gt_stage1_stage2_train_test.mp4'):
        destination = output / 'grids/previous_mixed' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    publish_json(output / 'grids/level_order.json', {'row_order': ['GT', 'Stage1', 'Stage2'], 'grids': order})
    publish_json(output / 'support_override_complete.json', {
        'task': task, 'support_levels': requested, 'output': str(output),
        'grid_order': 'ascending_numeric_action_level', 'source_outputs_modified': False,
    })
    print(f'[override] finished {task}: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'regrid'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['door', 'ball'], required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--reuse-output-job-id', action='store_true',
                        help='Explicitly continue the output directory of an earlier allocation')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this preparation/rendering on a compute allocation')
    if args.job_id != os.environ['SLURM_JOB_ID'] and not args.reuse_output_job_id:
        raise ValueError('Output job ID must match the current allocation')
    {'prepare': prepare, 'regrid': regrid}[args.command](args.config.resolve(), args.task, args.job_id)


if __name__ == '__main__':
    main()
