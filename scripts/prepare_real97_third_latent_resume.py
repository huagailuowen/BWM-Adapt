#!/usr/bin/env python3
"""Append one independent latent per physical environment to a frozen dual run."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import yaml
from scripts.evaluation.infer_real97_standard_reference import copy_cached


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    own = parser.parse_args()
    config = yaml.safe_load(own.config.read_text())
    flat = config['training']
    output = own.output.resolve()
    source_model = Path(flat['ckpt_path'])
    source_table = Path(flat['grouped_context_resume_context_table'])
    source = Path(flat['dataset_metadata_path']).parent
    complete = source_model.parent / ('.' + source_model.name + '.complete')
    if (source_model.stem != 'step-6500' or source_table.name != 'step-6500.context_table.json'
            or not all(p.is_file() for p in (source_model, source_table, complete))):
        raise RuntimeError('An exact, complete step6500 model/context pair is required')
    summary = read_json(source / 'manifest_summary.json')
    aliases_payload = read_json(source / 'latent_aliases.json')
    aliases = copy.deepcopy(aliases_payload['records'])
    table = read_json(source_table)
    physical = summary['physical_environment_ids']
    n = len(physical)
    old_n = 2 * n
    if n not in (8, 10) or len(aliases) != old_n or len(table['records']) != old_n:
        raise RuntimeError('Expected an eight/ten physical-environment dual-latent source')
    if sorted(float(r['friction_mu']) for r in table['records']) != list(range(old_n)):
        raise RuntimeError('Source context group IDs are not the original dense dual IDs')
    if sorted(physical.values()) != list(range(n)):
        raise RuntimeError('Physical environment IDs must be dense')
    frozen = output / 'input_manifest'
    frozen.mkdir()
    snapshot = output / 'source_snapshot_step6500'
    snapshot.mkdir()
    os.link(source_model, snapshot / source_model.name)
    shutil.copy2(source_table, snapshot / source_table.name)
    shutil.copy2(complete, snapshot / complete.name)
    shutil.copy2(own.config, output / 'submitted_config.yaml')
    for name in ['test.jsonl', 'ood.jsonl', 'episode_split.jsonl', 'chunk_events.jsonl',
                 'action_stats.json', 'stage_files.txt']:
        shutil.copy2(source / name, frozen / name)
    rng = random.Random(int(flat['seed']) + 6500)
    tokens = int(flat['physical_context_tokens'])
    dim = int(flat['physical_context_dim'])
    for name, index in sorted(physical.items(), key=lambda item: item[1]):
        group = old_n + int(index)
        aliases.append(dict(virtual_group_id=group, virtual_environment=name + '__z2',
                            physical_environment_index=index, physical_environment=name,
                            latent_replica=2, chunk_pool='all_train_chunks_of_physical_environment'))
        table['records'].append(dict(friction_mu=float(group),
            context=[[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(tokens)]))
    table.update(num_groups=3 * n, context_shape=[3 * n, tokens, dim])
    expanded_table = frozen / 'resume_step6500_expanded.context_table.json'
    write_json(expanded_table, table)
    train_hash = hashlib.sha256()
    total_rows = 0
    new_rows = 0
    with (source / 'train.jsonl').open() as src, (frozen / 'train.jsonl').open('xb') as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['dataset_split'] != 'train':
                raise RuntimeError('Non-training row in source training manifest')
            items = [row]
            if int(row['latent_replica']) == 0:
                item = copy.deepcopy(row)
                physical_id = int(row['physical_environment_index'])
                alias = aliases[old_n + physical_id]
                base_sample_id = row.get('source_sample_id', row['sample_id'].rsplit('__z', 1)[0])
                item.update(sample_id=base_sample_id + '__z2', source_sample_id=base_sample_id,
                            environment=alias['virtual_environment'], environment_index=alias['virtual_group_id'],
                            virtual_environment_id=alias['virtual_group_id'], latent_replica=2,
                            friction_mu=float(alias['virtual_group_id']))
                items.append(item)
                new_rows += 1
            for item in items:
                encoded = (json.dumps(item, separators=(',', ':')) + '\n').encode()
                dst.write(encoded)
                train_hash.update(encoded)
                total_rows += 1
    if total_rows != 3 * new_rows:
        raise RuntimeError('Source replicas do not have equally sized physical chunk pools')
    source_counts = summary['train_episodes_per_environment']
    summary.update(version='20260918_third_latent_resume6500_v1', source_manifest=str(source),
                   train_manifest_sha256=train_hash.hexdigest(),
                   environment_ids={a['virtual_environment']: a['virtual_group_id'] for a in aliases},
                   virtual_environment_count=3 * n, latent_replicas_per_physical_environment=3,
                   virtual_training_episode_count=3 * int(summary['episode_counts']['train']),
                   train_episodes_per_environment={
                       a['virtual_environment']: source_counts[a['physical_environment'] + '__z0'] for a in aliases},
                   chunk_counts={**summary['chunk_counts'], 'train': total_rows},
                   batch_per_gpu=[6, 10], curriculum_groups=None,
                   continuation_schedule='6501-6700 new replica only; 6701-7100 joint all replicas and model')
    write_json(frozen / 'manifest_summary.json', summary)
    aliases_payload.update(records=aliases, source_manifest=str(source),
                           original_dual_ids_and_vectors_preserved=True,
                           newly_added_group_ids=list(range(old_n, 3 * n)),
                           new_initialization='iid Uniform(-1,1); seed=training_seed+6500')
    write_json(frozen / 'latent_aliases.json', aliases_payload)
    provenance = dict(source_model=str(source_model), source_context_table=str(source_table),
        source_step=6500, target_step=7100, additional_updates=600,
        physical_environment_count=n, old_latent_count=old_n, new_latent_count=n, total_latent_count=3 * n,
        phases=[dict(first_step=6501, last_step=6700, phase='new_context',
                     sampled_groups=list(range(old_n, 3 * n)), context_lr=0.03, model_lr=0.0,
                     old_contexts_frozen=True, model_frozen=True),
                dict(first_step=6701, last_step=7100, phase='joint',
                     sampled_groups=list(range(3 * n)), context_lr=0.005, model_lr=flat['learning_rate'])],
        optimizer_state='fresh AdamW, same weight-resume policy as 6000-to-6500; no reset at phase transition',
        old_outputs_modified=False, original_data_splits_unchanged=True,
        world_size=2, batch_per_rank=[6, 10], keep_last_ordinary_pairs=1,
        source_snapshot_outside_checkpoint_retention=str(snapshot),
        model_sha256_not_recomputed=True,
        source_context_sha256=hashlib.sha256(source_table.read_bytes()).hexdigest())
    write_json(output / 'resume_provenance.json', provenance)
    write_json(frozen / 'continuation_curriculum.json', provenance['phases'])
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    cache.mkdir(parents=True, exist_ok=True)
    wan = cache / 'Wan2.2-TI2V-5B'
    copy_cached(Path('models/Wan2.2-TI2V-5B').resolve(), wan, directory=True)
    identity = summary['source_dataset_cache_identity']
    tag = identity['task'] + '_' + identity['version'] + '_' + identity['train_manifest_sha256'][:12]
    local_data = cache / 'datasets_real' / tag
    copy_cached(Path(flat['dataset_base_path']), local_data, directory=True, files=frozen / 'stage_files.txt')
    stat = source_model.stat()
    tag = hashlib.sha256(f'{source_model}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:16]
    local_checkpoint = cache / 'resume_checkpoints' / tag / source_model.name
    copy_cached(snapshot / source_model.name, local_checkpoint)
    flat.update(dataset_base_path=str(local_data), dataset_metadata_path=str(frozen / 'train.jsonl'),
                action_stat_path=str(frozen / 'action_stats.json'), ckpt_path=str(local_checkpoint),
                grouped_context_resume_context_table=str(expanded_table),
                model_paths=str(wan), output_path=str(output))
    (output / 'runtime.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print('[continuation_prepared] ' + json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
