"""One CSV of active training codes and formal inference trajectories.

PCA is fitted independently per task, using only its active training codes.
Inference codes never affect the fitted basis. Original 32-D vectors are retained.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_TASK_NAMES = {'event80': 'friction'}

COLOR_SPECS = {
    'event80': ('continuous_absolute', 'friction_mu', None),
    'gravity': ('ordinal', 'gravity_mps2', None),
    'mass_collision': ('ordinal', 'target_mass_kg', None),
    'light_switch': ('categorical_environment', 'causal_class', None),
    'mass_balance': ('ordinal', 'mass_ratio', None),
    'mass_friction': ('bivariate_ordinal', 'target_table_friction_mu', 'target_mass_kg'),
}


def jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-8)


def physical(row, task):
    names = {
        'event80': ['mu_index', 'friction_mu'],
        'gravity': ['gravity_index', 'gravity_mps2'],
        'mass_collision': ['target_mass_index', 'target_mass_kg'],
        'light_switch': ['context_group_id', 'causal_class', 'red_controls_lamp', 'blue_controls_lamp'],
        'mass_balance': ['ratio_index', 'mass_ratio'],
        'mass_friction': ['environment_group_id', 'environment_id', 'target_mass_kg', 'target_table_friction_mu'],
    }[task]
    return {name: row[name] for name in names if name in row}


def annotate_color_encodings(rows, task):
    """Record the exact task-local color semantics used by the formal figure."""
    scheme, primary_key, secondary_key = COLOR_SPECS[task]
    physical_rows = [json.loads(row['physical_parameters_json']) for row in rows]
    training_physical = [
        values for row, values in zip(rows, physical_rows)
        if row['phase'] == 'training_time'
    ]
    if scheme == 'categorical_environment':
        class_order = ['neither', 'red_only', 'blue_only', 'both']
        observed = {str(values[primary_key]) for values in training_physical}
        primary_values = [value for value in class_order if value in observed]
    else:
        primary_values = sorted({float(values[primary_key]) for values in training_physical})
    primary_rank = {value: index for index, value in enumerate(primary_values)}
    secondary_values = [] if secondary_key is None else sorted(
        {float(values[secondary_key]) for values in training_physical}
    )
    def ordinal_rank(value, reference):
        """Place held-out physics between equally spaced training ranks."""
        value = float(value)
        index = bisect.bisect_left(reference, value)
        if index < len(reference) and close(reference[index], value):
            return float(index)
        if index == 0:
            return 0.0
        if index == len(reference):
            return float(len(reference) - 1)
        return float(index) - 0.5
    for row, values in zip(rows, physical_rows):
        primary_value = str(values[primary_key]) if scheme == 'categorical_environment' else float(values[primary_key])
        secondary_value = '' if secondary_key is None else float(values[secondary_key])
        if scheme == 'categorical_environment':
            mapped_primary_rank = primary_rank[primary_value]
            primary_count = len(primary_values)
        else:
            mapped_primary_rank = ordinal_rank(primary_value, primary_values)
            primary_count = len(primary_values)
        row.update({
            'color_scheme': scheme,
            'color_reference': 'training_causal_environment' if scheme == 'categorical_environment' else 'active_training_physical_values_only',
            'color_tone': 'base' if row['phase'] == 'training_time' else 'dark',
            'color_primary_key': primary_key,
            'color_primary_value': primary_value,
            'color_primary_rank': mapped_primary_rank,
            'color_primary_count': primary_count,
            'color_primary_min': primary_values[0],
            'color_primary_max': primary_values[-1],
            'color_secondary_key': secondary_key or '',
            'color_secondary_value': secondary_value,
            'color_secondary_rank': '' if secondary_key is None else ordinal_rank(secondary_value, secondary_values),
            'color_secondary_count': '' if secondary_key is None else len(secondary_values),
            'color_secondary_min': '' if secondary_key is None else secondary_values[0],
            'color_secondary_max': '' if secondary_key is None else secondary_values[-1],
        })


def task_rows(spec):
    task = spec['task']
    manifest = yaml.safe_load((ROOT / spec['active_manifest']).read_text())['selection']
    ids, env_key = manifest['active_environment_ids'], manifest['environment_key']
    training = jsonl(spec['train_metadata'])
    evaluation = training if spec['eval_metadata'] == spec['train_metadata'] else jsonl(spec['eval_metadata'])
    active = {}
    for row in training:
        if any(close(row[env_key], value) for value in ids):
            active.setdefault(float(row['friction_mu']), row)
    if len(active) != manifest['active_environment_count']:
        raise ValueError(f'{task}: active metadata count {len(active)} does not match manifest.')
    source = ROOT / spec['context_table']
    payload = source.read_bytes()
    table = json.loads(payload)['records']
    selected = []
    for key, meta in sorted(active.items()):
        matches = [(i, row) for i, row in enumerate(table) if close(row['friction_mu'], key)]
        if len(matches) != 1:
            raise ValueError(f'{task}: no unique learned code for physical/group value {key}.')
        index, row = matches[0]
        z = np.asarray(row['context'], dtype=np.float64).reshape(-1)
        if z.size != 32 or not np.isfinite(z).all():
            raise ValueError(f'{task}: expected a finite 32-D training code.')
        selected.append((index, key, meta, z))
    matrix = np.stack([item[3] for item in selected])
    mean = matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    basis = vt[:3].copy()
    for component in basis:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    denominator = float(np.square(singular).sum())
    ratios = np.square(singular[:3]) / denominator if denominator else np.zeros(3)
    trajectory_path = Path(spec['result_root']) / 'context_trajectory.jsonl'
    trajectory_bytes = (ROOT / trajectory_path).read_bytes()
    common = {
        'task': task, 'checkpoint_step': spec['checkpoint_step'], 'latent_dim': 32,
        'active_training_groups': len(selected), 'pca_fit': 'active_training_only_per_task_centered_unscaled',
        'pca_sign_rule': 'largest_absolute_loading_positive',
        'PC1_explained_variance_ratio': ratios[0], 'PC2_explained_variance_ratio': ratios[1],
        'PC3_explained_variance_ratio': ratios[2],
        'context_table_path': spec['context_table'], 'context_table_sha256': hashlib.sha256(payload).hexdigest(),
        'active_manifest_path': spec['active_manifest'], 'trajectory_path': str(trajectory_path),
        'trajectory_sha256': hashlib.sha256(trajectory_bytes).hexdigest(),
        'result_root': spec['result_root'], 'domain_note': spec.get('domain_note', ''),
    }
    def project(z, **values):
        coordinates = (z - mean) @ basis.T
        return dict(common, **values, PC1=coordinates[0], PC2=coordinates[1], PC3=coordinates[2],
                    latent_l2_norm=float(np.linalg.norm(z)), **{f'Z_{i:02d}': float(v) for i, v in enumerate(z)})
    output = []
    for index, key, meta, z in selected:
        output.append(project(z, phase='training_time', record_role='learned_environment_code',
                              domain='id', context_group_domain='id', context_table_row=index,
                              environment_id=meta[env_key], context_lookup_value=key, inner_step='',
                              is_final=True, sample_index='', sample_id='', support_size='',
                              support_indices_json='[]', physical_parameters_json=json.dumps(physical(meta, task), sort_keys=True),
                              support_physics_json='[]'))
    # If a resumed inference logged the same step again, the last complete
    # recorded value wins. Preserve the actual support set as the episode key.
    records = {}
    for line in trajectory_bytes.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        supports = tuple(sorted(map(int, row.get('support_indices') or [row['sample_index']])))
        records[(supports, int(row['inner_step']))] = row
    final_steps = {}
    for supports, step in records:
        final_steps[supports] = max(step, final_steps.get(supports, step))
    for (supports, step), row in sorted(records.items()):
        z = np.asarray(row['context_flat'], dtype=np.float64).reshape(-1)
        if z.size != 32 or not np.isfinite(z).all():
            raise ValueError(f'{task}: non-finite or non-32-D inference code.')
        meta = evaluation[int(row['sample_index'])]
        key = float(meta['friction_mu'])
        group_domain = 'id' if any(close(key, value) for value in active) else 'ood'
        role = 'initial' if step == 0 else ('final' if step == final_steps[supports] else 'intermediate')
        output.append(project(z, phase='inference_time', record_role=role,
                              domain=spec.get('inference_domain', group_domain), context_group_domain=group_domain,
                              context_table_row='', environment_id=meta[env_key], context_lookup_value=key,
                              inner_step=step, is_final=step == final_steps[supports],
                              sample_index=row['sample_index'], sample_id=row.get('sample_id', ''),
                              support_size=len(supports), support_indices_json=json.dumps(supports),
                              physical_parameters_json=json.dumps(physical(meta, task), sort_keys=True),
                              support_physics_json=json.dumps([physical(evaluation[i], task) for i in supports], sort_keys=True)))
    annotate_color_encodings(output, task)
    print(json.dumps({'task': task, 'training_rows': len(selected), 'adaptation_episodes': len(final_steps),
                      'inference_rows': len(records), 'pca_explained_variance_ratio': ratios.tolist()}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/evaluation/sim_ours_formal_latents_pca.yaml')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run PCA and postprocessing on a Slurm compute node.')
    config = yaml.safe_load((ROOT / args.config).read_text())
    internal_rows = [row for spec in config['tasks'] for row in task_rows(spec)]
    output = ROOT / config['output_csv']
    output.parent.mkdir(parents=True, exist_ok=True)
    internal_output = output.with_name(output.stem + '_internal.csv')
    internal_temporary = internal_output.with_name(
        internal_output.name + f'.partial-{os.environ["SLURM_JOB_ID"]}'
    )
    with internal_temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(internal_rows[0]))
        writer.writeheader()
        writer.writerows(internal_rows)
    os.replace(internal_temporary, internal_output)
    public_rows = [
        row for row in internal_rows
        if row['phase'] == 'training_time' or row['is_final'] is True
    ]
    rows = [{
        'task': PUBLIC_TASK_NAMES.get(row['task'], row['task']),
        'split': 'train' if row['phase'] == 'training_time' else 'test',
        'physical_parameters': json.dumps(
            json.loads(row['physical_parameters_json']), sort_keys=True, separators=(',', ':')
        ),
        'latent': json.dumps(
            [row[f'Z_{index:02d}'] for index in range(32)], separators=(',', ':')
        ),
        'PC1': row['PC1'],
        'PC2': row['PC2'],
        'PC3': row['PC3'],
    } for row in public_rows]
    temporary = output.with_name(output.name + f'.partial-{os.environ["SLURM_JOB_ID"]}')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(f'[done] {output} rows={len(rows)} columns={list(rows[0])}', flush=True)
    print(f'[done] {internal_output} rows={len(internal_rows)} internal_render_data=true', flush=True)


if __name__ == '__main__':
    main()
