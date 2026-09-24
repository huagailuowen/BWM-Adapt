"""Extend the frozen collision K4 supports to K8 without changing prior runs."""
import copy
import json
from pathlib import Path

import yaml

root = Path('results/mass_collision/noleak_highmass2x_support_k1_k2_k4_all_model_predictions_v1')
methods = root / 'methods/ours_action8_highmass2x'
old = methods / 'k4/step_4300/seed_20260827'
out = methods / 'k8/step_4300/seed_20260827'
rows = [json.loads(line) for line in Path(
    'data/mass_collision_noleak_all30_mainview_bwm_full61_20260828/train.jsonl'
).read_text().splitlines() if line.strip()]
plan = copy.deepcopy(json.loads((old / 'manifest.json').read_text()))
supports = []
for item in plan:
    pool = list(map(int, item['all_action_indices']))
    chosen = list(map(int, item['support_indices']))
    if len(pool) != 9 or len(chosen) != 4:
        raise ValueError('Expected frozen nine-action pool and four supports')
    while len(chosen) < 8:
        remaining = [i for i in pool if i not in chosen]
        chosen.append(max(remaining, key=lambda i: (
            min(abs(int(rows[i]['speed_index']) - int(rows[j]['speed_index'])) for j in chosen),
            -int(rows[i]['speed_index']),
        )))
    query = [i for i in pool if i not in chosen]
    item.update(support_indices=chosen, nested_support_indices=chosen,
                query_indices=query, common_query_indices=query, target_indices=pool)
    supports.extend(rows[i] for i in chosen)
out.mkdir(parents=True, exist_ok=True)
(out / 'transfer').mkdir(exist_ok=True)
(out / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
(out / 'support_metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in supports))
for mode in ('action', 'video'):
    cfg = yaml.safe_load((old / f'{mode}_evaluation.yaml').read_text())
    cfg.update(method_name='ours_k8', support_size=8,
               use_model_predictions_for_support=True,
               output_dir=str(out / ('action_evaluation' if mode == 'action' else 'all_action_video_metrics/action_evaluation')))
    cfg['transfer_plans'] = [{
        'path': str(out / 'transfer' / f'{mode}_plan.json'),
        'domain_by_source': {int(r['source_index']): r['domain'] for r in plan},
    }]
    if mode == 'video':
        cfg['video_metrics_include_support_predictions'] = True
    (out / f'{mode}_evaluation.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
(out / 'protocol.json').write_text(json.dumps({
    'support_size': 8, 'reference_k4_manifest': str(old / 'manifest.json'),
    'additional_support_selection': 'greedy_farthest_speed_index_lower_speed_tiebreak',
    'support_loss_reduction': 'mean', 'spatial_loss_mode': 'none',
    'checkpoint': 'outputs/mass20of30_collision_noleak_mainview_c32_oldrandom_8action_highmass2x_s1_108592/step-4300.safetensors',
    'inner_steps': 40, 'inner_lr_schedule': '3.0:10,1.5:10,0.5:10,0.15:10',
    'action_outcomes': 'all nine model predictions, including support actions; GT only scores decisions',
    'video_prediction_count_per_environment': 9,
    'video_metrics_include_support_predictions': True,
    'disjoint_query_count_per_environment': 1,
    'prior_k1_k2_k4_outputs_unchanged': True,
}, indent=2) + '\n')
print(f'Prepared K8: {len(plan)} environments, {len(supports)} support clips, output={out}')
