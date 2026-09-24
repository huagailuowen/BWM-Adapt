"""Prepare fixed nearest-action support diagnostics without selecting on scores."""
import argparse
import copy
import json
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-size', type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    k = args.support_size
    root = Path('results/pushbox_friction_event80/event80_nearby_informative_support_k2_k4_mean_allpred_v1')
    formal = Path('results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1')
    source_manifest = formal / 'protocol/support_query_manifest.json'
    manifest = copy.deepcopy(json.loads(source_manifest.read_text()))
    metadata_path = Path('data/push_box_bwm_event_tap_segmented80_10action_A500_offset160_stop_65_105_20260705/train.jsonl')
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    out = root / f'methods/ours/k{k}/step_7272/seed_20260708'
    out.mkdir(parents=True, exist_ok=True)
    supports = []
    plans = []
    for env in manifest['environments']:
        anchor = int(env['support_indices'][0])
        pool = list(dict.fromkeys(map(int, env['support_indices'] + env['query_indices'])))
        anchor_action = int(metadata[anchor]['action_id'])
        others = sorted((i for i in pool if i != anchor), key=lambda i: (
            abs(int(metadata[i]['action_id']) - anchor_action), int(metadata[i]['action_id'])
        ))
        chosen = [anchor] + others[:k - 1]
        if len(pool) != 10 or len(chosen) != k:
            raise ValueError(f'Invalid action pool for {env["environment_id"]}')
        ordered = sorted(pool, key=lambda i: int(metadata[i]['action_id']))
        env.update(support_indices=chosen, query_indices=ordered,
                   support_actions=[int(metadata[i]['action_id']) for i in chosen],
                   support_query_disjoint=False,
                   support_selection='nearest_action_to_original_informative_support_lower_action_tiebreak')
        supports.extend(metadata[i] for i in chosen)
        plans.append({'source_index': anchor, 'support_indices': chosen,
                      'target_indices': ordered, 'domain': env['domain']})
    manifest.update(support_size=k, support_query_disjoint=False,
                    evaluation_id=f'event80_nearby_k{k}_mean_allpred_v1',
                    protocol='Fixed nearest-action support; all ten model predictions evaluated; support loss mean.',
                    action_outcome_source='model_prediction_for_every_candidate',
                    base_protocol=str(source_manifest))
    (out / 'support_query_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'support_metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in supports))
    (out / 'all_actions_plan.json').write_text(json.dumps(plans, indent=2) + '\n')
    table = json.loads((formal / 'methods/ours_inner_schedule_ablation/proportional_60/step_7272/seed_20260708/active_context_table.json').read_text())
    if len(table['records']) != 35:
        raise ValueError('Expected 35 active training contexts')
    (out / 'active_context_table.json').write_text(json.dumps(table) + '\n')
    cfg = yaml.safe_load(Path('configs/evaluation/event80/informative_support_complete_benchmark.yaml').read_text())
    cfg.update(benchmark_root=str(root), protocol_path=str(out / 'support_query_manifest.json'),
               output_root=str(out / 'metrics/complete_v1'), use_model_predictions_for_support=True,
               methods={f'ours_k{k}_nearby_mean_allpred': str(out.relative_to(root))})
    (out / 'metric_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = {
        'out': str(out),
        'sources': ','.join(str(p['source_index']) for p in plans),
        'targets': ';'.join(str(p['source_index']) + ':' + ','.join(map(str, p['target_indices'])) for p in plans),
        'active': ','.join(format(float(r['friction_mu']), '.10g') for r in table['records']),
    }
    (out / 'prepared.json').write_text(json.dumps(env, indent=2) + '\n')
    print(json.dumps({'k': k, 'output': str(out), 'supports': [
        {'mu': e['friction_mu'], 'actions': e['support_actions']} for e in manifest['environments']
    ]}))


if __name__ == '__main__':
    main()
