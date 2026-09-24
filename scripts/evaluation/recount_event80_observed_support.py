"""Keep the original K1 observed candidate and 90 queries fixed for every K."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Video metrics must run on a compute node')
    formal = Path('results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1')
    nearby = Path('results/pushbox_friction_event80/event80_nearby_informative_support_k2_k4_mean_allpred_v1')
    report = Path('results/support_number_analysis/event80_nearby_frozen_k1_observed_protocol_v1')
    report.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(Path('configs/evaluation/event80/informative_support_complete_benchmark.yaml').read_text())
    scores = []
    for k in (1, 2, 4):
        benchmark = formal if k == 1 else nearby
        relative = Path('methods/ours/step_7272/seed_20260708') if k == 1 else Path(f'methods/ours/k{k}/step_7272/seed_20260708')
        source = formal / 'protocol/support_query_manifest.json'
        protocol = copy.deepcopy(json.loads(source.read_text()))
        adaptation_manifest_path = source if k == 1 else nearby / relative / 'support_query_manifest.json'
        adaptation = json.loads(adaptation_manifest_path.read_text())
        adaptation_by_env = {e['environment_id']: e for e in adaptation['environments']}
        for env in protocol['environments']:
            supports = set(map(int, env['support_indices']))
            pool = list(dict.fromkeys(map(int, env['support_indices'] + env['query_indices'])))
            adapted_supports = list(map(int, adaptation_by_env[env['environment_id']]['support_indices']))
            if len(supports) != 1 or len(pool) != 10 or len(adapted_supports) != k or not supports.issubset(adapted_supports):
                raise ValueError(f'K{k}: unexpected support/candidate count')
            env['query_indices'] = [i for i in pool if i not in supports]
            env['support_query_disjoint'] = True
            env['adaptation_support_indices'] = adapted_supports
            env['adaptation_query_overlap_indices'] = [i for i in adapted_supports if i not in supports]
        protocol.update(support_size=1, adaptation_support_size=k, support_query_disjoint=True,
                        adaptation_support_query_disjoint=(k == 1), query_overlap_diagnostic=(k > 1),
                        protocol='Only the original K1 support uses an observed outcome; the same nine remaining actions use predictions for every K, including additional adaptation supports.',
                        action_outcome_source='fixed_k1_observed_support_plus_nine_model_predictions')
        out = report / f'k{k}'
        out.mkdir(exist_ok=True)
        manifest_path = out / 'support_query_manifest.json'
        manifest_path.write_text(json.dumps(protocol, indent=2) + '\n')
        cfg = copy.deepcopy(template)
        cfg.update(benchmark_root=str(benchmark), protocol_path=str(manifest_path),
                   output_root=str(out / 'metrics'), use_model_predictions_for_support=False,
                   methods={f'ours_k{k}_observed_support': str(relative)})
        config_path = out / 'metric_config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        subprocess.run([sys.executable, 'scripts/evaluation/evaluate_event80_benchmark.py',
                        '--config', str(config_path)], check=True)
        with (out / 'metrics/scoreboard.csv').open(newline='') as handle:
            score = next(csv.DictReader(handle))
        score.update(k=k, support_outcome_policy='only_original_k1_support_observed',
                     adaptation_support_size=k, observed_candidate_count_per_environment=1,
                     prediction_source=str(benchmark / relative),
                     metric_source=str(out / 'metrics/scoreboard.csv'))
        scores.append(score)
        with (report / 'scoreboard.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
        print(json.dumps(score), flush=True)
    (report / 'protocol.json').write_text(json.dumps({
        'job_id': os.environ['SLURM_JOB_ID'], 'support_sizes': [1, 2, 4],
        'action_selection': 'The original K1 support uses its observed outcome for every K. All other nine actions use predictions, including additional adaptation supports.',
        'query_gt_use': 'Only scores action decisions, never selects them.',
        'video_metric_counts': {'1': 90, '2': 90, '4': 90},
        'video_metric_comparison_note': 'Identical original K1 query clips for all K. Additional adaptation supports overlap this fixed evaluation set for K2/K4; this is a diagnostic, not disjoint-query generalization.',
        'k1_prediction_source': str(formal / 'methods/ours/step_7272/seed_20260708'),
        'k2_k4_prediction_source': str(nearby),
        'score_boundaries_changed': False, 'rollouts_regenerated': False,
        'support_loss_reduction': 'mean',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
