"""Recount existing K1/K2/K4 rollouts and enforce prediction-only candidates."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run the action recount on a compute node')
    root = Path('results/mass_collision/noleak_highmass2x_support_k1_k2_k4_all_model_predictions_v1')
    output = Path('results/support_number_analysis/mass_collision_prediction_only_recount')
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    reference_eligible = None
    for k in (1, 2, 4):
        run = root / f'methods/ours_action8_highmass2x/k{k}/step_4300/seed_20260827'
        config = run / 'action_evaluation.yaml'
        settings = yaml.safe_load(config.read_text())
        if settings.get('use_model_predictions_for_support') is not True:
            raise ValueError(f'K{k} does not require model predictions for support actions')
        subprocess.run([sys.executable, 'scripts/evaluation/evaluate_sim_action_selection.py',
                        '--config', str(config)], check=True)
        metrics = run / 'action_evaluation'
        candidates = [json.loads(line) for line in (metrics / 'candidate_outcomes.jsonl').read_text().splitlines()]
        if len(candidates) != 90:
            raise ValueError(f'K{k}: expected 90 action candidates, got {len(candidates)}')
        support_count = 0
        for candidate in candidates:
            if candidate['selection_source'] != 'model_prediction':
                raise ValueError(f'K{k}: non-prediction candidate found')
            for member in candidate['member_records']:
                if member['selection_source'] != 'model_prediction' or not member.get('prediction_path'):
                    raise ValueError(f'K{k}: member is not backed by a model prediction')
                if not Path(member['prediction_path']).is_file():
                    raise FileNotFoundError(member['prediction_path'])
                support_count += int(member['is_support'])
        if support_count != 10 * k:
            raise ValueError(f'K{k}: unexpected support prediction count {support_count}')
        decisions = [json.loads(line) for line in (metrics / 'decisions.jsonl').read_text().splitlines()]
        eligible = [r for r in decisions if r['oracle_reachable'] and r['candidate_set_complete']]
        keys = {(r['source_index'], r['target']['id']) for r in eligible}
        if reference_eligible is None:
            reference_eligible = keys
        elif keys != reference_eligible:
            raise ValueError('K settings do not share the same eligible environment-target pairs')
        summary = {
            'k': k, 'successes': sum(r['task_success'] for r in eligible),
            'eligible_decisions': len(eligible),
            'action_success': sum(r['task_success'] for r in eligible) / len(eligible),
            'candidate_predictions': len(candidates), 'support_action_predictions': support_count,
            'observed_gt_candidates': 0, 'result_source': str(metrics),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (output / 'summary.json').write_text(json.dumps({
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'selection_rule': 'Every candidate, including support actions, uses model prediction; GT scores only.',
        'videos_regenerated': False, 'support_loss_reduction': 'mean',
        'results': summaries,
    }, indent=2) + '\n')
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == '__main__':
    main()
