#!/usr/bin/env python3
"""Apply the formal static9/shared6 Soft scorer to the global-mean inference."""

import argparse
import json

import score_real97_soft_static9_standard_lora as formal


formal.OUT = formal.ROOT / 'outputs/eval_real97_soft_5500_simlr_globalmean_main_rules_20260922_v1'
formal.OLD = formal.ROOT / 'outputs/eval_real97_soft_5500_simlr_globalmean_20260922_v1'
formal.RUNS['ours'] = formal.ROOT / 'outputs/infer_real97_soft_5500_simlr_globalmean_20260922_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('track', 'images', 'aggregate'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    rows, frozen = formal.setup()
    if args.mode == 'track':
        formal.track(rows, frozen, args.shard)
    elif args.mode == 'images':
        formal.images(rows)
    else:
        formal.aggregate(rows, frozen)
        formal.write(formal.OUT / 'globalmean_main_rule_provenance.json', {
            'initialization': 'mean of all 18 training-time latent rows',
            'scoring_implementation': 'score_real97_soft_static9_standard_lora.py',
            'primary_cohort': 'fixed shared6 test12',
            'ade_mask': 'GT/newStandard/newLoRA/globalmeanOurs jointly valid',
            'onset_rule': '8px, 3 consecutive frames, within 6 degrees on 11 GT-positive queries',
            'source_inference': str(formal.RUNS['ours']),
            'source_basic_tracking': str(formal.OLD),
            'output': str(formal.OUT),
        })
        summary = json.loads((formal.OUT / 'summary.json').read_text())
        print(json.dumps(summary['groups']['test_shared6']['ours']), flush=True)


if __name__ == '__main__':
    main()
