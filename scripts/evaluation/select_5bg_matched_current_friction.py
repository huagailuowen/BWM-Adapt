#!/usr/bin/env python3
"""Old 5bg data, closest distinct friction levels within true ID/OOD sets."""
import itertools
import json
import os
from pathlib import Path
import sys

import select_shared_friction_informative_support as selector

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'results/pushbox_multibackground/matchedphysics30bg_active20_id5_ood5_k1_oracle_informative_support25_60_v1/protocol/support_query_manifest.json'


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Support video selection must run on a compute node')
    metadata = Path(sys.argv[sys.argv.index('--metadata-path') + 1])
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    mapping = {int(r['context_group_id']): float(r['physical_friction_mu'])
               for r in map(json.loads, metadata.read_text().splitlines()) if r}
    reference = json.loads(REFERENCE.read_text())
    environments = reference['environments'] if isinstance(reference, dict) else reference
    records = []

    def closest(values, count):
        domain = 'id' if not records else 'ood'
        target = sorted(float(e['friction_mu']) for e in environments if e['domain'] == domain)
        if len(target) != count:
            raise RuntimeError('Reference friction cohort size mismatch')
        candidates = sorted(values, key=lambda x: (mapping[x], x))
        chosen = min(itertools.combinations(candidates, count),
                     key=lambda xs: (sum(abs(mapping[x] - y) for x, y in zip(xs, target)), xs))
        records.append({'domain': domain, 'matches': [
            {'context_group_id': x, 'old_5bg_mu': mapping[x], 'reference_30bg_mu': y,
             'absolute_mu_difference': abs(mapping[x] - y)} for x, y in zip(chosen, target)]})
        return list(chosen)

    selector._evenly_spaced = closest
    selector.main()
    if len(records) != 2:
        raise RuntimeError('Expected one ID and one OOD selection')
    selector._write_json(output / 'friction_matching.json', {
        'reference': str(REFERENCE), 'source_data': 'original 5-background dataset',
        'assignment': 'minimum total absolute mu difference; distinct ordered levels within each domain',
        'id_definition': 'old model first 15 curriculum groups; never relabel by nearest reference',
        'matches': records})


if __name__ == '__main__':
    main()
