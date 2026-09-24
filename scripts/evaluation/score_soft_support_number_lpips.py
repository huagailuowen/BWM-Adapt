#!/usr/bin/env python3
"""Match formal Soft LPIPS on the frozen shared6/test12 cohort."""
import argparse
import os
from pathlib import Path

import summarize_real97_simlr as formal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, choices=[1, 2], required=True)
    parser.add_argument('--output-root', type=Path, default=None)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('LPIPS requires an allocated compute GPU')
    import score_real97_soft_final_lpips as lp
    if not lp.torch.cuda.is_available():
        raise RuntimeError('No allocated CUDA device')
    lp.torch.set_num_threads(4)
    lp.cv2.setNumThreads(1)
    output = (args.output_root / f'k{args.k}' if args.output_root is not None else
              formal.ROOT / 'results/support_number_analysis/real/soft' / f'k{args.k}')
    metrics = output / 'metrics'
    cohort = formal.read(formal.TABLE / 'soft_shared6/per_query.json')
    keys = {r['key'] for r in cohort}
    selected = [r for r in formal.read(metrics / 'per_query.json') if r['key'] in keys]
    if len(keys) != 12 or len(selected) != 12 or any(r['split'] != 'test' for r in selected):
        raise RuntimeError('Frozen Soft test12 cohort mismatch')
    model = lp.lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval().requires_grad_(False)
    records = []
    for row in selected:
        destination = metrics / 'lpips' / (row['key'] + '.json')
        indices = [i for i in row['row']['evaluation_frame_indices'] if i <= 28]
        if destination.exists():
            result = formal.read(destination)
            if result['indices'] != indices:
                raise RuntimeError('Cached LPIPS frame-window mismatch')
        else:
            gt = lp.frames(output / 'raw/gt' / (row['key'] + '.mp4'), indices)
            pred = lp.frames(output / 'raw/stage2' / (row['key'] + '.mp4'), indices)
            values = []
            with lp.torch.inference_mode():
                for start in range(0, len(indices), 8):
                    values.extend(model(gt[start:start+8], pred[start:start+8]).flatten().cpu().tolist())
            result = dict(key=row['key'], environment=row['environment'], split=row['split'],
                          mean=formal.mean(values), per_frame=values, indices=indices)
            formal.write(destination, result)
        records.append(result)
        print('[lpips]', args.k, row['key'], result['mean'], flush=True)
    formal.write(metrics / 'lpips_complete.json', dict(
        queries=12, records=records, mean=formal.mean([r['mean'] for r in records]),
        protocol='Alex v0.1, 512x256 INTER_LINEAR, future native3..84, no tracking mask',
        cohort='formal shared6/test12; equal per-query weighting', K=args.k))


if __name__ == '__main__':
    main()
