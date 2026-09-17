#!/usr/bin/env python3
"""Add matched LPIPS to the Soft TTT comparison after object scoring completes."""
import os
import cv2
import numpy as np
import torch
import lpips
import score_real97_soft_ttt_matched as scoring


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('LPIPS requires a GPU compute allocation')
    rows, frozen = scoring.setup()
    output = scoring.OUT
    summary = scoring.read(output / 'summary.json')
    if not summary.get('complete'):
        raise RuntimeError('Object-score aggregation must complete first')
    torch.set_num_threads(4)
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval()
    metric.requires_grad_(False)
    results = []
    for index, row in enumerate(rows):
        key, common, paths = scoring.prior.case(index, row)
        paths['ttt'] = scoring.RUN / 'raw/ttt' / (key + '.mp4')
        destination = output / 'cases' / key / 'lpips.json'
        signature = {kind: scoring.prior.sha(path) for kind, path in paths.items()}
        if destination.is_file():
            result = scoring.read(destination)
            if result['video_sha256'] != signature:
                raise RuntimeError('LPIPS input video changed: ' + key)
        else:
            wanted = common['evaluation_frame_indices']

            def load(kind):
                frames = scoring.base.decode(paths[kind])
                frames = frames[4:] if kind == 'standard' else frames[:29]
                if len(frames) != 29:
                    raise RuntimeError('Expected aligned native frames 0,3,...84')
                rgb = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in wanted])
                return torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).cuda().float() / 127.5 - 1

            gt = load('gt')
            scores = {}
            for kind in scoring.METHODS:
                prediction = load(kind)
                values = []
                with torch.inference_mode():
                    for start in range(0, len(wanted), 8):
                        values.extend(metric(gt[start:start+8], prediction[start:start+8]).flatten().cpu().tolist())
                scores[kind] = {'mean': float(np.mean(values)), 'per_frame': values}
            result = dict(index=index, key=key, environment=row['environment'], split=row['dataset_split'],
                          scores=scores, video_sha256=signature,
                          native_frames=[common['native_frame_indices'][i] for i in wanted])
            scoring.write(destination, result)
        results.append(result)
        print(f'[LPIPS] {key} {len(results)}/{len(rows)}', flush=True)
    for group, methods in summary['groups'].items():
        selected = [r for r in results if r['split'] == 'test']
        if group == 'test_shared6':
            selected = [r for r in selected if r['environment'] in scoring.SHARED]
        elif group.startswith('test/'):
            selected = [r for r in selected if r['environment'] == group.split('/', 1)[1]]
        elif group != 'test_all9':
            raise RuntimeError('Unexpected summary cohort: ' + group)
        for kind in scoring.METHODS:
            if len(selected) != methods[kind]['queries']:
                raise RuntimeError('Object/image cohort mismatch')
            methods[kind]['lpips'] = scoring.prior.mean([r['scores'][kind]['mean'] for r in selected])
            methods[kind]['lpips_queries'] = len(selected)
    summary.update(no_lpips=False, lpips_complete=True,
                   lpips_protocol='AlexNet v0.1, RGB [-1,1], 512x256 INTER_LINEAR; same native future frames 3..84; no detection mask')
    scoring.write(output / 'lpips_per_query.json', results)
    scoring.write(output / 'summary.json', summary)
    scoring.write(output / 'lpips_complete.json', {'queries': len(results), 'complete': True})
    print(summary['groups']['test_shared6'], flush=True)


if __name__ == '__main__':
    main()
