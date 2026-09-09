#!/usr/bin/env python3
"""Fail before model staging if an allocated GPU cannot execute CUDA work."""
import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected', type=int, required=True)
    args = parser.parse_args()
    print('[cuda_preflight] ' + json.dumps({
        'expected_devices': args.expected,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'SLURM_JOB_GPUS': os.environ.get('SLURM_JOB_GPUS'),
        'SLURM_STEP_GPUS': os.environ.get('SLURM_STEP_GPUS'),
        'torch_version': torch.__version__, 'cuda_build': torch.version.cuda,
    }), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Refusing CPU fallback; no training was started.')
    count = torch.cuda.device_count()
    if count != args.expected:
        raise RuntimeError(f'Expected {args.expected} visible GPUs, got {count}.')
    for index in range(count):
        with torch.cuda.device(index):
            properties = torch.cuda.get_device_properties(index)
            value = (torch.arange(16, device=f'cuda:{index}', dtype=torch.float32) + 1).sum()
            torch.cuda.synchronize(index)
            if value.item() != 136.0:
                raise RuntimeError(f'GPU {index} returned an incorrect CUDA result.')
            print('[cuda_preflight] ' + json.dumps({
                'device': index, 'name': properties.name,
                'uuid': str(getattr(properties, 'uuid', 'unavailable')),
                'cuda_execution': 'passed',
            }), flush=True)
    print('[cuda_preflight] PASS', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'[cuda_preflight] FAIL: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise SystemExit(2)
