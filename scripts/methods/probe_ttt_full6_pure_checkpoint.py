"""Isolated compute-node probe: explicit-state Wan-block recomputation.

This does not alter the default runner or any existing training process.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def install_pure_blocks(dit, installation):
    import torch
    from torch.utils.checkpoint import checkpoint
    from wan_video_action.methods.baselines.ttt_kvb.controller import TTTKVBMode
    from wan_video_action.methods.baselines.ttt_kvb.fast_weight import TTTKVBState

    controller = installation.controller
    ttt_indices = set(installation.layer_indices)
    for index, block in enumerate(dit.blocks):
        original = block.forward
        layer_id = f'block_{index:02d}' if index in ttt_indices else None

        def make_forward(original, layer_id):
            def forward(self, *args, **kwargs):
                if not torch.is_grad_enabled():
                    return original(*args, **kwargs)
                if layer_id is None:
                    return checkpoint(original, *args, use_reentrant=False, **kwargs)
                if controller.mode != TTTKVBMode.CAUSAL_SCAN:
                    raise RuntimeError('The probe only supports the causal full-gradient training path.')
                memory = self.self_attn.ttt_kvb_memory
                state = controller._states.get(layer_id)
                if state is None:
                    state = memory.initial_state(args[0].shape[0], args[0].device, differentiable=True)
                positional_count = len(args)

                def pure(*inputs, **call_kwargs):
                    # Each replay receives exactly its original state, not the
                    # controller's newer state from a later chunk or layer.
                    previous = (controller.mode, controller.differentiable,
                                controller.query_state_indices, controller._states,
                                controller._write_statistics)
                    local_states = {layer_id: TTTKVBState(*inputs[positional_count:])}
                    local_stats = {}
                    controller.mode = TTTKVBMode.CAUSAL_SCAN
                    controller.differentiable = True
                    controller.query_state_indices = None
                    controller._states = local_states
                    controller._write_statistics = local_stats
                    try:
                        output = original(*inputs[:positional_count], **call_kwargs)
                        return (output, *local_states[layer_id].tensors, local_stats.get(layer_id, []))
                    finally:
                        (controller.mode, controller.differentiable,
                         controller.query_state_indices, controller._states,
                         controller._write_statistics) = previous

                result = checkpoint(pure, *args, *state.tensors, use_reentrant=False, **kwargs)
                controller._states[layer_id] = TTTKVBState(*result[1:5])
                controller._write_statistics.setdefault(layer_id, []).extend(result[5])
                return result[0]
            return forward
        block.forward = MethodType(make_forward(original, layer_id), block)


def check_equivalence():
    import torch
    from torch import nn
    from wan_video_action.methods.baselines.ttt_kvb.wan_adapter import install_ttt_kvb

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim, self.num_heads = 32, 2
            self.q = nn.Linear(32, 32)
        def forward(self, x, freqs):
            return self.q(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        def forward(self, x, freqs):
            x = x + self.self_attn(x, freqs)
            return x + self.ff(x)

    records = []
    for dtype in (torch.float32, torch.bfloat16):
        values = []
        for optimized in (False, True):
            torch.manual_seed(20260918)
            dit = nn.Module()
            dit.blocks = nn.ModuleList([Block(), Block()])
            dit.to(device='cuda', dtype=dtype)
            installed = install_ttt_kvb(dit, layer_spec='0', gate_init=0.1,
                                       gate_vector=True, serial_after_attention=True,
                                       inner_batch_size=16, write_token_budget=0,
                                       scan_checkpoint_updates=2)
            if optimized:
                install_pure_blocks(dit, installed)
            inputs = [torch.randn(1, 48, 32, device='cuda', dtype=dtype, requires_grad=True) for _ in range(6)]
            installed.controller.reset(1)
            losses = []
            context = torch.autocast('cuda', dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
            with context:
                for x in inputs:
                    with installed.controller.causal_scan(differentiable=True):
                        y = x
                        for block in dit.blocks:
                            y = block(y, None)
                    losses.append(y.float().square().mean())
                loss = torch.stack(losses).mean()
            loss.backward()
            grads = {name: None if p.grad is None else p.grad.detach().float().cpu()
                     for name, p in dit.named_parameters()}
            values.append((float(loss.detach()), grads))
            installed.controller.clear()
            del dit, installed, losses, loss, inputs, y
        reference, candidate = values
        maximum = 0.0
        for name, expected in reference[1].items():
            actual = candidate[1][name]
            if expected is None or actual is None:
                if expected is not None or actual is not None:
                    raise AssertionError(f'Gradient presence mismatch: {name}')
                continue
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            maximum = max(maximum, float((actual - expected).abs().max()))
        if abs(reference[0] - candidate[0]) > 1e-6:
            raise AssertionError('Loss mismatch')
        records.append({'dtype': str(dtype), 'loss': reference[0], 'max_gradient_abs_error': maximum})
    print('[equivalence] ' + json.dumps(records), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'pure'], required=True)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--environments', type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'yejin-interactive':
        raise RuntimeError('This experiment requires the interactive GPU allocation.')
    os.chdir(ROOT)
    import torch
    import yaml
    output = ROOT / 'outputs/infra_probes' / f'ttt_full6_{os.environ["SLURM_JOB_ID"]}' / args.mode
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'pure':
        (output / 'equivalence.json').write_text(json.dumps(check_equivalence(), indent=2))
        torch.cuda.empty_cache()
    from scripts.methods import train_ttt_kvb_event80 as runner

    class ProbeLogger:
        def __init__(self, *unused, **kwargs):
            pass
        def on_step_end(self, *unused, **kwargs):
            pass
        def on_training_end(self, *unused, **kwargs):
            print('[probe_done] completed optimizer updates; no checkpoint writes', flush=True)
        def on_step_diagnostics(self, **kwargs):
            record = {
                'step': kwargs['step'], 'loss': float(kwargs['loss']),
                'seconds': time.monotonic() - kwargs['started_monotonic'],
                'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'allocated_cuda_gib': torch.cuda.memory_allocated() / 1024**3,
                'reserved_cuda_gib': torch.cuda.memory_reserved() / 1024**3,
                'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                'storage': kwargs.get('storage_statistics'),
            }
            with (output / 'updates.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print('[probe_update] ' + json.dumps(record), flush=True)

    runner.TimedRetentionModelLogger = ProbeLogger
    original_install = runner.install_ttt_kvb
    def install(dit, **kwargs):
        installed = original_install(dit, **kwargs)
        if args.mode == 'pure':
            install_pure_blocks(dit, installed)
        def before(module, inputs):
            print(f'[chunk_start] allocated_gib={torch.cuda.memory_allocated()/1024**3:.3f}', flush=True)
        def after(module, inputs, result):
            print(f'[chunk_end] allocated_gib={torch.cuda.memory_allocated()/1024**3:.3f}', flush=True)
        dit.blocks[0].register_forward_pre_hook(before)
        dit.blocks[-1].register_forward_hook(after)
        return installed
    runner.install_ttt_kvb = install
    base = ROOT / 'configs/train/train_mass_friction100_active84_ttt_kqv_prequential6_4envpergpu_2gpu_24h.yaml'
    config = yaml.safe_load(base.read_text())
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared'
    config['model']['model_paths'] = str(cache / 'Wan2.2-TI2V-5B')
    config['output']['ckpt_path'] = str(cache / 'BLM-step-12000.safetensors')
    config['output']['output_path'] = str(output)
    config['output']['log_steps'] = 1
    config['ttt_kvb'].update(ttt_max_updates=args.updates,
                             ttt_environments_per_rank=args.environments,
                             ttt_detach_every_chunks=0)
    runtime = output / 'launch_config.yaml'
    runtime.write_text(yaml.safe_dump(config, sort_keys=False))
    sys.argv = [runner.__file__, '--config', str(runtime)]
    runner.main()


if __name__ == '__main__':
    main()
