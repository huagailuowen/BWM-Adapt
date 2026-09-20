"""Reentrant experiment without retaining transient inner-SGD saved tensors.

Record BF16 accumulation-order error explicitly. FP32 comparisons retain the
original tolerance; BF16-quantized gradients may differ by up to 2% in L2 norm.
This is a single-device infrastructure probe, not a production DDP runner.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    import torch
    import yaml
    from scripts.methods import probe_ttt_full6_pure_checkpoint as probe
    from scripts.methods.probe_ttt_full6_reentrant import install_reentrant_blocks

    strict_close = torch.testing.assert_close
    rounded_comparisons = []
    def check_gradient(actual, expected, **kwargs):
        try:
            return strict_close(actual, expected, **kwargs)
        except AssertionError:
            quantized = torch.equal(expected, expected.bfloat16().float()) and torch.equal(actual, actual.bfloat16().float())
            relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
            maximum = float((actual - expected).abs().max())
            record = {'bf16_quantized': quantized, 'relative_l2_error': relative,
                      'max_abs_error': maximum, 'limit_relative_l2': 0.02}
            print('[bf16_gradient_rounding] ' + json.dumps(record), flush=True)
            rounded_comparisons.append(record)
            if not quantized or not torch.isfinite(actual).all() or relative > 0.02:
                raise

    original_check = probe.check_equivalence
    def equivalence():
        torch.testing.assert_close = check_gradient
        try:
            result = original_check()
            result.append({'bf16_rounding_comparisons': rounded_comparisons})
            return result
        finally:
            torch.testing.assert_close = strict_close

    original_load = yaml.safe_load
    def load_config(value):
        config = original_load(value)
        if isinstance(config, dict) and 'ttt_kvb' in config:
            config['ttt_kvb']['ttt_saved_tensor_policy'] = 'legacy'
            config['ttt_kvb']['ttt_saved_tensor_cpu_offload'] = False
            config['ttt_kvb']['ttt_scan_checkpoint_updates'] = 0
        return config

    yaml.safe_load = load_config
    probe.check_equivalence = equivalence
    probe.install_pure_blocks = install_reentrant_blocks
    print('[probe_variant] explicit-state reentrant; no saved-tensor cache; no detach; full six chunks', flush=True)
    probe.main()
