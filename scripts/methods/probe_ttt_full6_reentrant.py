"""Experimental reentrant checkpoint boundary around explicit-state Wan blocks."""
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def install_reentrant_blocks(dit, installation):
    import torch
    from torch.utils.checkpoint import checkpoint
    from wan_video_action.methods.baselines.ttt_kvb.controller import TTTKVBMode
    from wan_video_action.methods.baselines.ttt_kvb.fast_weight import TTTKVBState
    controller = installation.controller
    indices = set(installation.layer_indices)
    for index, block in enumerate(dit.blocks):
        original = block.forward
        layer_id = f'block_{index:02d}' if index in indices else None
        if layer_id is not None:
            # The entire block is now a recomputation unit; avoid nested scan
            # checkpoints. This changes storage, not the inner update equations.
            block.self_attn.ttt_kvb_memory.scan_checkpoint_updates = 0

        def make_forward(original, layer_id):
            def forward(self, *args, **kwargs):
                if not torch.is_grad_enabled():
                    return original(*args, **kwargs)
                if layer_id is None:
                    return checkpoint(original, *args, use_reentrant=False, **kwargs)
                if controller.mode != TTTKVBMode.CAUSAL_SCAN:
                    raise RuntimeError('Causal training only.')
                state = controller._states.get(layer_id)
                if state is None:
                    state = self.self_attn.ttt_kvb_memory.initial_state(
                        args[0].shape[0], args[0].device, differentiable=True)
                argument_count = len(args)
                keys = tuple(kwargs)
                state_offset = argument_count + len(keys)
                statistics = []

                def pure(*inputs):
                    replay = torch.is_grad_enabled()
                    previous = (controller.mode, controller.differentiable,
                                controller.query_state_indices, controller._states,
                                controller._write_statistics)
                    states = {layer_id: TTTKVBState(*inputs[state_offset:])}
                    stats = {}
                    controller.mode = TTTKVBMode.CAUSAL_SCAN
                    controller.differentiable = True
                    controller.query_state_indices = None
                    controller._states, controller._write_statistics = states, stats
                    try:
                        # Inner SGD requires autograd even in checkpoint's
                        # graph-free forward. Only a single block's graph is
                        # transiently built here; returned values are detached.
                        with torch.enable_grad():
                            output = original(*inputs[:argument_count], **dict(zip(keys, inputs[argument_count:state_offset])))
                            result = (output, *states[layer_id].tensors)
                        if replay:
                            return result
                        statistics.extend(stats.get(layer_id, []))
                        return tuple(value.detach() for value in result)
                    finally:
                        (controller.mode, controller.differentiable,
                         controller.query_state_indices, controller._states,
                         controller._write_statistics) = previous

                result = checkpoint(pure, *args, *kwargs.values(), *state.tensors,
                                    use_reentrant=True)
                controller._states[layer_id] = TTTKVBState(*result[1:])
                controller._write_statistics.setdefault(layer_id, []).extend(statistics)
                return result[0]
            return forward
        block.forward = MethodType(make_forward(original, layer_id), block)


if __name__ == '__main__':
    from scripts.methods import probe_ttt_full6_pure_checkpoint as probe
    probe.install_pure_blocks = install_reentrant_blocks
    probe.main()
