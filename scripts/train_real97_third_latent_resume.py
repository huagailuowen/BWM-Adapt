#!/usr/bin/env python3
"""Opt-in 6500 -> 7100 continuation: new replica only, then joint training."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import train_stage1_grouped_context as grouped


_original_optimizer = grouped._build_curriculum_optimizer
_model_parameter_ids = set()


class ContextOnlySafeStagedLogger(grouped.StagedGroupedContextModelLogger):
    """Publish small C-only snapshots without pretending a model file exists.

    This opt-in logger leaves all legacy trainers unchanged. Joint checkpoints
    still use the existing node-local staging and paired retention machinery.
    """

    def save_model(self, accelerator, model, file_name):
        unwrapped = accelerator.unwrap_model(model)
        context_names = {
            'friction_context_table.contexts',
            'friction_context_table.global_context',
            'background_context_table.contexts',
        }
        has_model_trainables = any(
            p.requires_grad for name, p in unwrapped.named_parameters()
            if name not in context_names
        )
        if has_model_trainables:
            return super().save_model(accelerator, model, file_name)
        output = Path(self.output_path)
        context_name = file_name.replace('.safetensors', '.context_table.json')
        # save_context_table already uses fsync plus atomic os.replace.
        # Do not emit a .safetensors.complete marker for a context-only save.
        self.save_context_table(accelerator, model, str(output / context_name))
        if accelerator.is_main_process:
            marker = output / ('.' + context_name + '.complete')
            temporary = marker.with_name(marker.name + f'.tmp-{os.getpid()}')
            with temporary.open('w') as handle:
                json.dump(dict(kind='context_only', context_table=context_name,
                               model_step=6500,
                               unchanged_model='source_snapshot_step6500/step-6500.safetensors'), handle)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, marker)
            print(f'[checkpoint] published context-only {context_name}; model remains step6500', flush=True)
        accelerator.wait_for_everyone()


def build_optimizer(model, args, accelerator):
    if (int(args.grouped_context_resume_step) != 6500
            or int(args.grouped_context_structured_updates) != 7100
            or int(args.grouped_context_friction_groups_per_update) != 6
            or int(args.grouped_context_actions_per_update) != 10):
        raise ValueError('This entrypoint requires resume6500, end7100, and 6x10 per rank')
    if float(args.grouped_context_weight_decay) != 0.0:
        raise ValueError('Context weight decay must be zero to keep old rows frozen')
    _model_parameter_ids.update(
        id(p) for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith(('friction_context_table.', 'background_context_table.'))
    )
    if not _model_parameter_ids:
        raise RuntimeError('No constructor-selected model parameters to train')
    return _original_optimizer(model, args, accelerator)


def continuation_phase(args, group_order, step):
    total = len(group_order)
    if total not in (24, 30):
        raise ValueError(f'Expected three latents for each of 8/10 environments, got {total}')
    old_count = 2 * (total // 3)
    if 6501 <= int(step) <= 6700:
        indices = list(range(old_count, total))
        return dict(round=1, phase='new_context', sample_group_indices=indices,
                    train_context_indices=indices, phase_start=6501, phase_end=6700,
                    active_count=total, new_count=len(indices))
    if 6701 <= int(step) <= 7100:
        indices = list(range(total))
        return dict(round=2, phase='joint', sample_group_indices=indices,
                    train_context_indices=indices, phase_start=6701, phase_end=7100,
                    active_count=total, new_count=0)
    raise ValueError(f'Unexpected continuation step: {step}')


def set_trainables(model, phase, freeze_background=False):
    if phase not in ('new_context', 'joint', 'model'):
        raise ValueError(f'Unexpected continuation phase: {phase}')
    for name, parameter in model.named_parameters():
        if name.endswith('friction_context_table.contexts'):
            parameter.requires_grad_(phase in ('new_context', 'joint'))
        else:
            # Restore only original model trainables, never the frozen VAE.
            parameter.requires_grad_(phase in ('joint', 'model') and id(parameter) in _model_parameter_ids)


if __name__ == '__main__':
    grouped.StagedGroupedContextModelLogger = ContextOnlySafeStagedLogger
    grouped._build_curriculum_optimizer = build_optimizer
    grouped._curriculum_phase_for_step = continuation_phase
    grouped._set_curriculum_requires_grad = set_trainables
    grouped.main()
