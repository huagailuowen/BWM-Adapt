#!/usr/bin/env python3
"""Opt-in 6000 -> 6500 dual-latent joint continuation; old trainers unchanged."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import train_stage1_grouped_context as grouped


def joint_phase(args, group_order, step):
    if not 6001 <= int(step) <= 6500:
        raise ValueError('Only joint updates 6001..6500 are supported')
    indices = list(range(len(group_order)))
    return dict(round=0, phase='joint', sample_group_indices=indices,
                train_context_indices=indices, phase_start=6001, phase_end=6500,
                active_count=len(indices), new_count=0)


def set_joint_trainables(model, phase, freeze_background=False):
    if phase not in ('joint', 'model'):
        raise ValueError('Unexpected phase in joint continuation: ' + phase)
    for name, parameter in model.named_parameters():
        if name.endswith('friction_context_table.contexts'):
            parameter.requires_grad_(phase == 'joint')
        elif name.startswith('background_context_table.'):
            parameter.requires_grad_(False)
        # Preserve constructor-selected model trainables. In particular, never
        # unfreeze the VAE or other frozen pipeline modules here.


if __name__ == '__main__':
    grouped._curriculum_phase_for_step = joint_phase
    grouped._set_curriculum_requires_grad = set_joint_trainables
    grouped.main()
