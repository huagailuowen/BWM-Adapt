#!/usr/bin/env python3
"""Resume a dual-latent model for 500 model-only updates, with immutable Z."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import train_stage1_grouped_context as grouped


def model_only_phase(args, group_order, step):
    if not 5501 <= int(step) <= 6000:
        raise ValueError("This entrypoint only supports updates 5501..6000")
    return {
        "round": 0, "phase": "model",
        "sample_group_indices": list(range(len(group_order))),
        "train_context_indices": [], "phase_start": 5501, "phase_end": 6000,
        "active_count": len(group_order), "new_count": 0,
    }


def freeze_context_phase(model, phase, freeze_background=False):
    if phase != "model":
        raise ValueError("Context-only or joint updates are forbidden in this continuation")
    for name, parameter in model.named_parameters():
        # Do not accidentally unfreeze the VAE or unused modules previously frozen
        # by the original model constructor. Keep only its model trainables.
        if name.startswith(("friction_context_table.", "background_context_table.")):
            parameter.requires_grad_(False)


class ImmutableContextTable(grouped.FrictionContextTable):
    def clamp_(self, *args, **kwargs):
        # Restored values are immutable, including values outside old clamp bounds.
        return None


if __name__ == "__main__":
    grouped.FrictionContextTable = ImmutableContextTable
    grouped._curriculum_phase_for_step = model_only_phase
    grouped._set_curriculum_requires_grad = freeze_context_phase
    grouped.main()

