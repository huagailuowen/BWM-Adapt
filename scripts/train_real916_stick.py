#!/usr/bin/env python3
"""Isolated real916 entrypoint: identical masked flow loss for Ours and Standard.

Legacy trainers and configurations are not modified. Class substitutions below
are local to this process and selected only by this new entrypoint.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from scripts import train_stage1_grouped_context as grouped
from scripts.methods import train_real97_standard_pooled as standard


def valid_future_flow_loss(pipe, inputs_shared, inputs_posi, inputs_nega):
    inputs = {**inputs_shared, **inputs_posi}
    valid_frames = int(inputs.pop("_stick_valid_video_frames"))
    if int(inputs["num_frames"]) != 33 or int(inputs["num_history_frames"]) != 1:
        raise ValueError("This entrypoint requires one condition plus 32 predicted frames")
    upper = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    lower = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(lower, upper, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    clean = inputs["input_latents"]
    if clean.shape[2] != 9:
        raise ValueError("Expected causal VAE layout: 1 + 8 latent temporal frames")
    condition = inputs.get("first_frame_latents")
    if condition is None or condition.shape[2] != 1:
        raise ValueError("Missing clean first-frame conditioning")
    # A causal future latent covers four sampled video frames. Drop the final
    # partially padded block too, rather than supervising repeated tail frames.
    valid_future = (valid_frames - 1) // 4
    if not 1 <= valid_future <= 8:
        raise ValueError(f"Invalid number of real future latent frames: {valid_future}")
    noise = torch.randn_like(clean)
    inputs["latents"] = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)
    inputs["latents"][:, :, :1] = condition
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    predicted = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(
        predicted[:, :, 1:1 + valid_future].float(),
        target[:, :, 1:1 + valid_future].float(),
    )
    return loss * pipe.scheduler.training_weight(timestep)


class ValidFutureMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task not in ("sft", "sft:train") or self.spatial_loss_mode != "none":
            raise ValueError("real916 Stick uses plain SFT without spatial ROI")
        self.task_to_loss[self.task] = valid_future_flow_loss

    def get_pipeline_inputs(self, data):
        shared, positive, negative = super().get_pipeline_inputs(data)
        shared["_stick_valid_video_frames"] = int(data["valid_video_frames"])
        return shared, positive, negative


class StickGroupedModule(ValidFutureMixin, grouped.GroupedContextStage1Module):
    pass


class StickStandardModule(ValidFutureMixin, standard.WanTrainingModule):
    pass


def main():
    method = os.environ["BWM_STICK916_METHOD"]
    if method == "ours":
        grouped.GroupedContextStage1Module = StickGroupedModule
        grouped.main()
    elif method == "standard":
        standard.WanTrainingModule = StickStandardModule
        os.environ["BWM_STANDARD_DEADLINE_EPOCH"] = os.environ["BWM_WALLCLOCK_CHECKPOINT_AT"]
        standard.main()
    else:
        raise ValueError(f"Unknown method {method!r}")


if __name__ == "__main__":
    main()

