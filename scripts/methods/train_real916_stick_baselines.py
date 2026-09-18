#!/usr/bin/env python3
"""Opt-in Stick 9/16 DINO/TTT baseline entrypoint; legacy losses are unchanged."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F


def main():
    method = os.environ["BWM_STICK916_BASELINE"]
    if method == "dino":
        from scripts.methods import train_real97_dinov2 as trainer
        from scripts.train_real916_stick import ValidFutureMixin

        class StickDINOQuery(ValidFutureMixin, trainer.WanTrainingModule):
            pass

        trainer.WanTrainingModule = StickDINOQuery
        trainer.main()
    elif method == "ttt":
        from scripts.methods import train_real97_ttt as trainer

        class StickTTT(trainer.legacy.OneMinuteTTTWanTrainingModule):
            def get_pipeline_inputs(self, data):
                shared, positive, negative = super().get_pipeline_inputs(data)
                shared["_stick_valid_video_frames"] = int(data["valid_video_frames"])
                return shared, positive, negative

            def _ttt_flow_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
                if self._ttt_timestep_index is None:
                    raise RuntimeError("Stream timestep was not set.")
                inputs = {**inputs_shared, **inputs_posi}
                valid_frames = int(inputs.pop("_stick_valid_video_frames"))
                valid_future = (valid_frames - 1) // 4
                clean = inputs["input_latents"]
                condition = inputs.get("first_frame_latents")
                if clean.shape[2] != 9 or condition is None or condition.shape[2] != 1:
                    raise ValueError("Stick requires 33 video frames and one observed frame.")
                if not 1 <= valid_future <= 8:
                    raise ValueError(f"No valid future or invalid frame count: {valid_frames}")
                timestep = pipe.scheduler.timesteps[self._ttt_timestep_index].reshape(1).to(
                    dtype=pipe.torch_dtype, device=pipe.device
                )
                noise = torch.randn_like(clean)
                inputs["latents"] = pipe.scheduler.add_noise(clean, noise, timestep)
                target = pipe.scheduler.training_target(clean, noise, timestep)
                inputs["latents"][:, :, :1] = condition
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
                stop = 1 + valid_future
                return F.mse_loss(
                    prediction[:, :, 1:stop].float(), target[:, :, 1:stop].float()
                ) * pipe.scheduler.training_weight(timestep)

        trainer.legacy.OneMinuteTTTWanTrainingModule = StickTTT
        trainer.main()
    else:
        raise ValueError(f"Unknown Stick baseline: {method}")


if __name__ == "__main__":
    main()
