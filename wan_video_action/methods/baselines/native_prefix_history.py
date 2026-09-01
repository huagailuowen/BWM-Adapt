"""Native Wan prefix-history baseline with an explicit latent-group reset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def build_event80_prefix_pair(support: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    support_video = torch.as_tensor(support["video"])
    query_video = torch.as_tensor(query["video"])
    if support_video.shape[2] != 41 or query_video.shape[2] != 41:
        raise ValueError("Event80 native prefix expects 41-frame support and query clips.")
    reset_anchor = query_video[:, :, 0:1].repeat(1, 1, 4, 1, 1)
    video = torch.cat((support_video, reset_anchor, query_video[:, :, 1:]), dim=2)

    support_action = torch.as_tensor(support["action"])
    query_action = torch.as_tensor(query["action"])
    reset_action = torch.zeros_like(query_action[:, :4])
    action = torch.cat((support_action, reset_action, query_action[:, :40]), dim=1)
    if video.shape[2] != 85 or action.shape[1] != 85:
        raise RuntimeError(f"Invalid prefix shapes video={video.shape}, action={action.shape}.")
    result = query.copy()
    result["video"] = video
    result["action"] = action
    return result


class PrefixSegmentEmbedding(nn.Module):
    """Marks 11 support groups, one reset group, and 10 query groups."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(3, int(dim)))
        self.register_buffer(
            "segment_ids",
            torch.tensor([0] * 11 + [1] + [2] * 10, dtype=torch.long),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        groups = int(self.segment_ids.numel())
        if tokens.shape[1] % groups:
            raise RuntimeError(
                f"Token count {tokens.shape[1]} is not divisible by {groups} temporal groups."
            )
        spatial = int(tokens.shape[1]) // groups
        ids = self.segment_ids.to(tokens.device).repeat_interleave(spatial)
        return tokens + self.embedding.to(tokens.dtype).index_select(0, ids).unsqueeze(0)


@dataclass(frozen=True)
class PrefixSegmentInstallation:
    module: PrefixSegmentEmbedding
    hook: Any


def install_prefix_segments(dit: nn.Module) -> PrefixSegmentInstallation:
    module = PrefixSegmentEmbedding(int(dit.dim)).to(
        device=dit.blocks[0].self_attn.q.weight.device,
        dtype=dit.blocks[0].self_attn.q.weight.dtype,
    )
    dit.add_module("native_prefix_segments", module)

    def inject(_block, args):
        return (module(args[0]), *args[1:])

    hook = dit.blocks[0].register_forward_pre_hook(inject)
    return PrefixSegmentInstallation(module=module, hook=hook)


class NativePrefixTrainingModule(nn.Module):
    def __init__(self, wan: nn.Module) -> None:
        super().__init__()
        self.wan = wan

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        inputs = self.wan.get_pipeline_inputs(data)
        inputs = self.wan.transfer_data_to_device(
            inputs, self.wan.pipe.device, self.wan.pipe.torch_dtype
        )
        for unit in self.wan.pipe.units:
            inputs = self.wan.pipe.unit_runner(unit, self.wan.pipe, *inputs)
        return self._query_only_flow_loss(*inputs)

    def _query_only_flow_loss(self, inputs_shared, inputs_posi, inputs_nega):
        pipe = self.wan.pipe
        inputs = {**inputs_shared, **inputs_posi}
        low = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
        high = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
        index = torch.randint(low, high, (1,))
        timestep = pipe.scheduler.timesteps[index].to(dtype=pipe.torch_dtype, device=pipe.device)
        noise = torch.randn_like(inputs["input_latents"])
        inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
        history_groups = int(inputs["fused_condition_latent_frames"])
        inputs["latents"][:, :, :history_groups] = inputs["first_frame_latents"]
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
        prediction = prediction[:, :, history_groups:]
        target = target[:, :, history_groups:]
        return F.mse_loss(prediction.float(), target.float()) * pipe.scheduler.training_weight(timestep)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        wan_state = {
            key[len("wan."):]: value
            for key, value in state_dict.items()
            if key.startswith("wan.")
        }
        exported = self.wan.export_trainable_state_dict(
            wan_state, remove_prefix=remove_prefix
        )
        return {f"wan.{key}": value for key, value in exported.items()}
