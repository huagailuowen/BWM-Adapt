"""Flamingo-style in-context world-model components.

Existing Wan blocks are not replaced. New GATED XATTN-DENSE modules live under
``dit.flamingo_icl`` and are activated only when this baseline is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
import yaml

from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="history_conditioned_wm",
    display_name="Flamingo In-Context WM",
    family=MethodFamily.BASELINE,
    summary=(
        "Resamples support video-action histories into fixed memory tokens and "
        "conditions frozen Wan blocks through zero-initialized gated cross-attention."
    ),
    training_protocol=(
        "Encode K disjoint support trajectories as read-only memory; every query "
        "attends to that memory without writing query information back."
    ),
    inference_protocol=(
        "Build one fixed 64-token memory from K supports and reuse it for every "
        "disjoint query trajectory in the same environment."
    ),
    adaptation_target=AdaptationTarget.HISTORY,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=(
        "Support and query trajectories remain disjoint.",
        "Query tokens never update support memory.",
        "The pretrained Wan and DINOv2 parameters remain frozen.",
    ),
)


class SquaredReLU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(value).square()


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim) * int(multiplier)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            SquaredReLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class FlamingoPerceiverAttention(nn.Module):
    """Original Flamingo resampler attention: latent Q, media+latent K/V."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")
        self.heads = int(heads)
        self.head_dim = int(dim) // self.heads
        self.scale = self.head_dim ** -0.5
        self.media_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, media: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        batch, latent_count, dim = latents.shape
        media = self.media_norm(media)
        normalized_latents = self.latent_norm(latents)
        key_value_input = torch.cat((media, normalized_latents), dim=1)
        query = self.to_q(normalized_latents)
        key, value = self.to_kv(key_value_input).chunk(2, dim=-1)
        query = query.view(batch, latent_count, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale,
            dim=-1,
        )
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).reshape(batch, latent_count, dim)
        return self.to_out(output)


class FlamingoPerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 1536,
        num_latents: int = 64,
        depth: int = 6,
        heads: int = 16,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) / math.sqrt(dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    (
                        FlamingoPerceiverAttention(dim, heads),
                        FeedForward(dim, multiplier=4),
                    )
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, media: torch.Tensor) -> torch.Tensor:
        latents = self.latents.unsqueeze(0).expand(media.shape[0], -1, -1)
        for attention, feed_forward in self.layers:
            latents = latents + attention(media, latents)
            latents = latents + feed_forward(latents)
        return self.output_norm(latents)


class ActionIntervalAdapter(nn.Module):
    """Encodes the complete action interval between two sampled video frames."""

    def __init__(self, action_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.step_encoder = nn.Sequential(
            nn.Linear(self.action_dim + 1, hidden_dim),
            SquaredReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action interval [T,{self.action_dim}], got {tuple(chunk.shape)}."
            )
        relative_time = torch.linspace(
            -1.0,
            1.0,
            steps=chunk.shape[0],
            device=chunk.device,
            dtype=chunk.dtype,
        ).unsqueeze(-1)
        encoded = self.step_encoder(torch.cat((chunk, relative_time), dim=-1))
        return self.output(encoded.mean(dim=0))


class FlamingoSupportEncoder(nn.Module):
    """Dense frozen-DINO support encoder followed by a Flamingo resampler."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        action_dim: int,
        sampled_frames: int = 8,
        memory_dim: int = 1536,
        num_latents: int = 64,
        resampler_layers: int = 6,
        heads: int = 16,
        action_hidden_dim: int = 256,
        max_support_trajectories: int = 2,
    ) -> None:
        super().__init__()
        self.sampled_frames = int(sampled_frames)
        self.action_dim = int(action_dim)
        self.memory_dim = int(memory_dim)
        self.max_support_trajectories = int(max_support_trajectories)
        if self.sampled_frames < 2:
            raise ValueError("sampled_frames must be at least two.")
        model_path = Path(model_path).expanduser()
        self.dino = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.dino.requires_grad_(False)
        self.dino.eval()
        dino_dim = int(self.dino.config.hidden_size)

        processor_path = model_path / "preprocessor_config.json"
        processor = {}
        if processor_path.is_file():
            with processor_path.open("r", encoding="utf-8") as handle:
                processor = json.load(handle)
        image_mean = processor.get("image_mean", [0.485, 0.456, 0.406])
        image_std = processor.get("image_std", [0.229, 0.224, 0.225])
        size = processor.get("size", {"shortest_edge": 224})
        self.image_size = int(
            size.get("height", size.get("shortest_edge", 224))
            if isinstance(size, dict)
            else size
        )
        self.register_buffer(
            "image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, memory_dim, bias=False),
        )
        self.action_adapter = ActionIntervalAdapter(
            action_dim=action_dim,
            hidden_dim=action_hidden_dim,
            output_dim=memory_dim,
        )
        self.temporal_embedding = nn.Parameter(
            torch.randn(self.sampled_frames, memory_dim) / math.sqrt(memory_dim)
        )
        self.segment_embedding = nn.Parameter(
            torch.randn(self.max_support_trajectories, memory_dim) / math.sqrt(memory_dim)
        )
        self.resampler = FlamingoPerceiverResampler(
            dim=memory_dim,
            num_latents=num_latents,
            depth=resampler_layers,
            heads=heads,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    @staticmethod
    def _canonical_video(video: Any) -> torch.Tensor:
        tensor = torch.as_tensor(video, dtype=torch.float32)
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim != 4 or tensor.shape[0] != 3:
            raise ValueError(
                "Support video must be [V,C,T,H,W] or [C,T,H,W] RGB; "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    def _canonical_actions(self, action: Any, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 1:
            tensor = tensor.view(-1, self.action_dim)
        if tensor.ndim != 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.shape[-1] != self.action_dim and tensor.shape[0] == self.action_dim:
            tensor = tensor.transpose(0, 1)
        if tensor.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {tuple(tensor.shape)}."
            )
        return tensor

    def extract_visual_features(
        self, video: Any
    ) -> tuple[torch.Tensor, tuple[int, ...], int]:
        video_tensor = self._canonical_video(video)
        frame_count = int(video_tensor.shape[1])
        indices_tensor = torch.linspace(
            0, frame_count - 1, steps=self.sampled_frames
        ).round().long()
        if int(torch.unique(indices_tensor).numel()) != self.sampled_frames:
            raise ValueError(
                f"Cannot select {self.sampled_frames} unique frames from T={frame_count}."
            )
        indices = tuple(int(value) for value in indices_tensor.tolist())
        device = next(self.dino.parameters()).device
        frames = video_tensor[:, indices].permute(1, 0, 2, 3).to(device=device)
        frames = (frames + 1.0) * 0.5
        frames = F.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        frames = (frames - self.image_mean.to(device)) / self.image_std.to(device)
        with torch.no_grad():
            dense = self.dino(pixel_values=frames).last_hidden_state[:, 1:].detach()
        return dense, indices, frame_count

    def _action_features(
        self,
        *,
        action: Any,
        frame_indices: tuple[int, ...],
        frame_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        actions = self._canonical_actions(action, device)
        action_count = int(actions.shape[0])
        features = [torch.zeros(self.memory_dim, device=device, dtype=actions.dtype)]
        denominator = max(int(frame_count) - 1, 1)
        for start_frame, end_frame in zip(frame_indices[:-1], frame_indices[1:]):
            start = int(round(start_frame / denominator * max(action_count - 1, 0)))
            end = int(round(end_frame / denominator * max(action_count - 1, 0)))
            end = max(start + 1, min(action_count, end))
            features.append(self.action_adapter(actions[start:end]))
        return torch.stack(features, dim=0)

    def project_supports(
        self,
        *,
        visual_features: list[torch.Tensor],
        actions: list[Any],
        frame_indices: list[tuple[int, ...]],
        frame_counts: list[int],
    ) -> torch.Tensor:
        support_count = len(visual_features)
        if not 1 <= support_count <= self.max_support_trajectories:
            raise ValueError(
                f"Expected 1..{self.max_support_trajectories} supports, got {support_count}."
            )
        media = []
        for segment_index, (dense, action, indices, frame_count) in enumerate(
            zip(visual_features, actions, frame_indices, frame_counts)
        ):
            projected = self.visual_projection(dense)
            action_features = self._action_features(
                action=action,
                frame_indices=indices,
                frame_count=frame_count,
                device=projected.device,
            )
            projected = projected + action_features[:, None, :]
            projected = projected + self.temporal_embedding[:, None, :]
            projected = projected + self.segment_embedding[segment_index][None, None, :]
            media.append(projected.reshape(-1, self.memory_dim))
        return self.resampler(torch.cat(media, dim=0).unsqueeze(0))


class GatedCrossAttentionDense(nn.Module):
    """Flamingo GATED XATTN-DENSE residual block with two zero gates."""

    def __init__(self, query_dim: int, memory_dim: int, heads: int = 16) -> None:
        super().__init__()
        if query_dim % heads:
            raise ValueError(f"query_dim={query_dim} must divide heads={heads}.")
        self.heads = int(heads)
        self.head_dim = int(query_dim) // self.heads
        self.scale = self.head_dim ** -0.5
        self.query_norm = nn.LayerNorm(query_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_kv = nn.Linear(memory_dim, 2 * query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim, bias=False)
        self.feed_forward = FeedForward(query_dim, multiplier=4)
        self.alpha_xattn = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.alpha_dense = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, query_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, token_count, dim = query_tokens.shape
        if memory.shape[0] == 1 and batch != 1:
            memory = memory.expand(batch, -1, -1)
        if memory.shape[0] != batch:
            raise ValueError(
                f"Memory batch {memory.shape[0]} does not match query batch {batch}."
            )
        query = self.to_q(self.query_norm(query_tokens))
        key, value = self.to_kv(self.memory_norm(memory)).chunk(2, dim=-1)
        query = query.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        attended = attended.transpose(1, 2).reshape(batch, token_count, dim)
        attended = self.to_out(attended)
        query_tokens = query_tokens + torch.tanh(self.alpha_xattn) * attended
        query_tokens = query_tokens + torch.tanh(self.alpha_dense) * self.feed_forward(
            query_tokens
        )
        return query_tokens


class FlamingoMemoryController:
    def __init__(self) -> None:
        self.memory: torch.Tensor | None = None

    def set(self, memory: torch.Tensor) -> None:
        self.memory = memory

    def clear(self) -> None:
        self.memory = None

    def require(self) -> torch.Tensor:
        if self.memory is None:
            raise RuntimeError("Flamingo support memory is not set for this query forward.")
        return self.memory


@dataclass
class FlamingoInstallation:
    adapters: nn.ModuleDict
    controller: FlamingoMemoryController
    block_indices: tuple[int, ...]
    hook_handles: tuple[Any, ...]

    def clear_memory(self) -> None:
        self.controller.clear()


def install_flamingo_history(
    dit: nn.Module,
    *,
    memory_dim: int = 1536,
    heads: int = 16,
    insertion_frequency: int = 4,
) -> FlamingoInstallation:
    if hasattr(dit, "flamingo_icl"):
        raise RuntimeError("Flamingo ICL adapters are already installed on this DiT.")
    blocks = list(dit.blocks)
    block_indices = tuple(range(0, len(blocks), int(insertion_frequency)))
    adapters = nn.ModuleDict(
        {
            str(index): GatedCrossAttentionDense(
                query_dim=int(dit.dim), memory_dim=memory_dim, heads=heads
            )
            for index in block_indices
        }
    )
    dit.add_module("flamingo_icl", adapters)
    controller = FlamingoMemoryController()
    handles = []
    for index in block_indices:
        adapter = adapters[str(index)]

        def inject(_module, args, *, selected_adapter=adapter):
            memory = controller.require()
            return (selected_adapter(args[0], memory), *args[1:])

        handles.append(blocks[index].register_forward_pre_hook(inject))
    return FlamingoInstallation(
        adapters=adapters,
        controller=controller,
        block_indices=block_indices,
        hook_handles=tuple(handles),
    )


@dataclass(frozen=True)
class HistoryEpisode:
    environment_id: int
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]


class Event80HistorySampler:
    """Samples six distinct actions, then forms a random-order K-support episode."""

    def __init__(
        self,
        *,
        metadata_path: str | Path,
        active_environment_manifest: str | Path,
        seed: int,
        support_sizes: Iterable[int] = (1, 2),
        chunks_per_environment: int = 6,
        environment_key: str = "mu_index",
        action_key: str = "action_id",
    ) -> None:
        self.seed = int(seed)
        self.support_sizes = tuple(sorted({int(value) for value in support_sizes}))
        self.chunks_per_environment = int(chunks_per_environment)
        if not self.support_sizes or min(self.support_sizes) < 1:
            raise ValueError("support_sizes must contain positive integers.")
        if max(self.support_sizes) >= self.chunks_per_environment:
            raise ValueError("Every episode needs at least one disjoint query.")
        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        with Path(active_environment_manifest).open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        self.active_environment_ids = tuple(
            int(value) for value in manifest["selection"]["active_environment_ids"]
        )
        grouped: dict[int, list[int]] = {
            value: [] for value in self.active_environment_ids
        }
        for index, row in enumerate(rows):
            environment_id = int(row[environment_key])
            if environment_id in grouped:
                grouped[environment_id].append(index)
        for environment_id, indices in grouped.items():
            actions = [int(rows[index][action_key]) for index in indices]
            if len(indices) < self.chunks_per_environment or len(actions) != len(set(actions)):
                raise ValueError(
                    f"Environment {environment_id} needs {self.chunks_per_environment} "
                    f"unique actions; got actions={actions}."
                )
        self.grouped_indices = grouped

    def sample(
        self,
        *,
        step: int,
        process_index: int,
        num_processes: int,
        environments_per_rank: int,
    ) -> tuple[HistoryEpisode, ...]:
        global_count = int(num_processes) * int(environments_per_rank)
        if not 1 <= global_count <= len(self.active_environment_ids):
            raise ValueError(f"Invalid global environment count {global_count}.")
        rng = random.Random(self.seed + int(step) * 104729)
        selected = rng.sample(list(self.active_environment_ids), global_count)
        episodes = []
        for environment_id in selected:
            ordered = rng.sample(
                self.grouped_indices[environment_id], self.chunks_per_environment
            )
            support_size = rng.choice(self.support_sizes)
            episodes.append(
                HistoryEpisode(
                    environment_id=environment_id,
                    support_indices=tuple(ordered[:support_size]),
                    query_indices=tuple(ordered[support_size:]),
                )
            )
        start = int(process_index) * int(environments_per_rank)
        return tuple(episodes[start : start + int(environments_per_rank)])


class FlamingoHistoryTrainingModule(nn.Module):
    def __init__(
        self,
        *,
        wan: nn.Module,
        support_encoder: FlamingoSupportEncoder,
        installation: FlamingoInstallation,
    ) -> None:
        super().__init__()
        self.wan = wan
        self.support_encoder = support_encoder
        self.installation = installation

    def extract_support_visual(self, video: Any):
        return self.support_encoder.extract_visual_features(video)

    def forward(
        self,
        *,
        query_data: dict[str, Any],
        support_visual_features: list[torch.Tensor],
        support_actions: list[Any],
        support_frame_indices: list[tuple[int, ...]],
        support_frame_counts: list[int],
    ) -> torch.Tensor:
        memory = self.support_encoder.project_supports(
            visual_features=support_visual_features,
            actions=support_actions,
            frame_indices=support_frame_indices,
            frame_counts=support_frame_counts,
        )
        self.installation.controller.set(memory)
        return self.wan(query_data)

    def clear_memory(self) -> None:
        self.installation.clear_memory()

    def gate_statistics(self) -> dict[str, float]:
        values = []
        for adapter in self.installation.adapters.values():
            values.extend(
                (
                    torch.tanh(adapter.alpha_xattn.detach()).abs().float(),
                    torch.tanh(adapter.alpha_dense.detach()).abs().float(),
                )
            )
        stacked = torch.stack(values)
        return {
            "gate_abs_mean": float(stacked.mean().item()),
            "gate_abs_max": float(stacked.max().item()),
        }

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        del remove_prefix
        return {
            key: value
            for key, value in state_dict.items()
            if (
                key.startswith("support_encoder.")
                and not key.startswith("support_encoder.dino.")
            )
            or key.startswith("wan.pipe.dit.flamingo_icl.")
        }
