"""Frozen-DINOv2 amortized environment-code components for Event80."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import yaml
from transformers import AutoModel


@dataclass(frozen=True)
class SupportQueryEpisodeIndices:
    environment_id: int
    support_index: int
    query_indices: tuple[int, ...]


class Event80K1Sampler:
    """Samples one support and every remaining action as read-only queries."""

    def __init__(
        self,
        *,
        metadata_path: str | Path,
        active_environment_manifest: str | Path,
        seed: int,
        queries_per_environment: int = 0,
        environment_key: str = "mu_index",
        action_key: str = "action_id",
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.environment_key = str(environment_key)
        self.action_key = str(action_key)
        self.seed = int(seed)
        self.queries_per_environment = int(queries_per_environment)
        if self.queries_per_environment < 0:
            raise ValueError("queries_per_environment must be non-negative.")
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        with Path(active_environment_manifest).open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        self.active_environment_ids = tuple(
            int(value)
            for value in manifest["selection"]["active_environment_ids"]
        )
        grouped: dict[int, list[int]] = {value: [] for value in self.active_environment_ids}
        for index, row in enumerate(self.rows):
            environment_id = int(row[self.environment_key])
            if environment_id in grouped:
                grouped[environment_id].append(index)
        for environment_id, indices in grouped.items():
            indices.sort(key=lambda index: (int(self.rows[index][self.action_key]), index))
            actions = [int(self.rows[index][self.action_key]) for index in indices]
            if len(indices) < 2 or len(actions) != len(set(actions)):
                raise ValueError(
                    f"Environment {environment_id} needs at least two unique actions; "
                    f"got actions={actions}."
                )
        self.grouped_indices = grouped

    def sample(
        self,
        *,
        step: int,
        process_index: int,
        num_processes: int,
        environments_per_rank: int,
    ) -> tuple[SupportQueryEpisodeIndices, ...]:
        global_count = int(num_processes) * int(environments_per_rank)
        if global_count <= 0 or global_count > len(self.active_environment_ids):
            raise ValueError(
                f"Invalid global environment count {global_count} for "
                f"{len(self.active_environment_ids)} active environments."
            )
        rng = random.Random(self.seed + int(step) * 104729)
        selected = rng.sample(list(self.active_environment_ids), global_count)
        episodes = []
        for environment_id in selected:
            indices = self.grouped_indices[environment_id]
            support_index = indices[rng.randrange(len(indices))]
            remaining_queries = [index for index in indices if index != support_index]
            if self.queries_per_environment:
                if self.queries_per_environment > len(remaining_queries):
                    raise ValueError(
                        f"Environment {environment_id} has only {len(remaining_queries)} "
                        f"disjoint queries, but {self.queries_per_environment} were requested."
                    )
                remaining_queries = rng.sample(
                    remaining_queries,
                    self.queries_per_environment,
                )
                remaining_queries.sort(
                    key=lambda index: (int(self.rows[index][self.action_key]), index)
                )
            query_indices = tuple(remaining_queries)
            episodes.append(
                SupportQueryEpisodeIndices(
                    environment_id=environment_id,
                    support_index=support_index,
                    query_indices=query_indices,
                )
            )
        start = int(process_index) * int(environments_per_rank)
        end = start + int(environments_per_rank)
        return tuple(episodes[start:end])


class ActionChunkEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.step_projection = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for chunk in chunks:
            if chunk.ndim != 2 or chunk.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected action chunk [T,{self.action_dim}], got {tuple(chunk.shape)}."
                )
            encoded = self.step_projection(chunk).unsqueeze(0)
            _, hidden = self.temporal_encoder(encoded)
            outputs.append(hidden[-1, 0])
        return torch.stack(outputs, dim=0)


class DINOv2AmortizedContextEncoder(nn.Module):
    """Maps one observed support video/action trajectory to a bounded 32-D Z."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        sampled_frames: int = 8,
        action_dim: int = 14,
        hidden_dim: int = 256,
        action_hidden_dim: int = 128,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).expanduser())
        self.sampled_frames = int(sampled_frames)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        if self.sampled_frames < 2:
            raise ValueError("sampled_frames must be at least two.")
        self.dino = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.dino.requires_grad_(False)
        self.dino.eval()
        dino_dim = int(self.dino.config.hidden_size)

        processor_path = Path(self.model_path) / "preprocessor_config.json"
        processor = {}
        if processor_path.is_file():
            with processor_path.open("r", encoding="utf-8") as handle:
                processor = json.load(handle)
        image_mean = processor.get("image_mean", [0.485, 0.456, 0.406])
        image_std = processor.get("image_std", [0.229, 0.224, 0.225])
        size = processor.get("size", {"shortest_edge": 224})
        if isinstance(size, dict):
            image_size = int(size.get("height", size.get("shortest_edge", 224)))
        else:
            image_size = int(size)
        self.image_size = image_size
        self.register_buffer(
            "image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(2 * dino_dim),
            nn.Linear(2 * dino_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_encoder = ActionChunkEncoder(self.action_dim, action_hidden_dim)
        self.transition_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + action_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    @staticmethod
    def _canonical_video(video: Any) -> torch.Tensor:
        tensor = torch.as_tensor(video, dtype=torch.float32)
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(
                "Support video must have shape [V,C,T,H,W] or [C,T,H,W], "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != 3:
            raise ValueError(f"Support video must be RGB, got shape={tuple(tensor.shape)}.")
        return tensor

    def extract_visual_features(
        self, video: Any
    ) -> tuple[torch.Tensor, tuple[int, ...], int]:
        video_tensor = self._canonical_video(video)
        frame_count = int(video_tensor.shape[1])
        indices_tensor = torch.linspace(
            0, frame_count - 1, steps=self.sampled_frames
        ).round().to(dtype=torch.long)
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
        frames = (frames - self.image_mean.to(device=device)) / self.image_std.to(device=device)
        with torch.no_grad():
            hidden = self.dino(pixel_values=frames).last_hidden_state
            cls = hidden[:, 0]
            mean_patch = hidden[:, 1:].mean(dim=1)
            features = torch.cat((cls, mean_patch), dim=-1).detach()
        return features, indices, frame_count

    def _canonical_actions(self, action: Any, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 1:
            if tensor.numel() % self.action_dim != 0:
                raise ValueError(f"Cannot reshape action vector of length {tensor.numel()}.")
            tensor = tensor.view(-1, self.action_dim)
        if tensor.ndim != 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.shape[-1] != self.action_dim and tensor.shape[0] == self.action_dim:
            tensor = tensor.transpose(0, 1)
        if tensor.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got shape={tuple(tensor.shape)}."
            )
        return tensor

    def project_support(
        self,
        *,
        visual_features: torch.Tensor,
        action: Any,
        frame_indices: tuple[int, ...],
        frame_count: int,
    ) -> torch.Tensor:
        device = visual_features.device
        actions = self._canonical_actions(action, device)
        action_count = int(actions.shape[0])
        chunks = []
        for start_frame, end_frame in zip(frame_indices[:-1], frame_indices[1:]):
            denominator = max(int(frame_count) - 1, 1)
            start = int(round(start_frame / denominator * max(action_count - 1, 0)))
            end = int(round(end_frame / denominator * max(action_count - 1, 0)))
            end = max(start + 1, min(action_count, end))
            chunks.append(actions[start:end])
        frame_features = self.visual_projection(visual_features)
        action_features = self.action_encoder(chunks)
        transition_features = self.transition_projection(
            torch.cat(
                (
                    frame_features[:-1],
                    frame_features[1:] - frame_features[:-1],
                    action_features,
                ),
                dim=-1,
            )
        )
        code = self.output_head(transition_features.mean(dim=0, keepdim=True))
        return torch.sigmoid(code)

    def forward(self, video: Any, action: Any) -> torch.Tensor:
        features, indices, frame_count = self.extract_visual_features(video)
        return self.project_support(
            visual_features=features,
            action=action,
            frame_indices=indices,
            frame_count=frame_count,
        )
