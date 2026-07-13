#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch

import train_stage2_ttt as base_stage2


def _as_int(value, default: int = 0) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().flatten()[0].item())
    if isinstance(value, (list, tuple)) and value:
        return _as_int(value[0], default)
    if value is None:
        return int(default)
    return int(value)


class TimeSeriesStage2TTTTrainingModule(base_stage2.Stage2TTTTrainingModule):
    """Stage2 TTT with an episode-level per-frame latent table.

    The old stage2 trainer adapts one C tensor shared by every support/query chunk.
    For time-varying state memory, this class adapts a full table shaped
    [episode_total_frames, physical_context_dim]. Each chunk receives the rows
    listed in metadata["physical_context_frame_indices"], so the 11 DiT condition
    tokens remain aligned to absolute episode time.
    """

    def __init__(self, *args, stage2_args, **kwargs):
        super().__init__(*args, stage2_args=stage2_args, **kwargs)
        self.trajectory_log_interval = int(getattr(stage2_args, "stage2_trajectory_log_interval", 20) or 0)
        rank = int(os.environ.get("RANK", "0"))
        output_path = Path(str(getattr(stage2_args, "output_path", ".") or "."))
        self.trajectory_log_path = output_path / f"stage2_trajectory_samples_rank{rank}.jsonl"
        self._trajectory_log_counter = 0
        self._last_context0 = None
        self._last_context_adapted = None

    def _task_items(self, task: dict) -> list[dict]:
        return list(task.get("support", [])) + list(task.get("query", []))

    def _infer_active_total_frames(self, task: dict) -> int:
        total = 0
        for item in self._task_items(task):
            total = max(total, _as_int(item.get("total_frames"), 0))
            indices = item.get("physical_context_frame_indices")
            if isinstance(indices, torch.Tensor):
                if indices.numel() > 0:
                    total = max(total, int(indices.detach().cpu().max().item()) + 1)
            elif isinstance(indices, (list, tuple)) and indices:
                total = max(total, max(int(v) for v in indices) + 1)
        if total <= 0:
            default_context = self.pipe.physical_context_encoder.default_context
            total = int(default_context.shape[0]) if default_context.ndim > 1 else 1
        return total

    def forward(self, task: dict) -> torch.Tensor:
        self._active_episode_total_frames = self._infer_active_total_frames(task)
        try:
            loss = super().forward(task)
            context_adapted = getattr(self, "_last_context_adapted", None)
            if context_adapted is not None:
                self._maybe_log_trajectory_sample(task, context_adapted)
            return loss
        finally:
            self._active_episode_total_frames = None

    def _adapt_context(self, support: list[dict], context0: torch.Tensor) -> torch.Tensor:
        context_adapted = super()._adapt_context(support, context0)
        self._last_context0 = context0.detach()
        self._last_context_adapted = context_adapted.detach()
        return context_adapted

    def _context0(self) -> torch.Tensor:
        total_frames = getattr(self, "_active_episode_total_frames", None)
        if total_frames is None:
            return super()._context0()
        default_context = self.pipe.physical_context_encoder.default_context
        context_dim = int(default_context.shape[-1])
        if self.context_init_mode == "uniform":
            context = torch.empty(
                (int(total_frames), context_dim),
                dtype=default_context.dtype,
                device=default_context.device,
            ).uniform_(self.context_init_min, self.context_init_max)
            return context.requires_grad_(True)
        if default_context.ndim == 1:
            seed = default_context.view(1, context_dim)
        else:
            seed = default_context.mean(dim=0, keepdim=True)
        return seed.expand(int(total_frames), context_dim).clone().requires_grad_(True)

    def _context_frame_indices(self, data: dict, table_len: int, token_count: int) -> torch.Tensor:
        raw = data.get("physical_context_frame_indices")
        if isinstance(raw, torch.Tensor):
            indices = [int(v) for v in raw.detach().cpu().flatten().tolist()]
        elif isinstance(raw, (list, tuple)):
            indices = [int(v) for v in raw]
        else:
            start = _as_int(data.get("start_frame"), 0)
            stride = _as_int(data.get("physical_context_stride"), 4)
            indices = [start + stride * i for i in range(token_count)]
        if not indices:
            indices = [0]
        if len(indices) < token_count:
            indices = indices + [indices[-1]] * (token_count - len(indices))
        elif len(indices) > token_count:
            indices = indices[:token_count]
        max_index = max(0, int(table_len) - 1)
        indices = [min(max(0, index), max_index) for index in indices]
        return torch.tensor(indices, dtype=torch.long)

    def _compose_physical_context(self, data: dict, physical_context: torch.Tensor) -> torch.Tensor:
        default_context = self.pipe.physical_context_encoder.default_context
        default_tokens = int(default_context.shape[0]) if default_context.ndim > 1 else 1
        is_episode_table = physical_context.ndim == 2 and int(physical_context.shape[0]) != default_tokens
        if not is_episode_table:
            return super()._compose_physical_context(data, physical_context)
        indices = self._context_frame_indices(
            data,
            table_len=int(physical_context.shape[0]),
            token_count=default_tokens,
        ).to(device=physical_context.device)
        chunk_context = physical_context.index_select(0, indices)
        return super()._compose_physical_context(data, chunk_context)

    @staticmethod
    def _nested_tensor(raw, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if raw is None:
            return None
        if isinstance(raw, torch.Tensor):
            return raw.detach().to(device=device, dtype=dtype)
        return torch.tensor(raw, device=device, dtype=dtype)

    @staticmethod
    def _to_jsonable(tensor: torch.Tensor) -> list:
        return tensor.detach().float().cpu().tolist()

    @staticmethod
    def _norm_xy_to_raw(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 0.60 - 0.30

    def _target_token_context(self, data: dict, *, token_count: int, device: torch.device, dtype: torch.dtype):
        target = self._nested_tensor(data.get("target_physical_context"), device=device, dtype=dtype)
        if target is None:
            return None
        target = target.reshape(-1, int(target.shape[-1]))
        if target.shape[0] < token_count:
            pad = target[-1:].expand(token_count - int(target.shape[0]), int(target.shape[-1]))
            target = torch.cat([target, pad], dim=0)
        return target[:token_count]

    def _target_window_context(self, data: dict, *, device: torch.device, dtype: torch.dtype):
        target = self._nested_tensor(data.get("target_payload_xy_norm_window"), device=device, dtype=dtype)
        if target is None:
            return None
        return target.reshape(-1, int(target.shape[-1]))

    def _learned_token_context(self, data: dict, context_table: torch.Tensor, *, token_count: int) -> torch.Tensor:
        indices = self._context_frame_indices(
            data,
            table_len=int(context_table.shape[0]),
            token_count=token_count,
        ).to(device=context_table.device)
        return context_table.index_select(0, indices)

    def _learned_window_context(self, data: dict, context_table: torch.Tensor) -> torch.Tensor:
        start = _as_int(data.get("start_frame"), 0)
        end = _as_int(data.get("end_frame"), start)
        if end < start:
            end = start
        indices = torch.arange(start, end + 1, dtype=torch.long, device=context_table.device)
        indices = torch.clamp(indices, 0, max(0, int(context_table.shape[0]) - 1))
        return context_table.index_select(0, indices)

    def _trajectory_record_for_item(self, kind: str, data: dict, context_table: torch.Tensor) -> dict:
        default_context = self.pipe.physical_context_encoder.default_context
        token_count = int(default_context.shape[0]) if default_context.ndim > 1 else 1
        learned_tokens = self._learned_token_context(data, context_table, token_count=token_count)
        learned_window = self._learned_window_context(data, context_table)
        true_tokens = self._target_token_context(
            data,
            token_count=token_count,
            device=context_table.device,
            dtype=context_table.dtype,
        )
        true_window = self._target_window_context(data, device=context_table.device, dtype=context_table.dtype)
        record = {
            "kind": kind,
            "sample_id": data.get("sample_id"),
            "episode_index": _as_int(data.get("episode_index"), -1),
            "chunk_id": _as_int(data.get("chunk_id"), -1),
            "start_frame": _as_int(data.get("start_frame"), 0),
            "end_frame": _as_int(data.get("end_frame"), 0),
            "token_frame_indices": self._context_frame_indices(
                data,
                table_len=int(context_table.shape[0]),
                token_count=token_count,
            ).tolist(),
            "learned_token_context_norm": self._to_jsonable(learned_tokens),
            "learned_token_xy_raw": self._to_jsonable(self._norm_xy_to_raw(learned_tokens)),
            "learned_window_context_norm": self._to_jsonable(learned_window),
            "learned_window_xy_raw": self._to_jsonable(self._norm_xy_to_raw(learned_window)),
        }
        if true_tokens is not None:
            token_delta = learned_tokens.float() - true_tokens.float()
            record["true_token_context_norm"] = self._to_jsonable(true_tokens)
            record["true_token_xy_raw"] = self._to_jsonable(self._norm_xy_to_raw(true_tokens))
            record["token_mae"] = float(token_delta.abs().mean().detach().cpu())
            record["token_mse"] = float((token_delta ** 2).mean().detach().cpu())
        if true_window is not None:
            n = min(int(true_window.shape[0]), int(learned_window.shape[0]))
            window_delta = learned_window[:n].float() - true_window[:n].float()
            record["true_window_context_norm"] = self._to_jsonable(true_window)
            record["true_window_xy_raw"] = self._to_jsonable(self._norm_xy_to_raw(true_window))
            record["window_mae"] = float(window_delta.abs().mean().detach().cpu())
            record["window_mse"] = float((window_delta ** 2).mean().detach().cpu())
        return record

    def _maybe_log_trajectory_sample(self, task: dict, context_adapted: torch.Tensor) -> None:
        if context_adapted.ndim != 2:
            return
        self._trajectory_log_counter += 1
        support = task.get("support", [])
        query = task.get("query", [])
        samples = []
        if support:
            samples.append(("support", support[0]))
        if query:
            samples.append(("query", query[0]))
        if not samples:
            return

        records = [
            self._trajectory_record_for_item(kind, data, context_adapted)
            for kind, data in samples
        ]
        for record in records:
            prefix = f"traj_{record['kind']}"
            if "token_mae" in record:
                self.last_metrics[f"{prefix}_token_mae"] = float(record["token_mae"])
            if "window_mae" in record:
                self.last_metrics[f"{prefix}_window_mae"] = float(record["window_mae"])

        interval = self.trajectory_log_interval
        if interval <= 0:
            return
        if self._trajectory_log_counter != 1 and self._trajectory_log_counter % interval != 0:
            return
        self.trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "local_step": self._trajectory_log_counter,
            "group_key": task.get("group_key"),
            "records": records,
        }
        with self.trajectory_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


base_stage2.Stage2TTTTrainingModule = TimeSeriesStage2TTTTrainingModule


if __name__ == "__main__":
    base_stage2.main()
