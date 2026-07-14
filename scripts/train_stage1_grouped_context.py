#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import accelerate
import torch
import torch.nn as nn
from tqdm import tqdm

from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from wan_video_action.data import RoboTwinUnifiedDataset
from wan_video_action.data.data_utils import pack_paths
from wan_video_action.data.operators import LoadCobotAction, ResolvePromptEmbPath, create_video_operator
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed

from train import TimedRetentionModelLogger, WanTrainingModule, wan_parser


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _unique_friction_values(metadata_path: str) -> list[float]:
    values = sorted({float(row["friction_mu"]) for row in _read_jsonl(metadata_path)})
    if not values:
        raise ValueError(f"No friction_mu values found in {metadata_path}.")
    return values


def _as_float_tensor(value, *, device: torch.device | str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32).flatten()
    if isinstance(value, (list, tuple)):
        return torch.tensor([float(item) for item in value], device=device, dtype=torch.float32).flatten()
    return torch.tensor([float(value)], device=device, dtype=torch.float32)


class FrictionContextTable(nn.Module):
    def __init__(
        self,
        *,
        friction_values: list[float],
        context_dim: int,
        num_tokens: int,
        init_mode: str,
        init_value: float,
        init_std: float,
        init_min: float,
        init_max: float,
    ):
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}.")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        values = torch.tensor([float(value) for value in friction_values], dtype=torch.float32)
        self.register_buffer("friction_values", values, persistent=True)
        shape = (len(friction_values), int(num_tokens), int(context_dim))
        mode = str(init_mode).strip().lower()
        if mode == "uniform":
            contexts = torch.empty(shape, dtype=torch.float32)
            contexts.uniform_(float(init_min), float(init_max))
        elif mode == "shared_uniform":
            base_context = torch.empty((1, int(num_tokens), int(context_dim)), dtype=torch.float32)
            base_context.uniform_(float(init_min), float(init_max))
            contexts = base_context.repeat(len(friction_values), 1, 1)
        elif mode in ("shared_normal", "shared_gaussian"):
            base_context = torch.full((1, int(num_tokens), int(context_dim)), float(init_value), dtype=torch.float32)
            if init_std > 0:
                base_context.normal_(mean=float(init_value), std=float(init_std))
            contexts = base_context.repeat(len(friction_values), 1, 1)
        elif mode in ("normal", "gaussian"):
            contexts = torch.full(shape, float(init_value), dtype=torch.float32)
            if init_std > 0:
                contexts.normal_(mean=float(init_value), std=float(init_std))
        elif mode in ("constant", "fixed"):
            contexts = torch.full(shape, float(init_value), dtype=torch.float32)
        elif mode in ("ordered_linear", "linear_ordered"):
            ordered = torch.linspace(float(init_min), float(init_max), steps=len(friction_values), dtype=torch.float32)
            contexts = torch.full(shape, float(init_value), dtype=torch.float32)
            contexts[..., 0] = ordered[:, None].repeat(1, int(num_tokens))
        elif mode in ("ordered_initial_random_rest", "curriculum_ordered_initial_random_rest"):
            contexts = torch.empty(shape, dtype=torch.float32)
            contexts.uniform_(float(init_min), float(init_max))
        else:
            raise ValueError(f"Unsupported grouped_context_init_mode={init_mode!r}.")
        self.contexts = nn.Parameter(contexts)

    def lookup(self, friction_mu, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        query = _as_float_tensor(friction_mu, device=self.friction_values.device)
        distances = torch.abs(query[:, None] - self.friction_values[None, :])
        indices = torch.argmin(distances, dim=1)
        context = self.contexts[indices]
        return context.to(device=device, dtype=dtype)

    @torch.no_grad()
    def clamp_(self, min_value: float | None, max_value: float | None) -> None:
        if min_value is None and max_value is None:
            return
        self.contexts.clamp_(min=min_value, max=max_value)

    def to_records(self) -> list[dict]:
        values = self.friction_values.detach().float().cpu().tolist()
        contexts = self.contexts.detach().float().cpu().tolist()
        return [
            {
                "friction_mu": float(mu),
                "context": context,
            }
            for mu, context in zip(values, contexts)
        ]


class GroupedContextStage1Module(WanTrainingModule):
    def __init__(self, *args, grouped_args: argparse.Namespace, friction_values: list[float], **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.pipe, "physical_context_encoder", None) is None:
            raise ValueError("Grouped-C stage1 requires physical_context_mode != 'none'.")
        self.friction_context_table = FrictionContextTable(
            friction_values=friction_values,
            context_dim=int(grouped_args.physical_context_dim),
            num_tokens=int(grouped_args.physical_context_tokens),
            init_mode=str(grouped_args.grouped_context_init_mode),
            init_value=float(grouped_args.grouped_context_init_value),
            init_std=float(grouped_args.grouped_context_init_std),
            init_min=float(grouped_args.grouped_context_init_min),
            init_max=float(grouped_args.grouped_context_init_max),
        )

    def get_pipeline_inputs(self, data):
        data = data.copy()
        data["physical_context"] = self.friction_context_table.lookup(
            data["friction_mu"],
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )
        return super().get_pipeline_inputs(data)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        keep_names = set(trainable_names)
        keep_names.add("friction_context_table.friction_values")
        exported = {
            key: value
            for key, value in state_dict.items()
            if key in keep_names
        }
        if remove_prefix:
            prefix = str(remove_prefix)
            exported = {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in exported.items()
            }
        return exported


class GroupedContextModelLogger(TimedRetentionModelLogger):
    def save_model(self, accelerator, model, file_name):
        unwrapped = accelerator.unwrap_model(model)
        has_non_context_trainable = any(
            param.requires_grad
            for name, param in unwrapped.named_parameters()
            if name != "friction_context_table.contexts"
        )
        if has_non_context_trainable:
            super().save_model(accelerator, model, file_name)
        else:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(
                    "[checkpoint] skipped model checkpoint during context-only phase; "
                    "saving context table only",
                    flush=True,
                )
        if not accelerator.is_main_process:
            return
        table = getattr(unwrapped, "friction_context_table", None)
        if table is None:
            return
        path = os.path.join(self.output_path, file_name.replace(".safetensors", ".context_table.json"))
        self.save_context_table(accelerator, model, path)
        if has_non_context_trainable:
            self._prune_context_tables_without_checkpoints()

    def save_context_table(self, accelerator, model, path: str) -> None:
        if not accelerator.is_main_process:
            return
        table = getattr(accelerator.unwrap_model(model), "friction_context_table", None)
        if table is None:
            return
        payload = {
            "num_groups": int(table.friction_values.numel()),
            "context_shape": list(table.contexts.shape),
            "records": table.to_records(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[checkpoint] saved {path}", flush=True)

    def _prune_context_tables_without_checkpoints(self):
        checkpoint_stems = {
            name[:-len(".safetensors")]
            for name in os.listdir(self.output_path)
            if name.startswith("step-") and name.endswith(".safetensors")
        }
        for path in Path(self.output_path).glob("step-*.context_table.json"):
            stem = path.name[:-len(".context_table.json")]
            if stem in checkpoint_stems:
                continue
            try:
                path.unlink()
                print(f"[checkpoint] pruned {path}", flush=True)
            except FileNotFoundError:
                pass


def add_grouped_context_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("grouped_context_stage1")
    group.add_argument("--grouped_context_init_mode", type=str, default="normal")
    group.add_argument("--grouped_context_init_value", type=float, default=0.5)
    group.add_argument("--grouped_context_init_std", type=float, default=0.02)
    group.add_argument("--grouped_context_init_min", type=float, default=0.0)
    group.add_argument("--grouped_context_init_max", type=float, default=1.0)
    group.add_argument("--grouped_context_clamp_min", type=float, default=0.0)
    group.add_argument("--grouped_context_clamp_max", type=float, default=1.0)
    group.add_argument("--grouped_context_lr", type=float, default=None)
    group.add_argument("--grouped_context_new_context_lr", type=float, default=None)
    group.add_argument("--grouped_context_lr_schedule", type=str, default=None)
    group.add_argument("--grouped_context_model_lr_warmup_steps", type=int, default=0)
    group.add_argument("--grouped_context_alternating_interval", type=int, default=0)
    group.add_argument("--grouped_context_alternating_start", type=str, default="model")
    group.add_argument("--grouped_context_alternating_warmup_steps", type=int, default=0)
    group.add_argument("--grouped_context_weight_decay", type=float, default=0.0)
    group.add_argument("--grouped_context_structured_updates", type=int, default=0)
    group.add_argument("--grouped_context_friction_groups_per_update", type=int, default=4)
    group.add_argument("--grouped_context_actions_per_update", type=int, default=4)
    group.add_argument("--grouped_context_microbatches_per_update", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_initial_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_add_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_total_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_initial_model_steps", type=int, default=300)
    group.add_argument("--grouped_context_curriculum_new_context_steps", type=int, default=200)
    group.add_argument("--grouped_context_curriculum_mid_context_steps", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_all_context_steps", type=int, default=200)
    group.add_argument("--grouped_context_curriculum_model_steps", type=int, default=200)
    group.add_argument("--grouped_context_curriculum_variant", type=str, default="default")
    group.add_argument("--grouped_context_mid_context_lr", type=float, default=None)
    group.add_argument("--grouped_context_post_curriculum_cycle_steps", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_rest_init_min", type=float, default=0.4)
    group.add_argument("--grouped_context_curriculum_rest_init_max", type=float, default=0.6)
    group.add_argument("--grouped_context_curriculum_initial_jitter", type=float, default=0.0)
    group.add_argument("--grouped_context_curriculum_initial_refinement_steps", type=int, default=0)
    group.add_argument("--frame_stride", type=int, default=1)
    return parser


def _metadata_index(metadata_path: str) -> tuple[dict[float, dict[int, list[int]]], list[dict]]:
    rows = _read_jsonl(metadata_path)
    grouped: dict[float, dict[int, list[int]]] = {}
    for index, row in enumerate(rows):
        mu = float(row["friction_mu"])
        action_id = int(row.get("action_id", 0))
        grouped.setdefault(mu, {}).setdefault(action_id, []).append(index)
    return grouped, rows


def _sample_structured_indices(
    *,
    grouped_indices: dict[float, dict[int, list[int]]],
    rows: list[dict],
    rng: random.Random,
    friction_groups: int,
    actions_per_update: int,
    allowed_friction_values: list[float] | None = None,
) -> list[int]:
    allowed_values = None
    if allowed_friction_values is not None:
        allowed_values = [float(value) for value in allowed_friction_values]
    def is_allowed(mu: float) -> bool:
        if allowed_values is None:
            return True
        return any(abs(float(mu) - value) <= 1e-5 for value in allowed_values)
    valid_friction_values = [
        mu
        for mu, by_action in grouped_indices.items()
        if len(by_action) >= actions_per_update and is_allowed(float(mu))
    ]
    if len(valid_friction_values) < friction_groups:
        raise ValueError(
            f"Need at least {friction_groups} friction groups with {actions_per_update} actions, "
            f"got {len(valid_friction_values)}."
        )
    selected_mu = rng.sample(valid_friction_values, friction_groups)
    common_actions = set(grouped_indices[selected_mu[0]])
    for mu in selected_mu[1:]:
        common_actions &= set(grouped_indices[mu])
    if len(common_actions) < actions_per_update:
        raise ValueError(
            f"Selected friction groups share only {len(common_actions)} actions; "
            f"need {actions_per_update}. groups={selected_mu}"
        )
    selected_actions = rng.sample(sorted(common_actions), actions_per_update)

    indices = []
    for action_id in selected_actions:
        for mu in selected_mu:
            candidates = grouped_indices[mu][action_id]
            indices.append(rng.choice(candidates))
    rng.shuffle(indices)
    return indices


def _nested_uniform_group_order(num_groups: int, initial_groups: int, total_groups: int) -> list[int]:
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}.")
    total = min(int(total_groups) if total_groups > 0 else num_groups, num_groups)
    initial = min(max(int(initial_groups), 1), total)
    selected: list[int] = []

    def add(index: int) -> None:
        index = max(0, min(num_groups - 1, int(index)))
        if index not in selected:
            selected.append(index)

    if initial == 1:
        add((num_groups - 1) // 2)
    else:
        for i in range(initial):
            add(round(i * (num_groups - 1) / max(initial - 1, 1)))
    while len(selected) < initial:
        candidates = [index for index in range(num_groups) if index not in selected]
        best = max(candidates, key=lambda index: min(abs(index - old) for old in selected))
        add(best)

    while len(selected) < total:
        candidates = [index for index in range(num_groups) if index not in selected]
        center = (num_groups - 1) / 2.0
        best = max(
            candidates,
            key=lambda index: (
                min(abs(index - old) for old in selected),
                -abs(float(index) - center),
                -index,
            ),
        )
        add(best)
    return selected


def _curriculum_phase_for_step(args, group_order: list[int], step: int) -> dict:
    initial_groups = int(args.grouped_context_curriculum_initial_groups)
    add_groups = int(args.grouped_context_curriculum_add_groups)
    total_groups = int(args.grouped_context_curriculum_total_groups) or len(group_order)
    initial_steps = int(args.grouped_context_curriculum_initial_model_steps)
    new_context_steps = int(args.grouped_context_curriculum_new_context_steps)
    mid_context_steps = int(getattr(args, "grouped_context_curriculum_mid_context_steps", 0) or 0)
    all_context_steps = int(args.grouped_context_curriculum_all_context_steps)
    model_steps = int(args.grouped_context_curriculum_model_steps)
    post_cycle_steps = int(getattr(args, "grouped_context_post_curriculum_cycle_steps", 0) or 0)
    initial_refinement_steps = int(
        getattr(args, "grouped_context_curriculum_initial_refinement_steps", 0) or 0
    )
    variant = str(getattr(args, "grouped_context_curriculum_variant", "default") or "default").strip().lower()
    two_new_context = variant in ("two_new_context", "high_model_mid", "high_model_mid_new")
    total_groups = min(total_groups, len(group_order))

    if initial_groups <= 0 or add_groups <= 0:
        raise ValueError("curriculum mode requires positive initial_groups and add_groups.")
    initial_groups = min(initial_groups, total_groups)
    current_step = int(step)
    if current_step <= initial_steps:
        active = group_order[:initial_groups]
        return {
            "round": 0,
            "phase": "model",
            "sample_group_indices": active,
            "train_context_indices": [],
            "phase_start": 1,
            "phase_end": initial_steps,
            "active_count": len(active),
            "new_count": len(active),
        }

    offset = initial_steps
    selected_count = 0
    round_id = 0
    while selected_count < total_groups:
        round_id += 1
        if selected_count == 0:
            new_start = 0
            new_end = initial_groups
        else:
            new_start = selected_count
            new_end = min(selected_count + add_groups, total_groups)
        new_indices = group_order[new_start:new_end]
        active_indices = group_order[:new_end]
        if two_new_context:
            phases = [
                ("new_context", new_context_steps, new_indices, new_indices),
                ("model", model_steps, active_indices, []),
                ("new_context_mid", mid_context_steps or new_context_steps, new_indices, new_indices),
                ("all_context", all_context_steps, active_indices, active_indices),
                ("model", model_steps, active_indices, []),
                ("all_context", all_context_steps, active_indices, active_indices),
                ("model", model_steps, active_indices, []),
            ]
        else:
            phases = [
                ("new_context", new_context_steps, new_indices, new_indices),
                ("all_context", all_context_steps, active_indices, active_indices),
                ("model", model_steps, active_indices, []),
                ("all_context", all_context_steps, active_indices, active_indices),
                ("model", model_steps, active_indices, []),
            ]
        if round_id == 1 and initial_refinement_steps > 0:
            refinement_cycle_steps = all_context_steps + model_steps
            if refinement_cycle_steps <= 0 or initial_refinement_steps % refinement_cycle_steps != 0:
                raise ValueError(
                    "grouped_context_curriculum_initial_refinement_steps must be divisible by "
                    "all_context_steps + model_steps."
                )
            for _ in range(initial_refinement_steps // refinement_cycle_steps):
                phases.extend(
                    [
                        ("all_context", all_context_steps, active_indices, active_indices),
                        ("model", model_steps, active_indices, []),
                    ]
                )
        for phase, duration, sample_indices, train_indices in phases:
            start = offset + 1
            end = offset + int(duration)
            if start <= current_step <= end:
                return {
                    "round": round_id,
                    "phase": phase,
                    "sample_group_indices": sample_indices,
                    "train_context_indices": train_indices,
                    "phase_start": start,
                    "phase_end": end,
                    "active_count": len(active_indices),
                    "new_count": len(new_indices),
                }
            offset = end
        selected_count = new_end

    active = group_order[:total_groups]
    if post_cycle_steps > 0:
        post_context_steps = post_cycle_steps
        post_model_steps = model_steps
        cycle_duration = post_context_steps + post_model_steps
        cycle_offset = max(0, current_step - offset - 1)
        cycle_index = cycle_offset // cycle_duration
        cycle_start = offset + cycle_index * cycle_duration + 1
        local_offset = cycle_offset % cycle_duration
        if local_offset < post_context_steps:
            phase = "all_context"
            phase_start = cycle_start
            phase_end = cycle_start + post_context_steps - 1
            train_indices = active
        else:
            phase = "model"
            phase_start = cycle_start + post_context_steps
            phase_end = cycle_start + cycle_duration - 1
            train_indices = []
        return {
            "round": round_id + 1 + int(cycle_index),
            "phase": phase,
            "sample_group_indices": active,
            "train_context_indices": train_indices,
            "phase_start": phase_start,
            "phase_end": phase_end,
            "active_count": len(active),
            "new_count": 0,
        }
    return {
        "round": round_id + 1,
        "phase": "model",
        "sample_group_indices": active,
        "train_context_indices": [],
        "phase_start": offset + 1,
        "phase_end": current_step,
        "active_count": len(active),
        "new_count": 0,
    }


@torch.no_grad()
def _initialize_curriculum_contexts(table, args, group_order: list[int], total_groups: int) -> None:
    mode = str(getattr(args, "grouped_context_init_mode", "")).strip().lower()
    ordered_mode = mode in ("ordered_initial_random_rest", "curriculum_ordered_initial_random_rest")
    initial_jitter = float(getattr(args, "grouped_context_curriculum_initial_jitter", 0.0) or 0.0)
    if not ordered_mode and initial_jitter <= 0:
        return
    initial_groups = min(
        int(getattr(args, "grouped_context_curriculum_initial_groups", 0) or 0),
        int(total_groups),
        len(group_order),
    )
    if initial_groups <= 0:
        raise ValueError("ordered_initial_random_rest requires positive grouped_context_curriculum_initial_groups.")
    generator = torch.Generator(device=table.contexts.device)
    generator.manual_seed(int(getattr(args, "seed", 0)) + 7919)
    if initial_jitter > 0 and not ordered_mode:
        base_context = table.contexts[int(group_order[0])].detach().clone()
        clamp_min = getattr(args, "grouped_context_clamp_min", None)
        clamp_max = getattr(args, "grouped_context_clamp_max", None)
        for group_index in group_order[:initial_groups]:
            noise = torch.empty_like(base_context)
            noise.uniform_(-initial_jitter, initial_jitter, generator=generator)
            initialized = base_context + noise
            initialized.clamp_(min=clamp_min, max=clamp_max)
            table.contexts[int(group_index)].copy_(initialized)
        return
    rest_min = float(getattr(args, "grouped_context_curriculum_rest_init_min", 0.4))
    rest_max = float(getattr(args, "grouped_context_curriculum_rest_init_max", 0.6))
    table.contexts.uniform_(rest_min, rest_max, generator=generator)
    ordered = torch.linspace(
        float(getattr(args, "grouped_context_init_min", 0.2)),
        float(getattr(args, "grouped_context_init_max", 0.8)),
        steps=initial_groups,
        dtype=table.contexts.dtype,
        device=table.contexts.device,
    )
    for value, group_index in zip(ordered, group_order[:initial_groups]):
        table.contexts[int(group_index)].fill_(float(value))


def _build_optimizer(model, args):
    context_lr = getattr(args, "grouped_context_lr", None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "friction_context_table.contexts":
            context_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if other_params:
        param_groups.append(
            {
                "name": "model",
                "params": other_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if context_params:
        param_groups.append(
            {
                "name": "context",
                "params": context_params,
                "lr": float(context_lr),
                "weight_decay": float(getattr(args, "grouped_context_weight_decay", 0.0)),
            }
        )
    return torch.optim.AdamW(param_groups)


def _segment_scheduled_lr(schedule: str | None, step: int, default_lr: float) -> float:
    if not schedule:
        return float(default_lr)
    current = max(0, int(step))
    for raw_segment in str(schedule).split(","):
        segment = raw_segment.strip()
        if not segment:
            continue
        parts = [part.strip() for part in segment.split(":")]
        mode = parts[0].lower()
        if mode == "warmup" and len(parts) == 3:
            end = int(float(parts[1]))
            target_lr = float(parts[2])
            if current <= end:
                return target_lr * (float(current) / max(float(end), 1.0))
        elif mode == "hold" and len(parts) == 4:
            start = int(float(parts[1]))
            end = int(float(parts[2]))
            lr = float(parts[3])
            if start <= current < end:
                return lr
        elif mode == "linear" and len(parts) == 5:
            start = int(float(parts[1]))
            end = int(float(parts[2]))
            start_lr = float(parts[3])
            end_lr = float(parts[4])
            if start <= current < end:
                alpha = (float(current) - float(start)) / max(float(end - start), 1.0)
                return start_lr + alpha * (end_lr - start_lr)
        elif mode == "after" and len(parts) == 3:
            start = int(float(parts[1]))
            lr = float(parts[2])
            if current >= start:
                return lr
        else:
            raise ValueError(
                "Invalid grouped_context_lr_schedule segment. Expected one of "
                "warmup:end:lr, hold:start:end:lr, linear:start:end:lr0:lr1, after:start:lr; "
                f"got {segment!r}."
            )
    return float(default_lr)


def _apply_scheduled_lrs(optimizer, args, step: int):
    context_lr = getattr(args, "grouped_context_lr", None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_lr = _segment_scheduled_lr(
        getattr(args, "grouped_context_lr_schedule", None),
        step,
        float(context_lr),
    )
    model_lr = float(args.learning_rate)
    warmup_steps = int(getattr(args, "grouped_context_model_lr_warmup_steps", 0) or 0)
    if warmup_steps > 0:
        model_lr = model_lr * min(float(step) / float(warmup_steps), 1.0)
    phase = _alternating_phase(args, step)
    if phase == "model":
        context_lr = 0.0
    elif phase == "context":
        model_lr = 0.0
    for group in optimizer.param_groups:
        if group.get("name") == "context":
            group["lr"] = context_lr
        else:
            group["lr"] = model_lr
    return model_lr, context_lr


def _alternating_phase(args, step: int) -> str | None:
    interval = int(getattr(args, "grouped_context_alternating_interval", 0) or 0)
    if interval <= 0:
        return None
    warmup_steps = int(getattr(args, "grouped_context_alternating_warmup_steps", 0) or 0)
    if int(step) <= warmup_steps:
        return "model"
    phase_index = (max(1, int(step) - warmup_steps) - 1) // interval
    start = str(getattr(args, "grouped_context_alternating_start", "model")).strip().lower()
    if start not in ("model", "context"):
        raise ValueError(f"grouped_context_alternating_start must be 'model' or 'context', got {start!r}.")
    model_phase = (phase_index % 2 == 0) if start == "model" else (phase_index % 2 == 1)
    return "model" if model_phase else "context"


def _set_alternating_requires_grad(model, args, step: int) -> str | None:
    phase = _alternating_phase(args, step)
    if phase is None:
        for param in model.parameters():
            param.requires_grad_(True)
        return None
    for name, param in model.named_parameters():
        is_context = name.endswith("friction_context_table.contexts") or name == "friction_context_table.contexts"
        param.requires_grad_(is_context if phase == "context" else not is_context)
    return phase


def _set_curriculum_requires_grad(model, phase: str) -> None:
    context_phase = phase in ("new_context", "new_context_mid", "all_context", "context")
    for name, param in model.named_parameters():
        is_context = name.endswith("friction_context_table.contexts") or name == "friction_context_table.contexts"
        param.requires_grad_(is_context if context_phase else not is_context)


def _apply_curriculum_lrs(optimizer, args, step: int, phase: str):
    model_lr = float(args.learning_rate)
    warmup_steps = int(getattr(args, "grouped_context_model_lr_warmup_steps", 0) or 0)
    if warmup_steps > 0:
        model_lr = model_lr * min(float(step) / float(warmup_steps), 1.0)
    context_lr = getattr(args, "grouped_context_lr", None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_lr = float(context_lr)
    if phase == "new_context" and getattr(args, "grouped_context_new_context_lr", None) is not None:
        context_lr = float(args.grouped_context_new_context_lr)
    if phase == "new_context_mid" and getattr(args, "grouped_context_mid_context_lr", None) is not None:
        context_lr = float(args.grouped_context_mid_context_lr)
    if phase in ("new_context", "new_context_mid", "all_context", "context"):
        model_lr = 0.0
    else:
        context_lr = 0.0
    for group in optimizer.param_groups:
        if group.get("name") == "context":
            group["lr"] = context_lr
        else:
            group["lr"] = model_lr
    return model_lr, context_lr


def _mask_context_grad(model, train_context_indices: list[int]) -> None:
    table = getattr(model, "friction_context_table", None)
    if table is None or table.contexts.grad is None:
        return
    mask = torch.zeros(table.contexts.shape[0], device=table.contexts.grad.device, dtype=table.contexts.grad.dtype)
    for index in train_context_indices:
        if 0 <= int(index) < mask.numel():
            mask[int(index)] = 1
    table.contexts.grad.mul_(mask[:, None, None])


def _log_context_table(accelerator, model, step: int, phase: str | None, reason: str) -> None:
    if not accelerator.is_main_process:
        return
    table = getattr(accelerator.unwrap_model(model), "friction_context_table", None)
    if table is None:
        return
    values = table.friction_values.detach().float().cpu().tolist()
    contexts = table.contexts.detach().float().cpu()
    print(f"[context_table] step={step} phase={phase or 'none'} reason={reason}", flush=True)
    for mu, context in zip(values, contexts):
        flat = context.flatten()
        if flat.numel() == 1:
            summary = f"c={float(flat[0]):.6f}"
        else:
            head = ",".join(f"{float(x):.4f}" for x in flat[: min(6, flat.numel())])
            summary = (
                f"mean={float(flat.mean()):.6f} std={float(flat.std(unbiased=False)):.6f} "
                f"norm={float(torch.linalg.vector_norm(flat)):.6f} head=[{head}]"
            )
        print(f"[context_table] mu={float(mu):.6f} {summary}", flush=True)


def _save_phase_context_table(accelerator, model, model_logger, step: int, phase: str | None, reason: str) -> None:
    if not accelerator.is_main_process:
        return
    safe_phase = str(phase or "none").replace("/", "_")
    safe_reason = str(reason or "phase").replace("/", "_")
    path = os.path.join(
        model_logger.output_path,
        f"phase-step-{int(step):06d}-{safe_phase}-{safe_reason}.context_table.json",
    )
    model_logger.save_context_table(accelerator, model, path)


def build_dataset(args, runtime_config):
    special_operator_map = {}
    if runtime_config["text_enabled"] and "prompt_emb" in runtime_config["data_file_keys"]:
        special_operator_map["prompt_emb"] = ResolvePromptEmbPath(base_path=args.dataset_base_path)

    with open(args.action_stat_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats

    dataset = RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=runtime_config["data_file_keys"],
        main_data_operator=create_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
        resize_mode=args.resize_mode,
            frame_stride=args.frame_stride,
            pad_short=args.pad_short_chunks,
        ),
        special_operator_map=special_operator_map,
    )
    pack_paths(
        dataset.data,
        ("video", "start_frame", "end_frame"),
        ("action", "start_frame", "end_frame"),
    )
    if "action" in runtime_config["data_file_keys"]:
        dataset.special_operator_map["action"] = LoadCobotAction(
            base_path=args.dataset_base_path,
            action_type=args.action_type,
            stat=stat,
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            pad_short=args.pad_short_chunks,
            output_dim=args.action_dim,
            frame_stride=args.frame_stride,
        )
    return dataset


def launch_grouped_stage1(accelerator, dataset, model, model_logger, args):
    optimizer = _build_optimizer(model, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        collate_fn=lambda items: items[0],
        num_workers=args.dataset_num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    for epoch_id in range(args.num_epochs):
        iterator = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for data in iterator:
            with accelerator.accumulate(model):
                step = int(getattr(model_logger, "step", 0)) + 1
                _apply_scheduled_lrs(optimizer, args, step)
                _set_alternating_requires_grad(model, args, step)
                optimizer.zero_grad(set_to_none=True)
                loss = model(data)
                accelerator.backward(loss)
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                accelerator.unwrap_model(model).friction_context_table.clamp_(
                    args.grouped_context_clamp_min,
                    args.grouped_context_clamp_max,
                )
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)


def launch_structured_grouped_stage1(accelerator, dataset, model, model_logger, args):
    grouped_indices, metadata_rows = _metadata_index(args.dataset_metadata_path)
    optimizer = _build_optimizer(model, args)
    model.to(device=accelerator.device)
    model, optimizer = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    updates = int(args.grouped_context_structured_updates)
    friction_groups = int(args.grouped_context_friction_groups_per_update)
    actions_per_update = int(args.grouped_context_actions_per_update)
    microbatches_per_update = int(args.grouped_context_microbatches_per_update)
    if microbatches_per_update <= 0:
        microbatches_per_update = friction_groups * actions_per_update
    if updates <= 0:
        raise ValueError("grouped_context_structured_updates must be positive in structured mode.")

    if accelerator.is_main_process:
        context_lr = args.grouped_context_lr if args.grouped_context_lr is not None else args.learning_rate
        print(
            "[structured_grouped_stage1] "
            f"updates={updates} friction_groups={friction_groups} actions_per_update={actions_per_update} "
            f"microbatches_per_update={microbatches_per_update} "
            f"model_lr={args.learning_rate:g} context_lr={float(context_lr):g}",
            flush=True,
        )

    iterator = tqdm(range(updates), disable=not accelerator.is_local_main_process)
    for update_idx in iterator:
        rng = random.Random(int(args.seed) + update_idx * max(1, accelerator.num_processes) + accelerator.process_index)
        step = update_idx + 1
        _apply_scheduled_lrs(optimizer, args, step)
        _set_alternating_requires_grad(model, args, step)
        sample_indices = _sample_structured_indices(
            grouped_indices=grouped_indices,
            rows=metadata_rows,
            rng=rng,
            friction_groups=friction_groups,
            actions_per_update=actions_per_update,
        )
        if len(sample_indices) > microbatches_per_update:
            sample_indices = sample_indices[:microbatches_per_update]
        optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for micro_idx, sample_index in enumerate(sample_indices):
            sync_context = (
                accelerator.no_sync(model)
                if micro_idx < len(sample_indices) - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                loss = model(dataset[sample_index])
                detached_losses.append(loss.detach())
                accelerator.backward(loss / len(sample_indices))
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        accelerator.unwrap_model(model).friction_context_table.clamp_(
            args.grouped_context_clamp_min,
            args.grouped_context_clamp_max,
        )
        phase = _alternating_phase(args, step)
        interval = int(getattr(args, "grouped_context_alternating_interval", 0) or 0)
        warmup_steps = int(getattr(args, "grouped_context_alternating_warmup_steps", 0) or 0)
        phase_end = step == warmup_steps or (
            step > warmup_steps and interval > 0 and (step - warmup_steps) % interval == 0
        )
        if interval > 0 and phase_end:
            _log_context_table(accelerator, model, step, phase, "phase_end")
            _save_phase_context_table(accelerator, model, model_logger, step, phase, "phase_end")
        mean_loss = torch.stack([loss.float() for loss in detached_losses]).mean()
        model_logger.on_step_end(accelerator, model, args.save_steps, loss=mean_loss)

    model_logger.on_training_end(accelerator, model, args.save_steps)


def launch_curriculum_grouped_stage1(accelerator, dataset, model, model_logger, args):
    grouped_indices, metadata_rows = _metadata_index(args.dataset_metadata_path)
    unwrapped = model
    friction_values = unwrapped.friction_context_table.friction_values.detach().float().cpu().tolist()
    total_groups = int(args.grouped_context_curriculum_total_groups) or len(friction_values)
    group_order = _nested_uniform_group_order(
        len(friction_values),
        int(args.grouped_context_curriculum_initial_groups),
        total_groups,
    )
    _initialize_curriculum_contexts(unwrapped.friction_context_table, args, group_order, total_groups)
    optimizer = _build_optimizer(model, args)
    model.to(device=accelerator.device)
    model, optimizer = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    initial_steps = int(args.grouped_context_curriculum_initial_model_steps)
    add_groups = int(args.grouped_context_curriculum_add_groups)
    groups_after_initial = max(0, min(total_groups, len(friction_values)) - int(args.grouped_context_curriculum_initial_groups))
    rounds = 1 + (groups_after_initial + max(add_groups, 1) - 1) // max(add_groups, 1)
    variant = str(getattr(args, "grouped_context_curriculum_variant", "default") or "default").strip().lower()
    if variant in ("two_new_context", "high_model_mid", "high_model_mid_new"):
        default_updates = initial_steps + rounds * (
            int(args.grouped_context_curriculum_new_context_steps)
            + int(getattr(args, "grouped_context_curriculum_mid_context_steps", 0) or int(args.grouped_context_curriculum_new_context_steps))
            + 2 * int(args.grouped_context_curriculum_all_context_steps)
            + 3 * int(args.grouped_context_curriculum_model_steps)
        )
    else:
        default_updates = initial_steps + rounds * (
            int(args.grouped_context_curriculum_new_context_steps)
            + 2 * int(args.grouped_context_curriculum_all_context_steps)
            + 2 * int(args.grouped_context_curriculum_model_steps)
        )
    default_updates += int(getattr(args, "grouped_context_curriculum_initial_refinement_steps", 0) or 0)
    updates = int(args.grouped_context_structured_updates) or default_updates
    actions_per_update = int(args.grouped_context_actions_per_update)
    friction_groups_per_update = int(args.grouped_context_friction_groups_per_update)
    microbatches_per_update = int(args.grouped_context_microbatches_per_update)
    if microbatches_per_update <= 0:
        microbatches_per_update = friction_groups_per_update * actions_per_update

    if accelerator.is_main_process:
        print(
            "[curriculum_grouped_stage1] "
            f"updates={updates} initial_groups={args.grouped_context_curriculum_initial_groups} "
            f"add_groups={add_groups} total_groups={total_groups} rounds={rounds} "
            f"variant={getattr(args, 'grouped_context_curriculum_variant', 'default')} "
            f"actions_per_update={actions_per_update} microbatches_per_update={microbatches_per_update}",
            flush=True,
        )
        selected = [friction_values[index] for index in group_order[: min(total_groups, len(group_order))]]
        print(
            "[curriculum_grouped_stage1] group_order_mu="
            + ",".join(f"{float(value):.6g}" for value in selected),
            flush=True,
        )

    iterator = tqdm(range(updates), disable=not accelerator.is_local_main_process)
    last_phase_key = None
    for update_idx in iterator:
        step = update_idx + 1
        phase_info = _curriculum_phase_for_step(args, group_order, step)
        phase = str(phase_info["phase"])
        phase_key = (phase_info["round"], phase, phase_info["phase_start"], phase_info["phase_end"])
        if accelerator.is_main_process and phase_key != last_phase_key:
            sample_mu = [friction_values[index] for index in phase_info["sample_group_indices"]]
            print(
                "[curriculum_phase] "
                f"step={step} round={phase_info['round']} phase={phase} "
                f"range={phase_info['phase_start']}-{phase_info['phase_end']} "
                f"active_count={phase_info['active_count']} new_count={phase_info['new_count']} "
                f"sample_mu={','.join(f'{float(value):.6g}' for value in sample_mu)}",
                flush=True,
            )
            last_phase_key = phase_key

        _set_curriculum_requires_grad(model, phase)
        _apply_curriculum_lrs(optimizer, args, step, phase)
        rng = random.Random(int(args.seed) + update_idx * max(1, accelerator.num_processes) + accelerator.process_index)
        sample_values = [friction_values[index] for index in phase_info["sample_group_indices"]]
        sample_friction_groups = min(friction_groups_per_update, len(sample_values))
        sample_indices = _sample_structured_indices(
            grouped_indices=grouped_indices,
            rows=metadata_rows,
            rng=rng,
            friction_groups=sample_friction_groups,
            actions_per_update=actions_per_update,
            allowed_friction_values=sample_values,
        )
        if len(sample_indices) > microbatches_per_update:
            sample_indices = sample_indices[:microbatches_per_update]
        optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for micro_idx, sample_index in enumerate(sample_indices):
            sync_context = (
                accelerator.no_sync(model)
                if micro_idx < len(sample_indices) - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                loss = model(dataset[sample_index])
                detached_losses.append(loss.detach())
                accelerator.backward(loss / len(sample_indices))
        if phase in ("new_context", "new_context_mid", "all_context"):
            _mask_context_grad(accelerator.unwrap_model(model), phase_info["train_context_indices"])
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        accelerator.unwrap_model(model).friction_context_table.clamp_(
            args.grouped_context_clamp_min,
            args.grouped_context_clamp_max,
        )
        if step == int(phase_info["phase_end"]):
            _log_context_table(accelerator, model, step, phase, "curriculum_phase_end")
            _save_phase_context_table(accelerator, model, model_logger, step, phase, "curriculum_phase_end")
        mean_loss = torch.stack([loss.float() for loss in detached_losses]).mean()
        model_logger.on_step_end(accelerator, model, args.save_steps, loss=mean_loss)

    model_logger.on_training_end(accelerator, model, args.save_steps)


def main() -> None:
    parser = add_grouped_context_config(wan_parser())
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)

    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    friction_values = _unique_friction_values(args.dataset_metadata_path)
    loggers = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}", False)]
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=loggers or None,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = build_dataset(args, runtime_config)
    model = GroupedContextStage1Module(
        model_paths=json.dumps(runtime_config["model_paths_list"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=runtime_config["tokenizer_path"],
        enable_text=args.enable_text,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        modules=runtime_config["modules"],
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        args=args,
        grouped_args=args,
        friction_values=friction_values,
    )
    if accelerator.is_main_process:
        print(
            f"Grouped-C Stage1: groups={len(friction_values)} "
            f"context_dim={args.physical_context_dim} tokens={args.physical_context_tokens}",
            flush=True,
        )

    model_logger = GroupedContextModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_minutes=args.checkpoint_save_minutes,
        keep_last=args.checkpoint_keep_last,
        log_steps=args.log_steps,
    )
    if int(getattr(args, "grouped_context_curriculum_initial_groups", 0) or 0) > 0:
        launch_curriculum_grouped_stage1(accelerator, dataset, model, model_logger, args)
    elif int(getattr(args, "grouped_context_structured_updates", 0) or 0) > 0:
        launch_structured_grouped_stage1(accelerator, dataset, model, model_logger, args)
    else:
        launch_grouped_stage1(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
