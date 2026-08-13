#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
from collections import defaultdict
import json
import os
import random
import shutil
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import accelerate
from accelerate.utils import InitProcessGroupKwargs
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from wan_video_action.data import RoboTwinUnifiedDataset
from wan_video_action.counterfactual_bridge import (
    CounterfactualSourceBank,
    parse_noise_bands,
    sample_noise_fraction,
    sample_nonlinear_bridge_condition,
)
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


@torch.no_grad()
def _load_grouped_context_table(model, path: str) -> bool:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    table = model.friction_context_table
    seen: set[int] = set()
    for record in payload.get("records", []):
        value = float(record["friction_mu"])
        distances = torch.abs(table.friction_values - value)
        index = int(torch.argmin(distances).item())
        if float(distances[index]) > 1e-5:
            raise ValueError(f"Context-table value {value} is absent from the current model table.")
        context = torch.tensor(
            record["context"],
            dtype=table.contexts.dtype,
            device=table.contexts.device,
        )
        if tuple(context.shape) != tuple(table.contexts[index].shape):
            raise ValueError(
                f"Context shape mismatch for group {value}: "
                f"checkpoint={tuple(context.shape)} model={tuple(table.contexts[index].shape)}"
            )
        table.contexts[index].copy_(context)
        seen.add(index)
    if len(seen) != int(table.friction_values.numel()):
        raise ValueError(
            f"Context table {path} restored {len(seen)} groups, "
            f"expected {int(table.friction_values.numel())}."
        )
    global_context = getattr(table, "global_context", None)
    saved_global_context = payload.get("global_context")
    if global_context is None or saved_global_context is None:
        return False
    restored_global = torch.tensor(
        saved_global_context,
        dtype=global_context.dtype,
        device=global_context.device,
    )
    if tuple(restored_global.shape) != tuple(global_context.shape):
        raise ValueError(
            f"Global context shape mismatch: checkpoint={tuple(restored_global.shape)} "
            f"model={tuple(global_context.shape)}"
        )
    global_context.copy_(restored_global)
    return True


@torch.no_grad()
def _load_background_context_table(model, path: str) -> bool:
    table = getattr(model, "background_context_table", None)
    if table is None:
        return False
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    section = payload.get("background_context_table")
    if not section:
        raise ValueError(f"Context table {path} does not contain background_context_table.")
    seen: set[int] = set()
    for record in section.get("records", []):
        value = float(record["background_index"])
        distances = torch.abs(table.friction_values - value)
        index = int(torch.argmin(distances).item())
        if float(distances[index]) > 1e-5:
            raise ValueError(f"Background-table value {value} is absent from the current model table.")
        context = torch.tensor(record["context"], dtype=table.contexts.dtype, device=table.contexts.device)
        if tuple(context.shape) != tuple(table.contexts[index].shape):
            raise ValueError(
                f"Background context shape mismatch for index {value}: "
                f"checkpoint={tuple(context.shape)} model={tuple(table.contexts[index].shape)}"
            )
        table.contexts[index].copy_(context)
        seen.add(index)
    if len(seen) != int(table.friction_values.numel()):
        raise ValueError(
            f"Background context table {path} restored {len(seen)} groups, "
            f"expected {int(table.friction_values.numel())}."
        )
    return True


def _unique_friction_values(metadata_path: str) -> list[float]:
    values = sorted({float(row["friction_mu"]) for row in _read_jsonl(metadata_path)})
    if not values:
        raise ValueError(f"No friction_mu values found in {metadata_path}.")
    return values


def _unique_background_values(metadata_path: str) -> list[float]:
    values = sorted({float(row["environment_index"]) for row in _read_jsonl(metadata_path)})
    if not values:
        raise ValueError(f"No environment_index values found in {metadata_path}.")
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
        enable_global_context: bool = False,
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
        if enable_global_context:
            self.global_context = nn.Parameter(contexts.detach().mean(dim=0).clone())
        else:
            self.register_parameter("global_context", None)

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
    def __init__(
        self,
        *args,
        grouped_args: argparse.Namespace,
        friction_values: list[float],
        background_values: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if getattr(self.pipe, "physical_context_encoder", None) is None:
            raise ValueError("Grouped-C stage1 requires physical_context_mode != 'none'.")
        self.bridge_enabled = bool(getattr(grouped_args, "grouped_context_bridge_enabled", False))
        self.self_correction_enabled = bool(
            getattr(grouped_args, "grouped_context_self_correction_enabled", False)
        )
        self.self_correction_sigma_min = float(
            getattr(grouped_args, "grouped_context_self_correction_sigma_min", 0.45)
        )
        self.self_correction_sigma_max = float(
            getattr(grouped_args, "grouped_context_self_correction_sigma_max", 0.85)
        )
        self.self_correction_source_mix = float(
            getattr(grouped_args, "grouped_context_self_correction_source_mix", 0.5)
        )
        self.last_self_correction_metrics: dict | None = None
        self.friction_context_table = FrictionContextTable(
            friction_values=friction_values,
            context_dim=int(grouped_args.physical_context_dim),
            num_tokens=int(grouped_args.physical_context_tokens),
            init_mode=str(grouped_args.grouped_context_init_mode),
            init_value=float(grouped_args.grouped_context_init_value),
            init_std=float(grouped_args.grouped_context_init_std),
            init_min=float(grouped_args.grouped_context_init_min),
            init_max=float(grouped_args.grouped_context_init_max),
            enable_global_context=self.bridge_enabled,
        )
        self.background_context_table = None
        if getattr(self.pipe, "background_context_encoder", None) is not None:
            if not background_values:
                raise ValueError("Background context is enabled, but no environment_index values were provided.")
            self.background_context_table = FrictionContextTable(
                friction_values=background_values,
                context_dim=int(grouped_args.background_context_dim),
                num_tokens=int(grouped_args.background_context_tokens),
                init_mode=str(grouped_args.grouped_context_init_mode),
                init_value=float(grouped_args.grouped_context_init_value),
                init_std=float(grouped_args.grouped_context_init_std),
                init_min=float(grouped_args.grouped_context_init_min),
                init_max=float(grouped_args.grouped_context_init_max),
            )

    def get_pipeline_inputs(self, data):
        data = data.copy()
        direct_physical_context = data.pop("_direct_physical_context", None)
        bridge_alpha = data.pop("_bridge_alpha", None)
        bridge_target_mu = data.pop("_bridge_target_mu", None)
        if direct_physical_context is not None:
            physical_context = torch.as_tensor(
                direct_physical_context,
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
        elif bridge_alpha is None:
            physical_context = self.friction_context_table.lookup(
                data["friction_mu"],
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
        else:
            if self.friction_context_table.global_context is None:
                raise RuntimeError("Bridge context requested without a global context parameter.")
            endpoint_context = self.friction_context_table.lookup(
                bridge_target_mu,
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
            global_context = self.friction_context_table.global_context.to(
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            ).unsqueeze(0)
            alpha = float(bridge_alpha)
            physical_context = (1.0 - alpha) * global_context + alpha * endpoint_context
        data["physical_context"] = physical_context
        if self.background_context_table is not None:
            data["background_context"] = self.background_context_table.lookup(
                data["environment_index"],
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
        return super().get_pipeline_inputs(data)

    def _prepare_pipeline_inputs(self, data):
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def _flow_match_loss_at_timestep(self, target_inputs, timestep_index: int):
        inputs_shared, inputs_posi, _ = target_inputs
        inputs_shared = inputs_shared.copy()
        timestep = self.pipe.scheduler.timesteps[timestep_index:timestep_index + 1].to(
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )
        noise = torch.randn_like(inputs_shared["input_latents"])
        inputs_shared["latents"] = self.pipe.scheduler.add_noise(
            inputs_shared["input_latents"],
            noise,
            timestep,
        )
        training_target = self.pipe.scheduler.training_target(
            inputs_shared["input_latents"],
            noise,
            timestep,
        )
        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        noise_pred = self.pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            timestep=timestep,
        )
        if "first_frame_latents" in inputs_shared:
            noise_pred = noise_pred[:, :, 1:]
            training_target = training_target[:, :, 1:]
        loss = F.mse_loss(noise_pred.float(), training_target.float())
        return loss * self.pipe.scheduler.training_weight(timestep)

    def _self_correction_flow_loss(
        self,
        target_inputs,
        donor_inputs,
        timestep_index: int,
        teacher_counterfactual: bool = False,
    ):
        inputs_shared, inputs_posi, _ = target_inputs
        donor_shared, _, _ = donor_inputs
        inputs_shared = inputs_shared.copy()
        target_latents = inputs_shared["input_latents"]
        donor_latents = donor_shared["input_latents"].detach()
        if tuple(target_latents.shape) != tuple(donor_latents.shape):
            raise ValueError(
                f"Self-correction target/donor latent shape mismatch: "
                f"target={tuple(target_latents.shape)} donor={tuple(donor_latents.shape)}"
            )

        sigma_value = float(self.pipe.scheduler.sigmas[int(timestep_index)])
        if not self.self_correction_sigma_min <= sigma_value <= self.self_correction_sigma_max:
            raise ValueError(
                f"Shared self-correction sigma {sigma_value} lies outside "
                f"[{self.self_correction_sigma_min}, {self.self_correction_sigma_max}]."
            )
        timestep = self.pipe.scheduler.timesteps[timestep_index:timestep_index + 1].to(
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )
        sigma = self.pipe.scheduler.sigmas[timestep_index].to(
            dtype=target_latents.dtype,
            device=target_latents.device,
        )
        gaussian_noise = torch.randn_like(target_latents)
        source_mix = float(self.self_correction_source_mix)
        if teacher_counterfactual:
            inputs_shared["latents"] = (
                (1.0 - sigma) * donor_latents + sigma * gaussian_noise
            )
            training_target = (
                inputs_shared["latents"] - target_latents
            ) / sigma.clamp_min(1e-6)
        else:
            structured_source = (1.0 - source_mix) * gaussian_noise + source_mix * donor_latents
            inputs_shared["latents"] = (
                (1.0 - sigma) * target_latents + sigma * structured_source
            )
            training_target = structured_source - target_latents

        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        noise_pred = self.pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            timestep=timestep,
        )
        if "first_frame_latents" in inputs_shared:
            noise_pred = noise_pred[:, :, 1:]
            training_target = training_target[:, :, 1:]
        loss = F.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        self.last_self_correction_metrics = {
            "sigma": float(sigma.detach().float().cpu()),
            "timestep": float(timestep.detach().float().cpu().item()),
            "source_mix": source_mix,
            "teacher_counterfactual": bool(teacher_counterfactual),
        }
        return loss

    def forward(self, data, inputs=None):
        donor_data = data.get("_self_correction_donor_data")
        timestep_index = data.get("_flow_timestep_index")
        if donor_data is None and timestep_index is None:
            self.last_self_correction_metrics = None
            return super().forward(data, inputs=inputs)
        if donor_data is not None and not self.self_correction_enabled:
            raise RuntimeError("Self-correction donor was supplied while the feature is disabled.")
        if inputs is not None:
            raise ValueError("Precomputed inputs are unsupported for shared-timestep training.")
        target_data = data.copy()
        target_data.pop("_self_correction_donor_data", None)
        target_data.pop("_flow_timestep_index", None)
        target_inputs = self._prepare_pipeline_inputs(target_data)
        if donor_data is None:
            self.last_self_correction_metrics = None
            return self._flow_match_loss_at_timestep(target_inputs, int(timestep_index))
        donor_data = donor_data.copy()
        teacher_counterfactual = bool(
            donor_data.pop("_teacher_counterfactual_source", False)
        )
        donor_data.pop("_teacher_source_environment", None)
        donor_data.pop("_teacher_source_path", None)
        donor_inputs = self._prepare_pipeline_inputs(donor_data)
        return self._self_correction_flow_loss(
            target_inputs,
            donor_inputs,
            int(timestep_index),
            teacher_counterfactual=teacher_counterfactual,
        )

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        keep_names = set(trainable_names)
        keep_names.add("friction_context_table.friction_values")
        keep_names.add("background_context_table.friction_values")
        keep_names.update(
            name
            for name in state_dict
            if name.startswith("pipe.background_context_encoder.")
        )
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
            if name not in (
                "friction_context_table.contexts",
                "friction_context_table.global_context",
                "background_context_table.contexts",
            )
        )
        completion_marker = Path(self.output_path) / f".{file_name}.complete"
        if accelerator.is_main_process:
            try:
                completion_marker.unlink()
            except FileNotFoundError:
                pass
        accelerator.wait_for_everyone()

        if has_non_context_trainable:
            super().save_model(accelerator, model, file_name)
        else:
            if accelerator.is_main_process:
                print(
                    "[checkpoint] skipped model checkpoint during context-only phase; "
                    "saving context table only",
                    flush=True,
                )

        if accelerator.is_main_process:
            table = getattr(unwrapped, "friction_context_table", None)
            if table is None:
                raise RuntimeError("Grouped-context checkpoint is missing its context table.")
            path = os.path.join(
                self.output_path,
                file_name.replace(".safetensors", ".context_table.json"),
            )
            self.save_context_table(accelerator, model, path)
            if has_non_context_trainable:
                self._prune_context_tables_without_checkpoints()
            temporary = completion_marker.with_name(
                f".{completion_marker.name}.tmp-{os.getpid()}"
            )
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(f"{file_name}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, completion_marker)
        else:
            deadline = time.monotonic() + 7200.0
            while not completion_marker.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for rank 0 to finish checkpoint {file_name}."
                    )
                time.sleep(2.0)
        accelerator.wait_for_everyone()

    def save_context_table(self, accelerator, model, path: str) -> None:
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        table = getattr(unwrapped, "friction_context_table", None)
        if table is None:
            return
        background_table = getattr(unwrapped, "background_context_table", None)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_groups": int(table.friction_values.numel()),
            "context_shape": list(table.contexts.shape),
            "records": table.to_records(),
        }
        if getattr(table, "global_context", None) is not None:
            payload["global_context"] = table.global_context.detach().float().cpu().tolist()
        if background_table is not None:
            payload["background_context_table"] = {
                "num_groups": int(background_table.friction_values.numel()),
                "context_shape": list(background_table.contexts.shape),
                "records": [
                    {
                        "background_index": float(record["friction_mu"]),
                        "context": record["context"],
                    }
                    for record in background_table.to_records()
                ],
            }
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        print(f"[checkpoint] saved {destination}", flush=True)

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


class StagedGroupedContextModelLogger(GroupedContextModelLogger):
    def __init__(self, *args, local_checkpoint_root: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_checkpoint_root = str(local_checkpoint_root)

    def save_model(self, accelerator, model, file_name):
        shared_output = Path(self.output_path)
        local_output = Path(self.local_checkpoint_root) / file_name.replace(".safetensors", "")
        if accelerator.is_main_process:
            shutil.rmtree(local_output, ignore_errors=True)
            local_output.mkdir(parents=True, exist_ok=False)
            print(
                f"[checkpoint] staging {file_name} on node-local {local_output}",
                flush=True,
            )
        accelerator.wait_for_everyone()

        local_logger = GroupedContextModelLogger(
            str(local_output),
            remove_prefix_in_ckpt=self.remove_prefix_in_ckpt,
            save_minutes=0,
            keep_last=0,
            log_steps=0,
        )
        local_logger.save_model(accelerator, model, file_name)

        context_name = file_name.replace(".safetensors", ".context_table.json")
        local_marker_name = f".{file_name}.complete"
        shared_marker = shared_output / local_marker_name
        if accelerator.is_main_process:
            temporary = shared_output / f".{file_name}.publish-{os.getpid()}"
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True, exist_ok=False)
            shutil.copy2(local_output / file_name, temporary / file_name)
            shutil.copy2(local_output / context_name, temporary / context_name)
            shutil.copy2(local_output / local_marker_name, temporary / local_marker_name)

            for final_path in (
                shared_output / file_name,
                shared_output / context_name,
                shared_marker,
            ):
                try:
                    final_path.unlink()
                except FileNotFoundError:
                    pass
            os.replace(temporary / file_name, shared_output / file_name)
            os.replace(temporary / context_name, shared_output / context_name)
            os.replace(temporary / local_marker_name, shared_marker)
            shutil.rmtree(temporary)
            shutil.rmtree(local_output, ignore_errors=True)
            self._prune_old_checkpoints()
            self._prune_context_tables_without_checkpoints()
            print(
                f"[checkpoint] published staged checkpoint "
                f"{shared_output / file_name}",
                flush=True,
            )
        else:
            deadline = time.monotonic() + 20.0 * 60.0 * 60.0
            while not shared_marker.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for staged checkpoint {shared_marker}."
                    )
                time.sleep(5.0)


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
    group.add_argument("--grouped_background_context_lr", type=float, default=None)
    group.add_argument("--grouped_background_freeze_after_steps", type=int, default=0)
    group.add_argument("--grouped_context_new_context_lr", type=float, default=None)
    group.add_argument("--grouped_context_lr_schedule", type=str, default=None)
    group.add_argument("--grouped_context_model_lr_warmup_steps", type=int, default=0)
    group.add_argument("--grouped_context_model_phase_warmup_steps", type=int, default=0)
    group.add_argument(
        "--grouped_context_reset_model_optimizer_state_on_phase_start",
        action="store_true",
        default=False,
    )
    group.add_argument("--grouped_context_post_curriculum_lr", type=float, default=None)
    group.add_argument("--grouped_context_post_curriculum_start_step", type=int, default=0)
    group.add_argument("--grouped_context_phase_max_displacement", type=float, default=0.0)
    group.add_argument("--grouped_context_phase_max_displacement_start_step", type=int, default=0)
    group.add_argument("--grouped_context_validation_interval", type=int, default=0)
    group.add_argument("--grouped_context_validation_seed", type=int, default=20260810)
    group.add_argument("--grouped_context_protected_checkpoint_steps", type=str, default=None)
    group.add_argument("--grouped_context_alternating_interval", type=int, default=0)
    group.add_argument("--grouped_context_alternating_start", type=str, default="model")
    group.add_argument("--grouped_context_alternating_warmup_steps", type=int, default=0)
    group.add_argument("--grouped_context_weight_decay", type=float, default=0.0)
    group.add_argument("--grouped_context_structured_updates", type=int, default=0)
    group.add_argument("--grouped_context_friction_groups_per_update", type=int, default=4)
    group.add_argument("--grouped_context_actions_per_update", type=int, default=4)
    group.add_argument("--grouped_context_microbatches_per_update", type=int, default=0)
    group.add_argument("--grouped_context_sampling_mode", type=str, default="common_actions")
    group.add_argument("--grouped_context_action_weight_path", type=str, default=None)
    group.add_argument(
        "--grouped_context_action_weight_column",
        type=str,
        default="primary_relative_to_uniform",
    )
    group.add_argument("--grouped_context_action_weight_contrast", type=float, default=1.0)
    group.add_argument("--grouped_context_action_weight_min_relative", type=float, default=0.0)
    group.add_argument("--grouped_context_action_weight_max_relative", type=float, default=1e9)
    group.add_argument("--grouped_context_episode_key", type=str, default="source_episode_index")
    group.add_argument("--grouped_context_window_kind_field", type=str, default="sampling_kind")
    group.add_argument("--grouped_context_preferred_window_kind", type=str, default="lift")
    group.add_argument("--grouped_context_preferred_window_probability", type=float, default=0.8)
    group.add_argument("--grouped_context_stratify_field", type=str, default=None)
    group.add_argument("--grouped_context_sampling_strata", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_initial_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_warmup_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_add_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_total_groups", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_strata", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_initial_model_steps", type=int, default=300)
    group.add_argument("--grouped_context_curriculum_assignment_model_steps", type=int, default=0)
    group.add_argument("--grouped_context_curriculum_initial_sample_mode", type=str, default="all")
    group.add_argument(
        "--grouped_context_curriculum_initial_sample_groups",
        type=str,
        default=None,
    )
    group.add_argument(
        "--grouped_context_curriculum_initial_sample_rank_field",
        type=str,
        default="physical_friction_mu",
    )
    group.add_argument(
        "--grouped_context_curriculum_random_context_warmup",
        action="store_true",
        default=False,
    )
    group.add_argument("--grouped_context_direct_random_steps", type=int, default=0)
    group.add_argument("--grouped_context_direct_random_min", type=float, default=0.0)
    group.add_argument("--grouped_context_direct_random_max", type=float, default=1.0)
    group.add_argument("--grouped_context_direct_random_pool_size", type=int, default=0)
    group.add_argument("--grouped_context_direct_random_resume_state", type=str, default=None)
    group.add_argument("--grouped_context_direct_random_local_checkpoint_root", type=str, default=None)
    group.add_argument(
        "--grouped_context_curriculum_shared_initial_friction",
        action="store_true",
        default=False,
    )
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
    group.add_argument("--grouped_context_resume_context_table", type=str, default=None)
    group.add_argument("--grouped_context_resume_step", type=int, default=0)
    group.add_argument("--grouped_context_initialize_optimizer_state", type=str, default=None)
    group.add_argument("--grouped_context_initial_context_pool_path", type=str, default=None)
    group.add_argument("--grouped_context_stage_checkpoints_locally", action="store_true", default=False)
    group.add_argument("--grouped_context_local_checkpoint_root", type=str, default=None)
    group.add_argument("--grouped_context_model_phase_checkpoint_policy", type=str, default="all")
    group.add_argument("--grouped_context_bridge_enabled", action="store_true", default=False)
    group.add_argument("--grouped_context_bridge_global_warmup_steps", type=int, default=300)
    group.add_argument("--grouped_context_bridge_training_steps", type=int, default=4000)
    group.add_argument("--grouped_context_bridge_endpoint_probability", type=float, default=0.4)
    group.add_argument("--grouped_context_bridge_alpha_levels", type=str, default="0.2,0.4,0.6,0.8")
    group.add_argument("--grouped_context_bridge_global_condition_repeats", type=int, default=4)
    group.add_argument("--grouped_context_bridge_chunks_per_env_per_rank", type=int, default=4)
    group.add_argument("--grouped_context_bridge_expected_world_size", type=int, default=4)
    group.add_argument("--grouped_context_bridge_global_warmup_lr", type=float, default=0.03)
    group.add_argument("--grouped_context_bridge_global_lr", type=float, default=0.01)
    group.add_argument("--grouped_context_bridge_center_reg_weight", type=float, default=0.01)
    group.add_argument("--grouped_context_bridge_metrics_log_steps", type=int, default=2)
    group.add_argument("--grouped_context_self_correction_enabled", action="store_true", default=False)
    group.add_argument("--grouped_context_self_correction_probability", type=float, default=0.1)
    group.add_argument("--grouped_context_self_correction_sigma_min", type=float, default=0.45)
    group.add_argument("--grouped_context_self_correction_sigma_max", type=float, default=0.85)
    group.add_argument("--grouped_context_self_correction_source_mix", type=float, default=0.5)
    group.add_argument("--grouped_context_counterfactual_enabled", action="store_true", default=False)
    group.add_argument("--grouped_context_counterfactual_manifest_path", type=str, default=None)
    group.add_argument("--grouped_context_counterfactual_raw_root", type=str, default=None)
    group.add_argument("--grouped_context_counterfactual_batch_fraction", type=float, default=0.5)
    group.add_argument(
        "--grouped_context_counterfactual_noise_bands",
        type=str,
        default="0.90:1.00:0.20,0.70:0.90:0.60,0.55:0.70:0.20",
    )
    group.add_argument("--grouped_context_bridge_curve_power", type=float, default=1.0)
    group.add_argument(
        "--grouped_context_bridge_sampling_mode",
        choices=("legacy", "nonlinear_40_30_30"),
        default="legacy",
    )
    group.add_argument(
        "--grouped_context_bridge_freeze_global_after_warmup",
        action="store_true",
        default=False,
    )
    group.add_argument("--grouped_context_bridge_timestep_buckets_per_update", type=int, default=1)
    group.add_argument(
        "--grouped_context_bridge_smoke_sequence",
        action="store_true",
        default=False,
    )
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


_ACTION_SAMPLING_WEIGHT_CACHE: dict[
    tuple[str, str, float, float, float],
    dict[int, dict[int, float]],
] = {}


def _load_action_sampling_weights(
    *,
    path: str,
    value_column: str,
    contrast: float,
    min_relative: float,
    max_relative: float,
) -> dict[int, dict[int, float]]:
    resolved_path = str(Path(path).expanduser().resolve())
    cache_key = (
        resolved_path,
        str(value_column),
        float(contrast),
        float(min_relative),
        float(max_relative),
    )
    cached = _ACTION_SAMPLING_WEIGHT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if contrast <= 0.0:
        raise ValueError(f"Action-weight contrast must be positive, got {contrast}.")
    if not (0.0 <= min_relative <= 1.0):
        raise ValueError(
            "Action-weight minimum relative probability must be in [0, 1], "
            f"got {min_relative}."
        )
    if max_relative < 1.0:
        raise ValueError(
            "Action-weight maximum relative probability must be at least 1, "
            f"got {max_relative}."
        )

    baseline_by_environment: dict[int, dict[int, float]] = {}
    with open(resolved_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"environment_group_id", "action_id", value_column}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise KeyError(
                f"Action-weight table {resolved_path} is missing columns "
                f"{sorted(missing_columns)}."
            )
        for row in reader:
            environment_id = int(round(float(row["environment_group_id"])))
            action_id = int(row["action_id"])
            value = float(row[value_column])
            if value < 0.0:
                raise ValueError(
                    f"Negative action weight for environment={environment_id}, "
                    f"action={action_id}: {value}."
                )
            by_action = baseline_by_environment.setdefault(environment_id, {})
            if action_id in by_action:
                raise ValueError(
                    f"Duplicate action weight for environment={environment_id}, "
                    f"action={action_id}."
                )
            by_action[action_id] = value
    if not baseline_by_environment:
        raise ValueError(f"Action-weight table {resolved_path} is empty.")

    probabilities_by_environment: dict[int, dict[int, float]] = {}
    relative_values: list[float] = []
    for environment_id, baseline_by_action in baseline_by_environment.items():
        action_ids = sorted(baseline_by_action)
        num_actions = len(action_ids)
        if num_actions == 0:
            continue
        raw_probabilities = [
            (1.0 + contrast * (baseline_by_action[action_id] - 1.0)) / num_actions
            for action_id in action_ids
        ]
        lower = min_relative / num_actions
        upper = max_relative / num_actions
        low_shift = min(lower - value for value in raw_probabilities) - 1.0
        high_shift = max(upper - value for value in raw_probabilities) + 1.0
        for _ in range(80):
            shift = 0.5 * (low_shift + high_shift)
            total = sum(
                min(upper, max(lower, value + shift))
                for value in raw_probabilities
            )
            if total < 1.0:
                low_shift = shift
            else:
                high_shift = shift
        shift = 0.5 * (low_shift + high_shift)
        probabilities = [
            min(upper, max(lower, value + shift))
            for value in raw_probabilities
        ]
        probabilities_by_environment[environment_id] = dict(zip(action_ids, probabilities))
        relative_values.extend(value * num_actions for value in probabilities)

    print(
        "[action-sampling] "
        f"table={resolved_path} environments={len(probabilities_by_environment)} "
        f"column={value_column} contrast={contrast:g} "
        f"normalized_relative_range=[{min(relative_values):.6f}, "
        f"{max(relative_values):.6f}]",
        flush=True,
    )
    _ACTION_SAMPLING_WEIGHT_CACHE[cache_key] = probabilities_by_environment
    return probabilities_by_environment


def _weighted_sample_without_replacement(
    *,
    rng: random.Random,
    values: list[int],
    weights: list[float],
    count: int,
) -> list[int]:
    if count > len(values):
        raise ValueError(f"Cannot sample {count} distinct values from {len(values)} candidates.")
    pool = list(zip(values, weights))
    selected: list[int] = []
    for _ in range(count):
        total = sum(weight for _, weight in pool)
        if total <= 0.0:
            raise ValueError("Weighted action sampler has no positive probability mass.")
        threshold = rng.random() * total
        cumulative = 0.0
        selected_index = len(pool) - 1
        for index, (_, weight) in enumerate(pool):
            cumulative += weight
            if threshold <= cumulative:
                selected_index = index
                break
        value, _ = pool.pop(selected_index)
        selected.append(value)
    return selected


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


def _sample_independent_window_indices(
    *,
    grouped_indices: dict[float, dict[int, list[int]]],
    rows: list[dict],
    rng: random.Random,
    friction_groups: int,
    chunks_per_group: int,
    allowed_friction_values: list[float] | None = None,
    stratify_field: str | None = None,
) -> list[int]:
    allowed_values = None
    if allowed_friction_values is not None:
        allowed_values = [float(value) for value in allowed_friction_values]

    def is_allowed(value: float) -> bool:
        if allowed_values is None:
            return True
        return any(abs(float(value) - allowed) <= 1e-5 for allowed in allowed_values)

    candidates_by_group = {
        float(value): [
            index
            for action_candidates in by_action.values()
            for index in action_candidates
        ]
        for value, by_action in grouped_indices.items()
        if is_allowed(float(value))
    }
    candidates_by_group = {
        value: indices
        for value, indices in candidates_by_group.items()
        if len(indices) >= chunks_per_group
    }
    values = sorted(candidates_by_group)
    if len(values) < friction_groups:
        raise ValueError(
            f"Need {friction_groups} eligible groups for independent-window sampling, "
            f"got {len(values)}."
        )

    selected_values: list[float] = []
    field = str(stratify_field or "").strip()
    if field:
        values_by_stratum: dict[str, list[float]] = {}
        for value in values:
            sample_index = candidates_by_group[value][0]
            if field not in rows[sample_index]:
                raise KeyError(f"Metadata row is missing stratification field {field!r}.")
            stratum = str(rows[sample_index][field])
            values_by_stratum.setdefault(stratum, []).append(value)
        strata = sorted(values_by_stratum)
        if len(strata) >= friction_groups:
            selected_strata = rng.sample(strata, friction_groups)
        else:
            selected_strata = strata
        selected_values.extend(rng.choice(values_by_stratum[stratum]) for stratum in selected_strata)

    remaining_values = [value for value in values if value not in selected_values]
    if len(selected_values) < friction_groups:
        selected_values.extend(
            rng.sample(remaining_values, friction_groups - len(selected_values))
        )

    indices: list[int] = []
    for value in selected_values:
        group_candidates = candidates_by_group[value]
        required_count = max(
            (
                int(rows[index].get("sampling_required_count", 0))
                for index in group_candidates
            ),
            default=0,
        )
        if required_count <= 0:
            indices.extend(rng.sample(group_candidates, chunks_per_group))
            continue
        if required_count > chunks_per_group:
            raise ValueError(
                f"sampling_required_count={required_count} exceeds "
                f"chunks_per_group={chunks_per_group} for group {value!r}."
            )

        required_candidates = [
            index
            for index in group_candidates
            if bool(rows[index].get("sampling_required_pool", False))
        ]

        def sample_distinct_episodes(
            candidates: list[int],
            count: int,
            used_episodes: set,
        ) -> list[int]:
            if count <= 0:
                return []
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            selected: list[int] = []
            for index in shuffled:
                episode_key = rows[index].get("episode_index", index)
                if episode_key in used_episodes:
                    continue
                selected.append(index)
                used_episodes.add(episode_key)
                if len(selected) == count:
                    return selected
            raise ValueError(
                f"Cannot sample {count} distinct episodes for group {value!r}; "
                f"only found {len(selected)}."
            )

        used_episodes: set = set()
        selected = sample_distinct_episodes(
            required_candidates,
            required_count,
            used_episodes,
        )
        selected_set = set(selected)
        remaining_candidates = [
            index for index in group_candidates if index not in selected_set
        ]
        selected.extend(
            sample_distinct_episodes(
                remaining_candidates,
                chunks_per_group - required_count,
                used_episodes,
            )
        )
        indices.extend(selected)
    rng.shuffle(indices)
    return indices


def _sample_episode_then_window_indices(
    *,
    grouped_indices: dict[float, dict[int, list[int]]],
    rows: list[dict],
    rng: random.Random,
    friction_groups: int,
    episodes_per_group: int,
    allowed_friction_values: list[float] | None = None,
    episode_key_field: str = "source_episode_index",
    window_kind_field: str = "sampling_kind",
    preferred_window_kind: str = "lift",
    preferred_window_probability: float = 0.8,
) -> list[int]:
    if not 0.0 <= float(preferred_window_probability) <= 1.0:
        raise ValueError(
            "grouped_context_preferred_window_probability must be in [0, 1], "
            f"got {preferred_window_probability}."
        )
    allowed_values = None
    if allowed_friction_values is not None:
        allowed_values = [float(value) for value in allowed_friction_values]

    def is_allowed(value: float) -> bool:
        if allowed_values is None:
            return True
        return any(abs(float(value) - allowed) <= 1e-5 for allowed in allowed_values)

    episodes_by_group: dict[float, dict[object, list[int]]] = {}
    for value, by_action in grouped_indices.items():
        value = float(value)
        if not is_allowed(value):
            continue
        by_episode: dict[object, list[int]] = {}
        for action_candidates in by_action.values():
            for index in action_candidates:
                if episode_key_field not in rows[index]:
                    raise KeyError(
                        f"Metadata row is missing episode key {episode_key_field!r}."
                    )
                episode_key = rows[index][episode_key_field]
                by_episode.setdefault(episode_key, []).append(index)
        if len(by_episode) >= episodes_per_group:
            episodes_by_group[value] = by_episode

    values = sorted(episodes_by_group)
    if len(values) < friction_groups:
        raise ValueError(
            f"Need {friction_groups} eligible groups for episode-then-window sampling, "
            f"got {len(values)}."
        )
    selected_values = rng.sample(values, friction_groups)

    indices: list[int] = []
    for value in selected_values:
        by_episode = episodes_by_group[value]
        episode_keys = sorted(by_episode, key=lambda item: str(item))
        selected_episodes = rng.sample(episode_keys, episodes_per_group)
        for episode_key in selected_episodes:
            episode_candidates = by_episode[episode_key]
            preferred_candidates = [
                index
                for index in episode_candidates
                if str(rows[index].get(window_kind_field, "")) == preferred_window_kind
            ]
            other_candidates = [
                index
                for index in episode_candidates
                if str(rows[index].get(window_kind_field, "")) != preferred_window_kind
            ]
            choose_preferred = rng.random() < float(preferred_window_probability)
            candidate_pool = preferred_candidates if choose_preferred else other_candidates
            if not candidate_pool:
                candidate_pool = other_candidates if choose_preferred else preferred_candidates
            if not candidate_pool:
                raise ValueError(
                    f"Group {value:g} episode {episode_key!r} has no candidate windows."
                )
            indices.append(rng.choice(candidate_pool))

    rng.shuffle(indices)
    return indices


def _sample_update_indices(
    *,
    grouped_indices: dict[float, dict[int, list[int]]],
    rows: list[dict],
    accelerator,
    args,
    update_idx: int,
    friction_groups: int,
    actions_per_update: int,
    microbatches_per_update: int,
    allowed_friction_values: list[float] | None = None,
) -> list[int]:
    mode = str(getattr(args, "grouped_context_sampling_mode", "common_actions") or "common_actions").strip().lower()
    if mode in ("balanced_repeated_groups", "balanced_group_slots"):
        rng = random.Random(
            int(args.seed)
            + int(update_idx) * max(1, int(accelerator.num_processes))
            + int(accelerator.process_index)
        )
        allowed_values = None
        if allowed_friction_values is not None:
            allowed_values = [float(value) for value in allowed_friction_values]

        def is_allowed(value: float) -> bool:
            if allowed_values is None:
                return True
            return any(abs(float(value) - allowed) <= 1e-5 for allowed in allowed_values)

        values = sorted(float(value) for value in grouped_indices if is_allowed(float(value)))
        if not values:
            raise ValueError("balanced_repeated_groups found no eligible environment groups.")
        if friction_groups < len(values):
            slot_values = rng.sample(values, friction_groups)
        else:
            slot_values = list(values)
            extra_slots = friction_groups - len(slot_values)
            extra_offset = (
                int(update_idx) * max(1, int(accelerator.num_processes))
                + int(accelerator.process_index)
            ) % len(values)
            slot_values.extend(
                values[(extra_offset + index) % len(values)]
                for index in range(extra_slots)
            )

        slot_counts: dict[float, int] = {}
        for value in slot_values:
            slot_counts[value] = slot_counts.get(value, 0) + 1
        sample_indices: list[int] = []
        for value, slot_count in slot_counts.items():
            candidates = [
                index
                for action_candidates in grouped_indices[value].values()
                for index in action_candidates
            ]
            requested = int(slot_count) * int(actions_per_update)
            if len(candidates) < requested:
                raise ValueError(
                    f"Environment {value:g} has {len(candidates)} windows, "
                    f"but {requested} distinct windows are required."
                )
            sample_indices.extend(rng.sample(candidates, requested))
        expected = int(friction_groups) * int(actions_per_update)
        if microbatches_per_update > 0 and int(microbatches_per_update) != expected:
            raise ValueError(
                f"balanced_repeated_groups requires microbatches_per_update={expected}, "
                f"got {microbatches_per_update}."
            )
        rng.shuffle(sample_indices)
        return sample_indices

    if mode in ("stratified_common_actions", "stratified_aligned_actions"):
        num_strata = int(getattr(args, "grouped_context_sampling_strata", 0) or 0)
        if num_strata <= 1:
            raise ValueError("stratified_common_actions requires grouped_context_sampling_strata > 1.")
        if int(friction_groups) < num_strata or int(friction_groups) % num_strata != 0:
            raise ValueError(
                "stratified_common_actions requires friction_groups to be a positive "
                "multiple of the number of strata: "
                f"friction_groups={friction_groups}, strata={num_strata}."
            )
        all_values = sorted(float(value) for value in grouped_indices)
        if len(all_values) % num_strata != 0:
            raise ValueError(
                f"Cannot split {len(all_values)} context groups into {num_strata} equal strata."
            )
        allowed = None
        if allowed_friction_values is not None:
            allowed = {float(value) for value in allowed_friction_values}
        rng = random.Random(
            int(args.seed)
            + int(update_idx) * max(1, int(accelerator.num_processes))
            + int(accelerator.process_index)
        )
        groups_per_stratum = len(all_values) // num_strata
        selected_per_stratum = int(friction_groups) // num_strata
        selected_values: list[float] = []
        for stratum in range(num_strata):
            start = stratum * groups_per_stratum
            end = start + groups_per_stratum
            candidates = [
                value
                for value in all_values[start:end]
                if allowed is None or value in allowed
            ]
            if len(candidates) < selected_per_stratum:
                raise ValueError(
                    f"Stratum {stratum} has {len(candidates)} active context groups, "
                    f"but {selected_per_stratum} are required; "
                    f"active_groups={len(allowed or all_values)}."
                )
            selected_values.extend(rng.sample(candidates, selected_per_stratum))

        common_actions = set(grouped_indices[selected_values[0]])
        for value in selected_values[1:]:
            common_actions.intersection_update(grouped_indices[value])
        if len(common_actions) < int(actions_per_update):
            raise ValueError(
                f"Only {len(common_actions)} common actions exist across the selected groups; "
                f"requested {actions_per_update}."
            )
        selected_actions = rng.sample(sorted(common_actions), int(actions_per_update))
        sample_indices = [
            rng.choice(grouped_indices[value][action_id])
            for value in selected_values
            for action_id in selected_actions
        ]
        expected = int(friction_groups) * int(actions_per_update)
        if int(microbatches_per_update) != expected:
            raise ValueError(
                "stratified_common_actions requires an exact logical batch: "
                f"microbatches_per_update={microbatches_per_update}, expected={expected}."
            )
        rng.shuffle(sample_indices)
        return sample_indices

    if mode in ("weighted_environment_actions", "weighted_actions_per_environment"):
        weight_path = str(
            getattr(args, "grouped_context_action_weight_path", "") or ""
        ).strip()
        if not weight_path:
            raise ValueError(
                "weighted_environment_actions requires grouped_context_action_weight_path."
            )
        action_probabilities = _load_action_sampling_weights(
            path=weight_path,
            value_column=str(args.grouped_context_action_weight_column),
            contrast=float(args.grouped_context_action_weight_contrast),
            min_relative=float(args.grouped_context_action_weight_min_relative),
            max_relative=float(args.grouped_context_action_weight_max_relative),
        )
        rng = random.Random(
            int(args.seed)
            + int(update_idx) * max(1, int(accelerator.num_processes))
            + int(accelerator.process_index)
        )
        allowed_values = None
        if allowed_friction_values is not None:
            allowed_values = [float(value) for value in allowed_friction_values]

        def is_allowed(value: float) -> bool:
            if allowed_values is None:
                return True
            return any(abs(float(value) - allowed) <= 1e-5 for allowed in allowed_values)

        valid_values: list[float] = []
        for value, by_action in grouped_indices.items():
            if not is_allowed(float(value)):
                continue
            environment_id = int(round(float(value)))
            if abs(float(value) - environment_id) > 1e-5:
                raise ValueError(
                    "weighted_environment_actions expects friction_mu to contain integer "
                    f"environment IDs, got {value}."
                )
            environment_weights = action_probabilities.get(environment_id, {})
            eligible_actions = set(by_action).intersection(environment_weights)
            if len(eligible_actions) >= int(actions_per_update):
                valid_values.append(float(value))
        if len(valid_values) < int(friction_groups):
            raise ValueError(
                f"Need {friction_groups} weighted environment groups with at least "
                f"{actions_per_update} actions, got {len(valid_values)}."
            )

        selected_values = rng.sample(valid_values, int(friction_groups))
        sample_indices: list[int] = []
        for value in selected_values:
            environment_id = int(round(value))
            environment_weights = action_probabilities[environment_id]
            action_ids = sorted(set(grouped_indices[value]).intersection(environment_weights))
            selected_actions = _weighted_sample_without_replacement(
                rng=rng,
                values=action_ids,
                weights=[environment_weights[action_id] for action_id in action_ids],
                count=int(actions_per_update),
            )
            sample_indices.extend(
                rng.choice(grouped_indices[value][action_id])
                for action_id in selected_actions
            )
        expected = int(friction_groups) * int(actions_per_update)
        if int(microbatches_per_update) != expected:
            raise ValueError(
                "weighted_environment_actions requires an exact logical batch: "
                f"microbatches_per_update={microbatches_per_update}, expected={expected}."
            )
        rng.shuffle(sample_indices)
        return sample_indices

    if mode in (
        "uniform_episode_then_window",
        "episode_then_window",
        "hierarchical_episode_windows",
    ):
        rng = random.Random(
            int(args.seed)
            + int(update_idx) * max(1, int(accelerator.num_processes))
            + int(accelerator.process_index)
        )
        sample_indices = _sample_episode_then_window_indices(
            grouped_indices=grouped_indices,
            rows=rows,
            rng=rng,
            friction_groups=friction_groups,
            episodes_per_group=actions_per_update,
            allowed_friction_values=allowed_friction_values,
            episode_key_field=str(args.grouped_context_episode_key),
            window_kind_field=str(args.grouped_context_window_kind_field),
            preferred_window_kind=str(args.grouped_context_preferred_window_kind),
            preferred_window_probability=float(
                args.grouped_context_preferred_window_probability
            ),
        )
        expected = int(friction_groups) * int(actions_per_update)
        if microbatches_per_update > 0 and int(microbatches_per_update) != expected:
            raise ValueError(
                "uniform_episode_then_window requires an exact logical batch: "
                f"microbatches_per_update={microbatches_per_update}, expected={expected}."
            )
        return sample_indices

    if mode in ("independent_windows", "all_windows", "random_windows"):
        # Every GPU independently constructs a complete logical batch:
        # 4 causal-environment groups x 4 arbitrary windows = 16 samples/rank.
        rng = random.Random(
            int(args.seed)
            + int(update_idx) * max(1, int(accelerator.num_processes))
            + int(accelerator.process_index)
        )
        sample_indices = _sample_independent_window_indices(
            grouped_indices=grouped_indices,
            rows=rows,
            rng=rng,
            friction_groups=friction_groups,
            chunks_per_group=actions_per_update,
            allowed_friction_values=allowed_friction_values,
            stratify_field=getattr(args, "grouped_context_stratify_field", None),
        )
        if microbatches_per_update > 0:
            sample_indices = sample_indices[:microbatches_per_update]
        return sample_indices

    if mode not in ("common_actions", "aligned_actions", "legacy"):
        raise ValueError(f"Unsupported grouped_context_sampling_mode={mode!r}.")
    rng = random.Random(
        int(args.seed)
        + int(update_idx) * max(1, int(accelerator.num_processes))
        + int(accelerator.process_index)
    )
    sample_indices = _sample_structured_indices(
        grouped_indices=grouped_indices,
        rows=rows,
        rng=rng,
        friction_groups=friction_groups,
        actions_per_update=actions_per_update,
        allowed_friction_values=allowed_friction_values,
    )
    if len(sample_indices) > microbatches_per_update:
        sample_indices = sample_indices[:microbatches_per_update]
    return sample_indices


def _covered_button_colors(row: dict) -> set[str]:
    value = row.get("covered_button_colors", [])
    if isinstance(value, str):
        value = [value]
    return {str(color).strip().lower() for color in value}


def _sample_bridge_rank_indices(
    *,
    grouped_indices: dict[float, dict[int, list[int]]],
    rows: list[dict],
    seed: int,
    update_idx: int,
    process_index: int,
    num_processes: int,
    chunks_per_env_per_rank: int,
) -> list[int]:
    if chunks_per_env_per_rank < 2:
        raise ValueError("Bridge sampling needs at least two chunks per environment and rank.")
    rng = random.Random(int(seed) + int(update_idx) * 104729)
    selected_for_rank: list[int] = []
    global_count = int(chunks_per_env_per_rank) * int(num_processes)
    extra_per_rank = int(chunks_per_env_per_rank) - 2

    for mu in sorted(grouped_indices):
        candidates = sorted({
            index
            for action_candidates in grouped_indices[mu].values()
            for index in action_candidates
        })
        red_only = [index for index in candidates if _covered_button_colors(rows[index]) == {"red"}]
        blue_only = [index for index in candidates if _covered_button_colors(rows[index]) == {"blue"}]
        both = [
            index
            for index in candidates
            if {"red", "blue"}.issubset(_covered_button_colors(rows[index]))
        ]
        used_indices: set[int] = set()
        used_episodes: set[int] = set()

        def take(pool: list[int], count: int) -> list[int]:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            chosen: list[int] = []
            for require_new_episode in (True, False):
                for index in shuffled:
                    if index in used_indices:
                        continue
                    episode = int(rows[index].get("episode_index", index))
                    if require_new_episode and episode in used_episodes:
                        continue
                    chosen.append(index)
                    used_indices.add(index)
                    used_episodes.add(episode)
                    if len(chosen) == count:
                        return chosen
            raise ValueError(
                f"Cannot draw {count} unique bridge chunks for environment {mu}; "
                f"selected={len(chosen)} pool={len(pool)}."
            )

        red = take(red_only, int(num_processes))
        blue = take(blue_only, int(num_processes))
        extra_count = global_count - 2 * int(num_processes)
        extras = take(both + candidates, extra_count)
        rank_start = int(process_index) * extra_per_rank
        rank_indices = [
            red[int(process_index)],
            blue[int(process_index)],
            *extras[rank_start:rank_start + extra_per_rank],
        ]
        if len(rank_indices) != int(chunks_per_env_per_rank):
            raise RuntimeError(
                f"Bridge rank shard has {len(rank_indices)} chunks, "
                f"expected {chunks_per_env_per_rank}."
            )
        selected_for_rank.extend(rank_indices)

    rng.shuffle(selected_for_rank)
    return selected_for_rank


def _sample_self_correction_donor_index(
    *,
    target_index: int,
    grouped_indices: dict[float, dict[int, list[int]]],
    rows: list[dict],
    rng: random.Random,
) -> int:
    target_row = rows[target_index]
    target_mu = float(target_row["friction_mu"])
    target_action = int(target_row.get("action_id", 0))
    target_colors = _covered_button_colors(target_row)
    candidate_values = [
        value for value in sorted(grouped_indices)
        if abs(float(value) - target_mu) > 1e-5
    ]
    rng.shuffle(candidate_values)

    fallback: list[int] = []
    for value in candidate_values:
        action_candidates = list(grouped_indices[value].get(target_action, []))
        rng.shuffle(action_candidates)
        matched = [
            index
            for index in action_candidates
            if _covered_button_colors(rows[index]) == target_colors
        ]
        if matched:
            return matched[0]
        fallback.extend(action_candidates)
    if not fallback:
        raise ValueError(f"No cross-environment donor exists for metadata index {target_index}.")
    return rng.choice(fallback)


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


def _random_stratified_group_order(
    *,
    num_groups: int,
    initial_groups: int,
    add_groups: int,
    total_groups: int,
    num_strata: int,
    shared_initial_across_strata: bool = False,
) -> list[int]:
    if num_strata <= 1:
        raise ValueError(f"num_strata must exceed one, got {num_strata}.")
    if num_groups <= 0 or num_groups % num_strata != 0:
        raise ValueError(
            f"num_groups={num_groups} must be positive and divisible by num_strata={num_strata}."
        )
    total = min(int(total_groups) if total_groups > 0 else num_groups, num_groups)
    for name, value in (
        ("initial_groups", initial_groups),
        ("add_groups", add_groups),
        ("total_groups", total),
    ):
        if int(value) <= 0 or int(value) % num_strata != 0:
            raise ValueError(f"{name}={value} must be positive and divisible by {num_strata}.")

    groups_per_stratum = num_groups // num_strata
    initial_per_stratum = int(initial_groups) // num_strata
    add_per_stratum = int(add_groups) // num_strata
    total_per_stratum = total // num_strata
    if initial_per_stratum > groups_per_stratum or total_per_stratum > groups_per_stratum:
        raise ValueError("Requested more groups per stratum than the dataset contains.")

    system_rng = random.SystemRandom()
    initial_by_stratum: list[list[int]] = []
    remaining_by_stratum: list[list[int]] = []
    shared_anchor_offsets: list[int] | None = None
    if shared_initial_across_strata:
        shared_anchor_offsets = []
        for bin_index in range(initial_per_stratum):
            bin_start = bin_index * groups_per_stratum // initial_per_stratum
            bin_end = (bin_index + 1) * groups_per_stratum // initial_per_stratum
            shared_anchor_offsets.append(system_rng.choice(range(bin_start, bin_end)))
    for stratum in range(num_strata):
        block_start = stratum * groups_per_stratum
        block = list(range(block_start, block_start + groups_per_stratum))
        if shared_anchor_offsets is not None:
            anchors = [block_start + offset for offset in shared_anchor_offsets]
        else:
            anchors = []
            for bin_index in range(initial_per_stratum):
                bin_start = bin_index * groups_per_stratum // initial_per_stratum
                bin_end = (bin_index + 1) * groups_per_stratum // initial_per_stratum
                anchors.append(system_rng.choice(block[bin_start:bin_end]))
        remaining = [index for index in block if index not in set(anchors)]
        system_rng.shuffle(remaining)
        initial_by_stratum.append(anchors)
        remaining_by_stratum.append(remaining)

    order = [
        index
        for stratum_indices in initial_by_stratum
        for index in stratum_indices
    ]
    selected_per_stratum = [initial_per_stratum] * num_strata
    offsets = [0] * num_strata
    while len(order) < total:
        for stratum in range(num_strata):
            remaining_needed = total_per_stratum - selected_per_stratum[stratum]
            take = min(add_per_stratum, remaining_needed)
            start = offsets[stratum]
            end = start + take
            order.extend(remaining_by_stratum[stratum][start:end])
            offsets[stratum] = end
            selected_per_stratum[stratum] += take
    if len(order) != total or len(set(order)) != total:
        raise RuntimeError(
            f"Invalid stratified curriculum order: length={len(order)} unique={len(set(order))}."
        )
    return order


def _load_or_create_curriculum_group_order(
    *,
    accelerator,
    output_path: str,
    num_groups: int,
    initial_groups: int,
    add_groups: int,
    total_groups: int,
    num_strata: int,
    shared_initial_across_strata: bool = False,
) -> list[int]:
    if num_strata <= 1:
        return _nested_uniform_group_order(num_groups, initial_groups, total_groups)

    destination = Path(output_path) / "curriculum_group_order.json"
    if accelerator.is_main_process:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            order = _random_stratified_group_order(
                num_groups=num_groups,
                initial_groups=initial_groups,
                add_groups=add_groups,
                total_groups=total_groups,
                num_strata=num_strata,
                shared_initial_across_strata=shared_initial_across_strata,
            )
            payload = {
                "sampling": "os_random_without_replacement",
                "num_groups": int(num_groups),
                "initial_groups": int(initial_groups),
                "add_groups": int(add_groups),
                "total_groups": int(total_groups),
                "num_strata": int(num_strata),
                "shared_initial_across_strata": bool(shared_initial_across_strata),
                "group_order": order,
            }
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            print(f"[curriculum] saved immutable random group order to {destination}", flush=True)
    accelerator.wait_for_everyone()
    with destination.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "num_groups": int(num_groups),
        "initial_groups": int(initial_groups),
        "add_groups": int(add_groups),
        "total_groups": int(total_groups),
        "num_strata": int(num_strata),
    }
    for key, value in expected.items():
        if int(payload.get(key, -1)) != value:
            raise ValueError(
                f"Persisted curriculum order has {key}={payload.get(key)!r}, expected {value}."
            )
    persisted_shared_initial = bool(payload.get("shared_initial_across_strata", False))
    if persisted_shared_initial != bool(shared_initial_across_strata):
        raise ValueError(
            "Persisted curriculum order has shared_initial_across_strata="
            f"{persisted_shared_initial}, expected {bool(shared_initial_across_strata)}."
        )
    order = [int(index) for index in payload["group_order"]]
    expected_total = min(int(total_groups), int(num_groups))
    if (
        len(order) != expected_total
        or len(set(order)) != expected_total
        or any(index < 0 or index >= num_groups for index in order)
    ):
        raise ValueError(f"Persisted curriculum order in {destination} is invalid.")
    return order


@torch.no_grad()
def _share_initial_contexts_across_strata(
    table,
    group_order: list[int],
    initial_groups: int,
    num_strata: int,
) -> dict[int, int]:
    if num_strata <= 1 or initial_groups <= 0 or initial_groups % num_strata != 0:
        raise ValueError(
            f"Cannot share initial contexts with initial_groups={initial_groups}, strata={num_strata}."
        )
    groups_per_stratum = initial_groups // num_strata
    aliases: dict[int, int] = {}
    for local_index in range(groups_per_stratum):
        canonical_index = int(group_order[local_index])
        for stratum in range(1, num_strata):
            alias_index = int(group_order[stratum * groups_per_stratum + local_index])
            table.contexts[alias_index].copy_(table.contexts[canonical_index])
            aliases[alias_index] = canonical_index
    return aliases


def _curriculum_phase_for_step(args, group_order: list[int], step: int) -> dict:
    initial_groups = int(args.grouped_context_curriculum_initial_groups)
    warmup_groups = int(
        getattr(args, "grouped_context_curriculum_warmup_groups", 0) or initial_groups
    )
    add_groups = int(args.grouped_context_curriculum_add_groups)
    total_groups = int(args.grouped_context_curriculum_total_groups) or len(group_order)
    initial_steps = int(args.grouped_context_curriculum_initial_model_steps)
    assignment_model_steps = int(
        getattr(args, "grouped_context_curriculum_assignment_model_steps", 0) or 0
    )
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
    warmup_groups = min(max(warmup_groups, initial_groups), total_groups)
    current_step = int(step)
    if current_step <= initial_steps:
        active = group_order[:warmup_groups]
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

    if current_step <= initial_steps + assignment_model_steps:
        active = group_order[:initial_groups]
        return {
            "round": 0,
            "phase": "model",
            "sample_group_indices": active,
            "train_context_indices": [],
            "phase_start": initial_steps + 1,
            "phase_end": initial_steps + assignment_model_steps,
            "active_count": len(active),
            "new_count": 0,
        }

    offset = initial_steps + assignment_model_steps
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
    background_context_params = []
    bridge_global_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "friction_context_table.contexts":
            context_params.append(param)
        elif name == "background_context_table.contexts":
            background_context_params.append(param)
        elif name == "friction_context_table.global_context":
            bridge_global_params.append(param)
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
    if background_context_params:
        background_context_lr = getattr(args, "grouped_background_context_lr", None)
        if background_context_lr is None:
            background_context_lr = context_lr
        param_groups.append(
            {
                "name": "background_context",
                "params": background_context_params,
                "lr": float(background_context_lr),
                "weight_decay": float(getattr(args, "grouped_context_weight_decay", 0.0)),
            }
        )
    if bridge_global_params:
        param_groups.append(
            {
                "name": "bridge_global",
                "params": bridge_global_params,
                "lr": float(getattr(args, "grouped_context_bridge_global_warmup_lr", context_lr)),
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(param_groups)


def _build_curriculum_optimizer(model, args, accelerator):
    state_path = getattr(args, "grouped_context_initialize_optimizer_state", None)
    if not state_path:
        return _build_optimizer(model, args)

    context_parameters = [model.friction_context_table.contexts]
    if model.background_context_table is not None:
        context_parameters.append(model.background_context_table.contexts)
    original_requires_grad = [parameter.requires_grad for parameter in context_parameters]
    for parameter in context_parameters:
        parameter.requires_grad_(False)
    optimizer = _build_optimizer(model, args)

    source = Path(state_path)
    if source.is_dir():
        source = source / "optimizer_scheduler.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    source_optimizer = payload["optimizer"]
    source_groups = [group.get("name") for group in source_optimizer["param_groups"]]
    if source_groups != ["model"]:
        raise ValueError(
            f"Expected one model optimizer group in {source}, got {source_groups}."
        )
    optimizer.load_state_dict(source_optimizer)

    for parameter, requires_grad in zip(context_parameters, original_requires_grad):
        parameter.requires_grad_(requires_grad)
    optimizer.add_param_group(
        {
            "name": "context",
            "params": [model.friction_context_table.contexts],
            "lr": float(args.grouped_context_lr),
            "weight_decay": float(getattr(args, "grouped_context_weight_decay", 0.0)),
        }
    )
    if model.background_context_table is not None:
        optimizer.add_param_group(
            {
                "name": "background_context",
                "params": [model.background_context_table.contexts],
                "lr": float(
                    getattr(args, "grouped_background_context_lr", None)
                    or args.grouped_context_lr
                ),
                "weight_decay": float(
                    getattr(args, "grouped_context_weight_decay", 0.0)
                ),
            }
        )
    if accelerator.is_main_process:
        print(
            f"[optimizer_init] restored model AdamW state from {source}; "
            "new environment-C parameter group starts with fresh moments",
            flush=True,
        )
    return optimizer


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


def _set_curriculum_requires_grad(model, phase: str, freeze_background: bool = False) -> None:
    context_phase = phase in ("new_context", "new_context_mid", "all_context", "context")
    for name, param in model.named_parameters():
        is_context = name.endswith("friction_context_table.contexts") or name == "friction_context_table.contexts"
        is_background_context = (
            name.endswith("background_context_table.contexts")
            or name == "background_context_table.contexts"
        )
        is_background_encoder = "background_context_encoder" in name
        if freeze_background and (is_background_context or is_background_encoder):
            param.requires_grad_(False)
        elif is_context:
            param.requires_grad_(context_phase)
        elif is_background_context:
            param.requires_grad_(phase in ("all_context", "context"))
        else:
            param.requires_grad_(not context_phase)


def _apply_curriculum_lrs(
    optimizer,
    args,
    step: int,
    phase: str,
    phase_start: int | None = None,
    freeze_background: bool = False,
):
    model_lr = float(args.learning_rate)
    warmup_steps = int(getattr(args, "grouped_context_model_lr_warmup_steps", 0) or 0)
    if warmup_steps > 0:
        model_lr = model_lr * min(float(step) / float(warmup_steps), 1.0)
    context_lr = getattr(args, "grouped_context_lr", None)
    if context_lr is None:
        context_lr = args.learning_rate
    context_lr = float(context_lr)
    background_context_lr = getattr(args, "grouped_background_context_lr", None)
    if background_context_lr is None:
        background_context_lr = context_lr
    background_context_lr = float(background_context_lr)
    if phase == "new_context" and getattr(args, "grouped_context_new_context_lr", None) is not None:
        context_lr = float(args.grouped_context_new_context_lr)
    if phase == "new_context_mid" and getattr(args, "grouped_context_mid_context_lr", None) is not None:
        context_lr = float(args.grouped_context_mid_context_lr)
    post_context_lr = getattr(args, "grouped_context_post_curriculum_lr", None)
    post_context_start = int(
        getattr(args, "grouped_context_post_curriculum_start_step", 0) or 0
    )
    if (
        phase in ("all_context", "context")
        and post_context_lr is not None
        and post_context_start > 0
        and int(step) >= post_context_start
    ):
        context_lr = float(post_context_lr)
    if phase in ("new_context", "new_context_mid", "all_context", "context"):
        model_lr = 0.0
    else:
        context_lr = 0.0
        phase_warmup_steps = int(
            getattr(args, "grouped_context_model_phase_warmup_steps", 0) or 0
        )
        if phase_warmup_steps > 0 and phase_start is not None and int(phase_start) > 1:
            local_step = max(1, int(step) - int(phase_start) + 1)
            model_lr *= min(float(local_step) / float(phase_warmup_steps), 1.0)
    if phase not in ("all_context", "context"):
        background_context_lr = 0.0
    if freeze_background:
        background_context_lr = 0.0
    for group in optimizer.param_groups:
        if group.get("name") == "context":
            group["lr"] = context_lr
        elif group.get("name") == "background_context":
            group["lr"] = background_context_lr
        else:
            group["lr"] = model_lr
    return model_lr, context_lr


def _clear_optimizer_group_state(optimizer, group_name: str) -> int:
    base_optimizer = optimizer
    visited = set()
    while hasattr(base_optimizer, "optimizer") and id(base_optimizer) not in visited:
        visited.add(id(base_optimizer))
        base_optimizer = base_optimizer.optimizer
    cleared = 0
    for group in base_optimizer.param_groups:
        if group.get("name") != group_name:
            continue
        for parameter in group["params"]:
            if parameter in base_optimizer.state:
                del base_optimizer.state[parameter]
                cleared += 1
    return cleared


def _mask_context_grad(model, train_context_indices: list[int]) -> None:
    table = getattr(model, "friction_context_table", None)
    if table is None or table.contexts.grad is None:
        return
    mask = torch.zeros(table.contexts.shape[0], device=table.contexts.grad.device, dtype=table.contexts.grad.dtype)
    for index in train_context_indices:
        if 0 <= int(index) < mask.numel():
            mask[int(index)] = 1
    table.contexts.grad.mul_(mask[:, None, None])


def _context_row_mask(table, train_context_indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(table.contexts.shape[0], device=table.contexts.device, dtype=torch.bool)
    for index in train_context_indices:
        if 0 <= int(index) < mask.numel():
            mask[int(index)] = True
    return mask


@torch.no_grad()
def _zero_masked_context_optimizer_state(optimizer, table, train_mask: torch.Tensor) -> None:
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    state = getattr(raw_optimizer, "state", {}).get(table.contexts, {})
    broadcast_mask = train_mask[:, None, None]
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if torch.is_tensor(value) and value.shape == table.contexts.shape:
            value.mul_(broadcast_mask.to(device=value.device, dtype=value.dtype))


@torch.no_grad()
def _project_context_displacement(table, anchor: torch.Tensor, max_displacement: float) -> torch.Tensor:
    delta = table.contexts - anchor
    flat = delta.flatten(start_dim=1)
    norms = torch.linalg.vector_norm(flat.float(), dim=1)
    scale = torch.clamp(float(max_displacement) / norms.clamp_min(1e-12), max=1.0)
    view_shape = (scale.shape[0],) + (1,) * (delta.ndim - 1)
    table.contexts.copy_(anchor + delta * scale.to(delta.dtype).view(view_shape))
    return torch.linalg.vector_norm((table.contexts - anchor).flatten(start_dim=1).float(), dim=1)


def _build_fixed_validation_indices(
    grouped_indices,
    accelerator,
    actions_per_group: int,
    seed: int,
) -> tuple[list[int], list]:
    all_values = sorted(grouped_indices)
    common_actions = set(grouped_indices[all_values[0]])
    for value in all_values[1:]:
        common_actions.intersection_update(grouped_indices[value])
    if len(common_actions) < int(actions_per_group):
        raise ValueError(
            f"Fixed validation needs {actions_per_group} common actions, got {len(common_actions)}."
        )
    rng = random.Random(int(seed))
    selected_actions = rng.sample(sorted(common_actions), int(actions_per_group))
    local_values = all_values[int(accelerator.process_index) :: int(accelerator.num_processes)]
    indices = [
        rng.choice(grouped_indices[value][action_id])
        for value in local_values
        for action_id in selected_actions
    ]
    return indices, selected_actions


@torch.no_grad()
def _run_fixed_validation(
    accelerator,
    model,
    dataset,
    sample_indices: list[int],
    seed: int,
) -> tuple[float, int]:
    python_state = random.getstate()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    losses = []
    try:
        for position, sample_index in enumerate(sample_indices):
            sample_seed = int(seed) + int(sample_index) * 1009 + int(position)
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            losses.append(model(dataset[sample_index]).detach().float())
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train(was_training)
    local_sum = torch.stack(losses).sum() if losses else torch.zeros((), device=accelerator.device)
    local_count = torch.tensor(float(len(losses)), device=accelerator.device)
    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum")
    return float((global_sum / global_count.clamp_min(1.0)).cpu()), int(global_count.item())


def _parse_step_set(raw_steps: str | None) -> set[int]:
    if not raw_steps:
        return set()
    return {int(part.strip()) for part in str(raw_steps).split(",") if part.strip()}


def _protect_paired_checkpoint(accelerator, model_logger, step: int) -> None:
    import os
    import shutil

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_root = Path(model_logger.output_path)
        protected_root = output_root / "protected_checkpoints" / f"step-{int(step)}"
        protected_root.mkdir(parents=True, exist_ok=True)
        sources = [
            output_root / f"step-{int(step)}.safetensors",
            output_root / f"step-{int(step)}.context_table.json",
        ]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"Cannot protect missing checkpoint artifact: {source}")
            destination = protected_root / source.name
            temporary = protected_root / f".{source.name}.tmp-{os.getpid()}"
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        marker = protected_root / ".complete"
        marker.write_text(f"protected_step={int(step)}\n", encoding="utf-8")
        print(f"[checkpoint] permanently protected step={step} at {protected_root}", flush=True)
    accelerator.wait_for_everyone()


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
    global_context = getattr(table, "global_context", None)
    if global_context is not None:
        flat = global_context.detach().float().cpu().flatten()
        head = ",".join(f"{float(x):.4f}" for x in flat[: min(6, flat.numel())])
        print(
            "[context_table] global "
            f"mean={float(flat.mean()):.6f} std={float(flat.std(unbiased=False)):.6f} "
            f"norm={float(torch.linalg.vector_norm(flat)):.6f} head=[{head}]",
            flush=True,
        )
    background_table = getattr(accelerator.unwrap_model(model), "background_context_table", None)
    if background_table is not None:
        background_values = background_table.friction_values.detach().float().cpu().tolist()
        background_contexts = background_table.contexts.detach().float().cpu()
        for background_index, context in zip(background_values, background_contexts):
            flat = context.flatten()
            head = ",".join(f"{float(x):.4f}" for x in flat[: min(6, flat.numel())])
            print(
                f"[background_context_table] index={int(background_index)} "
                f"mean={float(flat.mean()):.6f} std={float(flat.std(unbiased=False)):.6f} "
                f"norm={float(torch.linalg.vector_norm(flat)):.6f} head=[{head}]",
                flush=True,
            )


def _save_phase_context_table(accelerator, model, model_logger, step: int, phase: str | None, reason: str) -> None:
    if not accelerator.is_main_process:
        return
    safe_phase = str(phase or "none").replace("/", "_")
    safe_reason = str(reason or "phase").replace("/", "_")
    timestamp = datetime.now().strftime("%m%d-%H%M")
    path = os.path.join(
        model_logger.output_path,
        f"phase-{timestamp}-step-{int(step):06d}-{safe_phase}-{safe_reason}.context_table.json",
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


def _save_direct_random_training_state(
    accelerator,
    optimizer,
    scheduler,
    sample_generator,
    context_generator,
    context_pool,
    output_path: str,
    step: int,
) -> None:
    destination = Path(output_path) / f"step-{int(step)}.training_state"
    temporary = Path(output_path) / f".step-{int(step)}.training_state.tmp"
    if accelerator.is_main_process:
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite permanent training state {destination}."
            )
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()

    rank_state = {
        "step": int(step),
        "rank": int(accelerator.process_index),
        "world_size": int(accelerator.num_processes),
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
        "sample_generator_state": sample_generator.get_state(),
        "context_generator_state": context_generator.get_state(),
    }
    torch.save(
        rank_state,
        temporary / f"rank-{int(accelerator.process_index):04d}.rng.pt",
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        torch.save(
            {
                "step": int(step),
                "world_size": int(accelerator.num_processes),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            temporary / "optimizer_scheduler.pt",
        )
        if context_pool is not None:
            torch.save(
                context_pool.detach().float().cpu(),
                temporary / "context_pool.pt",
            )
        manifest = {
            "format": "bwm_direct_random_context_training_state_v1",
            "step": int(step),
            "world_size": int(accelerator.num_processes),
            "model_checkpoint": f"step-{int(step)}.safetensors",
            "optimizer_state": "optimizer_scheduler.pt",
            "rank_rng_pattern": "rank-{rank:04d}.rng.pt",
            "context_pool": "context_pool.pt" if context_pool is not None else None,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        complete_path = temporary / ".complete"
        with complete_path.open("w", encoding="utf-8") as handle:
            handle.write(f"step-{int(step)}\n")
            handle.flush()
            os.fsync(handle.fileno())
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.replace(temporary, destination)
        print(
            f"[training_state] permanently saved {destination} "
            "(optimizer, scheduler, and per-rank RNG)",
            flush=True,
        )
    accelerator.wait_for_everyone()


def _load_direct_random_training_state(
    accelerator,
    optimizer,
    scheduler,
    sample_generator,
    context_generator,
    state_path: str,
) -> tuple[int, torch.Tensor | None]:
    source = Path(state_path)
    if not (source / ".complete").is_file():
        raise FileNotFoundError(f"Incomplete direct-random training state: {source}")
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest["world_size"]) != int(accelerator.num_processes):
        raise ValueError(
            f"Resume world size {manifest['world_size']} does not match "
            f"current world size {accelerator.num_processes}."
        )
    shared_state = torch.load(
        source / "optimizer_scheduler.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(shared_state["optimizer"])
    scheduler.load_state_dict(shared_state["scheduler"])
    rank_state = torch.load(
        source / f"rank-{int(accelerator.process_index):04d}.rng.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(rank_state["python_rng_state"])
    torch.set_rng_state(rank_state["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(rank_state["torch_cuda_rng_state"])
    sample_generator.set_state(rank_state["sample_generator_state"])
    context_generator.set_state(rank_state["context_generator_state"])
    context_pool = None
    if manifest.get("context_pool"):
        context_pool = torch.load(
            source / str(manifest["context_pool"]),
            map_location="cpu",
            weights_only=False,
        ).float()
    step = int(manifest["step"])
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"[resume] restored model-paired optimizer/scheduler/RNG state "
            f"from {source} at step={step}",
            flush=True,
        )
    return step, context_pool


def _publish_direct_random_checkpoint(
    accelerator,
    local_output: Path,
    shared_output: Path,
    step: int,
) -> None:
    model_name = f"step-{int(step)}.safetensors"
    context_name = f"step-{int(step)}.context_table.json"
    state_name = f"step-{int(step)}.training_state"
    model_marker_name = f".{model_name}.complete"
    pair_marker = shared_output / f".step-{int(step)}.paired.complete"

    if accelerator.is_main_process:
        if pair_marker.exists():
            raise FileExistsError(
                f"Refusing to overwrite permanent paired checkpoint {pair_marker}."
            )
        temporary = shared_output / f".step-{int(step)}.pair.tmp-{os.getpid()}"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=False)

        shutil.copy2(local_output / model_name, temporary / model_name)
        shutil.copy2(local_output / context_name, temporary / context_name)
        shutil.copy2(local_output / model_marker_name, temporary / model_marker_name)
        shutil.copytree(local_output / state_name, temporary / state_name)

        final_model = shared_output / model_name
        final_context = shared_output / context_name
        final_model_marker = shared_output / model_marker_name
        final_state = shared_output / state_name
        for path in (final_model, final_context, final_model_marker):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if final_state.exists():
            shutil.rmtree(final_state)
        os.replace(temporary / model_name, final_model)
        os.replace(temporary / context_name, final_context)
        os.replace(temporary / model_marker_name, final_model_marker)
        os.replace(temporary / state_name, final_state)
        shutil.rmtree(temporary)

        pair_temporary = pair_marker.with_name(
            f".{pair_marker.name}.tmp-{os.getpid()}"
        )
        with pair_temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"step-{int(step)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pair_temporary, pair_marker)
        print(
            f"[checkpoint_bundle] permanently published paired step={step} "
            f"to {shared_output}",
            flush=True,
        )
    else:
        deadline = time.monotonic() + 20.0 * 60.0 * 60.0
        while not pair_marker.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for paired checkpoint {pair_marker}."
                )
            time.sleep(5.0)


def _save_direct_random_permanent_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    sample_generator,
    context_generator,
    context_pool,
    model_logger,
    args,
    step: int,
) -> None:
    configured_root = getattr(
        args,
        "grouped_context_direct_random_local_checkpoint_root",
        None,
    )
    local_root = Path(
        configured_root
        or os.environ.get("BWM_LOCAL_CHECKPOINT_ROOT")
        or (
            f"/tmp/{os.environ.get('USER', 'user')}/bwm_direct_random_checkpoints/"
            f"{os.environ.get('SLURM_JOB_ID', 'manual')}"
        )
    )
    local_output = local_root / f"step-{int(step)}"
    if accelerator.is_main_process:
        shutil.rmtree(local_output, ignore_errors=True)
        local_output.mkdir(parents=True, exist_ok=False)
        print(
            f"[checkpoint_bundle] staging step={step} on node-local {local_output}",
            flush=True,
        )
    accelerator.wait_for_everyone()

    local_logger = GroupedContextModelLogger(
        str(local_output),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_minutes=0,
        keep_last=0,
        log_steps=0,
    )
    local_logger.save_model(accelerator, model, f"step-{int(step)}.safetensors")
    _save_direct_random_training_state(
        accelerator,
        optimizer,
        scheduler,
        sample_generator,
        context_generator,
        context_pool,
        str(local_output),
        step,
    )
    _publish_direct_random_checkpoint(
        accelerator,
        local_output,
        Path(model_logger.output_path),
        step,
    )
    if accelerator.is_main_process:
        shutil.rmtree(local_output, ignore_errors=True)


def launch_direct_random_grouped_stage1(accelerator, dataset, model, model_logger, args):
    total_steps = int(args.grouped_context_direct_random_steps)
    local_batch_size = int(args.batch_size)
    if total_steps <= 0:
        raise ValueError("grouped_context_direct_random_steps must be positive.")
    if local_batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if int(args.gradient_accumulation_steps) != 1:
        raise ValueError("Direct-random batching requires gradient_accumulation_steps=1.")
    if int(getattr(model_logger, "keep_last", 0)) != 0:
        raise ValueError(
            "Direct-random permanent checkpoints require checkpoint_keep_last=0."
        )
    save_steps = int(args.save_steps or 0)
    if save_steps <= 0:
        raise ValueError("Direct-random training requires a positive save_steps interval.")

    model.friction_context_table.contexts.requires_grad_(False)
    if model.friction_context_table.global_context is not None:
        model.friction_context_table.global_context.requires_grad_(False)
    optimizer = _build_optimizer(model, args)
    warmup_steps = int(getattr(args, "grouped_context_model_lr_warmup_steps", 0) or 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=(
            (lambda current: min(1.0, float(current + 1) / float(warmup_steps)))
            if warmup_steps > 0
            else (lambda current: 1.0)
        ),
    )
    model.to(device=accelerator.device)
    model, optimizer = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    context_min = float(args.grouped_context_direct_random_min)
    context_max = float(args.grouped_context_direct_random_max)
    if context_max <= context_min:
        raise ValueError(
            f"Invalid direct-random C range [{context_min}, {context_max}]."
        )
    context_shape = (
        int(args.physical_context_tokens),
        int(args.physical_context_dim),
    )
    pool_size = int(getattr(args, "grouped_context_direct_random_pool_size", 0) or 0)
    if pool_size < 0:
        raise ValueError("grouped_context_direct_random_pool_size cannot be negative.")
    context_pool = None
    if pool_size > 0:
        pool_generator = torch.Generator(device="cpu")
        pool_generator.manual_seed(int(args.seed) + 43)
        context_pool = torch.rand(
            (pool_size, *context_shape),
            generator=pool_generator,
            dtype=torch.float32,
        )
        context_pool.mul_(context_max - context_min).add_(context_min)

    sample_generator = torch.Generator(device="cpu")
    context_generator = torch.Generator(device="cpu")
    rank_seed = int(args.seed) + int(accelerator.process_index) * 1_000_003
    sample_generator.manual_seed(rank_seed + 17)
    context_generator.manual_seed(rank_seed + 29)

    resume_step = 0
    resume_state = getattr(args, "grouped_context_direct_random_resume_state", None)
    if resume_state:
        resume_step, restored_pool = _load_direct_random_training_state(
            accelerator,
            optimizer,
            scheduler,
            sample_generator,
            context_generator,
            resume_state,
        )
        if restored_pool is not None:
            context_pool = restored_pool
        if pool_size > 0 and context_pool is None:
            raise ValueError(
                f"Resume state {resume_state} does not contain the requested context pool."
            )
    if context_pool is not None and int(context_pool.shape[0]) != pool_size:
        raise ValueError(
            f"Configured pool_size={pool_size}, restored pool has "
            f"shape={tuple(context_pool.shape)}."
        )
    model_logger.num_steps = resume_step
    if context_pool is not None and accelerator.is_main_process:
        pool_path = Path(model_logger.output_path) / "direct_context_pool.json"
        if not pool_path.exists():
            temporary = pool_path.with_name(f".{pool_path.name}.tmp-{os.getpid()}")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pool_size": pool_size,
                        "context_shape": list(context_shape),
                        "range": [context_min, context_max],
                        "contexts": context_pool.tolist(),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, pool_path)
    if accelerator.is_main_process:
        print(
            "[direct_random_context_stage1] "
            f"steps={total_steps} resume_step={resume_step} "
            f"dataset_size={len(dataset)} sampling=uniform_with_replacement "
            f"local_batch_size={local_batch_size} "
            f"global_batch_size={local_batch_size * int(accelerator.num_processes)} "
            f"context_shape={context_shape} context_range=[{context_min:g},{context_max:g}] "
            f"context_sampling={'fixed_pool_' + str(pool_size) if pool_size else 'continuous'} "
            f"save_steps={save_steps} warmup_steps={warmup_steps}",
            flush=True,
        )

    iterator = tqdm(
        range(resume_step, total_steps),
        disable=not accelerator.is_local_main_process,
    )
    for _ in iterator:
        sample_indices = torch.randint(
            low=0,
            high=len(dataset),
            size=(local_batch_size,),
            generator=sample_generator,
        ).tolist()
        optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for micro_idx, sample_index in enumerate(sample_indices):
            sync_context = (
                accelerator.no_sync(model)
                if micro_idx + 1 < local_batch_size
                else contextlib.nullcontext()
            )
            with sync_context:
                sample_data = dataset[int(sample_index)].copy()
                if context_pool is None:
                    direct_context = torch.rand(
                        context_shape,
                        generator=context_generator,
                        dtype=torch.float32,
                    )
                    direct_context.mul_(context_max - context_min).add_(context_min)
                else:
                    pool_index = int(
                        torch.randint(
                            0,
                            pool_size,
                            (1,),
                            generator=context_generator,
                        ).item()
                    )
                    direct_context = context_pool[pool_index].clone()
                sample_data["_direct_physical_context"] = direct_context
                loss = model(sample_data)
                detached_losses.append(loss.detach().float())
                accelerator.backward(loss / float(local_batch_size))
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        mean_loss = torch.stack(detached_losses).mean()
        model_logger.on_step_end(accelerator, model, None, loss=mean_loss)
        step = int(model_logger.num_steps)
        if step % save_steps == 0:
            _save_direct_random_permanent_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                sample_generator,
                context_generator,
                context_pool,
                model_logger,
                args,
                step,
            )

    if int(model_logger.num_steps) % save_steps != 0:
        step = int(model_logger.num_steps)
        _save_direct_random_permanent_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            sample_generator,
            context_generator,
            context_pool,
            model_logger,
            args,
            step,
        )


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

    resume_step = int(getattr(args, "grouped_context_resume_step", 0) or 0)
    iterator = tqdm(range(resume_step, updates), disable=not accelerator.is_local_main_process)
    for update_idx in iterator:
        step = update_idx + 1
        _apply_scheduled_lrs(optimizer, args, step)
        _set_alternating_requires_grad(model, args, step)
        sample_indices = _sample_update_indices(
            grouped_indices=grouped_indices,
            rows=metadata_rows,
            accelerator=accelerator,
            args=args,
            update_idx=update_idx,
            friction_groups=friction_groups,
            actions_per_update=actions_per_update,
            microbatches_per_update=microbatches_per_update,
        )
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
        background_context_table = getattr(
            accelerator.unwrap_model(model), "background_context_table", None
        )
        if background_context_table is not None:
            background_context_table.clamp_(
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


def _parse_bridge_alpha_levels(raw: str) -> list[float]:
    levels = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not levels:
        raise ValueError("grouped_context_bridge_alpha_levels cannot be empty.")
    if any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError(f"Bridge alpha levels must lie strictly inside (0,1), got {levels}.")
    return levels


def _set_bridge_requires_grad(
    model,
    phase: str,
    freeze_global_after_warmup: bool = False,
) -> None:
    for name, param in model.named_parameters():
        if name == "friction_context_table.contexts":
            param.requires_grad_(False)
        elif name == "friction_context_table.global_context":
            param.requires_grad_(
                phase == "global_warmup" or not freeze_global_after_warmup
            )
        else:
            param.requires_grad_(phase != "global_warmup")


def _apply_bridge_lrs(
    optimizer,
    args,
    phase: str,
    freeze_global_after_warmup: bool = False,
) -> tuple[float, float]:
    model_lr = 0.0 if phase == "global_warmup" else float(args.learning_rate)
    if phase == "global_warmup":
        global_lr = float(args.grouped_context_bridge_global_warmup_lr)
    elif freeze_global_after_warmup:
        global_lr = 0.0
    else:
        global_lr = float(args.grouped_context_bridge_global_lr)
    for group in optimizer.param_groups:
        if group.get("name") == "bridge_global":
            group["lr"] = global_lr
        elif group.get("name") == "context":
            group["lr"] = 0.0
        else:
            group["lr"] = model_lr
    return model_lr, global_lr


def _append_bridge_metrics(output_path: str, row: dict) -> None:
    destination = Path(output_path) / "bridge_metrics.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _sample_bridge_shared_timestep_index(
    *,
    scheduler,
    args,
    update_idx: int,
    self_correction: bool,
) -> int:
    total = len(scheduler.timesteps)
    lower = max(0, int(float(args.min_timestep_boundary) * total))
    upper = min(total, int(float(args.max_timestep_boundary) * total))
    candidates = list(range(lower, upper))
    if self_correction:
        sigma_min = float(args.grouped_context_self_correction_sigma_min)
        sigma_max = float(args.grouped_context_self_correction_sigma_max)
        candidates = [
            index
            for index in candidates
            if sigma_min <= float(scheduler.sigmas[index]) <= sigma_max
        ]
    if not candidates:
        raise ValueError(
            f"No shared timestep candidate: boundary=[{lower},{upper}) "
            f"self_correction={self_correction}."
        )
    rng = random.Random(int(args.seed) + int(update_idx) * 130363 + 29)
    return rng.choice(candidates)


def launch_bridge_grouped_stage1(accelerator, dataset, model, model_logger, args):
    grouped_indices, metadata_rows = _metadata_index(args.dataset_metadata_path)
    friction_values = sorted(float(value) for value in grouped_indices)
    if len(friction_values) != 4:
        raise ValueError(f"Global bridge training currently requires exactly four environments, got {friction_values}.")
    expected_world_size = int(args.grouped_context_bridge_expected_world_size)
    if int(accelerator.num_processes) != expected_world_size:
        raise ValueError(
            f"Bridge experiment requires world_size={expected_world_size}, "
            f"got {accelerator.num_processes}."
        )
    endpoint_probability = float(args.grouped_context_bridge_endpoint_probability)
    if not 0.0 <= endpoint_probability <= 1.0:
        raise ValueError(
            "grouped_context_bridge_endpoint_probability must lie in [0,1], "
            f"got {endpoint_probability}."
        )

    table = model.friction_context_table
    if table.global_context is None:
        raise RuntimeError("Bridge training requires friction_context_table.global_context.")
    table.contexts.requires_grad_(False)
    table.global_context.requires_grad_(True)
    optimizer = _build_optimizer(model, args)
    model.to(device=accelerator.device)
    model, optimizer = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    warmup_steps = int(args.grouped_context_bridge_global_warmup_steps)
    bridge_steps = int(args.grouped_context_bridge_training_steps)
    total_updates = warmup_steps + bridge_steps
    chunks_per_env_per_rank = int(args.grouped_context_bridge_chunks_per_env_per_rank)
    alpha_levels = _parse_bridge_alpha_levels(args.grouped_context_bridge_alpha_levels)
    global_repeats = int(args.grouped_context_bridge_global_condition_repeats)
    if global_repeats <= 0:
        raise ValueError("grouped_context_bridge_global_condition_repeats must be positive.")
    bridge_conditions: list[tuple[int | None, float]] = [(None, 0.0)] * global_repeats
    bridge_conditions.extend(
        (target_index, alpha)
        for target_index in range(4)
        for alpha in alpha_levels
    )
    center_reg_weight = float(args.grouped_context_bridge_center_reg_weight)
    metrics_log_steps = max(1, int(args.grouped_context_bridge_metrics_log_steps))
    self_correction_enabled = bool(args.grouped_context_self_correction_enabled)
    self_correction_probability = float(args.grouped_context_self_correction_probability)
    if not 0.0 <= self_correction_probability <= 1.0:
        raise ValueError(
            "grouped_context_self_correction_probability must lie in [0,1], "
            f"got {self_correction_probability}."
        )
    counterfactual_enabled = bool(args.grouped_context_counterfactual_enabled)
    if counterfactual_enabled and not self_correction_enabled:
        raise ValueError(
            "Teacher counterfactual training requires grouped_context_self_correction_enabled."
        )
    counterfactual_batch_fraction = float(args.grouped_context_counterfactual_batch_fraction)
    if not 0.0 < counterfactual_batch_fraction <= 1.0:
        raise ValueError("grouped_context_counterfactual_batch_fraction must lie in (0,1].")
    counterfactual_bank = None
    counterfactual_noise_bands = ()
    if counterfactual_enabled:
        if not args.grouped_context_counterfactual_manifest_path:
            raise ValueError("Counterfactual Teacher manifest path is required.")
        if not args.grouped_context_counterfactual_raw_root:
            raise ValueError("Counterfactual Teacher raw root is required.")
        counterfactual_bank = CounterfactualSourceBank(
            args.grouped_context_counterfactual_manifest_path,
            args.grouped_context_counterfactual_raw_root,
        )
        counterfactual_noise_bands = parse_noise_bands(
            args.grouped_context_counterfactual_noise_bands
        )
    sampling_mode = str(args.grouped_context_bridge_sampling_mode)
    curve_power = float(args.grouped_context_bridge_curve_power)
    freeze_global_after_warmup = bool(
        args.grouped_context_bridge_freeze_global_after_warmup
    )
    timestep_buckets = max(
        1,
        int(args.grouped_context_bridge_timestep_buckets_per_update),
    )
    smoke_sequence = bool(args.grouped_context_bridge_smoke_sequence)
    if smoke_sequence and (warmup_steps != 1 or bridge_steps != 5):
        raise ValueError(
            "Bridge smoke sequence requires exactly 1 warmup update and 5 bridge updates."
        )

    if accelerator.is_main_process:
        print(
            "[global_bridge_stage1] "
            f"warmup_steps={warmup_steps} bridge_steps={bridge_steps} total_updates={total_updates} "
            f"endpoint_probability={endpoint_probability:.3f} world_size={accelerator.num_processes} "
            f"chunks_per_env_per_rank={chunks_per_env_per_rank} "
            f"global_chunks_per_update={4 * chunks_per_env_per_rank * accelerator.num_processes} "
            f"per_env_global={chunks_per_env_per_rank * accelerator.num_processes} "
            f"alpha_levels={alpha_levels} global_condition_repeats={global_repeats} "
            f"shared_timestep=true self_correction_enabled={self_correction_enabled} "
            f"self_correction_probability={self_correction_probability:g} "
            f"counterfactual_enabled={counterfactual_enabled} "
            f"counterfactual_batch_fraction={counterfactual_batch_fraction:g} "
            f"sampling_mode={sampling_mode} curve_power={curve_power:g} "
            f"timestep_buckets={timestep_buckets} "
            f"freeze_global_after_warmup={freeze_global_after_warmup} "
            f"smoke_sequence={smoke_sequence}",
            flush=True,
        )

    endpoint_updates = 0
    bridge_updates = 0
    iterator = tqdm(range(total_updates), disable=not accelerator.is_local_main_process)
    for update_idx in iterator:
        step = update_idx + 1
        if step <= warmup_steps:
            phase = "global_warmup"
            condition_kind = "global"
            target_index = None
            target_mu = friction_values[0]
            bridge_position = 0.0
            alpha = 0.0
            weights = [0.25] * 4
        else:
            phase = "mixed_bridge"
            post_index = step - warmup_steps - 1
            condition_rng = random.Random(int(args.seed) + int(post_index) * 179424673 + 97)
            if sampling_mode == "nonlinear_40_30_30" or smoke_sequence:
                forced_kind = None
                if smoke_sequence:
                    forced_kind = (
                        "endpoint",
                        "near_global",
                        "near_global",
                        "interior",
                        "interior",
                    )[post_index]
                condition = sample_nonlinear_bridge_condition(
                    condition_rng,
                    endpoint_probability=endpoint_probability,
                    curve_power=curve_power,
                    target_count=4,
                    forced_kind=forced_kind,
                )
                condition_kind = condition.kind
                target_index = int(condition.target_index)
                target_mu = friction_values[target_index]
                bridge_position = float(condition.position)
                alpha = float(condition.alpha)
                weights = list(condition.weights)
                if condition_kind == "endpoint":
                    endpoint_updates += 1
                else:
                    bridge_updates += 1
            elif condition_rng.random() < endpoint_probability:
                condition_kind = "endpoint"
                target_index = condition_rng.randrange(4)
                target_mu = friction_values[int(target_index)]
                bridge_position = 1.0
                alpha = 1.0
                weights = [
                    1.0 if env_index == int(target_index) else 0.0
                    for env_index in range(4)
                ]
                endpoint_updates += 1
            else:
                target_index, alpha = condition_rng.choice(bridge_conditions)
                bridge_position = float(alpha)
                if target_index is None:
                    condition_kind = "global"
                    target_mu = friction_values[0]
                    weights = [0.25] * 4
                else:
                    condition_kind = "interpolation"
                    target_mu = friction_values[int(target_index)]
                    weights = [
                        (1.0 - float(alpha)) / 4.0
                        + (float(alpha) if env_index == int(target_index) else 0.0)
                        for env_index in range(4)
                    ]
                bridge_updates += 1

        unwrapped = accelerator.unwrap_model(model)
        _set_bridge_requires_grad(
            unwrapped,
            phase,
            freeze_global_after_warmup=freeze_global_after_warmup,
        )
        model_lr, global_lr = _apply_bridge_lrs(
            optimizer,
            args,
            phase,
            freeze_global_after_warmup=freeze_global_after_warmup,
        )
        correction_rng = random.Random(int(args.seed) + int(update_idx) * 15485863 + 43)
        if smoke_sequence and step > warmup_steps:
            self_correction_update = (step - warmup_steps - 1) in {2, 4}
        else:
            self_correction_update = (
                self_correction_enabled
                and step > warmup_steps
                and correction_rng.random() < self_correction_probability
            )
        shared_timestep_indices = []
        for bucket_index in range(timestep_buckets):
            timestep_seed_index = update_idx * timestep_buckets + bucket_index
            if counterfactual_enabled and self_correction_update:
                sigma_rng = random.Random(
                    int(args.seed) + timestep_seed_index * 67867967 + 211
                )
                desired_sigma = sample_noise_fraction(
                    sigma_rng,
                    counterfactual_noise_bands,
                )
                scheduler_sigmas = unwrapped.pipe.scheduler.sigmas
                timestep_index = min(
                    range(len(scheduler_sigmas)),
                    key=lambda index: abs(float(scheduler_sigmas[index]) - desired_sigma),
                )
            else:
                timestep_index = _sample_bridge_shared_timestep_index(
                    scheduler=unwrapped.pipe.scheduler,
                    args=args,
                    update_idx=timestep_seed_index,
                    self_correction=self_correction_update,
                )
            shared_timestep_indices.append(int(timestep_index))
        shared_sigmas = [
            float(unwrapped.pipe.scheduler.sigmas[index])
            for index in shared_timestep_indices
        ]
        shared_timestep_index = shared_timestep_indices[0]
        shared_sigma = shared_sigmas[0]
        sampling_grouped_indices = grouped_indices
        if counterfactual_enabled and self_correction_update:
            sampling_grouped_indices = counterfactual_bank.restrict_grouped_indices(
                grouped_indices
            )
        sample_indices = _sample_bridge_rank_indices(
            grouped_indices=sampling_grouped_indices,
            rows=metadata_rows,
            seed=int(args.seed),
            update_idx=update_idx,
            process_index=int(accelerator.process_index),
            num_processes=int(accelerator.num_processes),
            chunks_per_env_per_rank=chunks_per_env_per_rank,
        )
        expected_local = 4 * chunks_per_env_per_rank
        if len(sample_indices) != expected_local:
            raise RuntimeError(f"Bridge local batch has {len(sample_indices)} samples, expected {expected_local}.")
        correction_micro_indices: set[int] = set()
        if self_correction_update:
            if counterfactual_enabled:
                for env_mu in friction_values:
                    env_positions = [
                        position
                        for position, sample_index in enumerate(sample_indices)
                        if abs(float(metadata_rows[sample_index]["friction_mu"]) - env_mu) <= 1e-5
                    ]
                    count = max(
                        1,
                        int(round(len(env_positions) * counterfactual_batch_fraction)),
                    )
                    correction_micro_indices.update(env_positions[:count])
            else:
                correction_micro_indices.update(range(len(sample_indices)))

        optimizer.zero_grad(set_to_none=True)
        local_loss_sums = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
        local_counts = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
        local_standard_loss_sums = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
        local_standard_counts = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
        local_correction_loss_sums = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
        local_correction_counts = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
        center_reg = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        for micro_idx, sample_index in enumerate(sample_indices):
            row = metadata_rows[sample_index]
            env_mu = float(row["friction_mu"])
            env_index = min(range(4), key=lambda index: abs(friction_values[index] - env_mu))
            data = dataset[sample_index].copy()
            data["_flow_timestep_index"] = int(
                shared_timestep_indices[micro_idx % len(shared_timestep_indices)]
            )
            data["_bridge_alpha"] = float(alpha)
            data["_bridge_target_mu"] = float(target_mu)
            loss_scale = float(weights[env_index]) / float(chunks_per_env_per_rank)
            sample_is_correction = micro_idx in correction_micro_indices
            if sample_is_correction:
                donor_rng = random.Random(
                    int(args.seed)
                    + int(update_idx) * 32452843
                    + int(sample_index) * 49999
                    + int(accelerator.process_index)
                )
                if counterfactual_enabled:
                    data["_self_correction_donor_data"] = counterfactual_bank.materialize(
                        dataset=dataset,
                        target_data=data,
                        target_index=sample_index,
                        target_row=row,
                        rng=donor_rng,
                    )
                else:
                    donor_index = _sample_self_correction_donor_index(
                        target_index=sample_index,
                        grouped_indices=grouped_indices,
                        rows=metadata_rows,
                        rng=donor_rng,
                    )
                    data["_self_correction_donor_data"] = dataset[donor_index]

            sync_context = (
                accelerator.no_sync(model)
                if micro_idx < len(sample_indices) - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                loss = model(data)
                scaled_loss = loss * loss_scale
                if micro_idx == len(sample_indices) - 1 and float(alpha) < 1.0:
                    current_table = accelerator.unwrap_model(model).friction_context_table
                    center_reg = torch.mean(
                        (
                            current_table.global_context.float()
                            - current_table.contexts.detach().float().mean(dim=0)
                        )
                        ** 2
                    )
                    scaled_loss = scaled_loss + center_reg_weight * center_reg
                accelerator.backward(scaled_loss)
            local_loss_sums[env_index] += loss.detach().float()
            local_counts[env_index] += 1.0
            if sample_is_correction:
                local_correction_loss_sums[env_index] += loss.detach().float()
                local_correction_counts[env_index] += 1.0
            else:
                local_standard_loss_sums[env_index] += loss.detach().float()
                local_standard_counts[env_index] += 1.0

        current_table = accelerator.unwrap_model(model).friction_context_table
        global_grad = current_table.global_context.grad
        global_grad_norm = None
        gradient_cosine = None
        if global_grad is not None:
            global_grad_flat = global_grad.detach().float().flatten()
            global_grad_norm = float(torch.linalg.vector_norm(global_grad_flat).cpu())
            if target_index is not None:
                direction = (
                    current_table.contexts[int(target_index)].detach().float()
                    - current_table.global_context.detach().float()
                ).flatten()
                if float(torch.linalg.vector_norm(direction).cpu()) > 0 and global_grad_norm > 0:
                    gradient_cosine = float(
                        F.cosine_similarity(-global_grad_flat, direction, dim=0).cpu()
                    )

        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        current_table.clamp_(
            args.grouped_context_clamp_min,
            args.grouped_context_clamp_max,
        )
        with torch.no_grad():
            current_table.global_context.clamp_(
                min=args.grouped_context_clamp_min,
                max=args.grouped_context_clamp_max,
            )
        optimizer.zero_grad(set_to_none=True)

        global_loss_sums = accelerator.reduce(local_loss_sums, reduction="sum")
        global_counts = accelerator.reduce(local_counts, reduction="sum")
        global_standard_loss_sums = accelerator.reduce(local_standard_loss_sums, reduction="sum")
        global_standard_counts = accelerator.reduce(local_standard_counts, reduction="sum")
        global_correction_loss_sums = accelerator.reduce(local_correction_loss_sums, reduction="sum")
        global_correction_counts = accelerator.reduce(local_correction_counts, reduction="sum")
        per_env_losses = global_loss_sums / global_counts.clamp_min(1.0)
        per_env_standard_losses = (
            global_standard_loss_sums / global_standard_counts.clamp_min(1.0)
        )
        per_env_correction_losses = (
            global_correction_loss_sums / global_correction_counts.clamp_min(1.0)
        )
        weight_tensor = torch.tensor(weights, device=accelerator.device, dtype=torch.float32)
        weighted_flow_loss = torch.sum(per_env_losses * weight_tensor)
        reported_loss = weighted_flow_loss + (
            center_reg_weight * center_reg.detach().float()
            if float(alpha) < 1.0
            else 0.0
        )

        if accelerator.is_main_process and (
            step == 1
            or step % metrics_log_steps == 0
            or step == warmup_steps
            or step == total_updates
        ):
            endpoint_distances = [
                float(
                    torch.linalg.vector_norm(
                        current_table.global_context.detach().float()
                        - current_table.contexts[index].detach().float()
                    ).cpu()
                )
                for index in range(4)
            ]
            metrics = {
                "step": step,
                "phase": phase,
                "condition_kind": condition_kind,
                "target_environment_index": target_index,
                "target_environment_mu": target_mu,
                "bridge_position": float(bridge_position),
                "alpha": float(alpha),
                "weights": [float(value) for value in weights],
                "per_environment_flow_loss": [float(value.cpu()) for value in per_env_losses],
                "per_environment_standard_flow_loss": [
                    None if float(global_standard_counts[index].cpu()) == 0.0
                    else float(per_env_standard_losses[index].cpu())
                    for index in range(4)
                ],
                "per_environment_self_correction_loss": [
                    None if float(global_correction_counts[index].cpu()) == 0.0
                    else float(per_env_correction_losses[index].cpu())
                    for index in range(4)
                ],
                "weighted_flow_loss": float(weighted_flow_loss.cpu()),
                "center_reg": float(center_reg.detach().float().cpu()),
                "center_reg_weight": center_reg_weight,
                "reported_total_loss": float(reported_loss.cpu()),
                "model_lr": model_lr,
                "global_lr": global_lr,
                "global_context_mean": float(current_table.global_context.detach().float().mean().cpu()),
                "global_context_norm": float(
                    torch.linalg.vector_norm(current_table.global_context.detach().float()).cpu()
                ),
                "global_to_endpoint_l2": endpoint_distances,
                "global_grad_norm": global_grad_norm,
                "negative_grad_cosine_to_target_endpoint": gradient_cosine,
                "shared_timestep_index": int(shared_timestep_index),
                "shared_sigma": shared_sigma,
                "timestep_indices": shared_timestep_indices,
                "timestep_sigmas": shared_sigmas,
                "self_correction_update": self_correction_update,
                "self_correction_probability": self_correction_probability,
                "counterfactual_enabled": counterfactual_enabled,
                "counterfactual_examples": len(correction_micro_indices),
                "counterfactual_batch_fraction": (
                    float(len(correction_micro_indices)) / float(expected_local)
                ),
                "world_size": int(accelerator.num_processes),
                "local_batch_size": expected_local,
                "global_batch_size": expected_local * int(accelerator.num_processes),
                "global_chunks_per_environment": chunks_per_env_per_rank * int(accelerator.num_processes),
                "endpoint_probability": endpoint_probability,
                "realized_endpoint_fraction": (
                    float(endpoint_updates) / float(endpoint_updates + bridge_updates)
                    if endpoint_updates + bridge_updates > 0
                    else None
                ),
            }
            print("[bridge_metrics] " + json.dumps(metrics, sort_keys=True), flush=True)
            _append_bridge_metrics(model_logger.output_path, metrics)

        if step == warmup_steps:
            _log_context_table(accelerator, model, step, phase, "global_warmup_end")
            _save_phase_context_table(
                accelerator,
                model,
                model_logger,
                step,
                phase,
                "global_warmup_end",
            )
        model_logger.on_step_end(accelerator, model, args.save_steps, loss=reported_loss)

    if smoke_sequence:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(
                "[global_bridge_stage1] smoke sequence completed; "
                "skipping final training checkpoint",
                flush=True,
            )
        return
    model_logger.on_training_end(accelerator, model, args.save_steps)


def _assign_initial_contexts_from_pool(
    accelerator,
    table,
    group_order: list[int],
    initial_groups: int,
    pool_path: str,
    output_path: str,
) -> None:
    pool = torch.load(pool_path, map_location="cpu", weights_only=False).float()
    if pool.ndim != 3 or tuple(pool.shape[1:]) != tuple(table.contexts.shape[1:]):
        raise ValueError(
            f"Context pool shape {tuple(pool.shape)} is incompatible with "
            f"table shape {tuple(table.contexts.shape)}."
        )
    count = min(int(initial_groups), len(group_order))
    if count > int(pool.shape[0]):
        raise ValueError(
            f"Cannot assign {count} initial groups from pool_size={int(pool.shape[0])}."
        )
    assignment_path = Path(output_path) / "initial_context_pool_assignment.json"
    if accelerator.is_main_process and not assignment_path.exists():
        selected_pool_indices = random.SystemRandom().sample(
            range(int(pool.shape[0])),
            count,
        )
        records = []
        table_values = table.friction_values.detach().float().cpu().tolist()
        for group_index, pool_index in zip(group_order[:count], selected_pool_indices):
            records.append(
                {
                    "group_index": int(group_index),
                    "group_value": float(table_values[int(group_index)]),
                    "pool_index": int(pool_index),
                }
            )
        temporary = assignment_path.with_name(
            f".{assignment_path.name}.tmp-{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_pool": str(pool_path),
                    "pool_size": int(pool.shape[0]),
                    "records": records,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, assignment_path)
    accelerator.wait_for_everyone()
    with assignment_path.open("r", encoding="utf-8") as handle:
        assignment = json.load(handle)
    with torch.no_grad():
        for record in assignment["records"]:
            table.contexts[int(record["group_index"])].copy_(
                pool[int(record["pool_index"])].to(
                    device=table.contexts.device,
                    dtype=table.contexts.dtype,
                )
            )
    if accelerator.is_main_process:
        print(
            f"[context_pool] assigned {len(assignment['records'])} initial groups "
            f"from fixed pool {pool_path}; mapping={assignment_path}",
            flush=True,
        )


def launch_curriculum_grouped_stage1(accelerator, dataset, model, model_logger, args):
    grouped_indices, metadata_rows = _metadata_index(args.dataset_metadata_path)
    unwrapped = model
    friction_values = unwrapped.friction_context_table.friction_values.detach().float().cpu().tolist()
    total_groups = int(args.grouped_context_curriculum_total_groups) or len(friction_values)
    group_order = _load_or_create_curriculum_group_order(
        accelerator=accelerator,
        output_path=model_logger.output_path,
        num_groups=len(friction_values),
        initial_groups=int(args.grouped_context_curriculum_initial_groups),
        add_groups=int(args.grouped_context_curriculum_add_groups),
        total_groups=total_groups,
        num_strata=int(getattr(args, "grouped_context_curriculum_strata", 0) or 0),
        shared_initial_across_strata=bool(
            getattr(args, "grouped_context_curriculum_shared_initial_friction", False)
        ),
    )
    _initialize_curriculum_contexts(unwrapped.friction_context_table, args, group_order, total_groups)
    initial_context_pool_path = getattr(
        args,
        "grouped_context_initial_context_pool_path",
        None,
    )
    if initial_context_pool_path and not getattr(
        args,
        "grouped_context_resume_context_table",
        None,
    ):
        _assign_initial_contexts_from_pool(
            accelerator,
            unwrapped.friction_context_table,
            group_order,
            int(args.grouped_context_curriculum_initial_groups),
            initial_context_pool_path,
            model_logger.output_path,
        )
    shared_initial = bool(
        getattr(args, "grouped_context_curriculum_shared_initial_friction", False)
    )
    random_context_warmup = bool(
        getattr(args, "grouped_context_curriculum_random_context_warmup", False)
    )
    if shared_initial and random_context_warmup:
        raise ValueError(
            "shared_initial_friction and random_context_warmup are mutually exclusive."
        )
    initial_sample_group_indices = None
    initial_sample_mode = str(
        getattr(args, "grouped_context_curriculum_initial_sample_mode", "all") or "all"
    ).strip().lower()
    if initial_sample_mode not in ("all", "explicit", "stratum_median"):
        raise ValueError(
            "grouped_context_curriculum_initial_sample_mode must be all, explicit, or "
            "stratum_median, "
            f"got {initial_sample_mode!r}."
        )
    if initial_sample_mode == "explicit":
        raw_sample_groups = str(
            getattr(args, "grouped_context_curriculum_initial_sample_groups", None) or ""
        )
        requested_group_values = [
            float(value.strip())
            for value in raw_sample_groups.split(",")
            if value.strip()
        ]
        if not requested_group_values:
            raise ValueError(
                "explicit initial sampling requires "
                "grouped_context_curriculum_initial_sample_groups."
            )
        initial_group_count = min(
            int(args.grouped_context_curriculum_initial_groups),
            len(group_order),
        )
        active_initial_indices = set(group_order[:initial_group_count])
        initial_sample_group_indices = []
        for requested_value in requested_group_values:
            distances = [
                abs(float(group_value) - requested_value)
                for group_value in friction_values
            ]
            group_index = min(range(len(distances)), key=distances.__getitem__)
            if distances[group_index] > 1e-5:
                raise ValueError(
                    f"Explicit initial sample group {requested_value:g} is absent from "
                    "the context table."
                )
            if group_index not in active_initial_indices:
                raise ValueError(
                    f"Explicit initial sample group {requested_value:g} is not among "
                    f"the first {initial_group_count} active curriculum groups."
                )
            initial_sample_group_indices.append(group_index)
        if len(set(initial_sample_group_indices)) != len(initial_sample_group_indices):
            raise ValueError("Explicit initial sample groups must be unique.")
        expected_strata = int(getattr(args, "grouped_context_curriculum_strata", 0) or 0)
        if expected_strata > 0:
            stratify_field = str(
                getattr(args, "grouped_context_stratify_field", None) or "environment_index"
            )
            row_by_group = {}
            for row in metadata_rows:
                row_by_group.setdefault(float(row["friction_mu"]), row)
            selected_strata = {
                row_by_group[float(friction_values[group_index])][stratify_field]
                for group_index in initial_sample_group_indices
            }
            if len(selected_strata) != expected_strata:
                raise ValueError(
                    f"Explicit initial sampling covers {len(selected_strata)} strata, "
                    f"expected {expected_strata}."
                )
    elif initial_sample_mode == "stratum_median":
        initial_group_count = min(
            int(args.grouped_context_curriculum_initial_groups),
            len(group_order),
        )
        stratify_field = str(
            getattr(args, "grouped_context_stratify_field", None) or "environment_index"
        )
        rank_field = str(
            getattr(
                args,
                "grouped_context_curriculum_initial_sample_rank_field",
                "physical_friction_mu",
            )
        )
        row_by_group = {}
        for row in metadata_rows:
            row_by_group.setdefault(float(row["friction_mu"]), row)
        groups_by_stratum = defaultdict(list)
        for group_index in group_order[:initial_group_count]:
            group_value = float(friction_values[int(group_index)])
            row = row_by_group[group_value]
            groups_by_stratum[row[stratify_field]].append(
                (float(row[rank_field]), int(group_index))
            )
        initial_sample_group_indices = []
        for stratum in sorted(groups_by_stratum, key=str):
            ranked = sorted(groups_by_stratum[stratum])
            initial_sample_group_indices.append(ranked[len(ranked) // 2][1])
        expected_strata = int(getattr(args, "grouped_context_curriculum_strata", 0) or 0)
        if expected_strata > 0 and len(initial_sample_group_indices) != expected_strata:
            raise ValueError(
                f"Initial median sampling selected {len(initial_sample_group_indices)} strata, "
                f"expected {expected_strata}."
            )
    shared_context_indices: dict[int, int] = {}
    if shared_initial:
        shared_context_indices = _share_initial_contexts_across_strata(
            unwrapped.friction_context_table,
            group_order,
            int(args.grouped_context_curriculum_initial_groups),
            int(getattr(args, "grouped_context_curriculum_strata", 0) or 0),
        )
    metadata_group_values = sorted(float(value) for value in grouped_indices)
    if len(metadata_group_values) != len(friction_values):
        raise ValueError(
            f"Metadata has {len(metadata_group_values)} groups but the context table has "
            f"{len(friction_values)}."
        )
    shared_context_values = {
        metadata_group_values[alias_index]: metadata_group_values[canonical_index]
        for alias_index, canonical_index in shared_context_indices.items()
    }
    optimizer = _build_curriculum_optimizer(model, args, accelerator)
    model.to(device=accelerator.device)
    model, optimizer = accelerator.prepare(model, optimizer)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    initial_steps = int(args.grouped_context_curriculum_initial_model_steps)
    assignment_model_steps = int(
        getattr(args, "grouped_context_curriculum_assignment_model_steps", 0) or 0
    )
    add_groups = int(args.grouped_context_curriculum_add_groups)
    groups_after_initial = max(0, min(total_groups, len(friction_values)) - int(args.grouped_context_curriculum_initial_groups))
    rounds = 1 + (groups_after_initial + max(add_groups, 1) - 1) // max(add_groups, 1)
    variant = str(getattr(args, "grouped_context_curriculum_variant", "default") or "default").strip().lower()
    if variant in ("two_new_context", "high_model_mid", "high_model_mid_new"):
        curriculum_cycle_steps = (
            int(args.grouped_context_curriculum_new_context_steps)
            + int(getattr(args, "grouped_context_curriculum_mid_context_steps", 0) or int(args.grouped_context_curriculum_new_context_steps))
            + 2 * int(args.grouped_context_curriculum_all_context_steps)
            + 3 * int(args.grouped_context_curriculum_model_steps)
        )
    else:
        curriculum_cycle_steps = (
            int(args.grouped_context_curriculum_new_context_steps)
            + 2 * int(args.grouped_context_curriculum_all_context_steps)
            + 2 * int(args.grouped_context_curriculum_model_steps)
        )
    default_updates = initial_steps + assignment_model_steps + rounds * curriculum_cycle_steps
    default_updates += int(getattr(args, "grouped_context_curriculum_initial_refinement_steps", 0) or 0)
    updates = int(args.grouped_context_structured_updates) or default_updates
    actions_per_update = int(args.grouped_context_actions_per_update)
    friction_groups_per_update = int(args.grouped_context_friction_groups_per_update)
    microbatches_per_update = int(args.grouped_context_microbatches_per_update)
    if microbatches_per_update <= 0:
        microbatches_per_update = friction_groups_per_update * actions_per_update
    validation_interval = int(getattr(args, "grouped_context_validation_interval", 0) or 0)
    validation_seed = int(getattr(args, "grouped_context_validation_seed", 20260810))
    validation_indices: list[int] = []
    validation_actions: list = []
    if validation_interval > 0:
        validation_indices, validation_actions = _build_fixed_validation_indices(
            grouped_indices,
            accelerator,
            actions_per_update,
            validation_seed,
        )
    protected_checkpoint_steps = _parse_step_set(
        getattr(args, "grouped_context_protected_checkpoint_steps", None)
    )

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
        if validation_interval > 0:
            print(
                "[validation_setup] "
                f"interval={validation_interval} seed={validation_seed} "
                f"actions={','.join(str(value) for value in validation_actions)}",
                flush=True,
            )
        if initial_sample_group_indices is not None:
            initial_sample_values = [
                friction_values[index] for index in initial_sample_group_indices
            ]
            print(
                "[curriculum_grouped_stage1] initial_model_sample_mu="
                + ",".join(f"{float(value):.6g}" for value in initial_sample_values),
                flush=True,
            )

    resume_step = int(getattr(args, "grouped_context_resume_step", 0) or 0)
    iterator = tqdm(range(resume_step, updates), disable=not accelerator.is_local_main_process)
    last_phase_key = None
    last_forced_model_step = None
    context_phase_anchor = None
    context_phase_limit = float(
        getattr(args, "grouped_context_phase_max_displacement", 0.0) or 0.0
    )
    context_phase_limit_start = int(
        getattr(args, "grouped_context_phase_max_displacement_start_step", 0) or 0
    )
    for update_idx in iterator:
        step = update_idx + 1
        phase_info = _curriculum_phase_for_step(args, group_order, step)
        phase = str(phase_info["phase"])
        phase_sample_group_indices = phase_info["sample_group_indices"]
        if initial_sample_group_indices is not None and step <= initial_steps:
            phase_sample_group_indices = initial_sample_group_indices
        phase_key = (phase_info["round"], phase, phase_info["phase_start"], phase_info["phase_end"])
        phase_changed = phase_key != last_phase_key
        if phase_changed:
            context_phase = phase in ("new_context", "new_context_mid", "all_context", "context")
            if (
                context_phase
                and context_phase_limit > 0.0
                and int(phase_info["phase_start"]) >= context_phase_limit_start
            ):
                context_phase_anchor = (
                    accelerator.unwrap_model(model)
                    .friction_context_table.contexts.detach()
                    .clone()
                )
            else:
                context_phase_anchor = None
        if accelerator.is_main_process and phase_changed:
            sample_mu = [friction_values[index] for index in phase_sample_group_indices]
            print(
                "[curriculum_phase] "
                f"step={step} round={phase_info['round']} phase={phase} "
                f"range={phase_info['phase_start']}-{phase_info['phase_end']} "
                f"active_count={phase_info['active_count']} new_count={phase_info['new_count']} "
                f"sample_mu={','.join(f'{float(value):.6g}' for value in sample_mu)}",
                flush=True,
            )
        if (
            phase_changed
            and phase == "model"
            and bool(
                getattr(
                    args,
                    "grouped_context_reset_model_optimizer_state_on_phase_start",
                    False,
                )
            )
        ):
            cleared_states = _clear_optimizer_group_state(optimizer, "model")
            if accelerator.is_main_process:
                print(
                    "[optimizer_reset] "
                    f"step={step} group=model cleared_parameter_states={cleared_states}",
                    flush=True,
                )
        last_phase_key = phase_key

        background_freeze_after = int(
            getattr(args, "grouped_background_freeze_after_steps", 0) or 0
        )
        freeze_background = (
            background_freeze_after > 0 and step > background_freeze_after
        )
        if (
            accelerator.is_main_process
            and background_freeze_after > 0
            and step == background_freeze_after + 1
        ):
            print(
                "[background_context] permanently froze Background Z and "
                f"background_context_encoder after step={background_freeze_after}",
                flush=True,
            )
        _set_curriculum_requires_grad(model, phase, freeze_background=freeze_background)
        current_model_lr, current_context_lr = _apply_curriculum_lrs(
            optimizer,
            args,
            step,
            phase,
            phase_start=int(phase_info["phase_start"]),
            freeze_background=freeze_background,
        )
        if accelerator.is_main_process and phase_changed:
            print(
                f"[phase_lr] step={step} model_lr={current_model_lr:.8g} "
                f"context_lr={current_context_lr:.8g}",
                flush=True,
            )
        sample_values = [friction_values[index] for index in phase_sample_group_indices]
        sample_friction_groups = min(friction_groups_per_update, len(sample_values))
        sample_indices = _sample_update_indices(
            grouped_indices=grouped_indices,
            rows=metadata_rows,
            accelerator=accelerator,
            args=args,
            update_idx=update_idx,
            friction_groups=sample_friction_groups,
            actions_per_update=actions_per_update,
            microbatches_per_update=microbatches_per_update,
            allowed_friction_values=sample_values,
        )
        optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for micro_idx, sample_index in enumerate(sample_indices):
            sync_context = (
                accelerator.no_sync(model)
                if micro_idx < len(sample_indices) - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                sample_data = dataset[sample_index]
                if random_context_warmup and step <= initial_steps:
                    random_context = random.Random(
                        int(args.seed)
                        + int(step) * 104729
                        + int(accelerator.process_index) * 1009
                        + int(micro_idx)
                    )
                    pool_size = min(
                        int(args.grouped_context_curriculum_initial_groups),
                        len(group_order),
                    )
                    borrowed_index = group_order[random_context.randrange(pool_size)]
                    sample_data = sample_data.copy()
                    sample_data["friction_mu"] = float(friction_values[borrowed_index])
                elif shared_context_values:
                    sample_group_value = float(metadata_rows[sample_index]["friction_mu"])
                    canonical_value = shared_context_values.get(sample_group_value)
                    if canonical_value is not None:
                        sample_data = sample_data.copy()
                        sample_data["friction_mu"] = canonical_value
                loss = model(sample_data)
                detached_losses.append(loss.detach())
                accelerator.backward(loss / len(sample_indices))
        frozen_context_snapshot = None
        frozen_context_mask = None
        if phase in ("new_context", "new_context_mid", "all_context"):
            unwrapped_model = accelerator.unwrap_model(model)
            _mask_context_grad(unwrapped_model, phase_info["train_context_indices"])
            context_table = unwrapped_model.friction_context_table
            train_mask = _context_row_mask(context_table, phase_info["train_context_indices"])
            if not bool(train_mask.all()):
                frozen_context_snapshot = context_table.contexts.detach().clone()
                frozen_context_mask = ~train_mask
                _zero_masked_context_optimizer_state(optimizer, context_table, train_mask)
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if frozen_context_snapshot is not None:
            context_table = accelerator.unwrap_model(model).friction_context_table
            with torch.no_grad():
                context_table.contexts[frozen_context_mask].copy_(
                    frozen_context_snapshot[frozen_context_mask]
                )
        if shared_context_indices:
            context_table = accelerator.unwrap_model(model).friction_context_table
            with torch.no_grad():
                for alias_index, canonical_index in shared_context_indices.items():
                    context_table.contexts[alias_index].copy_(
                        context_table.contexts[canonical_index]
                    )
        accelerator.unwrap_model(model).friction_context_table.clamp_(
            args.grouped_context_clamp_min,
            args.grouped_context_clamp_max,
        )
        if context_phase_anchor is not None:
            _project_context_displacement(
                accelerator.unwrap_model(model).friction_context_table,
                context_phase_anchor,
                context_phase_limit,
            )
        if (
            random_context_warmup
            and step == initial_steps
            and accelerator.is_main_process
        ):
            print(
                "[curriculum] random C borrowing ended; the initial active groups now "
                "permanently use their distinct table entries",
                flush=True,
            )
        if step == int(phase_info["phase_end"]):
            checkpoint_policy = str(
                getattr(args, "grouped_context_model_phase_checkpoint_policy", "all")
                or "all"
            ).strip().lower()
            cycle_offset = initial_steps + assignment_model_steps
            is_cycle_end = (
                step > cycle_offset
                and (step - cycle_offset) % int(curriculum_cycle_steps) == 0
            )
            should_checkpoint_model_phase = phase == "model" and (
                checkpoint_policy == "all"
                or (checkpoint_policy == "cycle_end" and is_cycle_end)
            )
            if checkpoint_policy not in ("all", "cycle_end", "none"):
                raise ValueError(
                    "grouped_context_model_phase_checkpoint_policy must be "
                    f"all, cycle_end, or none; got {checkpoint_policy!r}."
                )
            if should_checkpoint_model_phase:
                regular_save_due = (
                    int(args.save_steps) > 0
                    and step % int(args.save_steps) == 0
                )
                if not regular_save_due:
                    if accelerator.is_main_process:
                        print(
                            "[checkpoint] forcing paired model/context save at "
                            f"model phase end step={step}",
                            flush=True,
                        )
                    model_logger.save_model(accelerator, model, f"step-{step}.safetensors")
                last_forced_model_step = step
            _log_context_table(accelerator, model, step, phase, "curriculum_phase_end")
            _save_phase_context_table(accelerator, model, model_logger, step, phase, "curriculum_phase_end")
            if context_phase_anchor is not None and accelerator.is_main_process:
                displacement = torch.linalg.vector_norm(
                    (
                        accelerator.unwrap_model(model).friction_context_table.contexts
                        - context_phase_anchor
                    ).flatten(start_dim=1).float(),
                    dim=1,
                )
                print(
                    f"[context_displacement] step={step} limit={context_phase_limit:.6f} "
                    f"mean={float(displacement.mean().cpu()):.6f} "
                    f"max={float(displacement.max().cpu()):.6f}",
                    flush=True,
                )
            if should_checkpoint_model_phase and step in protected_checkpoint_steps:
                _protect_paired_checkpoint(accelerator, model_logger, step)
        mean_loss = torch.stack([loss.float() for loss in detached_losses]).mean()
        model_logger.on_step_end(accelerator, model, args.save_steps, loss=mean_loss)
        if validation_interval > 0 and step % validation_interval == 0:
            validation_loss, validation_count = _run_fixed_validation(
                accelerator,
                model,
                dataset,
                validation_indices,
                validation_seed,
            )
            if accelerator.is_main_process:
                print(
                    f"[validation] step={step} loss={validation_loss:.6f} "
                    f"count={validation_count}",
                    flush=True,
                )

    if last_forced_model_step != model_logger.num_steps:
        model_logger.on_training_end(accelerator, model, args.save_steps)


def main() -> None:
    parser = add_grouped_context_config(wan_parser())
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)

    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    friction_values = _unique_friction_values(args.dataset_metadata_path)
    background_values = (
        _unique_background_values(args.dataset_metadata_path)
        if bool(getattr(args, "background_context_enabled", False))
        else []
    )
    loggers = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}", False)]
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=loggers or None,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            ),
            InitProcessGroupKwargs(timeout=timedelta(hours=1)),
        ],
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
        background_values=background_values,
    )
    restored_global_context = False
    if getattr(args, "grouped_context_resume_context_table", None):
        restored_global_context = _load_grouped_context_table(
            model,
            args.grouped_context_resume_context_table,
        )
        _load_background_context_table(model, args.grouped_context_resume_context_table)
        if accelerator.is_main_process:
            print(
                f"[resume] restored context table from {args.grouped_context_resume_context_table} "
                f"at step={int(args.grouped_context_resume_step)}",
                flush=True,
            )
    if bool(getattr(args, "grouped_context_bridge_enabled", False)) and not restored_global_context:
        with torch.no_grad():
            model.friction_context_table.global_context.copy_(
                model.friction_context_table.contexts.detach().mean(dim=0)
            )
        if accelerator.is_main_process:
            print(
                "[global_bridge_stage1] initialized global context from the mean of four endpoints",
                flush=True,
            )
    if accelerator.is_main_process:
        print(
            f"Grouped-C Stage1: groups={len(friction_values)} "
            f"context_dim={args.physical_context_dim} tokens={args.physical_context_tokens} "
            f"background_groups={len(background_values)}",
            flush=True,
        )

    if bool(getattr(args, "grouped_context_stage_checkpoints_locally", False)):
        local_checkpoint_root = (
            getattr(args, "grouped_context_local_checkpoint_root", None)
            or os.environ.get("BWM_LOCAL_CHECKPOINT_ROOT")
        )
        if not local_checkpoint_root:
            raise ValueError(
                "Local checkpoint staging is enabled but no local root was configured."
            )
        model_logger = StagedGroupedContextModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
            save_minutes=args.checkpoint_save_minutes,
            keep_last=args.checkpoint_keep_last,
            log_steps=args.log_steps,
            local_checkpoint_root=local_checkpoint_root,
        )
    else:
        model_logger = GroupedContextModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
            save_minutes=args.checkpoint_save_minutes,
            keep_last=args.checkpoint_keep_last,
            log_steps=args.log_steps,
        )
    model_logger.num_steps = int(getattr(args, "grouped_context_resume_step", 0) or 0)
    if int(getattr(args, "grouped_context_direct_random_steps", 0) or 0) > 0:
        launch_direct_random_grouped_stage1(accelerator, dataset, model, model_logger, args)
    elif bool(getattr(args, "grouped_context_bridge_enabled", False)):
        launch_bridge_grouped_stage1(accelerator, dataset, model, model_logger, args)
    elif int(getattr(args, "grouped_context_curriculum_initial_groups", 0) or 0) > 0:
        launch_curriculum_grouped_stage1(accelerator, dataset, model, model_logger, args)
    elif int(getattr(args, "grouped_context_structured_updates", 0) or 0) > 0:
        launch_structured_grouped_stage1(accelerator, dataset, model, model_logger, args)
    else:
        launch_grouped_stage1(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
