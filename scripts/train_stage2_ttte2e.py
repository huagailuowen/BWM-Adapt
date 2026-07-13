#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import accelerate
import torch
from tqdm import tqdm

from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from train import TimedRetentionModelLogger, WanTrainingModule, wan_parser
from train_stage2_ttt import PushBoxMetaTaskDataset, add_stage2_config, build_dataset
from wan_video_action.data.operators import LoadCobotAction, ResolvePromptEmbPath, create_video_operator
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


TTTE2E_PARAM_SCOPES = ("gates", "lowrank", "all")
TTTE2E_OUTER_DIT_LAYER_SCOPES = ("none", "all", "last_half", "last_third")
TTTE2E_INNER_OPTIMIZERS = ("sgd", "adam")
TTTE2E_OUTER_OBJECTIVES = ("adapted_loss", "relative_improve", "hybrid_loss_relative_improve")
TTTE2E_TASK_MODES = ("support_query", "episode_sequence")
TTTE2E_SEQUENCE_CHUNK_MODES = ("random_chunk", "all_chunks_mean")


def _adapter_param_allowed(name: str, scope: str) -> bool:
    if scope == "gates":
        return name.endswith(".gate")
    if scope == "lowrank":
        return name.endswith(".gate") or name.endswith(".down.weight") or name.endswith(".up.weight")
    if scope == "all":
        return True
    raise ValueError(f"Unsupported TTT-E2E parameter scope: {scope}.")


def _clip_grad_tensors(
    grads: tuple[torch.Tensor | None, ...],
    max_norm: float,
) -> tuple[torch.Tensor | None, ...]:
    if max_norm <= 0:
        return grads
    valid = [grad for grad in grads if grad is not None]
    if not valid:
        return grads
    norm_sq = torch.stack([grad.detach().float().pow(2).sum() for grad in valid]).sum()
    grad_norm = torch.sqrt(norm_sq).clamp_min(1e-6)
    scale = min(1.0, float(max_norm) / float(grad_norm.cpu()))
    return tuple(None if grad is None else grad * scale for grad in grads)


def _parse_inner_lr_schedule(raw: str, inner_steps: int, default_lr: float) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(float(default_lr) for _ in range(int(inner_steps)))

    values: list[float] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "@" in item:
            count_text, lr_text = item.split("@", 1)
            values.extend([float(lr_text)] * int(count_text))
        else:
            values.append(float(item))

    if len(values) != int(inner_steps):
        raise ValueError(
            f"ttte2e_inner_lr_schedule expands to {len(values)} values, "
            f"but stage2_inner_steps={inner_steps}."
        )
    return tuple(values)


def _freeze_dit_except_layer_scope(dit: torch.nn.Module, scope: str) -> tuple[int, int]:
    if scope == "none":
        dit.requires_grad_(False)
        return 0, len(getattr(dit, "blocks", ()))
    if scope == "all":
        return len(getattr(dit, "blocks", ())), len(getattr(dit, "blocks", ()))
    if scope == "last_half":
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            raise ValueError("outer DiT layer scope requires pipe.dit.blocks.")
        num_blocks = len(blocks)
        start = num_blocks // 2
        dit.requires_grad_(False)
        for block in blocks[start:]:
            block.requires_grad_(True)
        return num_blocks - start, num_blocks
    if scope != "last_third":
        raise ValueError(f"Unsupported outer DiT layer scope: {scope}.")

    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise ValueError("outer DiT layer scope requires pipe.dit.blocks.")
    num_blocks = len(blocks)
    start = (num_blocks * 2) // 3
    dit.requires_grad_(False)
    for block in blocks[start:]:
        block.requires_grad_(True)
    return num_blocks - start, num_blocks


class FunctionalAdapterBank(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, fast_params: dict[str, torch.Tensor]):
        super().__init__()
        self.base = base
        self.fast_params = fast_params

    def forward(
        self,
        layer_index: int,
        x: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.func.functional_call(
            self.base,
            self.fast_params,
            (layer_index, x),
            {"conditioning": conditioning},
            strict=False,
        )


class SequentialEpisodeMetaTaskDataset(PushBoxMetaTaskDataset):
    def __init__(
        self,
        *,
        unique_episodes: int,
        duplicate_episodes: int,
        episode_chunk_mode: str,
        **kwargs,
    ):
        super().__init__(support_min=1, support_max=1, query_min=1, query_max=1, **kwargs)
        self.unique_episodes = int(unique_episodes)
        self.duplicate_episodes = int(duplicate_episodes)
        self.episode_chunk_mode = str(episode_chunk_mode)
        if self.unique_episodes <= 0:
            raise ValueError("unique_episodes must be positive.")
        if self.duplicate_episodes < 0:
            raise ValueError("duplicate_episodes must be non-negative.")
        if self.episode_chunk_mode not in TTTE2E_SEQUENCE_CHUNK_MODES:
            raise ValueError(f"Unsupported episode_chunk_mode={self.episode_chunk_mode}.")

        episode_groups = []
        for key, rows in self.groups:
            episodes = defaultdict(list)
            for row in rows:
                episode_key = row.get("episode_index", row.get("sample_id"))
                episodes[episode_key].append(row)
            if episodes:
                ordered = sorted(episodes.items(), key=lambda item: str(item[0]))
                episode_groups.append((key, ordered))
        if not episode_groups:
            raise ValueError("No non-empty episode groups were built for sequential TTT-E2E.")
        self.episode_groups = episode_groups
        print(
            "Sequential TTT-E2E episode dataset: "
            f"groups={len(self.episode_groups)} unique={self.unique_episodes} "
            f"duplicate={self.duplicate_episodes} chunk_mode={self.episode_chunk_mode}",
            flush=True,
        )
        for key, episodes in self.episode_groups:
            chunks = sum(len(rows) for _, rows in episodes)
            print(f"  sequence_group={key}: episodes={len(episodes)}, chunks={chunks}", flush=True)

    def __getitem__(self, data_id: int) -> dict:
        rng = random.Random(self.seed + int(data_id))
        key, episodes = self.episode_groups[int(data_id) % len(self.episode_groups)]
        pool = list(episodes)
        rng.shuffle(pool)
        selected = pool[: min(self.unique_episodes, len(pool))]
        duplicated = rng.sample(selected, min(self.duplicate_episodes, len(selected))) if selected else []
        sequence = selected + duplicated
        rng.shuffle(sequence)

        entries = []
        for episode_id, rows in sequence:
            row_pool = list(rows)
            rng.shuffle(row_pool)
            chosen_rows = row_pool[:1] if self.episode_chunk_mode == "random_chunk" else row_pool
            entries.append(
                {
                    "episode_index": episode_id,
                    "sample_ids": [row.get("sample_id") for row in chosen_rows],
                    "items": [self._process_row(row) for row in chosen_rows],
                }
            )
        return {
            "group_key": list(key),
            "sequence_episode_ids": [entry["episode_index"] for entry in entries],
            "sequence": entries,
        }


def build_ttte2e_dataset(args, runtime_config):
    if str(getattr(args, "ttte2e_task_mode", "support_query")) != "episode_sequence":
        return build_dataset(args, runtime_config)

    special_operator_map = {}
    if runtime_config["text_enabled"] and "prompt_emb" in runtime_config["data_file_keys"]:
        special_operator_map["prompt_emb"] = ResolvePromptEmbPath(base_path=args.dataset_base_path)

    with open(args.action_stat_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats

    if "action" in runtime_config["data_file_keys"]:
        special_operator_map["action"] = LoadCobotAction(
            base_path=args.dataset_base_path,
            action_type=args.action_type,
            stat=stat,
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            pad_short=args.pad_short_chunks,
            output_dim=args.action_dim,
        )

    return SequentialEpisodeMetaTaskDataset(
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
            pad_short=args.pad_short_chunks,
        ),
        special_operator_map=special_operator_map,
        seed=args.seed,
        group_keys=args.stage2_group_keys,
        tasks_per_epoch=args.stage2_tasks_per_epoch,
        unique_episodes=args.ttte2e_sequence_unique_episodes,
        duplicate_episodes=args.ttte2e_sequence_duplicate_episodes,
        episode_chunk_mode=args.ttte2e_sequence_episode_chunk_mode,
    )


class Stage2TTTE2ETrainingModule(WanTrainingModule):
    """
    Mild TTT-E2E analogue for BWM.

    Original TTT-E2E updates suffix-block MLP weights with the task loss and
    meta-trains their initialization through the unrolled inner updates. For the
    5B video model we keep the backbone frozen and use late-block low-rank
    residual adapters as the fast weights.
    """

    def __init__(self, *args, stage2_args: argparse.Namespace, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_steps = int(stage2_args.stage2_inner_steps)
        self.inner_lr = float(stage2_args.ttte2e_inner_lr)
        self.inner_lr_schedule = _parse_inner_lr_schedule(
            str(stage2_args.ttte2e_inner_lr_schedule),
            self.inner_steps,
            self.inner_lr,
        )
        self.inner_optimizer = str(stage2_args.ttte2e_inner_optimizer)
        self.inner_grad_clip = float(stage2_args.ttte2e_inner_grad_clip)
        self.inner_reg_weight = float(stage2_args.ttte2e_inner_reg_weight)
        self.outer_reg_weight = float(stage2_args.ttte2e_outer_reg_weight)
        self.outer_objective = str(stage2_args.ttte2e_outer_objective)
        self.relative_improve_weight = float(stage2_args.ttte2e_relative_improve_weight)
        self.inner_fix_support_noise = bool(stage2_args.ttte2e_inner_fix_support_noise)
        self.inner_converge = bool(stage2_args.ttte2e_inner_converge)
        self.inner_converge_min_steps = max(1, int(stage2_args.ttte2e_inner_converge_min_steps))
        self.inner_converge_patience = max(1, int(stage2_args.ttte2e_inner_converge_patience))
        self.inner_converge_rel_tol = float(stage2_args.ttte2e_inner_converge_rel_tol)
        self.inner_adam_beta1 = float(stage2_args.ttte2e_inner_adam_beta1)
        self.inner_adam_beta2 = float(stage2_args.ttte2e_inner_adam_beta2)
        self.inner_adam_eps = float(stage2_args.ttte2e_inner_adam_eps)
        self.second_order = bool(stage2_args.ttte2e_second_order)
        self.task_mode = str(stage2_args.ttte2e_task_mode)
        self.sequence_outer_timesteps = max(1, int(stage2_args.ttte2e_sequence_outer_timesteps))
        self.sequence_weight_start = float(stage2_args.ttte2e_sequence_weight_start)
        self.sequence_weight_end = float(stage2_args.ttte2e_sequence_weight_end)
        self.sequence_low_weight_count = max(0, int(stage2_args.ttte2e_sequence_low_weight_count))
        self.sequence_save_on_cpu = bool(stage2_args.ttte2e_sequence_save_on_cpu)
        self.sequence_streaming_backward = bool(stage2_args.ttte2e_sequence_streaming_backward)
        self.query_weight = float(stage2_args.stage2_query_weight)
        self.show_eval_weight = float(stage2_args.stage2_show_eval_weight)
        self.gap_weight = float(stage2_args.stage2_gap_weight)
        self.gap_margin = float(stage2_args.stage2_gap_margin)
        self.improvement_eps = float(stage2_args.stage2_improvement_eps)
        self.param_scope = str(stage2_args.ttte2e_inner_param_scope)
        self.outer_dit_layers = str(stage2_args.ttte2e_outer_dit_layers)
        self.last_metrics = {}
        self._last_inner_steps_used = 0
        self._last_inner_converged = False
        self._last_inner_start_loss = 0.0
        self._last_inner_final_loss = 0.0
        self._last_inner_rel_drop = 0.0
        self._last_inner_lr_used_last = 0.0

        if getattr(self.pipe, "physical_context_encoder", None) is not None:
            raise ValueError(
                "TTT-E2E mild branch must not use latent C. Set physical_context_mode='none'."
            )
        if getattr(self.pipe, "physical_adapter_bank", None) is None:
            raise ValueError(
                "TTT-E2E mild branch requires physical_adapter_mode='residual'."
            )
        if self.task_mode == "episode_sequence" and not self.second_order:
            print(
                "WARNING: episode_sequence is running in first-order mode. "
                "Later outer losses still see the fast-weight trajectory, but "
                "gradients through inner gradients are approximated.",
                flush=True,
            )

        if self.outer_dit_layers != "all":
            selected_blocks, total_blocks = _freeze_dit_except_layer_scope(
                self.pipe.dit,
                self.outer_dit_layers,
            )
            print(
                f"TTT-E2E outer DiT trainable scope: {self.outer_dit_layers} "
                f"({selected_blocks}/{total_blocks} blocks)",
                flush=True,
            )

        selected = []
        for name, param in self.pipe.physical_adapter_bank.named_parameters():
            is_selected = _adapter_param_allowed(name, self.param_scope)
            param.requires_grad_(is_selected)
            if is_selected:
                selected.append(name)
        if not selected:
            raise ValueError(f"No adapter parameters selected for scope={self.param_scope}.")
        self.adapter_param_names = tuple(selected)

    def _adapter_params(self) -> dict[str, torch.nn.Parameter]:
        return {
            name: param
            for name, param in self.pipe.physical_adapter_bank.named_parameters()
            if name in self.adapter_param_names
        }

    @staticmethod
    def _mean_stack(items: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(items).mean()

    def _adapter_reg(
        self,
        fast_params: dict[str, torch.Tensor],
        base_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for name, fast in fast_params.items():
            base = base_params[name].to(device=fast.device, dtype=fast.dtype)
            losses.append(torch.mean((fast.float() - base.float()) ** 2))
        return torch.stack(losses).mean()

    def _adapter_delta_norm(
        self,
        fast_params: dict[str, torch.Tensor],
        base_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = []
        for name, fast in fast_params.items():
            base = base_params[name].to(device=fast.device, dtype=fast.dtype)
            parts.append((fast.float() - base.float()).pow(2).sum())
        return torch.sqrt(torch.stack(parts).sum().clamp_min(1e-12))

    @staticmethod
    def _param_l2_norm(params: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [value.detach().float().pow(2).sum() for value in params.values()]
        return torch.sqrt(torch.stack(parts).sum().clamp_min(1e-12))

    @staticmethod
    def _gate_stats(params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = [value.detach().float().reshape(-1) for name, value in params.items() if name.endswith(".gate")]
        if not gates:
            device = next(iter(params.values())).device
            zero = torch.zeros((), device=device)
            return zero, zero, zero
        values = torch.cat(gates)
        return values.abs().mean(), values.abs().max(), values.mean()

    def _prepare_loss_inputs(self, data: dict):
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    @staticmethod
    def _shared_inputs(inputs):
        return inputs[0] if isinstance(inputs, tuple) else inputs

    def _sample_timestep(self, inputs: dict) -> torch.Tensor:
        inputs = self._shared_inputs(inputs)
        max_idx = int(inputs.get("max_timestep_boundary", 1) * len(self.pipe.scheduler.timesteps))
        min_idx = int(inputs.get("min_timestep_boundary", 0) * len(self.pipe.scheduler.timesteps))
        max_idx = max(min_idx + 1, max_idx)
        timestep_id = torch.randint(min_idx, max_idx, (1,))
        return self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _flow_match_loss_from_inputs(
        self,
        inputs,
        *,
        fast_adapter_params: dict[str, torch.Tensor] | None = None,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = dict(self._shared_inputs(inputs))
        if timestep is None:
            timestep = self._sample_timestep(inputs)
        if noise is None:
            noise = torch.randn_like(inputs["input_latents"])

        inputs["latents"] = self.pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        training_target = self.pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
        if "first_frame_latents" in inputs:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        if fast_adapter_params is not None:
            models["physical_adapter_bank"] = FunctionalAdapterBank(
                self.pipe.physical_adapter_bank,
                fast_adapter_params,
            )
        noise_pred = self.pipe.model_fn(**models, **inputs, timestep=timestep)

        if "first_frame_latents" in inputs:
            noise_pred = noise_pred[:, :, 1:]
            training_target = training_target[:, :, 1:]

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        return loss * self.pipe.scheduler.training_weight(timestep)

    def _sft_loss(
        self,
        data: dict,
        *,
        fast_adapter_params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        inputs = self._prepare_loss_inputs(data)
        return self._flow_match_loss_from_inputs(inputs, fast_adapter_params=fast_adapter_params)

    def _support_loss(
        self,
        support: list[dict],
        fast_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._mean_stack([self._sft_loss(item, fast_adapter_params=fast_params) for item in support])

    def _multi_timestep_loss(
        self,
        data: dict,
        *,
        fast_adapter_params: dict[str, torch.Tensor],
        timestep_samples: int,
    ) -> torch.Tensor:
        inputs = self._prepare_loss_inputs(data)
        return self._mean_stack(
            [
                self._flow_match_loss_from_inputs(inputs, fast_adapter_params=fast_adapter_params)
                for _ in range(max(1, int(timestep_samples)))
            ]
        )

    def _sequence_entry_outer_loss(
        self,
        items: list[dict],
        fast_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._mean_stack(
            [
                self._multi_timestep_loss(
                    item,
                    fast_adapter_params=fast_params,
                    timestep_samples=self.sequence_outer_timesteps,
                )
                for item in items
            ]
        )

    def _sequence_weight(self, index: int, length: int) -> float:
        if index < self.sequence_low_weight_count:
            return self.sequence_weight_start
        remaining = max(1, int(length) - self.sequence_low_weight_count - 1)
        progress = float(index - self.sequence_low_weight_count) / float(remaining)
        progress = min(1.0, max(0.0, progress))
        return self.sequence_weight_start + progress * (self.sequence_weight_end - self.sequence_weight_start)

    def _adapt_adapter_params_from(
        self,
        items: list[dict],
        fast_params: dict[str, torch.Tensor],
        base_params: dict[str, torch.nn.Parameter],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        last_inner_loss = None
        loss_history = []
        steps_used = 0
        adam_m: dict[str, torch.Tensor] = {}
        adam_v: dict[str, torch.Tensor] = {}
        if self.inner_optimizer == "adam":
            adam_m = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in fast_params.items()}
            adam_v = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in fast_params.items()}

        for step_idx in range(self.inner_steps):
            support_loss = self._support_loss(items, fast_params)
            inner_loss = support_loss
            if self.inner_reg_weight > 0:
                inner_loss = inner_loss + self.inner_reg_weight * self._adapter_reg(fast_params, base_params)
            loss_history.append(float(inner_loss.detach().float().cpu()))
            grads = torch.autograd.grad(
                inner_loss,
                tuple(fast_params.values()),
                create_graph=self.second_order,
                allow_unused=True,
            )
            grads = _clip_grad_tensors(grads, self.inner_grad_clip)
            next_params = {}
            lr = float(self.inner_lr_schedule[step_idx])
            for (name, value), grad in zip(fast_params.items(), grads):
                if grad is None:
                    grad = torch.zeros_like(value)
                if self.inner_optimizer == "adam":
                    grad_f = grad.float()
                    adam_m[name] = self.inner_adam_beta1 * adam_m[name] + (1.0 - self.inner_adam_beta1) * grad_f
                    adam_v[name] = self.inner_adam_beta2 * adam_v[name] + (1.0 - self.inner_adam_beta2) * grad_f.pow(2)
                    bias_correction1 = 1.0 - self.inner_adam_beta1 ** (step_idx + 1)
                    bias_correction2 = 1.0 - self.inner_adam_beta2 ** (step_idx + 1)
                    update = (adam_m[name] / bias_correction1) / (
                        (adam_v[name] / bias_correction2).sqrt() + self.inner_adam_eps
                    )
                    next_params[name] = (value.float() - lr * update).to(dtype=value.dtype)
                else:
                    next_params[name] = value - lr * grad
            fast_params = next_params
            last_inner_loss = inner_loss
            steps_used = step_idx + 1

        if last_inner_loss is None:
            last_inner_loss = torch.zeros((), device=self.pipe.device, dtype=torch.float32)
        if loss_history:
            self._last_inner_start_loss = loss_history[0]
            self._last_inner_final_loss = loss_history[-1]
            self._last_inner_rel_drop = (loss_history[0] - loss_history[-1]) / (abs(loss_history[0]) + 1e-8)
        else:
            self._last_inner_start_loss = 0.0
            self._last_inner_final_loss = float(last_inner_loss.detach().float().cpu())
            self._last_inner_rel_drop = 0.0
        self._last_inner_steps_used = steps_used
        self._last_inner_converged = False
        self._last_inner_lr_used_last = float(self.inner_lr_schedule[max(0, min(steps_used, len(self.inner_lr_schedule)) - 1)])
        return fast_params, last_inner_loss

    def _prepare_support_objective(self, support: list[dict]) -> list[tuple[object, torch.Tensor, torch.Tensor]]:
        objective = []
        with torch.no_grad():
            for item in support:
                inputs = self._prepare_loss_inputs(item)
                timestep = self._sample_timestep(inputs)
                noise = torch.randn_like(self._shared_inputs(inputs)["input_latents"])
                objective.append((inputs, timestep, noise))
        return objective

    def _support_objective_loss(
        self,
        objective: list[tuple[object, torch.Tensor, torch.Tensor]],
        fast_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._mean_stack(
            [
                self._flow_match_loss_from_inputs(
                    inputs,
                    fast_adapter_params=fast_params,
                    timestep=timestep,
                    noise=noise,
                )
                for inputs, timestep, noise in objective
            ]
        )

    def _adapt_adapter_params(
        self,
        support: list[dict],
        base_params: dict[str, torch.nn.Parameter],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        fast_params: dict[str, torch.Tensor] = {name: param for name, param in base_params.items()}
        last_inner_loss = None
        support_objective = self._prepare_support_objective(support) if self.inner_fix_support_noise else None
        loss_history = []
        steps_used = 0
        converged = False
        adam_m: dict[str, torch.Tensor] = {}
        adam_v: dict[str, torch.Tensor] = {}
        if self.inner_optimizer == "adam":
            adam_m = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in fast_params.items()}
            adam_v = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in fast_params.items()}
        for step_idx in range(self.inner_steps):
            if support_objective is None:
                support_loss = self._support_loss(support, fast_params)
            else:
                support_loss = self._support_objective_loss(support_objective, fast_params)
            inner_loss = support_loss
            if self.inner_reg_weight > 0:
                inner_loss = inner_loss + self.inner_reg_weight * self._adapter_reg(fast_params, base_params)
            loss_history.append(float(inner_loss.detach().float().cpu()))
            grads = torch.autograd.grad(
                inner_loss,
                tuple(fast_params.values()),
                create_graph=self.second_order,
                allow_unused=True,
            )
            grads = _clip_grad_tensors(grads, self.inner_grad_clip)
            next_params = {}
            lr = float(self.inner_lr_schedule[step_idx])
            for (name, value), grad in zip(fast_params.items(), grads):
                if grad is None:
                    grad = torch.zeros_like(value)
                if self.inner_optimizer == "adam":
                    grad_f = grad.float()
                    adam_m[name] = self.inner_adam_beta1 * adam_m[name] + (1.0 - self.inner_adam_beta1) * grad_f
                    adam_v[name] = self.inner_adam_beta2 * adam_v[name] + (1.0 - self.inner_adam_beta2) * grad_f.pow(2)
                    bias_correction1 = 1.0 - self.inner_adam_beta1 ** (step_idx + 1)
                    bias_correction2 = 1.0 - self.inner_adam_beta2 ** (step_idx + 1)
                    update = (adam_m[name] / bias_correction1) / (
                        (adam_v[name] / bias_correction2).sqrt() + self.inner_adam_eps
                    )
                    next_params[name] = (value.float() - lr * update).to(dtype=value.dtype)
                else:
                    next_params[name] = value - lr * grad
            fast_params = next_params
            last_inner_loss = inner_loss
            steps_used = step_idx + 1
            if self.inner_converge and steps_used >= self.inner_converge_min_steps:
                if len(loss_history) > self.inner_converge_patience:
                    previous = loss_history[-self.inner_converge_patience - 1]
                    current = loss_history[-1]
                    rel_drop = (previous - current) / (abs(previous) + 1e-8)
                    if rel_drop < self.inner_converge_rel_tol:
                        converged = True
                        break
        if last_inner_loss is None:
            last_inner_loss = torch.zeros((), device=self.pipe.device, dtype=torch.float32)
        if loss_history:
            self._last_inner_start_loss = loss_history[0]
            self._last_inner_final_loss = loss_history[-1]
            self._last_inner_rel_drop = (loss_history[0] - loss_history[-1]) / (abs(loss_history[0]) + 1e-8)
        else:
            self._last_inner_start_loss = 0.0
            self._last_inner_final_loss = float(last_inner_loss.detach().float().cpu())
            self._last_inner_rel_drop = 0.0
        self._last_inner_steps_used = steps_used
        self._last_inner_converged = converged
        self._last_inner_lr_used_last = float(self.inner_lr_schedule[max(0, min(steps_used, len(self.inner_lr_schedule)) - 1)])
        return fast_params, last_inner_loss

    def _paired_baseline_adapted_loss(
        self,
        data: dict,
        fast_params: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            baseline_inputs = self._prepare_loss_inputs(data)
            timestep = self._sample_timestep(baseline_inputs)
            noise = torch.randn_like(self._shared_inputs(baseline_inputs)["input_latents"])
            baseline_loss = self._flow_match_loss_from_inputs(
                baseline_inputs,
                fast_adapter_params=None,
                timestep=timestep,
                noise=noise,
            )
        adapted_inputs = self._prepare_loss_inputs(data)
        adapted_loss = self._flow_match_loss_from_inputs(
            adapted_inputs,
            fast_adapter_params=fast_params,
            timestep=timestep,
            noise=noise,
        )
        return baseline_loss.detach(), adapted_loss

    def _forward_episode_sequence_impl(self, task: dict) -> torch.Tensor:
        sequence = task["sequence"]
        if not sequence:
            raise ValueError("episode_sequence task is empty.")

        base_params = self._adapter_params()
        fast_params: dict[str, torch.Tensor] = {name: param for name, param in base_params.items()}
        weighted_losses = []
        weights = []
        raw_losses = []
        inner_losses = []

        for index, entry in enumerate(sequence):
            items = entry["items"]
            outer_loss = self._sequence_entry_outer_loss(items, fast_params)
            weight = self._sequence_weight(index, len(sequence))
            weighted_losses.append(outer_loss * weight)
            weights.append(weight)
            raw_losses.append(outer_loss.detach())
            fast_params, inner_loss = self._adapt_adapter_params_from(items, fast_params, base_params)
            inner_losses.append(inner_loss.detach())

        weight_sum = max(1e-8, float(sum(weights)))
        sequence_loss = torch.stack(weighted_losses).sum() / weight_sum
        adapter_reg = self._adapter_reg(fast_params, base_params)
        adapter_delta_norm = self._adapter_delta_norm(fast_params, base_params)
        base_param_norm = self._param_l2_norm(base_params)
        base_gate_abs_mean, base_gate_abs_max, base_gate_mean = self._gate_stats(base_params)
        fast_gate_abs_mean, fast_gate_abs_max, fast_gate_mean = self._gate_stats(fast_params)

        loss = sequence_loss
        if self.outer_reg_weight > 0:
            loss = loss + self.outer_reg_weight * adapter_reg

        raw_stack = torch.stack(raw_losses)
        inner_stack = torch.stack(inner_losses) if inner_losses else torch.zeros((1,), device=self.pipe.device)
        split = min(self.sequence_low_weight_count, raw_stack.numel())
        first_loss = raw_stack[:split].mean() if split > 0 else raw_stack.mean()
        later_loss = raw_stack[split:].mean() if split < raw_stack.numel() else raw_stack.mean()

        self.last_metrics = {
            "loss": float(loss.detach().float().cpu()),
            "seq_outer": float(sequence_loss.detach().float().cpu()),
            "seq_first": float(first_loss.detach().float().cpu()),
            "seq_later": float(later_loss.detach().float().cpu()),
            "seq_len": float(len(sequence)),
            "seq_weight_first": float(weights[0]),
            "seq_weight_last": float(weights[-1]),
            "outer_time_samples": float(self.sequence_outer_timesteps),
            "query": float(sequence_loss.detach().float().cpu()),
            "query_base": 0.0,
            "show": float(first_loss.detach().float().cpu()),
            "show_base": 0.0,
            "gap": 0.0,
            "rel_show_imp": 0.0,
            "rel_query_imp": 0.0,
            "inner": float(inner_stack.mean().float().cpu()),
            "inner_lr_first": float(self.inner_lr_schedule[0]),
            "inner_lr_last": float(self.inner_lr_schedule[-1]),
            "inner_lr_used_last": self._last_inner_lr_used_last,
            "inner_steps_used": float(self._last_inner_steps_used),
            "inner_converged": 0.0,
            "relative_objective": 0.0,
            "hybrid_objective": 0.0,
            "relative_improve_weight": self.relative_improve_weight,
            "inner_start": self._last_inner_start_loss,
            "inner_final": self._last_inner_final_loss,
            "inner_rel_drop": self._last_inner_rel_drop,
            "adapter_reg": float(adapter_reg.detach().float().cpu()),
            "adapter_delta_norm": float(adapter_delta_norm.detach().float().cpu()),
            "adapter_delta_rel": float((adapter_delta_norm.detach() / base_param_norm.clamp_min(1e-12)).float().cpu()),
            "base_gate_abs_mean": float(base_gate_abs_mean.float().cpu()),
            "base_gate_abs_max": float(base_gate_abs_max.float().cpu()),
            "base_gate_mean": float(base_gate_mean.float().cpu()),
            "fast_gate_abs_mean": float(fast_gate_abs_mean.float().cpu()),
            "fast_gate_abs_max": float(fast_gate_abs_max.float().cpu()),
            "fast_gate_mean": float(fast_gate_mean.float().cpu()),
        }
        return loss

    def _forward_episode_sequence(self, task: dict) -> torch.Tensor:
        if self.sequence_save_on_cpu:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return self._forward_episode_sequence_impl(task)
        return self._forward_episode_sequence_impl(task)

    def backward_episode_sequence(self, task: dict) -> torch.Tensor:
        sequence = task["sequence"]
        if not sequence:
            raise ValueError("episode_sequence task is empty.")

        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if self.sequence_save_on_cpu else nullcontext()
        with context:
            base_params = self._adapter_params()
            fast_params: dict[str, torch.Tensor] = {name: param for name, param in base_params.items()}
            weights = [self._sequence_weight(index, len(sequence)) for index in range(len(sequence))]
            weight_sum = max(1e-8, float(sum(weights)))
            raw_losses = []
            inner_losses = []

            for index, entry in enumerate(sequence):
                items = entry["items"]
                outer_loss = self._sequence_entry_outer_loss(items, fast_params)
                loss_part = outer_loss * (float(weights[index]) / weight_sum)
                loss_part.backward(retain_graph=True)
                raw_losses.append(outer_loss.detach())
                del loss_part, outer_loss
                fast_params, inner_loss = self._adapt_adapter_params_from(items, fast_params, base_params)
                inner_losses.append(inner_loss.detach())

            raw_stack = torch.stack(raw_losses)
            sequence_loss = torch.sum(
                torch.stack(
                    [
                        raw_stack[index] * (float(weights[index]) / weight_sum)
                        for index in range(raw_stack.numel())
                    ]
                )
            )
            adapter_reg = self._adapter_reg(fast_params, base_params)
            adapter_delta_norm = self._adapter_delta_norm(fast_params, base_params)
            base_param_norm = self._param_l2_norm(base_params)
            base_gate_abs_mean, base_gate_abs_max, base_gate_mean = self._gate_stats(base_params)
            fast_gate_abs_mean, fast_gate_abs_max, fast_gate_mean = self._gate_stats(fast_params)

            loss_for_log = sequence_loss.detach()
            if self.outer_reg_weight > 0:
                reg_loss = self.outer_reg_weight * adapter_reg
                reg_loss.backward()
                loss_for_log = loss_for_log + reg_loss.detach()

            inner_stack = torch.stack(inner_losses) if inner_losses else torch.zeros((1,), device=self.pipe.device)
            split = min(self.sequence_low_weight_count, raw_stack.numel())
            first_loss = raw_stack[:split].mean() if split > 0 else raw_stack.mean()
            later_loss = raw_stack[split:].mean() if split < raw_stack.numel() else raw_stack.mean()

            self.last_metrics = {
                "loss": float(loss_for_log.float().cpu()),
                "seq_outer": float(sequence_loss.float().cpu()),
                "seq_first": float(first_loss.float().cpu()),
                "seq_later": float(later_loss.float().cpu()),
                "seq_len": float(len(sequence)),
                "seq_weight_first": float(weights[0]),
                "seq_weight_last": float(weights[-1]),
                "outer_time_samples": float(self.sequence_outer_timesteps),
                "query": float(sequence_loss.float().cpu()),
                "query_base": 0.0,
                "show": float(first_loss.float().cpu()),
                "show_base": 0.0,
                "gap": 0.0,
                "rel_show_imp": 0.0,
                "rel_query_imp": 0.0,
                "inner": float(inner_stack.mean().float().cpu()),
                "inner_lr_first": float(self.inner_lr_schedule[0]),
                "inner_lr_last": float(self.inner_lr_schedule[-1]),
                "inner_lr_used_last": self._last_inner_lr_used_last,
                "inner_steps_used": float(self._last_inner_steps_used),
                "inner_converged": 0.0,
                "relative_objective": 0.0,
                "hybrid_objective": 0.0,
                "relative_improve_weight": self.relative_improve_weight,
                "inner_start": self._last_inner_start_loss,
                "inner_final": self._last_inner_final_loss,
                "inner_rel_drop": self._last_inner_rel_drop,
                "adapter_reg": float(adapter_reg.detach().float().cpu()),
                "adapter_delta_norm": float(adapter_delta_norm.detach().float().cpu()),
                "adapter_delta_rel": float((adapter_delta_norm.detach() / base_param_norm.clamp_min(1e-12)).float().cpu()),
                "base_gate_abs_mean": float(base_gate_abs_mean.float().cpu()),
                "base_gate_abs_max": float(base_gate_abs_max.float().cpu()),
                "base_gate_mean": float(base_gate_mean.float().cpu()),
                "fast_gate_abs_mean": float(fast_gate_abs_mean.float().cpu()),
                "fast_gate_abs_max": float(fast_gate_abs_max.float().cpu()),
                "fast_gate_mean": float(fast_gate_mean.float().cpu()),
            }
            return loss_for_log

    def forward(self, task: dict) -> torch.Tensor:
        if self.task_mode == "episode_sequence":
            return self._forward_episode_sequence(task)

        support = task["support"]
        query = task["query"]
        base_params = self._adapter_params()
        fast_params, inner_loss = self._adapt_adapter_params(support, base_params)

        show_baseline = []
        show_adapted = []
        for item in support:
            baseline_loss, adapted_loss = self._paired_baseline_adapted_loss(item, fast_params)
            show_baseline.append(baseline_loss)
            show_adapted.append(adapted_loss)

        query_baseline = []
        query_adapted = []
        for item in query:
            baseline_loss, adapted_loss = self._paired_baseline_adapted_loss(item, fast_params)
            query_baseline.append(baseline_loss)
            query_adapted.append(adapted_loss)

        show_baseline_loss = self._mean_stack(show_baseline)
        show_adapted_loss = self._mean_stack(show_adapted)
        query_baseline_loss = self._mean_stack(query_baseline)
        query_adapted_loss = self._mean_stack(query_adapted)

        rel_show_imp = (show_baseline_loss - show_adapted_loss.detach()) / (
            show_baseline_loss.abs() + self.improvement_eps
        )
        rel_query_imp = (query_baseline_loss - query_adapted_loss) / (
            query_baseline_loss.abs() + self.improvement_eps
        )
        rel_show_imp_for_loss = (show_baseline_loss.detach() - show_adapted_loss) / (
            show_baseline_loss.detach().abs() + self.improvement_eps
        )
        rel_query_imp_for_loss = (query_baseline_loss.detach() - query_adapted_loss) / (
            query_baseline_loss.detach().abs() + self.improvement_eps
        )
        gap_loss = torch.relu(rel_show_imp.detach() - rel_query_imp - self.gap_margin) ** 2
        adapter_reg = self._adapter_reg(fast_params, base_params)
        adapter_delta_norm = self._adapter_delta_norm(fast_params, base_params)
        base_param_norm = self._param_l2_norm(base_params)
        base_gate_abs_mean, base_gate_abs_max, base_gate_mean = self._gate_stats(base_params)
        fast_gate_abs_mean, fast_gate_abs_max, fast_gate_mean = self._gate_stats(fast_params)

        if self.outer_objective == "relative_improve":
            loss = -self.query_weight * rel_query_imp_for_loss
            if self.show_eval_weight > 0:
                loss = loss - self.show_eval_weight * rel_show_imp_for_loss
        elif self.outer_objective == "hybrid_loss_relative_improve":
            loss = self.query_weight * query_adapted_loss
            if self.show_eval_weight > 0:
                loss = loss + self.show_eval_weight * show_adapted_loss
            loss = loss - self.relative_improve_weight * (
                self.query_weight * rel_query_imp_for_loss
                + self.show_eval_weight * rel_show_imp_for_loss
            )
        else:
            loss = self.query_weight * query_adapted_loss
            if self.show_eval_weight > 0:
                loss = loss + self.show_eval_weight * show_adapted_loss
        if self.gap_weight > 0:
            loss = loss + self.gap_weight * gap_loss
        if self.outer_reg_weight > 0:
            loss = loss + self.outer_reg_weight * adapter_reg

        self.last_metrics = {
            "loss": float(loss.detach().float().cpu()),
            "query_base": float(query_baseline_loss.detach().float().cpu()),
            "query": float(query_adapted_loss.detach().float().cpu()),
            "show_base": float(show_baseline_loss.detach().float().cpu()),
            "show": float(show_adapted_loss.detach().float().cpu()),
            "gap": float(gap_loss.detach().float().cpu()),
            "rel_show_imp": float(rel_show_imp.detach().float().cpu()),
            "rel_query_imp": float(rel_query_imp.detach().float().cpu()),
            "inner": float(inner_loss.detach().float().cpu()),
            "inner_lr_first": float(self.inner_lr_schedule[0]),
            "inner_lr_last": float(self.inner_lr_schedule[-1]),
            "inner_lr_used_last": self._last_inner_lr_used_last,
            "inner_steps_used": float(self._last_inner_steps_used),
            "inner_converged": float(self._last_inner_converged),
            "relative_objective": float(self.outer_objective == "relative_improve"),
            "hybrid_objective": float(self.outer_objective == "hybrid_loss_relative_improve"),
            "relative_improve_weight": self.relative_improve_weight,
            "inner_start": self._last_inner_start_loss,
            "inner_final": self._last_inner_final_loss,
            "inner_rel_drop": self._last_inner_rel_drop,
            "adapter_reg": float(adapter_reg.detach().float().cpu()),
            "adapter_delta_norm": float(adapter_delta_norm.detach().float().cpu()),
            "adapter_delta_rel": float((adapter_delta_norm.detach() / base_param_norm.clamp_min(1e-12)).float().cpu()),
            "base_gate_abs_mean": float(base_gate_abs_mean.float().cpu()),
            "base_gate_abs_max": float(base_gate_abs_max.float().cpu()),
            "base_gate_mean": float(base_gate_mean.float().cpu()),
            "fast_gate_abs_mean": float(fast_gate_abs_mean.float().cpu()),
            "fast_gate_abs_max": float(fast_gate_abs_max.float().cpu()),
            "fast_gate_mean": float(fast_gate_mean.float().cpu()),
        }
        return loss


def add_ttte2e_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("stage2_ttte2e")
    group.add_argument("--ttte2e_task_mode", type=str, default="support_query", choices=TTTE2E_TASK_MODES)
    group.add_argument("--ttte2e_inner_param_scope", type=str, default="lowrank", choices=TTTE2E_PARAM_SCOPES)
    group.add_argument("--ttte2e_inner_lr", type=float, default=1e-3)
    group.add_argument(
        "--ttte2e_inner_lr_schedule",
        type=str,
        default="",
        help='Optional comma schedule for inner LR, e.g. "5@0.12,5@0.04" or ten comma-separated LR values.',
    )
    group.add_argument("--ttte2e_inner_optimizer", type=str, default="sgd", choices=TTTE2E_INNER_OPTIMIZERS)
    group.add_argument("--ttte2e_inner_grad_clip", type=float, default=0.1)
    group.add_argument("--ttte2e_inner_reg_weight", type=float, default=1e-4)
    group.add_argument("--ttte2e_outer_reg_weight", type=float, default=1e-4)
    group.add_argument("--ttte2e_outer_objective", type=str, default="adapted_loss", choices=TTTE2E_OUTER_OBJECTIVES)
    group.add_argument("--ttte2e_relative_improve_weight", type=float, default=1.0)
    group.add_argument(
        "--ttte2e_inner_fix_support_noise",
        action="store_true",
        default=False,
        help="Reuse one fixed timestep/noise draw per support sample during the inner loop.",
    )
    group.add_argument(
        "--ttte2e_inner_converge",
        action="store_true",
        default=False,
        help="Stop the inner loop early when fixed support objective improvement has plateaued.",
    )
    group.add_argument("--ttte2e_inner_converge_min_steps", type=int, default=10)
    group.add_argument("--ttte2e_inner_converge_patience", type=int, default=5)
    group.add_argument("--ttte2e_inner_converge_rel_tol", type=float, default=1e-4)
    group.add_argument("--ttte2e_inner_adam_beta1", type=float, default=0.9)
    group.add_argument("--ttte2e_inner_adam_beta2", type=float, default=0.999)
    group.add_argument("--ttte2e_inner_adam_eps", type=float, default=1e-8)
    group.add_argument("--ttte2e_sequence_unique_episodes", type=int, default=10)
    group.add_argument("--ttte2e_sequence_duplicate_episodes", type=int, default=5)
    group.add_argument("--ttte2e_sequence_episode_chunk_mode", type=str, default="random_chunk", choices=TTTE2E_SEQUENCE_CHUNK_MODES)
    group.add_argument("--ttte2e_sequence_outer_timesteps", type=int, default=1)
    group.add_argument("--ttte2e_sequence_low_weight_count", type=int, default=3)
    group.add_argument("--ttte2e_sequence_weight_start", type=float, default=0.4)
    group.add_argument("--ttte2e_sequence_weight_end", type=float, default=1.0)
    group.add_argument(
        "--ttte2e_sequence_save_on_cpu",
        action="store_true",
        default=False,
        help="Offload saved autograd tensors to CPU in episode_sequence mode to keep exact second-order gradients under GPU memory limits.",
    )
    group.add_argument(
        "--ttte2e_sequence_streaming_backward",
        action="store_true",
        default=False,
        help="In episode_sequence mode, backward each episode outer loss immediately and keep only the fast-weight trajectory graph.",
    )
    group.add_argument(
        "--ttte2e_outer_dit_layers",
        type=str,
        default="none",
        choices=TTTE2E_OUTER_DIT_LAYER_SCOPES,
        help="Optional DiT parameter subset updated by the outer loop. Inner loop still only adapts adapter fast weights.",
    )
    group.add_argument(
        "--ttte2e_second_order",
        action="store_true",
        default=False,
        help="Use second-order gradients through inner updates. Default is first-order for memory.",
    )
    return parser


def launch_stage2_ttte2e_training(accelerator, dataset, model, model_logger, args):
    streaming_backward = (
        str(getattr(args, "ttte2e_task_mode", "support_query")) == "episode_sequence"
        and bool(getattr(args, "ttte2e_sequence_streaming_backward", False))
    )
    if streaming_backward:
        if accelerator.num_processes != 1:
            raise ValueError("ttte2e_sequence_streaming_backward currently requires one process/GPU.")
        if int(args.gradient_accumulation_steps) != 1:
            raise ValueError("ttte2e_sequence_streaming_backward requires gradient_accumulation_steps=1.")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(getattr(args, "stage2_outer_warmup_steps", 0) or 0)
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda items: items[0],
        num_workers=args.dataset_num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    for epoch_id in range(args.num_epochs):
        iterator = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for task in iterator:
            with accelerator.accumulate(model):
                if streaming_backward:
                    loss = accelerator.unwrap_model(model).backward_episode_sequence(task)
                else:
                    loss = model(task)
                    accelerator.backward(loss)
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                if accelerator.is_main_process and model_logger.log_steps > 0 and model_logger.num_steps % model_logger.log_steps == 0:
                    metrics = accelerator.unwrap_model(model).last_metrics
                    print(
                        "[stage2_ttte2e] "
                        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                        flush=True,
                    )
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)


def main() -> None:
    parser = add_ttte2e_config(add_stage2_config(wan_parser()))
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)

    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    loggers = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}", False)]
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=loggers or None,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = build_ttte2e_dataset(args, runtime_config)
    model = Stage2TTTE2ETrainingModule(
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
        stage2_args=args,
    )
    if accelerator.is_main_process:
        names = sorted(model.trainable_param_names())
        print(f"TTT-E2E mild trainable parameter tensors: {len(names)}", flush=True)
        for name in names[:80]:
            print(f"  {name}", flush=True)
        if len(names) > 80:
            print(f"  ... {len(names) - 80} more", flush=True)

    model_logger = TimedRetentionModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_minutes=args.checkpoint_save_minutes,
        keep_last=args.checkpoint_keep_last,
        log_steps=args.log_steps,
    )
    launch_stage2_ttte2e_training(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
