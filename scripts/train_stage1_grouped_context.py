#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
        init_value: float,
        init_std: float,
    ):
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}.")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        values = torch.tensor([float(value) for value in friction_values], dtype=torch.float32)
        self.register_buffer("friction_values", values, persistent=True)
        contexts = torch.full(
            (len(friction_values), int(num_tokens), int(context_dim)),
            float(init_value),
            dtype=torch.float32,
        )
        if init_std > 0:
            contexts.normal_(mean=float(init_value), std=float(init_std))
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
            init_value=float(grouped_args.grouped_context_init_value),
            init_std=float(grouped_args.grouped_context_init_std),
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
        super().save_model(accelerator, model, file_name)
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        table = getattr(unwrapped, "friction_context_table", None)
        if table is None:
            return
        path = os.path.join(self.output_path, file_name.replace(".safetensors", ".context_table.json"))
        payload = {
            "num_groups": int(table.friction_values.numel()),
            "context_shape": list(table.contexts.shape),
            "records": table.to_records(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[checkpoint] saved {path}", flush=True)


def add_grouped_context_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("grouped_context_stage1")
    group.add_argument("--grouped_context_init_value", type=float, default=0.5)
    group.add_argument("--grouped_context_init_std", type=float, default=0.02)
    group.add_argument("--grouped_context_clamp_min", type=float, default=0.0)
    group.add_argument("--grouped_context_clamp_max", type=float, default=1.0)
    return parser


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
        )
    return dataset


def launch_grouped_stage1(accelerator, dataset, model, model_logger, args):
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
    launch_grouped_stage1(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
