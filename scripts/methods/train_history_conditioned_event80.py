#!/usr/bin/env python3
"""Flamingo-style K={1,2} in-context baseline training on Event80."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json

import accelerate
from accelerate.utils import InitProcessGroupKwargs
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from scripts.train import TimedRetentionModelLogger, WanTrainingModule, wan_parser
from scripts.train_stage1_grouped_context import build_dataset
from wan_video_action.methods.baselines.history_conditioned import (
    Event80HistorySampler,
    FlamingoHistoryTrainingModule,
    FlamingoSupportEncoder,
    install_flamingo_history,
)
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


def add_history_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("flamingo_history_context")
    group.add_argument("--history_dinov2_model_path", type=str, required=False)
    group.add_argument("--history_active_environment_manifest", type=str, required=False)
    group.add_argument("--history_support_sizes", type=str, default="1,2")
    group.add_argument("--history_chunks_per_environment", type=int, default=6)
    group.add_argument("--history_environments_per_rank", type=int, default=1)
    group.add_argument("--history_sampled_frames", type=int, default=8)
    group.add_argument("--history_memory_dim", type=int, default=1536)
    group.add_argument("--history_num_latents", type=int, default=64)
    group.add_argument("--history_resampler_layers", type=int, default=6)
    group.add_argument("--history_heads", type=int, default=16)
    group.add_argument("--history_insertion_frequency", type=int, default=4)
    group.add_argument("--history_module_learning_rate", type=float, default=1e-4)
    group.add_argument("--history_gate_learning_rate", type=float, default=1e-3)
    group.add_argument("--history_max_updates", type=int, default=50000)
    return parser


def parse_support_sizes(value) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else str(value).split(",")
    parsed = tuple(sorted({int(item) for item in values}))
    if not parsed:
        raise ValueError("history_support_sizes cannot be empty.")
    return parsed


def main() -> None:
    parser = add_history_config(wan_parser())
    parser.add_argument("--frame_stride", type=int, default=1)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if not args.history_dinov2_model_path or not args.history_active_environment_manifest:
        raise ValueError("DINOv2 path and active-environment manifest are required.")
    support_sizes = parse_support_sizes(args.history_support_sizes)

    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            ),
            InitProcessGroupKwargs(timeout=timedelta(hours=1)),
        ],
    )
    dataset = build_dataset(args, runtime_config)
    sampler = Event80HistorySampler(
        metadata_path=args.dataset_metadata_path,
        active_environment_manifest=args.history_active_environment_manifest,
        seed=args.seed,
        support_sizes=support_sizes,
        chunks_per_environment=args.history_chunks_per_environment,
    )
    wan = WanTrainingModule(
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
    )
    wan.requires_grad_(False)
    support_encoder = FlamingoSupportEncoder(
        model_path=args.history_dinov2_model_path,
        action_dim=args.action_dim,
        sampled_frames=args.history_sampled_frames,
        memory_dim=args.history_memory_dim,
        num_latents=args.history_num_latents,
        resampler_layers=args.history_resampler_layers,
        heads=args.history_heads,
        max_support_trajectories=max(support_sizes),
    )
    installation = install_flamingo_history(
        wan.pipe.dit,
        memory_dim=args.history_memory_dim,
        heads=args.history_heads,
        insertion_frequency=args.history_insertion_frequency,
    )
    model = FlamingoHistoryTrainingModule(
        wan=wan,
        support_encoder=support_encoder,
        installation=installation,
    )
    support_parameters = [
        parameter
        for name, parameter in model.support_encoder.named_parameters()
        if parameter.requires_grad and not name.startswith("dino.")
    ]
    gate_parameters = []
    adapter_parameters = []
    for name, parameter in installation.adapters.named_parameters():
        if name.endswith("alpha_xattn") or name.endswith("alpha_dense"):
            gate_parameters.append(parameter)
        else:
            adapter_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "support_resampler",
                "params": support_parameters,
                "lr": float(args.history_module_learning_rate),
                "weight_decay": 0.0,
            },
            {
                "name": "gated_xattn_dense",
                "params": adapter_parameters,
                "lr": float(args.history_module_learning_rate),
                "weight_decay": 0.1,
            },
            {
                "name": "zero_gates",
                "params": gate_parameters,
                "lr": float(args.history_gate_learning_rate),
                "weight_decay": 0.0,
            },
        ]
    )
    warmup_steps = int(getattr(args, "stage1_warmup_steps", 100) or 0)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            min(1.0, float(step + 1) / float(warmup_steps)) if warmup_steps else 1.0
        ),
    )
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    model_logger = TimedRetentionModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_minutes=args.checkpoint_save_minutes,
        keep_last=args.checkpoint_keep_last,
        log_steps=args.log_steps,
    )
    if accelerator.is_main_process:
        print(
            "[flamingo_icl] "
            f"support_sizes={support_sizes} chunks_per_environment="
            f"{args.history_chunks_per_environment} global_environments="
            f"{accelerator.num_processes * args.history_environments_per_rank} "
            f"memory={args.history_num_latents}x{args.history_memory_dim} "
            f"insert_every={args.history_insertion_frequency} "
            f"blocks={installation.block_indices} wan_frozen=true dino_frozen=true",
            flush=True,
        )

    iterator = tqdm(
        range(int(args.history_max_updates)),
        disable=not accelerator.is_local_main_process,
    )
    for update_index in iterator:
        step = update_index + 1
        episodes = sampler.sample(
            step=step,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            environments_per_rank=args.history_environments_per_rank,
        )
        optimizer.zero_grad(set_to_none=True)
        total_queries = sum(len(episode.query_indices) for episode in episodes)
        detached_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        query_counter = 0
        sampled_k = []
        for episode in episodes:
            supports = [dataset[index] for index in episode.support_indices]
            sampled_k.append(len(supports))
            unwrapped = accelerator.unwrap_model(model)
            support_visual_features = []
            support_frame_indices = []
            support_frame_counts = []
            for support in supports:
                visual, frame_indices, frame_count = unwrapped.extract_support_visual(
                    support["video"]
                )
                support_visual_features.append(visual)
                support_frame_indices.append(frame_indices)
                support_frame_counts.append(frame_count)
            support_actions = [support["action"] for support in supports]
            for query_index in episode.query_indices:
                query_counter += 1
                sync_context = (
                    accelerator.no_sync(model)
                    if query_counter < total_queries
                    else nullcontext()
                )
                with sync_context:
                    loss = model(
                        query_data=dataset[query_index],
                        support_visual_features=support_visual_features,
                        support_actions=support_actions,
                        support_frame_indices=support_frame_indices,
                        support_frame_counts=support_frame_counts,
                    )
                    accelerator.backward(loss / float(total_queries))
                accelerator.unwrap_model(model).clear_memory()
                detached_loss += loss.detach().float() / float(total_queries)
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if step % int(args.log_steps) == 0 and accelerator.is_main_process:
            stats = accelerator.unwrap_model(model).gate_statistics()
            print(
                "[flamingo_metrics] "
                f"step={step} loss={float(detached_loss.item()):.6f} "
                f"K={sampled_k} gate_abs_mean={stats['gate_abs_mean']:.8f} "
                f"gate_abs_max={stats['gate_abs_max']:.8f}",
                flush=True,
            )
        model_logger.on_step_end(
            accelerator,
            model,
            args.save_steps,
            loss=detached_loss,
        )
    model_logger.on_training_end(accelerator, model, args.save_steps)


if __name__ == "__main__":
    main()
