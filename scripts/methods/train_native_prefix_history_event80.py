#!/usr/bin/env python3
"""Train the Event80 native prefix-history baseline."""

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
from wan_video_action.methods.baselines.dinov2_amortized import Event80K1Sampler
from wan_video_action.methods.baselines.native_prefix_history import (
    NativePrefixTrainingModule,
    build_event80_prefix_pair,
    install_prefix_segments,
)
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


def add_prefix_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("native_prefix_history")
    group.add_argument("--prefix_active_environment_manifest", type=str, required=False)
    group.add_argument("--prefix_environments_per_rank", type=int, default=3)
    group.add_argument("--prefix_queries_per_environment", type=int, default=5)
    group.add_argument("--prefix_num_history_frames", type=int, default=45)
    group.add_argument("--prefix_segment_learning_rate", type=float, default=1e-4)
    group.add_argument("--prefix_max_updates", type=int, default=50000)
    return parser


def main() -> None:
    parser = add_prefix_config(wan_parser())
    parser.add_argument("--frame_stride", type=int, default=1)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if not args.prefix_active_environment_manifest:
        raise ValueError("prefix_active_environment_manifest is required.")

    set_global_seed(args.seed)
    runtime = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(hours=1)),
        ],
    )
    dataset = build_dataset(args, runtime)
    sampler = Event80K1Sampler(
        metadata_path=args.dataset_metadata_path,
        active_environment_manifest=args.prefix_active_environment_manifest,
        seed=args.seed,
        queries_per_environment=args.prefix_queries_per_environment,
    )
    wan = WanTrainingModule(
        model_paths=json.dumps(runtime["model_paths_list"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=runtime["tokenizer_path"],
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
        modules=runtime["modules"],
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.prefix_num_history_frames,
        args=args,
    )
    installation = install_prefix_segments(wan.pipe.dit)
    model = NativePrefixTrainingModule(wan)
    segment_ids = {id(parameter) for parameter in installation.module.parameters()}
    wan_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in segment_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": wan_parameters, "lr": float(args.learning_rate), "weight_decay": args.weight_decay},
            {"params": list(installation.module.parameters()), "lr": float(args.prefix_segment_learning_rate), "weight_decay": 0.0},
        ]
    )
    warmup = int(args.stage1_warmup_steps or 0)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    logger = TimedRetentionModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_minutes=args.checkpoint_save_minutes,
        keep_last=args.checkpoint_keep_last,
        log_steps=args.log_steps,
    )
    if accelerator.is_main_process:
        print(
            f"[native_prefix] global_envs={accelerator.num_processes * args.prefix_environments_per_rank} "
            f"per_rank_envs={args.prefix_environments_per_rank} K=1 Q={args.prefix_queries_per_environment} "
            "layout=41_support+4_reset_Q0+40_query_future total_frames=85 history_frames=45",
            flush=True,
        )

    iterator = tqdm(range(int(args.prefix_max_updates)), disable=not accelerator.is_local_main_process)
    for update_index in iterator:
        step = update_index + 1
        episodes = sampler.sample(
            step=step,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            environments_per_rank=args.prefix_environments_per_rank,
        )
        total_queries = sum(len(episode.query_indices) for episode in episodes)
        optimizer.zero_grad(set_to_none=True)
        detached = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        counter = 0
        for episode in episodes:
            support = dataset[episode.support_index]
            for query_index in episode.query_indices:
                counter += 1
                context = accelerator.no_sync(model) if counter < total_queries else nullcontext()
                with context:
                    pair = build_event80_prefix_pair(support, dataset[query_index])
                    loss = model(pair)
                    accelerator.backward(loss / float(total_queries))
                detached += loss.detach().float() / float(total_queries)
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        logger.on_step_end(accelerator, model, args.save_steps, loss=detached)
    logger.on_training_end(accelerator, model, args.save_steps)


if __name__ == "__main__":
    main()
