#!/usr/bin/env python3
"""Practical six-chunk prequential TTT-KVB training on Event80."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
from accelerate.utils import InitProcessGroupKwargs
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from scripts.train import TimedRetentionModelLogger, WanTrainingModule, wan_parser
from scripts.train_stage1_grouped_context import build_dataset
from wan_video_action.methods.baselines.ttt_kvb.prequential import (
    Event80PrequentialSampler,
)
from wan_video_action.methods.baselines.ttt_kvb.wan_adapter import install_ttt_kvb
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


def add_ttt_kvb_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("ttt_kvb_prequential")
    group.add_argument("--ttt_active_environment_manifest", type=str, required=False)
    group.add_argument("--ttt_sequence_length", type=int, default=6)
    group.add_argument("--ttt_environments_per_rank", type=int, default=1)
    group.add_argument("--ttt_layers", type=str, default="uniform:8")
    group.add_argument("--ttt_expansion", type=int, default=4)
    group.add_argument("--ttt_base_inner_lr", type=float, default=0.1)
    group.add_argument("--ttt_inner_batch_size", type=int, default=64)
    group.add_argument("--ttt_write_token_budget", type=int, default=512)
    group.add_argument("--ttt_gate_init", type=float, default=0.01)
    group.add_argument("--ttt_slow_learning_rate", type=float, default=1.0e-4)
    group.add_argument("--ttt_max_updates", type=int, default=50000)
    return parser


def main() -> None:
    parser = add_ttt_kvb_config(wan_parser())
    parser.add_argument("--frame_stride", type=int, default=1)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if not args.ttt_active_environment_manifest:
        raise ValueError("ttt_active_environment_manifest is required.")
    if int(args.ttt_write_token_budget) % int(args.ttt_inner_batch_size) != 0:
        raise ValueError("TTT token budget must be divisible by the inner batch size.")

    # TTT-KVB keeps an explicit mutable fast state outside the Wan block inputs.
    # Activation-checkpoint replay cannot reproduce that stateful execution
    # contract reliably, so this dedicated runner always uses a single forward.
    if args.use_gradient_checkpointing or args.use_gradient_checkpointing_offload:
        print(
            "[ttt_kvb] disabling activation checkpointing for state-consistent "
            "fast-weight query replay",
            flush=True,
        )
        args.use_gradient_checkpointing = False
        args.use_gradient_checkpointing_offload = False

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
    sampler = Event80PrequentialSampler(
        metadata_path=args.dataset_metadata_path,
        active_environment_manifest=args.ttt_active_environment_manifest,
        seed=args.seed,
        sequence_length=args.ttt_sequence_length,
    )
    model = WanTrainingModule(
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
    installation = install_ttt_kvb(
        model.pipe.dit,
        layer_spec=args.ttt_layers,
        expansion=args.ttt_expansion,
        base_inner_lr=args.ttt_base_inner_lr,
        inner_batch_size=args.ttt_inner_batch_size,
        write_token_budget=args.ttt_write_token_budget,
        gate_init=args.ttt_gate_init,
    )
    controller = installation.controller
    ttt_parameters = list(installation.ttt_parameters())
    ttt_parameter_ids = {id(parameter) for parameter in ttt_parameters}
    wan_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in ttt_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "wan",
                "params": wan_parameters,
                "lr": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
            },
            {
                "name": "ttt_slow",
                "params": ttt_parameters,
                "lr": float(args.ttt_slow_learning_rate),
                "weight_decay": 0.01,
            },
        ]
    )
    warmup_steps = int(getattr(args, "stage1_warmup_steps", 0) or 0)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            min(1.0, float(step + 1) / float(warmup_steps))
            if warmup_steps > 0
            else 1.0
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

    updates_per_chunk = int(args.ttt_write_token_budget) // int(args.ttt_inner_batch_size)
    if accelerator.is_main_process:
        print(
            "[ttt_kvb_prequential] "
            f"layers={installation.layer_indices} sequence_length={args.ttt_sequence_length} "
            f"environments_per_rank={args.ttt_environments_per_rank} "
            f"updates_per_chunk={updates_per_chunk} "
            f"layer_local_updates_per_stream="
            f"{updates_per_chunk * int(args.ttt_sequence_length) * len(installation.layer_indices)} "
            f"gate_init={args.ttt_gate_init} base_inner_lr={args.ttt_base_inner_lr}",
            flush=True,
        )

    iterator = tqdm(
        range(int(args.ttt_max_updates)),
        disable=not accelerator.is_local_main_process,
    )
    sequence_length = int(args.ttt_sequence_length)
    for update_index in iterator:
        step = update_index + 1
        episodes = sampler.sample(
            step=step,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            environments_per_rank=args.ttt_environments_per_rank,
        )
        optimizer.zero_grad(set_to_none=True)
        local_position_losses = torch.zeros(
            sequence_length, device=accelerator.device, dtype=torch.float32
        )
        for episode in episodes:
            controller.reset(batch_size=1)
            unwrapped_model = accelerator.unwrap_model(model)
            for position, sample_index in enumerate(episode.indices):
                chunk = dataset[sample_index]
                is_final_query = position == sequence_length - 1
                sync_context = (
                    nullcontext() if is_final_query else accelerator.no_sync(model)
                )
                with sync_context:
                    with controller.query_read():
                        loss = model(chunk)
                        accelerator.backward(
                            loss / float(sequence_length * len(episodes)),
                            retain_graph=not is_final_query,
                        )
                local_position_losses[position] += loss.detach().float() / float(
                    len(episodes)
                )

                # The current chunk is observed only after its prediction. Its write
                # can therefore affect later chunks, never its own query loss.
                with controller.support_write(differentiable=not is_final_query):
                    # Support video tokens are observations, not an outer-loss
                    # path. TTTMLPMemory locally enables autograd for its FP32
                    # fast-state update while the frozen support backbone stays
                    # graph-free.
                    with torch.no_grad():
                        unwrapped_model(chunk)

            if int(args.log_steps) > 0 and step % int(args.log_steps) == 0:
                state_norms = controller.state_norms()
                mean_state_norm = (
                    sum(state_norms.values()) / float(len(state_norms))
                    if state_norms
                    else 0.0
                )
                print(
                    f"[ttt_stream] rank={accelerator.process_index} step={step} "
                    f"environment={episode.environment_id} actions={episode.action_ids} "
                    f"mean_state_norm={mean_state_norm:.6f}",
                    flush=True,
                )
            controller.clear()

        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        gathered_losses = accelerator.gather(local_position_losses)
        gathered_losses = gathered_losses.reshape(-1, sequence_length).mean(dim=0)
        detached_loss = gathered_losses.mean()
        if accelerator.is_main_process and int(args.log_steps) > 0:
            if step % int(args.log_steps) == 0:
                position_text = " ".join(
                    f"p{position + 1}={value:.6f}"
                    for position, value in enumerate(gathered_losses.tolist())
                )
                print(
                    f"[ttt_train] step={step} loss={detached_loss.item():.6f} "
                    f"{position_text}",
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
