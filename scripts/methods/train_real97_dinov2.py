#!/usr/bin/env python3
"""Real97 train-only DINO concat-MLP baseline with allocation-aware saves."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import accelerate
from accelerate.utils import InitProcessGroupKwargs
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from safetensors.torch import save_file
import torch
from torch.optim.lr_scheduler import LambdaLR

from scripts.train import WanTrainingModule, wan_parser
from scripts.train_stage1_grouped_context import build_dataset
from scripts.methods.train_dinov2_event80 import (
    DINOv2WanTrainingModule,
    add_dinov2_config,
)
from scripts.methods.train_real97_standard_pooled import PooledEpisodeSampler
from wan_video_action.methods.baselines.dinov2_amortized import DINOv2AmortizedContextEncoder
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + f".{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_checkpoint(accelerator, model, args, step, *, protected=False, reason="periodic"):
    """Save Wan and its matching DINO head together; never prune protected/."""
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        directory = Path(args.output_path)
        if protected:
            directory = directory / "protected"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"step-{step}.safetensors"
        metadata_path = directory / f"step-{step}.metadata.json"
        marker = directory / f".step-{step}.safetensors.complete"
        if target.exists() or metadata_path.exists() or marker.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint bundle: {target}")
        exported = accelerator.unwrap_model(model).export_trainable_state_dict(
            state, remove_prefix=args.remove_prefix_in_ckpt,
        )
        exported = {
            key: value.detach().to(device="cpu").contiguous()
            for key, value in exported.items()
        }
        if not any(key.startswith("wan.") for key in exported):
            raise RuntimeError("Checkpoint is missing world-model parameters.")
        if not any(key.startswith("support_encoder.") for key in exported):
            raise RuntimeError("Checkpoint is missing its matching DINO context head.")
        temporary = directory / f".step-{step}.{os.getpid()}.partial"
        save_file(exported, str(temporary), metadata={
            "step": str(step), "method": "dinov2_concat_mlp", "reason": reason,
            "context_storage": "support_encoder_in_same_file_not_environment_table",
        })
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        atomic_json(metadata_path, {
            "step": step, "reason": reason, "protected": protected,
            "method": "dinov2_concat_mlp", "model_and_head_same_checkpoint": True,
            "context_table": False, "full_optimizer_resume_state": False,
            "support_k_choices": [int(value.strip()) for value in str(args.dinov2_support_k_choices).split(",")],
            "query_topup_from_support": args.dinov2_query_topup_from_support,
            "query_min_count": int(args.dinov2_queries_per_environment) if args.dinov2_query_topup_from_support else 0,
            "context_dim": args.dinov2_output_dim,
            "checkpoint": str(target), "args": vars(args),
        })
        atomic_json(marker, {"step": step, "checkpoint": target.name, "reason": reason})
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print(f"[checkpoint] saved={target} reason={reason}", flush=True)
        if not protected:
            completed = sorted(
                (path for path in directory.glob("step-*.safetensors")
                 if path.stem.removeprefix("step-").isdigit()
                 and (directory / ("." + path.name + ".complete")).is_file()),
                key=lambda path: int(path.stem.removeprefix("step-")),
            )
            for old in completed[:-int(args.checkpoint_keep_last)]:
                old.unlink()
                (directory / f"{old.stem}.metadata.json").unlink(missing_ok=True)
                (directory / ("." + old.name + ".complete")).unlink()
    del state
    accelerator.wait_for_everyone()


def main():
    parser = add_dinov2_config(wan_parser())
    parser.add_argument(
        "--dinov2_query_topup_from_support", action="store_true", default=False,
        help="Opt in to K=1..4 and reuse selected supports to reach the minimum query count.",
    )
    if "--frame_stride" not in parser._option_string_actions:
        parser.add_argument("--frame_stride", type=int, default=1)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    choices = tuple(int(value.strip()) for value in str(args.dinov2_support_k_choices).split(","))
    query_topup = bool(args.dinov2_query_topup_from_support)
    expected_choices = (1, 2, 3, 4) if query_topup else (1, 2)
    if choices != expected_choices or args.dinov2_aggregation_mode != "concat_mlp":
        raise ValueError(f"This runner requires concat_mlp and support K choices {expected_choices}.")
    min_queries = int(args.dinov2_queries_per_environment) if query_topup else 0
    if query_topup and min_queries != 4:
        raise ValueError("The support-reuse branch requires a minimum of four queries.")
    if int(args.num_history_frames) != 1:
        raise ValueError("This Ball/Door entrypoint does not yet support Soft history5.")
    if args.spatial_loss_mode != "none":
        raise ValueError("Real97 DINO Ball/Door must not use ROI weighting.")
    if not (args.video_light_augmentation_enabled and args.dinov2_support_light_augmentation_enabled):
        raise ValueError("Both query and DINO support augmentation must be enabled.")
    if int(args.checkpoint_keep_last) < 1:
        raise ValueError("checkpoint_keep_last must be positive.")
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Exactly two CUDA ranks are required; no CPU fallback.")
    deadline = float(os.environ["BWM_DINO_DEADLINE_EPOCH"])
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1, mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            ),
            InitProcessGroupKwargs(timeout=timedelta(hours=1)),
        ],
    )
    sampler_args = vars(args).copy()
    sampler_args.update(
        standard_environments_per_rank=args.dinov2_environments_per_rank,
        standard_episodes_per_environment=args.dinov2_trajectories_per_environment,
        standard_window_kind_field=args.dinov2_window_kind_field,
        standard_preferred_window_kind=args.dinov2_preferred_window_kind,
        standard_preferred_window_probability=args.dinov2_preferred_window_probability,
    )
    # Reuse the current Standard/Ours rank-independent episode-first sampling.
    # This also rejects every non-train row before loading any model.
    sampler = PooledEpisodeSampler(SimpleNamespace(**sampler_args))
    if sampler.episodes_per_env <= max(choices):
        raise ValueError("Each environment must have at least one disjoint query.")
    if query_topup and sampler.episodes_per_env != 6:
        raise ValueError("The support-reuse branch requires six distinct episodes per environment.")
    if any(row.get("action_semantics") != args.action_type for row in sampler.rows):
        raise ValueError("Manifest action semantics do not match the training config.")
    dataset = build_dataset(args, runtime_config)
    wan = WanTrainingModule(
        model_paths=json.dumps(runtime_config["model_paths_list"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=runtime_config["tokenizer_path"], enable_text=args.enable_text,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs, modules=runtime_config["modules"],
        fp8_models=args.fp8_models, offload_models=args.offload_models,
        ckpt_path=args.ckpt_path, task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames, args=args,
    )
    encoder = DINOv2AmortizedContextEncoder(
        model_path=args.dinov2_model_path, sampled_frames=args.dinov2_sampled_frames,
        temporal_stride=args.dinov2_temporal_stride, action_dim=args.action_dim,
        hidden_dim=args.dinov2_hidden_dim, action_hidden_dim=args.dinov2_action_hidden_dim,
        output_dim=args.dinov2_output_dim, temporal_layers=args.dinov2_temporal_layers,
        temporal_heads=args.dinov2_temporal_heads,
        aggregation_mode=args.dinov2_aggregation_mode,
        mlp_hidden_dim=args.dinov2_mlp_hidden_dim,
    )
    model = DINOv2WanTrainingModule(
        wan=wan, support_encoder=encoder, support_light_augmentation_enabled=True,
    )
    optimizer = torch.optim.AdamW([
        {"name": "wan", "params": [p for p in wan.parameters() if p.requires_grad],
         "lr": float(args.learning_rate), "weight_decay": float(args.weight_decay)},
        {"name": "amortized_head", "params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": float(args.dinov2_head_learning_rate), "weight_decay": 0.01},
    ])
    warmup = int(args.stage1_warmup_steps or 0)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda index: min(1.0, (index + 1) / warmup) if warmup else 1.0,
    )
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    model.train()
    output = Path(args.output_path)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        run_lock = (output / ".training.lock").open("a")
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "dinov2_run_settings.json").open("x", encoding="utf-8") as handle:
            json.dump({
                "method": "dinov2_concat_mlp", "train_environments": sorted(sampler.groups),
                "sampling": "independent_per_rank_uniform_environment_episode_then_window",
                "environments_per_rank": sampler.envs_per_rank,
                "trajectories_per_environment": sampler.episodes_per_env,
                "support_k_choices": choices, "support_selection": "uniform_without_replacement",
                "query_count": "max(total_trajectories_minus_K,4)" if query_topup else "total_trajectories_minus_K",
                "query_topup_from_support": query_topup, "query_min_count": min_queries,
                "query_topup_selection": "uniform_without_replacement_from_selected_supports" if query_topup else "none",
                "world_size": 2,
                "support_query_episode_disjoint": not query_topup, "context_table": False,
                "deadline_epoch": deadline, "args": vars(args),
            }, handle, indent=2, default=str)
        print(
            f"[dino_real97] per_rank={sampler.envs_per_rank}x{sampler.episodes_per_env} "
            f"support_K={','.join(map(str, choices))} query={'max(remaining,4)' if query_topup else 'remaining'} "
            f"query_topup_from_support={query_topup} max_updates={args.dinov2_max_updates} "
            f"deadline_epoch={deadline} support_augmentation=True", flush=True,
        )
    stop_flag = torch.zeros((), device=accelerator.device, dtype=torch.int32)
    previous_time = time.monotonic()
    for step in range(1, int(args.dinov2_max_updates) + 1):
        selected = sampler.sample(step, accelerator.process_index)
        groups = {}
        for index in selected:
            groups.setdefault(int(sampler.rows[index]["environment_index"]), []).append(index)
        rng = random.Random(int(args.seed) + step * 15485863 + accelerator.process_index * 32452843)
        episodes = []
        for env, indices in groups.items():
            support_indices = rng.sample(indices, rng.choice(choices))
            support_set = set(support_indices)
            query_indices = [index for index in indices if index not in support_set]
            if query_topup and len(query_indices) < min_queries:
                query_indices.extend(rng.sample(support_indices, min_queries - len(query_indices)))
            episodes.append((env, support_indices, query_indices))
        total_queries = sum(len(query) for _, _, query in episodes)
        total_supports = sum(len(support) for _, support, _ in episodes)
        total_unique = sum(len(set(support) | set(query)) for _, support, query in episodes)
        total_reused = sum(len(set(support) & set(query)) for _, support, query in episodes)
        optimizer.zero_grad(set_to_none=True)
        detached_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        query_counter = 0
        for env, support_indices, query_indices in episodes:
            supports = tuple(dataset[index] for index in support_indices)
            unwrapped = accelerator.unwrap_model(model)
            # One temporally coherent augmentation draw and frozen DINO pass per
            # selected support. Reuse features for all queries in this update.
            visual = tuple(unwrapped.extract_support_visual(row["video"]) for row in supports)
            query_weight = 1.0 / (len(episodes) * len(query_indices))
            for query_index in query_indices:
                query_counter += 1
                sync = nullcontext() if query_counter == total_queries else accelerator.no_sync(model)
                with sync:
                    loss = model(
                        query_data=dataset[query_index],
                        support_visual_features=tuple(item[0] for item in visual),
                        support_action=tuple(row["action"] for row in supports),
                        support_frame_indices=tuple(item[1] for item in visual),
                        support_frame_count=tuple(item[2] for item in visual),
                    )
                    accelerator.backward(loss * query_weight)
                detached_loss += loss.detach().float() * query_weight
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        # In-memory clock only; one tiny collective at the optimizer boundary.
        if accelerator.is_main_process:
            stop_flag.fill_(int(time.time() >= deadline))
        torch.distributed.broadcast(stop_flag, src=0)
        should_stop = bool(stop_flag.item())
        if step % max(1, int(args.log_steps)) == 0:
            values = torch.tensor(
                [float(total_supports), float(total_queries), float(total_unique), float(total_reused)],
                device=accelerator.device, dtype=torch.float32,
            )
            counts = accelerator.reduce(values, reduction="sum").tolist()
            loss_value = accelerator.reduce(detached_loss, reduction="mean").item()
            now = time.monotonic()
            if accelerator.is_main_process:
                record = {
                    "step": step, "query_loss": loss_value, "loss": loss_value,
                    "lr_wan": optimizer.param_groups[0]["lr"],
                    "lr_head": optimizer.param_groups[1]["lr"],
                    "global_support_chunks": int(counts[0]),
                    "global_query_chunks": int(counts[1]),
                    "global_total_chunks": int(sum(counts[:2])),
                    "global_unique_chunks": int(counts[2]),
                    "global_reused_support_queries": int(counts[3]),
                    "seconds_per_step": (now - previous_time) / max(1, int(args.log_steps)),
                    "timestamp": time.time(),
                }
                with (output / "loss.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                with (output / "sampled_episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "step": step, "rank": 0,
                        "environments": [{
                            "environment_index": env,
                            "support": [sampler.rows[i]["sample_id"] for i in support],
                            "query": [sampler.rows[i]["sample_id"] for i in query],
                            "reused_support_as_query": [
                                sampler.rows[i]["sample_id"] for i in query if i in set(support)
                            ],
                        } for env, support, query in episodes],
                    }) + "\n")
                print("[dino_train] " + json.dumps(record), flush=True)
            previous_time = now
        final = step == int(args.dinov2_max_updates)
        if step == 2300 or should_stop or final:
            reason = "wallclock_23h30" if should_stop else ("final" if final else "step2300")
            save_checkpoint(accelerator, model, args, step, protected=True, reason=reason)
        elif step % int(args.save_steps) == 0:
            save_checkpoint(accelerator, model, args, step)
        if should_stop or final:
            break
    accelerator.end_training()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
