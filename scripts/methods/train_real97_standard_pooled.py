#!/usr/bin/env python3
"""Train-only real97 pooled baseline, with no context table or curriculum."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys
import time

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
from wan_video_action.methods.baselines.window_sampling import WindowIndexSelector
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


class PooledEpisodeSampler:
    def __init__(self, args):
        with open(args.dataset_metadata_path, encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.groups = {}
        for index, row in enumerate(self.rows):
            if row.get("dataset_split") != "train":
                raise ValueError(f"Non-training row in pooled manifest: {index}")
            env = int(row["environment_index"])
            episode = int(row["episode_index"])
            self.groups.setdefault(env, {}).setdefault(episode, []).append(index)
        self.envs_per_rank = int(args.standard_environments_per_rank)
        self.episodes_per_env = int(args.standard_episodes_per_environment)
        if len(self.groups) < self.envs_per_rank:
            raise ValueError("Not enough training environments for the local batch.")
        for env, episodes in self.groups.items():
            if len(episodes) < self.episodes_per_env:
                raise ValueError(f"Environment {env} has insufficient distinct train episodes.")
        self.windows = WindowIndexSelector(
            self.rows, mode="uniform_episode_then_window",
            kind_field=args.standard_window_kind_field,
            preferred_kind=args.standard_preferred_window_kind,
            preferred_probability=args.standard_preferred_window_probability,
        )
        self.seed = int(args.seed)

    def sample(self, step, rank):
        rng = random.Random(self.seed + int(step) * 104729 + int(rank) * 1000003)
        result = []
        for env in rng.sample(sorted(self.groups), self.envs_per_rank):
            episodes = self.groups[env]
            for episode in rng.sample(sorted(episodes), self.episodes_per_env):
                result.append(self.windows.choose(rng, episodes[episode]))
        return result


def save_checkpoint(accelerator, model, args, step, *, protected=False, reason="periodic"):
    """Publish complete files atomically; prune only this run's ordinary files."""
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        directory = Path(args.output_path)
        if protected:
            directory = directory / "protected"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"step-{step}.safetensors"
        if target.exists():
            if not protected:
                raise FileExistsError(f"Refusing to overwrite {target}")
            print(f"[checkpoint] protected file already exists; retaining {target}", flush=True)
        else:
            exported = accelerator.unwrap_model(model).export_trainable_state_dict(
                state, remove_prefix=args.remove_prefix_in_ckpt,
            )
            exported = {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in exported.items()
            }
            temporary = directory / f".step-{step}.{os.getpid()}.partial"
            save_file(
                exported, str(temporary),
                metadata={"step": str(step), "method": "standard_pooled", "reason": reason},
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            print(f"[checkpoint] saved={target} reason={reason}", flush=True)
            if not protected:
                candidates = sorted(
                    (path for path in directory.glob("step-*.safetensors")
                     if path.stem.removeprefix("step-").isdigit()),
                    key=lambda path: int(path.stem.removeprefix("step-")),
                )
                for old in candidates[:-2]:
                    old.unlink()
    del state
    accelerator.wait_for_everyone()


def main():
    parser = wan_parser()
    if "--frame_stride" not in parser._option_string_actions:
        parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--standard_max_updates", type=int, default=5500)
    parser.add_argument("--standard_environments_per_rank", type=int, default=3)
    parser.add_argument("--standard_episodes_per_environment", type=int, default=8)
    parser.add_argument("--standard_window_kind_field", default="sampling_kind")
    parser.add_argument("--standard_preferred_window_kind", default="precise")
    parser.add_argument("--standard_preferred_window_probability", type=float, default=0.5)
    parser.add_argument("--standard_history_anchor_windows", action="store_true", default=False)
    parser.add_argument("--standard_eef_canonicalize", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Each standard real97 experiment requires exactly two ranks.")
    deadline = float(os.environ["BWM_STANDARD_DEADLINE_EPOCH"])
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            ),
            InitProcessGroupKwargs(timeout=timedelta(hours=1)),
        ],
    )
    sampler = PooledEpisodeSampler(args)
    dataset = build_dataset(args, runtime_config)
    if args.standard_eef_canonicalize:
        if args.action_type != "eef_target":
            raise ValueError("Target EEF canonicalization requires action_type=eef_target")
        from wan_video_action.data.history_prefix import CanonicalTargetEEF
        with open(args.action_stat_path, encoding="utf-8") as handle:
            eef_stats = json.load(handle)["eef_target"]
        dataset.special_operator_map["action"] = CanonicalTargetEEF(args, eef_stats)
    if args.standard_history_anchor_windows:
        from wan_video_action.data.history_prefix import HistoryPrefixDataset
        dataset = HistoryPrefixDataset(dataset, args)
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
    if args.standard_history_anchor_windows:
        if args.task not in ("sft", "sft:train") or args.spatial_loss_mode != "none":
            raise ValueError("History pooled training currently requires SFT without ROI")
        from wan_video_action.methods.baselines.history_loss import history_prefix_flow_loss
        wan.task_to_loss[args.task] = history_prefix_flow_loss
    model = wan
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay),
    )
    warmup = int(getattr(args, "stage1_warmup_steps", 100) or 0)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda index: min(1.0, (index + 1) / warmup) if warmup else 1.0,
    )
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    model.train()
    if accelerator.is_main_process:
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "standard_run_settings.json").open("x", encoding="utf-8") as handle:
            json.dump({
                "method": "standard_pooled", "context_table": False,
                "train_environments": sorted(sampler.groups),
                "environments_per_rank": sampler.envs_per_rank,
                "episodes_per_environment": sampler.episodes_per_env,
                "world_size": 2, "max_updates": args.standard_max_updates,
                "deadline_epoch": deadline, "args": vars(args),
            }, handle, indent=2, default=str)
        print(
            f"[standard] local_batch={sampler.envs_per_rank * sampler.episodes_per_env} "
            f"global_batch={2 * sampler.envs_per_rank * sampler.episodes_per_env} "
            f"max_updates={args.standard_max_updates} deadline_epoch={deadline}",
            flush=True,
        )
    stop_flag = torch.zeros((), device=accelerator.device, dtype=torch.int32)
    previous_time = time.monotonic()
    for step in range(1, int(args.standard_max_updates) + 1):
        indices = sampler.sample(step, accelerator.process_index)
        optimizer.zero_grad(set_to_none=True)
        detached_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        for position, index in enumerate(indices):
            sync = nullcontext() if position == len(indices) - 1 else accelerator.no_sync(model)
            with sync:
                loss = model(dataset[index])
                accelerator.backward(loss / len(indices))
            detached_loss += loss.detach().float() / len(indices)
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        # One tiny collective at an optimizer boundary; no filesystem/Slurm polls.
        if accelerator.is_main_process:
            stop_flag.fill_(int(time.time() >= deadline))
        torch.distributed.broadcast(stop_flag, src=0)
        should_stop = bool(stop_flag.item())
        if step % max(1, int(args.log_steps)) == 0:
            loss_value = accelerator.reduce(detached_loss, reduction="mean").item()
            now = time.monotonic()
            if accelerator.is_main_process:
                record = {
                    "step": step, "loss": loss_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds_per_step": (now - previous_time) / max(1, int(args.log_steps)),
                    "timestamp": time.time(),
                }
                with (Path(args.output_path) / "loss.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print("[standard_train] " + json.dumps(record), flush=True)
            previous_time = now
        final = step == int(args.standard_max_updates)
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
