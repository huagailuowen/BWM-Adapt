#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
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


def _as_keys(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


class PushBoxMetaTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        base_path: str,
        metadata_path: str,
        data_file_keys: Iterable[str],
        main_data_operator,
        special_operator_map: dict | None,
        repeat: int,
        seed: int,
        group_keys: str,
        tasks_per_epoch: int,
        support_min: int,
        support_max: int,
        query_min: int,
        query_max: int,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.data_file_keys = tuple(data_file_keys)
        self.main_data_operator = main_data_operator
        self.special_operator_map = {} if special_operator_map is None else dict(special_operator_map)
        self.repeat = int(repeat)
        self.seed = int(seed)
        self.group_keys = _as_keys(group_keys)
        self.support_min = int(support_min)
        self.support_max = int(support_max)
        self.query_min = int(query_min)
        self.query_max = int(query_max)
        if self.support_min <= 0 or self.query_min <= 0:
            raise ValueError("support_min and query_min must be positive.")
        if self.support_max < self.support_min or self.query_max < self.query_min:
            raise ValueError("support/query max must be >= min.")

        rows = _read_jsonl(metadata_path)
        groups = defaultdict(list)
        for row in rows:
            key = tuple(row.get(name) for name in self.group_keys)
            groups[key].append(row)

        min_required = self.support_min + self.query_min
        self.groups = [
            (key, value)
            for key, value in sorted(groups.items(), key=lambda item: str(item[0]))
            if len(value) >= min_required
        ]
        if not self.groups:
            raise ValueError(
                f"No groups in {metadata_path} have at least {min_required} samples "
                f"for group_keys={self.group_keys}."
            )

        self.rows = rows
        base_len = int(tasks_per_epoch) if int(tasks_per_epoch) > 0 else len(rows)
        self.length = max(1, base_len) * max(1, self.repeat)
        print(
            f"Stage2 meta dataset: rows={len(rows)}, groups={len(self.groups)}, "
            f"tasks={self.length}, group_keys={self.group_keys}"
        )
        for key, value in self.groups:
            episodes = len({row.get("episode_index") for row in value})
            print(f"  group={key}: chunks={len(value)}, episodes={episodes}")

    def __len__(self):
        return self.length

    def _resolve_frame_range(self, data: dict) -> tuple[int, int]:
        start_frame = int(data.get("start_frame", 0))
        if data.get("end_frame") is not None:
            end_frame = int(data["end_frame"])
        elif data.get("length") is not None:
            end_frame = start_frame + int(data["length"]) - 1
        else:
            raise KeyError("Metadata row must contain end_frame or length.")
        return start_frame, end_frame

    def _wrap_frame_range_metadata(self, data: dict, payload):
        start_frame, end_frame = self._resolve_frame_range(data)

        def wrap_item(item):
            if isinstance(item, str):
                return {"data": item, "start_frame": start_frame, "end_frame": end_frame}
            if isinstance(item, dict):
                wrapped = item.copy()
                wrapped["start_frame"] = start_frame
                wrapped["end_frame"] = end_frame
                return wrapped
            return item

        if isinstance(payload, (list, tuple)):
            return [wrap_item(item) for item in payload]
        return wrap_item(payload)

    def _process_row(self, row: dict) -> dict:
        data = row.copy()
        for key in self.data_file_keys:
            if key in self.special_operator_map:
                source = data[key] if key in data else data
                data[key] = self.special_operator_map[key](self._wrap_frame_range_metadata(data, source))
            elif key in data:
                data[key] = self.main_data_operator(self._wrap_frame_range_metadata(data, data[key]))
        return data

    def _sample_rows(self, rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
        rows = list(rows)
        rng.shuffle(rows)
        support_count = rng.randint(self.support_min, min(self.support_max, len(rows) - self.query_min))
        remaining_after_support = len(rows) - support_count
        query_count = rng.randint(self.query_min, min(self.query_max, remaining_after_support))

        support = rows[:support_count]
        support_ids = {row.get("sample_id") for row in support}
        support_episodes = {row.get("episode_index") for row in support}
        remaining = [row for row in rows if row.get("sample_id") not in support_ids]
        query_candidates = [row for row in remaining if row.get("episode_index") not in support_episodes]
        if len(query_candidates) < query_count:
            query_candidates = remaining
        return support, query_candidates[:query_count]

    def __getitem__(self, data_id: int) -> dict:
        rng = random.Random(self.seed + int(data_id))
        key, rows = self.groups[int(data_id) % len(self.groups)]
        support_rows, query_rows = self._sample_rows(rows, rng)
        return {
            "group_key": list(key),
            "support_ids": [row.get("sample_id") for row in support_rows],
            "query_ids": [row.get("sample_id") for row in query_rows],
            "support": [self._process_row(row) for row in support_rows],
            "query": [self._process_row(row) for row in query_rows],
        }


class Stage2TTTTrainingModule(WanTrainingModule):
    def __init__(self, *args, stage2_args: argparse.Namespace, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_steps = int(stage2_args.stage2_inner_steps)
        self.inner_lr = float(stage2_args.stage2_inner_lr)
        self.inner_grad_clip = float(stage2_args.stage2_inner_grad_clip)
        self.query_weight = float(stage2_args.stage2_query_weight)
        self.show_eval_weight = float(stage2_args.stage2_show_eval_weight)
        self.gap_weight = float(stage2_args.stage2_gap_weight)
        self.gap_margin = float(stage2_args.stage2_gap_margin)
        self.context_reg_weight = float(stage2_args.stage2_context_reg_weight)
        self.improvement_eps = float(stage2_args.stage2_improvement_eps)
        self.last_metrics = {}
        if getattr(self.pipe, "physical_context_encoder", None) is None:
            raise ValueError("Stage2 TTT requires physical_context_mode != 'none'.")

    def _context0(self) -> torch.Tensor:
        return self.pipe.physical_context_encoder.default_context

    def _context_reg(self, context: torch.Tensor) -> torch.Tensor:
        target = self._context0().detach().to(dtype=context.dtype, device=context.device)
        return torch.mean((context.float() - target.float()) ** 2)

    def _prepare_loss_inputs(self, data: dict, physical_context: torch.Tensor):
        data = data.copy()
        data["physical_context"] = physical_context
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
        inputs: dict,
        *,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = self._shared_inputs(inputs)
        inputs = dict(inputs)
        if timestep is None:
            timestep = self._sample_timestep(inputs)
        if noise is None:
            noise = torch.randn_like(inputs["input_latents"])

        inputs["latents"] = self.pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        training_target = self.pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
        if "first_frame_latents" in inputs:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        noise_pred = self.pipe.model_fn(**models, **inputs, timestep=timestep)

        if "first_frame_latents" in inputs:
            noise_pred = noise_pred[:, :, 1:]
            training_target = training_target[:, :, 1:]

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        return loss * self.pipe.scheduler.training_weight(timestep)

    def _sft_loss(self, data: dict, physical_context: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_loss_inputs(data, physical_context)
        return self._flow_match_loss_from_inputs(inputs)

    def _paired_baseline_adapted_loss(
        self,
        data: dict,
        context0: torch.Tensor,
        context_adapted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            baseline_inputs = self._prepare_loss_inputs(data, context0)
            timestep = self._sample_timestep(baseline_inputs)
            noise = torch.randn_like(self._shared_inputs(baseline_inputs)["input_latents"])
            baseline_loss = self._flow_match_loss_from_inputs(
                baseline_inputs,
                timestep=timestep,
                noise=noise,
            )
        adapted_inputs = self._prepare_loss_inputs(data, context_adapted)
        adapted_loss = self._flow_match_loss_from_inputs(
            adapted_inputs,
            timestep=timestep,
            noise=noise,
        )
        return baseline_loss.detach(), adapted_loss

    def _adapt_context(self, support: list[dict], context0: torch.Tensor) -> torch.Tensor:
        context = context0
        for _ in range(self.inner_steps):
            support_losses = [self._sft_loss(item, context) for item in support]
            support_loss = torch.stack(support_losses).mean()
            if self.context_reg_weight > 0:
                support_loss = support_loss + self.context_reg_weight * self._context_reg(context)
            (grad,) = torch.autograd.grad(support_loss, context, create_graph=False)
            if self.inner_grad_clip > 0:
                grad_norm = grad.float().norm().clamp_min(1e-6)
                grad = grad * min(1.0, self.inner_grad_clip / float(grad_norm.detach().cpu()))
            context = context - self.inner_lr * grad
        return context

    @staticmethod
    def _mean_stack(items: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(items).mean()

    def forward(self, task: dict) -> torch.Tensor:
        support = task["support"]
        query = task["query"]
        context0 = self._context0()
        context_adapted = self._adapt_context(support, context0)

        show_baseline = []
        show_adapted = []
        for item in support:
            baseline_loss, adapted_loss = self._paired_baseline_adapted_loss(item, context0, context_adapted)
            show_baseline.append(baseline_loss)
            show_adapted.append(adapted_loss)

        query_baseline = []
        query_adapted = []
        for item in query:
            baseline_loss, adapted_loss = self._paired_baseline_adapted_loss(item, context0, context_adapted)
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
        gap_loss = torch.relu(rel_show_imp.detach() - rel_query_imp - self.gap_margin) ** 2
        context_reg = self._context_reg(context_adapted)

        loss = self.query_weight * query_adapted_loss
        if self.show_eval_weight > 0:
            loss = loss + self.show_eval_weight * show_adapted_loss
        if self.gap_weight > 0:
            loss = loss + self.gap_weight * gap_loss
        if self.context_reg_weight > 0:
            loss = loss + self.context_reg_weight * context_reg

        self.last_metrics = {
            "loss": float(loss.detach().float().cpu()),
            "query": float(query_adapted_loss.detach().float().cpu()),
            "show": float(show_adapted_loss.detach().float().cpu()),
            "gap": float(gap_loss.detach().float().cpu()),
            "rel_show_imp": float(rel_show_imp.detach().float().cpu()),
            "rel_query_imp": float(rel_query_imp.detach().float().cpu()),
            "context_reg": float(context_reg.detach().float().cpu()),
        }
        return loss


def add_stage2_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("stage2_ttt")
    group.add_argument("--stage2_group_keys", type=str, default="source_split,friction_mu")
    group.add_argument("--stage2_tasks_per_epoch", type=int, default=200)
    group.add_argument("--stage2_support_min", type=int, default=1)
    group.add_argument("--stage2_support_max", type=int, default=3)
    group.add_argument("--stage2_query_min", type=int, default=4)
    group.add_argument("--stage2_query_max", type=int, default=6)
    group.add_argument("--stage2_inner_steps", type=int, default=1)
    group.add_argument("--stage2_inner_lr", type=float, default=0.05)
    group.add_argument("--stage2_inner_grad_clip", type=float, default=1.0)
    group.add_argument("--stage2_query_weight", type=float, default=1.0)
    group.add_argument("--stage2_show_eval_weight", type=float, default=0.1)
    group.add_argument("--stage2_gap_weight", type=float, default=0.2)
    group.add_argument("--stage2_gap_margin", type=float, default=0.05)
    group.add_argument("--stage2_context_reg_weight", type=float, default=1e-3)
    group.add_argument("--stage2_improvement_eps", type=float, default=1e-4)
    return parser


def build_dataset(args, runtime_config):
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

    return PushBoxMetaTaskDataset(
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
        support_min=args.stage2_support_min,
        support_max=args.stage2_support_max,
        query_min=args.stage2_query_min,
        query_max=args.stage2_query_max,
    )


def launch_stage2_training(accelerator, dataset, model, model_logger, args):
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
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
                        "[stage2] "
                        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                        flush=True,
                    )
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)


def main() -> None:
    parser = add_stage2_config(wan_parser())
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

    dataset = build_dataset(args, runtime_config)
    model = Stage2TTTTrainingModule(
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
        print(f"Trainable parameter tensors: {len(names)}", flush=True)
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
    launch_stage2_training(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
