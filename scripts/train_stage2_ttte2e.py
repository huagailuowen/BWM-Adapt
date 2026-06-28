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
from tqdm import tqdm

from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from train import TimedRetentionModelLogger, WanTrainingModule, wan_parser
from train_stage2_ttt import add_stage2_config, build_dataset
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed


TTTE2E_PARAM_SCOPES = ("gates", "lowrank", "all")


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
        self.inner_grad_clip = float(stage2_args.ttte2e_inner_grad_clip)
        self.inner_reg_weight = float(stage2_args.ttte2e_inner_reg_weight)
        self.outer_reg_weight = float(stage2_args.ttte2e_outer_reg_weight)
        self.second_order = bool(stage2_args.ttte2e_second_order)
        self.query_weight = float(stage2_args.stage2_query_weight)
        self.show_eval_weight = float(stage2_args.stage2_show_eval_weight)
        self.gap_weight = float(stage2_args.stage2_gap_weight)
        self.gap_margin = float(stage2_args.stage2_gap_margin)
        self.improvement_eps = float(stage2_args.stage2_improvement_eps)
        self.param_scope = str(stage2_args.ttte2e_inner_param_scope)
        self.last_metrics = {}

        if getattr(self.pipe, "physical_context_encoder", None) is not None:
            raise ValueError(
                "TTT-E2E mild branch must not use latent C. Set physical_context_mode='none'."
            )
        if getattr(self.pipe, "physical_adapter_bank", None) is None:
            raise ValueError(
                "TTT-E2E mild branch requires physical_adapter_mode='residual'."
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

    def _adapt_adapter_params(
        self,
        support: list[dict],
        base_params: dict[str, torch.nn.Parameter],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        fast_params: dict[str, torch.Tensor] = {name: param for name, param in base_params.items()}
        last_inner_loss = None
        for _ in range(self.inner_steps):
            support_loss = self._support_loss(support, fast_params)
            inner_loss = support_loss
            if self.inner_reg_weight > 0:
                inner_loss = inner_loss + self.inner_reg_weight * self._adapter_reg(fast_params, base_params)
            grads = torch.autograd.grad(
                inner_loss,
                tuple(fast_params.values()),
                create_graph=self.second_order,
                allow_unused=True,
            )
            grads = _clip_grad_tensors(grads, self.inner_grad_clip)
            next_params = {}
            for (name, value), grad in zip(fast_params.items(), grads):
                if grad is None:
                    grad = torch.zeros_like(value)
                next_params[name] = value - self.inner_lr * grad
            fast_params = next_params
            last_inner_loss = inner_loss
        if last_inner_loss is None:
            last_inner_loss = torch.zeros((), device=self.pipe.device, dtype=torch.float32)
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

    def forward(self, task: dict) -> torch.Tensor:
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
        gap_loss = torch.relu(rel_show_imp.detach() - rel_query_imp - self.gap_margin) ** 2
        adapter_reg = self._adapter_reg(fast_params, base_params)
        adapter_delta_norm = self._adapter_delta_norm(fast_params, base_params)

        loss = self.query_weight * query_adapted_loss
        if self.show_eval_weight > 0:
            loss = loss + self.show_eval_weight * show_adapted_loss
        if self.gap_weight > 0:
            loss = loss + self.gap_weight * gap_loss
        if self.outer_reg_weight > 0:
            loss = loss + self.outer_reg_weight * adapter_reg

        self.last_metrics = {
            "loss": float(loss.detach().float().cpu()),
            "query": float(query_adapted_loss.detach().float().cpu()),
            "show": float(show_adapted_loss.detach().float().cpu()),
            "gap": float(gap_loss.detach().float().cpu()),
            "rel_show_imp": float(rel_show_imp.detach().float().cpu()),
            "rel_query_imp": float(rel_query_imp.detach().float().cpu()),
            "inner": float(inner_loss.detach().float().cpu()),
            "adapter_reg": float(adapter_reg.detach().float().cpu()),
            "adapter_delta_norm": float(adapter_delta_norm.detach().float().cpu()),
        }
        return loss


def add_ttte2e_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("stage2_ttte2e")
    group.add_argument("--ttte2e_inner_param_scope", type=str, default="lowrank", choices=TTTE2E_PARAM_SCOPES)
    group.add_argument("--ttte2e_inner_lr", type=float, default=1e-3)
    group.add_argument("--ttte2e_inner_grad_clip", type=float, default=0.1)
    group.add_argument("--ttte2e_inner_reg_weight", type=float, default=1e-4)
    group.add_argument("--ttte2e_outer_reg_weight", type=float, default=1e-4)
    group.add_argument(
        "--ttte2e_second_order",
        action="store_true",
        default=False,
        help="Use second-order gradients through inner updates. Default is first-order for memory.",
    )
    return parser


def launch_stage2_ttte2e_training(accelerator, dataset, model, model_logger, args):
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

    dataset = build_dataset(args, runtime_config)
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
