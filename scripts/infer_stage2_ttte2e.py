#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import (  # noqa: E402
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from make_gt_pred_comparison import _default_pred_name  # noqa: E402
from train_stage2_ttt import add_stage2_config  # noqa: E402
from train_stage2_ttte2e import add_ttte2e_config  # noqa: E402
from wan_video_action.parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config  # noqa: E402
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402

from infer_stage2_ttt import (  # noqa: E402
    _adapter_reg_loss,
    _build_support_dataset,
    _clip_grad_list,
    _clone_named_params,
    _count_params,
    _freeze_pipe,
    _group_rows,
    _read_jsonl,
    _restore_named_params,
    _restore_requires_grad,
    _select_support_indices,
    _set_requires_grad,
    _shared_inputs,
    _video_geometry,
    _write_jsonl,
    _write_three_way_comparison,
)


def _adapter_param_allowed(name: str, scope: str) -> bool:
    if scope == "gates":
        return name.endswith(".gate")
    if scope == "lowrank":
        return name.endswith(".gate") or name.endswith(".down.weight") or name.endswith(".up.weight")
    if scope == "all":
        return True
    raise ValueError(f"Unsupported TTT-E2E parameter scope: {scope}.")


def _select_ttte2e_adapter_params(pipe, scope: str) -> list[tuple[str, torch.nn.Parameter]]:
    bank = getattr(pipe, "physical_adapter_bank", None)
    if bank is None:
        raise ValueError("TTT-E2E mild inference requires physical_adapter_mode='residual'.")
    selected = [
        (name, param)
        for name, param in bank.named_parameters()
        if _adapter_param_allowed(name, scope)
    ]
    if not selected:
        raise ValueError(f"No adapter parameters selected for scope={scope}.")
    return selected


def _prepare_loss_inputs(pipe, data: dict, args):
    height, width, num_frames, num_views = _video_geometry(data["video"])
    inputs_shared = {
        "input_video": data["video"],
        "action": data.get("action"),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_views": num_views,
        "num_history_frames": int(args.num_history_frames),
        "cfg_scale": 1,
        "tiled": False,
        "rand_device": pipe.device,
        "use_gradient_checkpointing": bool(args.use_gradient_checkpointing),
        "use_gradient_checkpointing_offload": bool(args.use_gradient_checkpointing_offload),
        "max_timestep_boundary": float(args.max_timestep_boundary),
        "min_timestep_boundary": float(args.min_timestep_boundary),
    }
    inputs_posi = {"prompt": data.get("prompt"), "prompt_emb": data.get("prompt_emb")}
    inputs_nega = {"negative_prompt": data.get("negative_prompt"), "prompt_emb": data.get("negative_prompt_emb")}
    inputs = (inputs_shared, inputs_posi, inputs_nega)
    for unit in pipe.units:
        inputs = pipe.unit_runner(unit, pipe, *inputs)
    return inputs


def _sample_timestep(pipe, inputs, args) -> torch.Tensor:
    inputs = _shared_inputs(inputs)
    max_idx = int(float(args.max_timestep_boundary) * len(pipe.scheduler.timesteps))
    min_idx = int(float(args.min_timestep_boundary) * len(pipe.scheduler.timesteps))
    max_idx = max(min_idx + 1, max_idx)
    timestep_id = torch.randint(min_idx, max_idx, (1,))
    return pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)


def _flow_match_loss(pipe, inputs, args) -> torch.Tensor:
    inputs = dict(_shared_inputs(inputs))
    timestep = _sample_timestep(pipe, inputs, args)
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    if "first_frame_latents" in inputs:
        pred = pred[:, :, 1:]
        target = target[:, :, 1:]
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    return loss * pipe.scheduler.training_weight(timestep)


def _parse_ttte2e_inner_lr_schedule(args) -> list[float]:
    raw = str(getattr(args, "ttte2e_inner_lr_schedule", "") or "").strip()
    if not raw:
        return [float(args.ttte2e_inner_lr)] * int(args.stage2_inner_steps)
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "@" in item:
            count_text, lr_text = item.split("@", 1)
            values.extend([float(lr_text)] * int(count_text))
        else:
            values.append(float(item))
    if len(values) != int(args.stage2_inner_steps):
        raise ValueError(
            f"ttte2e_inner_lr_schedule expands to {len(values)} values, "
            f"but stage2_inner_steps={args.stage2_inner_steps}."
        )
    return values


def _adapt_adapter_state(
    pipe,
    support_items: list[dict],
    args,
    *,
    adapter_named_params: list[tuple[str, torch.nn.Parameter]],
    base_adapter_state: dict[str, torch.Tensor],
) -> tuple[list[float], dict[str, torch.Tensor], dict[str, float]]:
    pipe.scheduler.set_timesteps(1000, training=True)
    inner_lrs = _parse_ttte2e_inner_lr_schedule(args)
    _restore_named_params(adapter_named_params, base_adapter_state)
    previous_requires_grad = _set_requires_grad(adapter_named_params, True)

    losses = []
    adapter_reg_value = 0.0
    try:
        for inner_idx, inner_lr in enumerate(inner_lrs):
            support_losses = []
            for item in support_items:
                inputs = _prepare_loss_inputs(pipe, item, args)
                support_losses.append(_flow_match_loss(pipe, inputs, args))
            support_loss = torch.stack(support_losses).mean()
            loss = support_loss
            if float(args.ttte2e_inner_reg_weight) > 0:
                adapter_reg = _adapter_reg_loss(adapter_named_params, base_adapter_state)
                adapter_reg_value = float(adapter_reg.detach().float().cpu())
                loss = loss + float(args.ttte2e_inner_reg_weight) * adapter_reg

            grads = torch.autograd.grad(
                loss,
                [param for _, param in adapter_named_params],
                create_graph=False,
                allow_unused=True,
            )
            grads = _clip_grad_list(list(grads), float(args.ttte2e_inner_grad_clip))
            with torch.no_grad():
                for (_, param), grad in zip(adapter_named_params, grads):
                    if grad is not None:
                        param.add_(grad, alpha=-float(inner_lr))
            losses.append(float(loss.detach().float().cpu()))
            print(
                f"[inner_ttte2e] step={inner_idx + 1} loss={losses[-1]:.6f} "
                f"support={float(support_loss.detach().float().cpu()):.6f} "
                f"adapter_reg={adapter_reg_value:.6f} inner_lr={float(inner_lr):.6f}",
                flush=True,
            )
    finally:
        _restore_requires_grad(previous_requires_grad)

    adapted_adapter_state = _clone_named_params(adapter_named_params)
    metrics = {
        "adapter_params": float(_count_params(adapter_named_params)),
        "adapter_reg": float(adapter_reg_value),
    }
    return losses, adapted_adapter_state, metrics


def parse_args():
    parser = argparse.ArgumentParser("Stage2 TTT-E2E mild test-time inference and comparison.")
    parser = add_ttte2e_config(add_stage2_config(add_general_config(parser)))
    parser.add_argument("--stage2_ckpt_path", type=str, required=True)
    parser.add_argument("--support_metadata_path", type=str, default="data/push_box_bwm_friction_9mu_450/train.jsonl")
    parser.add_argument("--support_count", type=int, default=2)
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--baseline_pred_dir", type=str, default=None)
    parser.add_argument("--comparison_output_path", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    if str(args.physical_context_mode).lower() != "none":
        raise ValueError("TTT-E2E mild inference expects physical_context_mode='none'.")

    query_rows = _read_jsonl(args.dataset_metadata_path)
    support_rows = _read_jsonl(args.support_metadata_path)
    support_groups = _group_rows(support_rows, args.stage2_group_keys)

    sample_indices = _parse_sample_indices(args.sample_indices)
    if sample_indices is None:
        start = int(args.start_index)
        end = len(query_rows) if not args.max_samples else min(len(query_rows), start + int(args.max_samples))
        sample_indices = list(range(start, end))
    if args.max_samples and args.sample_indices is not None:
        sample_indices = sample_indices[: int(args.max_samples)]

    support_plan = {}
    for sample_index in sample_indices:
        row = query_rows[int(sample_index)]
        group_key = tuple(row.get(name.strip()) for name in args.stage2_group_keys.split(",") if name.strip())
        if group_key not in support_groups:
            raise KeyError(f"No support rows for group={group_key}.")
        if group_key not in support_plan:
            support_plan[group_key] = _select_support_indices(
                support_rows=support_rows,
                support_groups=support_groups,
                group_key=group_key,
                support_count=int(args.support_count),
                seed=int(args.seed),
            )
        print(
            f"[plan_ttte2e] query={sample_index} group={group_key} "
            f"support={support_plan[group_key]}",
            flush=True,
        )

    output_path = Path(args.output_path)
    comparison_path = Path(args.comparison_output_path) if args.comparison_output_path else output_path.parent / "comparison_videos"
    plan_rows = []
    for group_key, indices in sorted(support_plan.items(), key=lambda item: str(item[0])):
        plan_rows.append(
            {
                "group_key": list(group_key),
                "support_indices": indices,
                "support_sample_ids": [support_rows[index].get("sample_id") for index in indices],
                "support_episode_indices": [support_rows[index].get("episode_index") for index in indices],
            }
        )
    _write_jsonl(output_path.parent / "support_plan.jsonl", plan_rows)
    if args.dry_run:
        return

    runtime_config = prepare_runtime_config(args)
    support_dataset = _build_support_dataset(args, runtime_config)
    query_dataset = build_infer_dataset(args)

    pipe = build_pipeline(args)
    load_checkpoint_weights(pipe, args.stage2_ckpt_path)
    _freeze_pipe(pipe)
    adapter_named_params = _select_ttte2e_adapter_params(pipe, args.ttte2e_inner_param_scope)
    base_adapter_state = _clone_named_params(adapter_named_params)
    print(
        f"[ttte2e_scope] scope={args.ttte2e_inner_param_scope} "
        f"adapter_tensors={len(adapter_named_params)} adapter_params={_count_params(adapter_named_params)} "
        f"lr={float(args.ttte2e_inner_lr):g} reg={float(args.ttte2e_inner_reg_weight):g}",
        flush=True,
    )

    adapter_cache: dict[tuple, tuple[list[float], dict[str, torch.Tensor], dict[str, float]]] = {}
    result_rows = []
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path.mkdir(parents=True, exist_ok=True)

    for sample_index in sample_indices:
        sample_index = int(sample_index)
        row = query_rows[sample_index]
        group_key = tuple(row.get(name.strip()) for name in args.stage2_group_keys.split(",") if name.strip())
        pred_name = _default_pred_name(sample_index, row)
        pred_path = output_path / pred_name

        if group_key not in adapter_cache:
            support_indices = support_plan[group_key]
            print(f"[adapt_ttte2e] group={group_key} support={support_indices}", flush=True)
            support_items = [support_dataset[index] for index in support_indices]
            adapter_cache[group_key] = _adapt_adapter_state(
                pipe,
                support_items,
                args,
                adapter_named_params=adapter_named_params,
                base_adapter_state=base_adapter_state,
            )
            torch.cuda.empty_cache()

        inner_losses, adapted_adapter_state, adapt_metrics = adapter_cache[group_key]
        _restore_named_params(adapter_named_params, adapted_adapter_state)
        if pred_path.exists() and args.skip_existing:
            print(f"[skip] existing prediction {pred_path}", flush=True)
        else:
            sample = query_dataset[sample_index]
            sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
            print(
                f"[sample_ttte2e] sample_index={sample_index} group={group_key} "
                f"episode={sample['episode_index']} output={pred_path}",
                flush=True,
            )
            _run_autoregressive(pipe=pipe, sample=sample, args=args)
            torch.cuda.empty_cache()

        comparison_file = None
        if args.baseline_pred_dir:
            stage1_pred_path = Path(args.baseline_pred_dir) / pred_name
            if not stage1_pred_path.exists():
                print(f"[comparison_skip] missing stage1 baseline {stage1_pred_path}", flush=True)
            elif not pred_path.exists():
                print(f"[comparison_skip] missing TTT-E2E prediction {pred_path}", flush=True)
            else:
                comparison_file = _write_three_way_comparison(
                    row=row,
                    sample_index=sample_index,
                    dataset_base_path=Path(args.dataset_base_path),
                    stage1_pred_path=stage1_pred_path,
                    stage2_pred_path=pred_path,
                    output_dir=comparison_path,
                    width=int(args.width),
                    height=int(args.height),
                    fps=int(args.fps),
                    quality=int(args.quality),
                )
                print(f"[comparison_ttte2e] {comparison_file}", flush=True)

        result_rows.append(
            {
                "sample_index": sample_index,
                "sample_id": row.get("sample_id"),
                "episode_index": row.get("episode_index"),
                "group_key": list(group_key),
                "support_indices": support_plan[group_key],
                "ttte2e_inner_param_scope": args.ttte2e_inner_param_scope,
                "inner_losses": inner_losses,
                "prediction_path": str(pred_path),
                "comparison_path": None if comparison_file is None else str(comparison_file),
                "adapter_param_count": int(adapt_metrics.get("adapter_params", 0)),
                "adapter_reg": float(adapt_metrics.get("adapter_reg", 0.0)),
            }
        )
        _write_jsonl(output_path.parent / "results.jsonl", result_rows)

    print(f"[done] predictions={output_path}", flush=True)
    print(f"[done] comparisons={comparison_path}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
