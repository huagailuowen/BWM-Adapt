#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

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
from make_gt_pred_comparison import (  # noqa: E402
    _default_pred_name,
    _read_gt_video,
    _read_pred_video,
    _resize_rgb,
)
from train_stage2_ttt import add_stage2_config  # noqa: E402
from wan_video_action.data import LoadCobotAction, RoboTwinUnifiedDataset, create_video_operator  # noqa: E402
from wan_video_action.parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config  # noqa: E402
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402

TTT_ADAPT_SCOPES = (
    "context",
    "adapter_gates",
    "context_adapter_gates",
    "adapter_lowrank",
    "context_adapter_lowrank",
    "adapter_all",
    "context_adapter_all",
)


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _build_support_dataset(args, runtime_config):
    with open(args.action_stat_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats

    special_operator_map = {}
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

    return RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.support_metadata_path,
        repeat=1,
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


def _group_rows(rows: list[dict], group_keys: str) -> dict[tuple, list[int]]:
    keys = tuple(part.strip() for part in group_keys.split(",") if part.strip())
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[tuple(row.get(key) for key in keys)].append(index)
    return groups


def _support_sort_key(row: dict):
    push_start = int(row.get("push_start", 60))
    start = int(row.get("start_frame", 0))
    is_push_focus = row.get("chunk_type") == "push_focus"
    return (
        0 if is_push_focus else 1,
        abs(start - max(0, push_start - 10)),
        int(row.get("episode_index", 0)),
        str(row.get("sample_id", "")),
    )


def _select_support_indices(
    *,
    support_rows: list[dict],
    support_groups: dict[tuple, list[int]],
    group_key: tuple,
    support_count: int,
    seed: int,
) -> list[int]:
    candidates = list(support_groups[group_key])
    stable_group_id = int(hashlib.sha1(str(group_key).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_group_id)
    candidates.sort(key=lambda index: _support_sort_key(support_rows[index]))

    selected = []
    used_episodes = set()
    for index in candidates:
        episode = support_rows[index].get("episode_index")
        if episode in used_episodes:
            continue
        selected.append(index)
        used_episodes.add(episode)
        if len(selected) >= support_count:
            return selected

    remaining = [index for index in candidates if index not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, support_count - len(selected))])
    return selected


def _freeze_pipe(pipe) -> None:
    for name in ("dit", "vae", "action_encoder", "physical_context_encoder", "physical_adapter_bank"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    pipe.eval()


def _uses_context_adaptation(scope: str) -> bool:
    return str(scope) == "context" or str(scope).startswith("context_")


def _adapter_kind(scope: str) -> str | None:
    scope = str(scope)
    if "adapter_gates" in scope:
        return "gates"
    if "adapter_lowrank" in scope:
        return "lowrank"
    if "adapter_all" in scope:
        return "all"
    return None


def _adapter_param_allowed(name: str, kind: str) -> bool:
    if kind == "gates":
        return name.endswith(".gate") or name.endswith(".cond_gate.weight")
    if kind == "lowrank":
        return (
            name.endswith(".gate")
            or name.endswith(".cond_gate.weight")
            or name.endswith(".down.weight")
            or name.endswith(".up.weight")
        )
    if kind == "all":
        return True
    raise ValueError(f"Unsupported adapter adaptation kind: {kind}.")


def _select_ttt_adapter_params(pipe, scope: str) -> list[tuple[str, torch.nn.Parameter]]:
    kind = _adapter_kind(scope)
    if kind is None:
        return []
    bank = getattr(pipe, "physical_adapter_bank", None)
    if bank is None:
        raise ValueError(f"ttt_adapt_scope={scope} requires physical_adapter_mode='residual'.")
    selected = [
        (name, param)
        for name, param in bank.named_parameters()
        if _adapter_param_allowed(name, kind)
    ]
    if not selected:
        raise ValueError(f"No adapter parameters selected for ttt_adapt_scope={scope}.")
    return selected


def _clone_named_params(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def _restore_named_params(
    named_params: list[tuple[str, torch.nn.Parameter]],
    state: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, param in named_params:
            if name not in state:
                raise KeyError(f"Missing adapted adapter parameter '{name}'.")
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def _set_requires_grad(
    named_params: list[tuple[str, torch.nn.Parameter]],
    enabled: bool,
) -> list[tuple[torch.nn.Parameter, bool]]:
    previous = []
    for _, param in named_params:
        previous.append((param, bool(param.requires_grad)))
        param.requires_grad_(enabled)
    return previous


def _restore_requires_grad(previous: list[tuple[torch.nn.Parameter, bool]]) -> None:
    for param, old_value in previous:
        param.requires_grad_(old_value)


def _adapter_reg_loss(
    named_params: list[tuple[str, torch.nn.Parameter]],
    base_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    if not named_params:
        raise ValueError("adapter_reg_loss requires non-empty adapter params.")
    losses = []
    for name, param in named_params:
        base = base_state[name].to(device=param.device, dtype=param.dtype)
        losses.append(torch.mean((param.float() - base.float()) ** 2))
    return torch.stack(losses).mean()


def _clip_grad_list(
    grads: list[torch.Tensor | None],
    max_norm: float,
) -> list[torch.Tensor | None]:
    if max_norm <= 0:
        return grads
    valid = [grad for grad in grads if grad is not None]
    if not valid:
        return grads
    norm_sq = torch.stack([grad.detach().float().pow(2).sum() for grad in valid]).sum()
    grad_norm = torch.sqrt(norm_sq).clamp_min(1e-6)
    scale = min(1.0, float(max_norm) / float(grad_norm.cpu()))
    return [None if grad is None else grad * scale for grad in grads]


def _count_params(named_params: list[tuple[str, torch.nn.Parameter]]) -> int:
    return int(sum(param.numel() for _, param in named_params))


def _video_geometry(video: torch.Tensor) -> tuple[int, int, int, int]:
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor (V,C,T,H,W), got {tuple(video.shape)}")
    num_views = int(video.shape[0])
    return int(video.shape[-2]) * num_views, int(video.shape[-1]), int(video.shape[2]), num_views


def _prepare_loss_inputs(pipe, data: dict, physical_context: torch.Tensor, args):
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
        "physical_context": physical_context,
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


def _shared_inputs(inputs):
    return inputs[0] if isinstance(inputs, tuple) else inputs


def _sample_timestep(pipe, inputs: dict, args) -> torch.Tensor:
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


def _context_reg(pipe, context: torch.Tensor) -> torch.Tensor:
    target = pipe.physical_context_encoder.default_context.detach().to(dtype=context.dtype, device=context.device)
    return torch.mean((context.float() - target.float()) ** 2)


def _parse_inner_lr_schedule(args) -> list[float]:
    raw = str(getattr(args, "stage2_inner_lr_schedule", "") or "").strip()
    if not raw:
        return [float(args.stage2_inner_lr)] * int(args.stage2_inner_steps)
    schedule = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            lr_text, count_text = item.split(":", 1)
        elif "x" in item:
            lr_text, count_text = item.split("x", 1)
        elif "*" in item:
            lr_text, count_text = item.split("*", 1)
        else:
            lr_text, count_text = item, "1"
        lr = float(lr_text.strip())
        count = int(count_text.strip())
        if count <= 0:
            raise ValueError(f"Invalid non-positive inner LR schedule count in {item!r}.")
        schedule.extend([lr] * count)
    if not schedule:
        raise ValueError(f"Empty stage2_inner_lr_schedule={raw!r}.")
    return schedule


def _clamp_context(context: torch.Tensor, args) -> torch.Tensor:
    clamp_min = getattr(args, "stage2_context_clamp_min", None)
    clamp_max = getattr(args, "stage2_context_clamp_max", None)
    if clamp_min is None and clamp_max is None:
        return context
    clamp_min = None if clamp_min is None else float(clamp_min)
    clamp_max = None if clamp_max is None else float(clamp_max)
    return torch.clamp(context, min=clamp_min, max=clamp_max)


def _adapt_ttt_state(
    pipe,
    support_items: list[dict],
    args,
    *,
    adapter_named_params: list[tuple[str, torch.nn.Parameter]],
    base_adapter_state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[float], dict[str, torch.Tensor], dict[str, float]]:
    if pipe.physical_context_encoder is None:
        raise ValueError("Stage2 TTT inference requires physical_context_mode != 'none'.")

    scope = str(args.ttt_adapt_scope)
    use_context = _uses_context_adaptation(scope)
    use_adapter = bool(adapter_named_params)
    inner_lrs = _parse_inner_lr_schedule(args)
    pipe.scheduler.set_timesteps(1000, training=True)
    context0 = pipe.physical_context_encoder.default_context.detach().clone().to(dtype=pipe.torch_dtype, device=pipe.device)
    context = context0.clone()
    if use_context:
        context = context.requires_grad_(True)
    if use_adapter:
        _restore_named_params(adapter_named_params, base_adapter_state)
    previous_requires_grad = _set_requires_grad(adapter_named_params, True)

    losses = []
    adapter_reg_value = 0.0
    try:
        for inner_idx, inner_lr in enumerate(inner_lrs):
            support_losses = []
            for item in support_items:
                inputs = _prepare_loss_inputs(pipe, item, context, args)
                support_losses.append(_flow_match_loss(pipe, inputs, args))
            support_loss = torch.stack(support_losses).mean()
            loss = support_loss
            context_reg_value = 0.0
            if use_context and float(args.stage2_context_reg_weight) > 0:
                context_reg = _context_reg(pipe, context)
                context_reg_value = float(context_reg.detach().float().cpu())
                loss = loss + float(args.stage2_context_reg_weight) * context_reg
            if use_adapter and float(args.ttt_adapter_reg_weight) > 0:
                adapter_reg = _adapter_reg_loss(adapter_named_params, base_adapter_state)
                adapter_reg_value = float(adapter_reg.detach().float().cpu())
                loss = loss + float(args.ttt_adapter_reg_weight) * adapter_reg

            grad_targets: list[torch.Tensor] = []
            if use_context:
                grad_targets.append(context)
            grad_targets.extend(param for _, param in adapter_named_params)
            grads = torch.autograd.grad(loss, grad_targets, create_graph=False, allow_unused=True)

            offset = 0
            if use_context:
                context_grad = grads[0]
                if context_grad is None:
                    context_grad = torch.zeros_like(context)
                if float(args.stage2_inner_grad_clip) > 0:
                    grad_norm = context_grad.float().norm().clamp_min(1e-6)
                    scale = min(1.0, float(args.stage2_inner_grad_clip) / float(grad_norm.detach().cpu()))
                    context_grad = context_grad * scale
                context = _clamp_context(context - float(inner_lr) * context_grad, args).detach().requires_grad_(True)
                offset = 1

            if use_adapter:
                adapter_grads = _clip_grad_list(list(grads[offset:]), float(args.ttt_adapter_grad_clip))
                with torch.no_grad():
                    for (_, param), grad in zip(adapter_named_params, adapter_grads):
                        if grad is not None:
                            param.add_(grad, alpha=-float(args.ttt_adapter_lr))

            losses.append(float(loss.detach().float().cpu()))
            print(
                f"[inner] step={inner_idx + 1} scope={scope} loss={losses[-1]:.6f} "
                f"support={float(support_loss.detach().float().cpu()):.6f} "
                f"context_reg={context_reg_value:.6f} adapter_reg={adapter_reg_value:.6f} "
                f"inner_lr={float(inner_lr):.6f} c_mean={float(context.detach().float().mean().cpu()):.6f}",
                flush=True,
            )
    finally:
        _restore_requires_grad(previous_requires_grad)

    adapted_adapter_state = _clone_named_params(adapter_named_params) if use_adapter else {}
    metrics = {
        "adapter_params": float(_count_params(adapter_named_params)),
        "adapter_reg": float(adapter_reg_value),
        "context_initial_mean": float(context0.detach().float().mean().cpu()),
        "context_adapted_mean": float(context.detach().float().mean().cpu()),
        "context_delta_mean": float((context.detach().float() - context0.detach().float()).mean().cpu()),
        "inner_steps": float(len(inner_lrs)),
    }
    return context.detach(), losses, adapted_adapter_state, metrics


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(88, 9 * len(label) + 12)
    draw.rectangle((0, 0, width, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _write_three_way_comparison(
    *,
    row: dict,
    sample_index: int,
    dataset_base_path: Path,
    stage1_pred_path: Path,
    stage2_pred_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
) -> Path:
    num_views = len(row["video"])
    target_width = width * num_views
    total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
    gt_frames = _read_gt_video(row, dataset_base_path, width, height, total_frames)
    stage1_frames = _read_pred_video(stage1_pred_path, target_width, height)
    stage2_frames = _read_pred_video(stage2_pred_path, target_width, height)
    total = min(len(gt_frames), len(stage1_frames), len(stage2_frames))
    if total <= 0:
        raise RuntimeError(f"No frames to compare for sample_index={sample_index}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    name = _default_pred_name(sample_index, row).replace(".mp4", "_gt_stage1_stage2ttt.mp4")
    output_path = output_dir / name
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=quality) as writer:
        for frame_id in range(total):
            gt = _draw_label(_resize_rgb(gt_frames[frame_id], target_width, height), "GT")
            stage1 = _draw_label(_resize_rgb(stage1_frames[frame_id], target_width, height), "STAGE1")
            stage2 = _draw_label(_resize_rgb(stage2_frames[frame_id], target_width, height), "STAGE2+TTT")
            writer.append_data(np.concatenate([gt, stage1, stage2], axis=0))
    return output_path


def _write_two_way_comparison(
    *,
    row: dict,
    sample_index: int,
    dataset_base_path: Path,
    pred_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    quality: int,
    pred_label: str,
) -> Path:
    num_views = len(row["video"])
    target_width = width * num_views
    total_frames = int(row.get("length", row["end_frame"] - row["start_frame"] + 1))
    gt_frames = _read_gt_video(row, dataset_base_path, width, height, total_frames)
    pred_frames = _read_pred_video(pred_path, target_width, height)
    total = min(len(gt_frames), len(pred_frames))
    if total <= 0:
        raise RuntimeError(f"No frames to compare for sample_index={sample_index}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    name = _default_pred_name(sample_index, row).replace(".mp4", f"_gt_{pred_label.lower()}.mp4")
    output_path = output_dir / name
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=quality) as writer:
        for frame_id in range(total):
            gt = _draw_label(_resize_rgb(gt_frames[frame_id], target_width, height), "GT")
            pred = _draw_label(_resize_rgb(pred_frames[frame_id], target_width, height), pred_label)
            writer.append_data(np.concatenate([gt, pred], axis=0))
    return output_path


def parse_args():
    parser = argparse.ArgumentParser("Stage2 TTT test-time inference and comparison.")
    parser = add_stage2_config(add_general_config(parser))
    parser.add_argument("--stage2_ckpt_path", type=str, required=True)
    parser.add_argument("--support_metadata_path", type=str, default="data/push_box_bwm_calibrated_v2_100pairs/train.jsonl")
    parser.add_argument("--support_count", type=int, default=2)
    parser.add_argument(
        "--ttt_adapt_scope",
        type=str,
        default="context",
        choices=TTT_ADAPT_SCOPES,
        help=(
            "Test-time trainable subset. Default 'context' preserves the medium-C inner loop. "
            "'context_adapter_gates' adds conservative local gate/cond-gate updates with far fewer "
            "parameters than adapting the whole adapter bank. These adapter scopes are ablations "
            "inside the latent-C branch; use scripts/infer_stage2_ttte2e.py for the separate "
            "TTT-E2E mild branch."
        ),
    )
    parser.add_argument("--ttt_adapter_lr", type=float, default=1e-3)
    parser.add_argument("--ttt_adapter_grad_clip", type=float, default=0.1)
    parser.add_argument("--ttt_adapter_reg_weight", type=float, default=1e-4)
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--baseline_pred_dir", type=str, default=None)
    parser.add_argument("--comparison_output_path", type=str, default=None)
    parser.add_argument("--render_support", action="store_true", default=False)
    parser.add_argument("--support_output_path", type=str, default=None)
    parser.add_argument("--support_comparison_output_path", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

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
            f"[plan] query={sample_index} group={group_key} "
            f"support={support_plan[group_key]}",
            flush=True,
        )

    output_path = Path(args.output_path)
    comparison_path = Path(args.comparison_output_path) if args.comparison_output_path else output_path.parent / "comparison_videos"
    support_output_path = Path(args.support_output_path) if args.support_output_path else output_path.parent / "show_raw"
    support_comparison_path = (
        Path(args.support_comparison_output_path)
        if args.support_comparison_output_path
        else output_path.parent / "show_comparison_videos"
    )
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
    adapter_named_params = _select_ttt_adapter_params(pipe, args.ttt_adapt_scope)
    base_adapter_state = _clone_named_params(adapter_named_params)
    print(
        f"[ttt_scope] scope={args.ttt_adapt_scope} "
        f"context={_uses_context_adaptation(args.ttt_adapt_scope)} "
        f"adapter_tensors={len(adapter_named_params)} adapter_params={_count_params(adapter_named_params)} "
        f"adapter_lr={float(args.ttt_adapter_lr):g} adapter_reg={float(args.ttt_adapter_reg_weight):g}",
        flush=True,
    )

    context_cache: dict[tuple, tuple[torch.Tensor, list[float], dict[str, torch.Tensor], dict[str, float]]] = {}
    rendered_support = set()
    result_rows = []
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path.mkdir(parents=True, exist_ok=True)
    if args.render_support:
        support_output_path.mkdir(parents=True, exist_ok=True)
        support_comparison_path.mkdir(parents=True, exist_ok=True)

    for sample_index in sample_indices:
        sample_index = int(sample_index)
        row = query_rows[sample_index]
        group_key = tuple(row.get(name.strip()) for name in args.stage2_group_keys.split(",") if name.strip())
        pred_name = _default_pred_name(sample_index, row)
        pred_path = output_path / pred_name

        if group_key not in context_cache:
            support_indices = support_plan[group_key]
            print(f"[adapt] group={group_key} support={support_indices}", flush=True)
            support_items = [support_dataset[index] for index in support_indices]
            context_cache[group_key] = _adapt_ttt_state(
                pipe,
                support_items,
                args,
                adapter_named_params=adapter_named_params,
                base_adapter_state=base_adapter_state,
            )
            torch.cuda.empty_cache()

        adapted_context, inner_losses, adapted_adapter_state, adapt_metrics = context_cache[group_key]
        if adapter_named_params:
            _restore_named_params(adapter_named_params, adapted_adapter_state)
        if args.render_support and group_key not in rendered_support:
            for support_index in support_plan[group_key]:
                support_row = support_rows[int(support_index)]
                support_pred_name = _default_pred_name(int(support_index), support_row)
                support_pred_path = support_output_path / support_pred_name
                if not (support_pred_path.exists() and args.skip_existing):
                    support_sample = query_dataset[int(support_index)]
                    support_sample = prepare_sample_for_rollout(support_sample, int(support_index), pipe, args)
                    support_sample["physical_context"] = adapted_context
                    support_sample["output_path"] = str(support_pred_path)
                    print(
                        f"[support_sample] group={group_key} support_index={support_index} "
                        f"episode={support_sample['episode_index']} output={support_pred_path}",
                        flush=True,
                    )
                    _run_autoregressive(pipe=pipe, sample=support_sample, args=args)
                    torch.cuda.empty_cache()
                support_comparison = _write_two_way_comparison(
                    row=support_row,
                    sample_index=int(support_index),
                    dataset_base_path=Path(args.dataset_base_path),
                    pred_path=support_pred_path,
                    output_dir=support_comparison_path,
                    width=int(args.width),
                    height=int(args.height),
                    fps=int(args.fps),
                    quality=int(args.quality),
                    pred_label="SHOW+TTT",
                )
                print(f"[support_comparison] {support_comparison}", flush=True)
            rendered_support.add(group_key)
        if pred_path.exists() and args.skip_existing:
            print(f"[skip] existing prediction {pred_path}", flush=True)
        else:
            sample = query_dataset[sample_index]
            sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
            sample["physical_context"] = adapted_context
            print(
                f"[sample] sample_index={sample_index} group={group_key} "
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
                print(f"[comparison_skip] missing stage2 prediction {pred_path}", flush=True)
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
                print(f"[comparison] {comparison_file}", flush=True)

        result_rows.append(
            {
                "sample_index": sample_index,
                "sample_id": row.get("sample_id"),
                "episode_index": row.get("episode_index"),
                "group_key": list(group_key),
                "friction_mu": row.get("friction_mu"),
                "target_c": None if row.get("friction_mu") is None else float(row.get("friction_mu")) / 0.25,
                "support_indices": support_plan[group_key],
                "ttt_adapt_scope": args.ttt_adapt_scope,
                "inner_losses": inner_losses,
                "prediction_path": str(pred_path),
                "comparison_path": None if comparison_file is None else str(comparison_file),
                "context_norm": float(adapted_context.float().norm().cpu()),
                "context_initial_mean": float(adapt_metrics.get("context_initial_mean", float("nan"))),
                "context_adapted_mean": float(adapt_metrics.get("context_adapted_mean", float("nan"))),
                "context_delta_mean": float(adapt_metrics.get("context_delta_mean", float("nan"))),
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
