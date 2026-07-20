#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from infer import _run_autoregressive, build_infer_dataset, build_pipeline, prepare_sample_for_rollout
from infer_stage2_ttt import (
    _build_support_dataset,
    _compose_known_physical_context,
    _default_pred_name,
    _freeze_pipe,
    _prepare_loss_inputs,
    _read_jsonl,
    _shared_inputs,
    parse_args,
)
from wan_video_action.parsers import prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
from wan_video_action.utils import set_global_seed


def _parse_cases(raw: str) -> list[tuple[int, int]]:
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        source_text, target_text = item.split(":", 1)
        cases.append((int(source_text), int(target_text)))
    if not cases:
        raise ValueError("ACTION_INV_CASES must contain source_action5:target pairs.")
    return cases


def _final_context(
    trajectory_rows: list[dict],
    sample_index: int,
    reference: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    rows = [row for row in trajectory_rows if int(row.get("sample_index", -1)) == sample_index]
    if not rows:
        raise ValueError(f"No Stage2 context trajectory found for sample_index={sample_index}.")
    row = max(rows, key=lambda value: int(value["inner_step"]))
    context = torch.tensor(row["context_flat"], dtype=torch.float32, device=device)
    context = context.reshape(reference.shape).detach()
    return context, int(row["inner_step"])


def _terminal_only_item(item: dict) -> dict:
    result = dict(item)
    video = item["video"].detach()
    if video.ndim != 5 or video.shape[2] < 2:
        raise ValueError(f"Expected video (V,C,T,H,W), got {tuple(video.shape)}.")
    # Only the observed first frame and requested final frame are retained.
    # Unknown intermediate frames are never copied from GT.
    terminal = video[:, :, -1:].clone()
    known_video = video[:, :, 0:1].expand_as(video).clone()
    known_video[:, :, -1:] = terminal
    result["video"] = known_video
    return result


def _fixed_timestep(pipe, fraction: float) -> torch.Tensor:
    fraction = min(1.0, max(0.0, float(fraction)))
    index = int(round(fraction * (len(pipe.scheduler.timesteps) - 1)))
    timestep = pipe.scheduler.timesteps[index]
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    return timestep.to(dtype=pipe.torch_dtype, device=pipe.device)


def _terminal_flow_loss(
    pipe,
    item: dict,
    action: torch.Tensor,
    context: torch.Tensor,
    args,
    *,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    loss_frames: str,
) -> torch.Tensor:
    data = dict(item)
    data["action"] = action
    inputs = dict(_shared_inputs(_prepare_loss_inputs(pipe, data, context, args)))
    if tuple(inputs["input_latents"].shape) != tuple(noise.shape):
        raise ValueError(
            f"Fixed noise shape {tuple(noise.shape)} does not match latent shape "
            f"{tuple(inputs['input_latents'].shape)}."
        )
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    if "first_frame_latents" in inputs:
        pred = pred[:, :, 1:]
        target = target[:, :, 1:]
    if pred.shape[2] < 1:
        raise ValueError("No future latent frame remains after removing history conditioning.")

    if loss_frames == "final":
        pred_for_loss = pred[:, :, -1:]
        target_for_loss = target[:, :, -1:]
    elif loss_frames == "all":
        pred_for_loss = pred
        target_for_loss = target
    else:
        raise ValueError(f"Unsupported ACTION_INV_LOSS_FRAMES={loss_frames!r}.")
    loss = torch.nn.functional.mse_loss(pred_for_loss.float(), target_for_loss.float())
    return loss * pipe.scheduler.training_weight(timestep).mean()


def _assemble_action(head: torch.Tensor, tail: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.cat([head, tail], dim=-1).to(dtype=dtype)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _render(
    *,
    pipe,
    query_dataset,
    metadata_row: dict,
    sample_index: int,
    action: torch.Tensor,
    context: torch.Tensor,
    output_dir: Path,
    args,
) -> Path:
    sample = prepare_sample_for_rollout(query_dataset[sample_index], sample_index, pipe, args)
    reference_action = sample["action"]
    sample["action"] = action.detach().to(
        device=reference_action.device,
        dtype=reference_action.dtype,
    )
    sample["physical_context"] = _compose_known_physical_context(sample, context, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _default_pred_name(sample_index, metadata_row)
    sample["output_path"] = str(output_path)
    _run_autoregressive(pipe=pipe, sample=sample, args=args)
    return output_path


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    cases = _parse_cases(os.environ.get("ACTION_INV_CASES", "205:208,175:178"))
    trajectory_path = Path(os.environ["ACTION_INV_CONTEXT_TRAJECTORY"])
    output_root = Path(args.output_path)
    timestep_fraction = float(os.environ.get("ACTION_INV_TIMESTEP_FRACTION", "0.5"))
    clamp_min = float(os.environ.get("ACTION_INV_CLAMP_MIN", "-1.0"))
    clamp_max = float(os.environ.get("ACTION_INV_CLAMP_MAX", "1.0"))
    effective_dims = int(os.environ.get("ACTION_INV_EFFECTIVE_DIMS", "8"))
    loss_frames = os.environ.get("ACTION_INV_LOSS_FRAMES", "final").strip().lower()
    if loss_frames not in {"final", "all"}:
        raise ValueError(f"ACTION_INV_LOSS_FRAMES must be final or all, got {loss_frames!r}.")
    optimized_dims_raw = os.environ.get("ACTION_INV_OPT_DIMS", "").strip()
    optimized_dims = (
        [int(value.strip()) for value in optimized_dims_raw.split(",") if value.strip()]
        if optimized_dims_raw
        else list(range(effective_dims))
    )
    learning_rates = [1.0] * 30 + [0.5] * 30 + [0.1] * 30

    output_root.mkdir(parents=True, exist_ok=True)
    runtime_config = prepare_runtime_config(args)
    optimization_dataset = _build_support_dataset(args, runtime_config)
    query_dataset = build_infer_dataset(args)
    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    trajectory_rows = _read_jsonl(trajectory_path)

    pipe = build_pipeline(args)
    load_checkpoint_weights(pipe, args.stage2_ckpt_path)
    _freeze_pipe(pipe)
    pipe.scheduler.set_timesteps(1000, training=True)
    context_reference = pipe.physical_context_encoder.default_context.detach()
    timestep = _fixed_timestep(pipe, timestep_fraction)

    result_rows = []
    all_step_rows = []
    for source_index, target_index in cases:
        source_row = metadata_rows[source_index]
        target_row = metadata_rows[target_index]
        if int(source_row.get("action_id", -1)) != 5:
            raise ValueError(f"Source sample {source_index} is not action-5: {source_row.get('action_id')}.")
        if float(source_row["friction_mu"]) != float(target_row["friction_mu"]):
            raise ValueError(
                f"Source/target friction mismatch: {source_row['friction_mu']} vs {target_row['friction_mu']}."
            )

        context, context_step = _final_context(
            trajectory_rows,
            source_index,
            context_reference,
            device=pipe.device,
        )
        source_item = optimization_dataset[source_index]
        target_item = optimization_dataset[target_index]
        loss_item = _terminal_only_item(target_item) if loss_frames == "final" else dict(target_item)
        source_action = torch.as_tensor(source_item["action"], device=pipe.device, dtype=torch.float32)
        target_action = torch.as_tensor(target_item["action"], device=pipe.device, dtype=torch.float32)
        if source_action.shape != target_action.shape or source_action.shape[-1] != int(args.action_dim):
            raise ValueError(
                f"Unexpected action shapes source={tuple(source_action.shape)} "
                f"target={tuple(target_action.shape)} action_dim={args.action_dim}."
            )
        if not 0 < effective_dims <= source_action.shape[-1]:
            raise ValueError(f"Invalid effective action dimensions: {effective_dims}.")
        if not optimized_dims or any(dim < 0 or dim >= effective_dims for dim in optimized_dims):
            raise ValueError(
                f"ACTION_INV_OPT_DIMS={optimized_dims} must select dimensions in [0, {effective_dims})."
            )

        action_head = source_action[..., :effective_dims].clone().requires_grad_(True)
        action_tail = source_action[..., effective_dims:].clone()
        initial_head = action_head.detach().clone()

        # The same noise and midpoint training timestep are used for all 90
        # updates, so changes in loss are caused by action rather than resampling.
        with torch.no_grad():
            seed = int(args.seed) + target_index
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            probe_inputs = dict(
                _shared_inputs(
                    _prepare_loss_inputs(
                        pipe,
                        {**loss_item, "action": source_action.to(dtype=pipe.torch_dtype)},
                        context,
                        args,
                    )
                )
            )
            fixed_noise = torch.randn(
                probe_inputs["input_latents"].shape,
                generator=generator,
                device=probe_inputs["input_latents"].device,
                dtype=probe_inputs["input_latents"].dtype,
            )
            gt_action_loss = _terminal_flow_loss(
                pipe,
                loss_item,
                target_action.to(dtype=pipe.torch_dtype),
                context,
                args,
                timestep=timestep,
                noise=fixed_noise,
                loss_frames=loss_frames,
            )

        case_rows = []
        initial_loss = None
        for update_step, lr in enumerate(learning_rates, start=1):
            action = _assemble_action(action_head, action_tail, pipe.torch_dtype)
            loss = _terminal_flow_loss(
                pipe,
                loss_item,
                action,
                context,
                args,
                timestep=timestep,
                noise=fixed_noise,
                loss_frames=loss_frames,
            )
            if initial_loss is None:
                initial_loss = float(loss.detach().cpu())
            grad = torch.autograd.grad(loss, action_head, retain_graph=False, create_graph=False)[0]
            grad_mask = torch.zeros_like(grad)
            grad_mask[..., optimized_dims] = 1
            grad = grad * grad_mask
            grad_norm = float(grad.detach().float().norm().cpu())
            with torch.no_grad():
                action_head.add_(grad, alpha=-float(lr))
                action_head.clamp_(min=clamp_min, max=clamp_max)

            step_row = {
                "source_index": source_index,
                "target_index": target_index,
                "friction_mu": float(target_row["friction_mu"]),
                "source_action_id": int(source_row["action_id"]),
                "target_action_id": int(target_row["action_id"]),
                "context_inner_step": context_step,
                "optimized_action_dims": optimized_dims,
                "action_update_step": update_step,
                "lr": float(lr),
                "final_frame_flow_loss": float(loss.detach().cpu()),
                "action_grad_norm": grad_norm,
                "action_delta_l2": float((action_head.detach() - initial_head).float().norm().cpu()),
                "action_first8_min": float(action_head.detach().min().cpu()),
                "action_first8_max": float(action_head.detach().max().cpu()),
                "action_first8": action_head.detach().float().cpu().tolist(),
            }
            case_rows.append(step_row)
            all_step_rows.append(step_row)
            print(
                f"[action_step] source={source_index} target={target_index} "
                f"step={update_step:03d}/090 lr={lr:g} loss={step_row['final_frame_flow_loss']:.8f} "
                f"grad={grad_norm:.6g} delta={step_row['action_delta_l2']:.6g}",
                flush=True,
            )
            _write_jsonl(output_root / "action_optimization_trajectory.jsonl", all_step_rows)

        learned_action = _assemble_action(action_head.detach(), action_tail, torch.float32)
        with torch.no_grad():
            final_loss = _terminal_flow_loss(
                pipe,
                loss_item,
                learned_action.to(dtype=pipe.torch_dtype),
                context,
                args,
                timestep=timestep,
                noise=fixed_noise,
                loss_frames=loss_frames,
            )

        action5_path = _render(
            pipe=pipe,
            query_dataset=query_dataset,
            metadata_row=target_row,
            sample_index=target_index,
            action=source_action,
            context=context,
            output_dir=output_root / "zstar_action5_raw",
            args=args,
        )
        torch.cuda.empty_cache()
        gt_action_path = _render(
            pipe=pipe,
            query_dataset=query_dataset,
            metadata_row=target_row,
            sample_index=target_index,
            action=target_action,
            context=context,
            output_dir=output_root / "zstar_gt_action_raw",
            args=args,
        )
        torch.cuda.empty_cache()
        learned_action_path = _render(
            pipe=pipe,
            query_dataset=query_dataset,
            metadata_row=target_row,
            sample_index=target_index,
            action=learned_action,
            context=context,
            output_dir=output_root / "zstar_learned_action_raw",
            args=args,
        )
        torch.cuda.empty_cache()

        case_dir = output_root / f"source{source_index:04d}_target{target_index:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_action": source_action.cpu(),
                "target_action": target_action.cpu(),
                "learned_action": learned_action.cpu(),
                "stage2_context": context.cpu(),
            },
            case_dir / "actions_and_context.pt",
        )
        result = {
            "source_index": source_index,
            "target_index": target_index,
            "friction_mu": float(target_row["friction_mu"]),
            "source_action_id": int(source_row["action_id"]),
            "target_action_id": int(target_row["action_id"]),
            "context_inner_step": context_step,
            "effective_action_dims": effective_dims,
            "optimized_action_dims": optimized_dims,
            "loss_frames": loss_frames,
            "lr_schedule": "1.0:30,0.5:30,0.1:30",
            "timestep_fraction": timestep_fraction,
            "initial_final_frame_flow_loss": initial_loss,
            "gt_action_final_frame_flow_loss": float(gt_action_loss.cpu()),
            "learned_action_final_frame_flow_loss": float(final_loss.cpu()),
            "learned_action_delta_l2": float((learned_action[..., :effective_dims] - initial_head).norm().cpu()),
            "gt_action_distance_from_initial_l2": float(
                (target_action[..., :effective_dims] - initial_head).norm().cpu()
            ),
            "learned_action_distance_to_gt_l2": float(
                (learned_action[..., :effective_dims] - target_action[..., :effective_dims]).norm().cpu()
            ),
            "action5_prediction": str(action5_path),
            "gt_action_prediction": str(gt_action_path),
            "learned_action_prediction": str(learned_action_path),
            "state_path": str(case_dir / "actions_and_context.pt"),
        }
        result_rows.append(result)
        _write_jsonl(output_root / "results.jsonl", result_rows)
        _write_jsonl(case_dir / "action_trajectory.jsonl", case_rows)
        print(f"[case_done] {json.dumps(result, sort_keys=True)}", flush=True)

    manifest = {
        "context_trajectory": str(trajectory_path),
        "checkpoint": str(args.stage2_ckpt_path),
        "cases": cases,
        "target_information": (
            "observed first frame plus target final frame only"
            if loss_frames == "final"
            else "complete target video clip"
        ),
        "loss": (
            "flow-matching MSE on final latent time slice only"
            if loss_frames == "final"
            else "flow-matching MSE on all future latent time slices"
        ),
        "effective_action_dimensions": effective_dims,
        "optimized_action_dimensions": optimized_dims,
        "fixed_action_dimensions": int(args.action_dim) - effective_dims,
        "lr_schedule": "1.0:30,0.5:30,0.1:30",
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] output={output_root}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
