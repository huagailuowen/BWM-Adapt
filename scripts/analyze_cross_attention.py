#!/usr/bin/env python3
"""
Summarize Wan DiT cross-attention from video latent tokens to condition tokens.

This script is intentionally analysis-only. It monkey-patches each DiT
CrossAttention module at runtime, runs standard FlowMatch SFT forwards on
selected dataset rows, and records how much attention each video query token
assigns to text, physical-C, action, and other condition-token ranges.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diffsynth.diffusion import FlowMatchSFTLoss  # noqa: E402
from diffsynth.models.wan_video_dit import flash_attention  # noqa: E402
from train import WanTrainingModule, wan_parser  # noqa: E402
from wan_video_action.data import RoboTwinUnifiedDataset  # noqa: E402
from wan_video_action.data.data_utils import pack_paths  # noqa: E402
from wan_video_action.data.operators import LoadCobotAction, ResolvePromptEmbPath, create_video_operator  # noqa: E402
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


def _parse_indices(indices: str, start_index: int, num_cases: int) -> list[int]:
    if indices.strip():
        return [int(item.strip()) for item in indices.split(",") if item.strip()]
    return list(range(int(start_index), int(start_index) + int(num_cases)))


def _as_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def build_dataset(args, runtime_config):
    special_operator_map = {}
    if runtime_config["text_enabled"] and "prompt_emb" in runtime_config["data_file_keys"]:
        special_operator_map["prompt_emb"] = ResolvePromptEmbPath(base_path=args.dataset_base_path)

    with open(args.action_stat_path, "r") as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats

    dataset = RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
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
    pack_paths(
        dataset.data,
        ("video", "start_frame", "end_frame"),
        ("action", "start_frame", "end_frame"),
    )

    if "action" in runtime_config["data_file_keys"]:
        dataset.special_operator_map["action"] = LoadCobotAction(
            base_path=args.dataset_base_path,
            action_type=args.action_type,
            stat=stat,
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            pad_short=args.pad_short_chunks,
            output_dim=args.action_dim,
        )
    return dataset


def build_model(args, runtime_config, device: str):
    args.use_gradient_checkpointing = False
    args.use_gradient_checkpointing_offload = False
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
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        modules=runtime_config["modules"],
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device=device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        args=args,
    )
    model.eval()
    return model


class CrossAttentionRecorder:
    def __init__(
        self,
        query_chunk_size: int = 1024,
        include_token_values: bool = False,
        include_query_maps: bool = False,
    ):
        self.query_chunk_size = int(query_chunk_size)
        self.include_token_values = bool(include_token_values)
        self.include_query_maps = bool(include_query_maps)
        self.current_sample: str | None = None
        self.current_ranges: dict[str, tuple[int, int]] = {}
        self.current_grid: tuple[int, int, int] | None = None
        self.layer_stats: dict[int, dict[str, Any]] = defaultdict(self._new_layer_stat)
        self.sample_layer_stats: dict[str, dict[int, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(self._new_layer_stat)
        )

    @staticmethod
    def _new_layer_stat() -> dict[str, Any]:
        return {
            "calls": 0,
            "query_tokens": 0,
            "context_tokens": 0,
            "denom": 0.0,
            "head_denom": 0.0,
            "groups": defaultdict(float),
            "heads": {},
            "token_sum": None,
            "token_denom": 0.0,
            "query_group_sum": {},
            "query_denom": None,
            "query_grid": None,
        }

    def set_context(
        self,
        sample_id: str,
        ranges: dict[str, tuple[int, int]],
        query_grid: tuple[int, int, int] | None = None,
    ):
        self.current_sample = sample_id
        self.current_ranges = dict(ranges)
        self.current_grid = query_grid

    def _add_query_map_stats(
        self,
        stat: dict[str, Any],
        group_name: str,
        part: torch.Tensor,
        query_offset: int,
        total_query_tokens: int,
    ):
        if not self.include_query_maps:
            return
        # part: B, H, Q_chunk, K_group. Sum over batch/head/group-token,
        # preserving the video latent query-token axis.
        bsz, heads, query_chunk, _ = part.shape
        if group_name not in stat["query_group_sum"]:
            stat["query_group_sum"][group_name] = torch.zeros(total_query_tokens, dtype=torch.float64)
        start = int(query_offset)
        end = start + int(query_chunk)
        stat["query_group_sum"][group_name][start:end] += part.sum(dim=(0, 1, 3)).detach().double().cpu()

    def _add_group_stats(
        self,
        stat: dict[str, Any],
        attn: torch.Tensor,
        total_context_tokens: int,
        query_offset: int,
        total_query_tokens: int,
    ):
        # attn: B, H, Q, K
        bsz, heads, query_tokens, _ = attn.shape
        stat["denom"] += float(bsz * heads * query_tokens)
        stat["head_denom"] += float(bsz * query_tokens)
        stat["query_tokens"] += int(query_tokens)
        stat["context_tokens"] = max(int(stat["context_tokens"]), int(total_context_tokens))

        covered = torch.zeros(total_context_tokens, dtype=torch.bool, device=attn.device)
        if self.include_query_maps:
            if stat["query_denom"] is None:
                stat["query_denom"] = torch.zeros(total_query_tokens, dtype=torch.float64)
            query_start = int(query_offset)
            query_end = query_start + int(query_tokens)
            stat["query_denom"][query_start:query_end] += float(bsz * heads)
            stat["query_grid"] = self.current_grid
        for group_name, (start, end) in self.current_ranges.items():
            start = max(0, int(start))
            end = min(total_context_tokens, int(end))
            if end <= start:
                continue
            covered[start:end] = True
            part = attn[..., start:end]
            stat["groups"][group_name] += float(part.sum().detach().cpu())
            self._add_query_map_stats(stat, group_name, part, query_offset, total_query_tokens)
            if group_name not in stat["heads"]:
                stat["heads"][group_name] = torch.zeros(heads, dtype=torch.float64)
            stat["heads"][group_name] += part.sum(dim=(0, 2, 3)).detach().double().cpu()

        if total_context_tokens > 0 and not bool(covered.all()):
            idx = torch.nonzero(~covered, as_tuple=False).flatten()
            part = attn.index_select(-1, idx)
            stat["groups"]["other"] += float(part.sum().detach().cpu())
            self._add_query_map_stats(stat, "other", part, query_offset, total_query_tokens)
            if "other" not in stat["heads"]:
                stat["heads"]["other"] = torch.zeros(heads, dtype=torch.float64)
            stat["heads"]["other"] += part.sum(dim=(0, 2, 3)).detach().double().cpu()

        if self.include_token_values:
            token_sum = attn.sum(dim=(0, 1, 2)).detach().double().cpu()
            if stat["token_sum"] is None:
                stat["token_sum"] = torch.zeros(total_context_tokens, dtype=torch.float64)
            if int(stat["token_sum"].numel()) < total_context_tokens:
                padded = torch.zeros(total_context_tokens, dtype=torch.float64)
                padded[: stat["token_sum"].numel()] = stat["token_sum"]
                stat["token_sum"] = padded
            stat["token_sum"][:total_context_tokens] += token_sum
            stat["token_denom"] += float(bsz * heads * query_tokens)

    def _record(
        self,
        layer_idx: int,
        attn: torch.Tensor,
        total_context_tokens: int,
        query_offset: int,
        total_query_tokens: int,
    ):
        self._add_group_stats(
            self.layer_stats[layer_idx],
            attn,
            total_context_tokens,
            query_offset,
            total_query_tokens,
        )
        if self.current_sample is not None:
            self._add_group_stats(
                self.sample_layer_stats[self.current_sample][layer_idx],
                attn,
                total_context_tokens,
                query_offset,
                total_query_tokens,
            )

    def cross_attention_forward(self, layer_idx: int, module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if module.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            img = None
            ctx = y

        q = module.norm_q(module.q(x))
        k = module.norm_k(module.k(ctx))
        v = module.v(ctx)

        bsz, query_tokens, dim = q.shape
        heads = module.num_heads
        head_dim = dim // heads
        context_tokens = int(k.shape[1])
        scale = 1.0 / math.sqrt(float(head_dim))

        q = q.reshape(bsz, query_tokens, heads, head_dim).transpose(1, 2)
        k = k.reshape(bsz, context_tokens, heads, head_dim).transpose(1, 2)
        v = v.reshape(bsz, context_tokens, heads, head_dim).transpose(1, 2)

        chunks = []
        chunk_size = max(1, self.query_chunk_size)
        for start in range(0, query_tokens, chunk_size):
            end = min(query_tokens, start + chunk_size)
            q_chunk = q[:, :, start:end, :]
            scores = torch.matmul(q_chunk.float(), k.float().transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            self._record(layer_idx, attn, context_tokens, start, query_tokens)
            out = torch.matmul(attn.to(v.dtype), v)
            chunks.append(out)

        out = torch.cat(chunks, dim=2).transpose(1, 2).reshape(bsz, query_tokens, dim)

        if img is not None:
            k_img = module.norm_k_img(module.k_img(img))
            v_img = module.v_img(img)
            out = out + flash_attention(q=module.norm_q(module.q(x)), k=k_img, v=v_img, num_heads=module.num_heads)

        return module.o(out)

    def install(self, dit):
        for layer_idx, block in enumerate(dit.blocks):
            cross_attn = getattr(block, "cross_attn", None)
            if cross_attn is None:
                continue

            def patched_forward(x, y, *, _layer_idx=layer_idx, _module=cross_attn):
                return self.cross_attention_forward(_layer_idx, _module, x, y)

            cross_attn.forward = patched_forward

    @staticmethod
    def _finalize_stat(stat: dict[str, Any]) -> dict[str, Any]:
        denom = max(float(stat["denom"]), 1.0)
        head_denom = max(float(stat["head_denom"]), 1.0)
        groups = {name: value / denom for name, value in sorted(stat["groups"].items())}
        heads = {
            name: [float(v) / head_denom for v in values.tolist()]
            for name, values in sorted(stat["heads"].items())
        }
        result = {
            "calls": int(stat["calls"]),
            "query_tokens_seen": int(stat["query_tokens"]),
            "context_tokens": int(stat["context_tokens"]),
            "groups": groups,
            "heads": heads,
        }
        if stat["token_sum"] is not None:
            token_denom = max(float(stat["token_denom"]), 1.0)
            result["tokens"] = [float(v) / token_denom for v in stat["token_sum"].tolist()]
        if stat["query_denom"] is not None:
            denom = stat["query_denom"].clamp_min(1.0)
            result["query_grid"] = list(stat["query_grid"]) if stat["query_grid"] is not None else None
            result["query_maps"] = {
                name: (values / denom).tolist()
                for name, values in sorted(stat["query_group_sum"].items())
            }
        return result

    def mark_layer_call(self, layer_idx: int):
        self.layer_stats[layer_idx]["calls"] += 1
        if self.current_sample is not None:
            self.sample_layer_stats[self.current_sample][layer_idx]["calls"] += 1

    def summarize(self) -> dict[str, Any]:
        return {
            "layers": {
                str(layer_idx): self._finalize_stat(stat)
                for layer_idx, stat in sorted(self.layer_stats.items())
            },
            "samples": {
                sample_id: {
                    str(layer_idx): self._finalize_stat(stat)
                    for layer_idx, stat in sorted(layer_stats.items())
                }
                for sample_id, layer_stats in sorted(self.sample_layer_stats.items())
            },
        }


def install_call_counter(recorder: CrossAttentionRecorder, dit):
    for layer_idx, block in enumerate(dit.blocks):
        cross_attn = getattr(block, "cross_attn", None)
        if cross_attn is None:
            continue
        original_forward = cross_attn.forward

        def wrapped_forward(x, y, *, _layer_idx=layer_idx, _original_forward=original_forward):
            recorder.mark_layer_call(_layer_idx)
            return _original_forward(x, y)

        cross_attn.forward = wrapped_forward


def run_units(model: WanTrainingModule, data: dict[str, Any]):
    inputs = model.get_pipeline_inputs(data)
    inputs = model.transfer_data_to_device(inputs, model.pipe.device, model.pipe.torch_dtype)
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    return inputs


def token_ranges_from_inputs(inputs, args) -> dict[str, tuple[int, int]]:
    inputs_shared, inputs_posi, _ = inputs
    context = inputs_posi.get("context", inputs_shared.get("context"))
    text_count = int(context.shape[1]) if isinstance(context, torch.Tensor) and context.ndim >= 3 else 0
    physical_emb = inputs_shared.get("physical_context_emb")
    physical_count = int(physical_emb.shape[1]) if isinstance(physical_emb, torch.Tensor) and physical_emb.ndim >= 3 else 0
    action_emb = inputs_shared.get("action_emb")
    action_count = int(action_emb.shape[1]) if isinstance(action_emb, torch.Tensor) and action_emb.ndim >= 3 else 0

    cursor = 0
    ranges: dict[str, tuple[int, int]] = {}
    if text_count:
        ranges["text"] = (cursor, cursor + text_count)
        cursor += text_count
    if physical_count:
        ranges["physical"] = (cursor, cursor + physical_count)
        cursor += physical_count
    if action_count:
        ranges["action"] = (cursor, cursor + action_count)
        cursor += action_count
    return ranges


def query_grid_from_inputs(inputs, model: WanTrainingModule) -> tuple[int, int, int] | None:
    inputs_shared, _, _ = inputs
    latents = inputs_shared.get("input_latents")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        return None
    patch_t, patch_h, patch_w = model.pipe.dit.patch_size
    return (
        int(latents.shape[2]) // int(patch_t),
        int(latents.shape[3]) // int(patch_h),
        int(latents.shape[4]) // int(patch_w),
    )


def sample_id_from_data(data: dict[str, Any], fallback_index: int) -> str:
    for key in ("sample_id", "case_id", "pair_id"):
        value = data.get(key)
        if value is not None:
            return str(value)
    return f"row_{fallback_index}"


def write_layer_csv(path: str | Path, summary: dict[str, Any]):
    rows = []
    for layer_idx, layer in summary["layers"].items():
        groups = layer.get("groups", {})
        rows.append(
            {
                "layer": layer_idx,
                "text": groups.get("text", 0.0),
                "physical": groups.get("physical", 0.0),
                "action": groups.get("action", 0.0),
                "other": groups.get("other", 0.0),
                "context_tokens": layer.get("context_tokens", 0),
                "query_tokens_seen": layer.get("query_tokens_seen", 0),
                "calls": layer.get("calls", 0),
            }
        )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "text", "physical", "action", "other", "context_tokens", "query_tokens_seen", "calls"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = wan_parser()
    group = parser.add_argument_group("cross_attention_analysis")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to analyze. Overrides --ckpt_path.")
    group.add_argument("--analysis_device", type=str, default=None, help="Device for analysis, e.g. cuda or cuda:0.")
    group.add_argument("--indices", type=str, default="", help="Comma-separated dataset row indices to analyze.")
    group.add_argument("--num_cases", type=int, default=4, help="Number of rows to analyze when --indices is empty.")
    group.add_argument("--num_repeats", type=int, default=1, help="Number of random timestep/noise forwards per row.")
    group.add_argument("--query_chunk_size", type=int, default=1024, help="Chunk size for query-token attention computation.")
    group.add_argument("--include_token_values", action="store_true", default=False, help="Also write per-condition-token averages.")
    group.add_argument(
        "--include_query_maps",
        action="store_true",
        default=False,
        help="Also write per-video-latent-query attention maps for text/physical/action groups.",
    )
    group.add_argument("--output_json", type=str, default="outputs/cross_attention_summary.json")
    group.add_argument("--output_csv", type=str, default="outputs/cross_attention_layers.csv")
    args = parser.parse_args()
    args = merge_yaml_and_args(args.config, parser, args)

    if args.checkpoint is not None:
        args.ckpt_path = args.checkpoint

    device = args.analysis_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[warn] CPU analysis will be very slow for Wan DiT.", flush=True)

    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    dataset = build_dataset(args, runtime_config)
    model = build_model(args, runtime_config, device)
    model.pipe.to(device=device, dtype=_dtype_from_arg(args.mixed_precision))
    model.pipe.scheduler.set_timesteps(1000, training=True)

    recorder = CrossAttentionRecorder(
        query_chunk_size=args.query_chunk_size,
        include_token_values=args.include_token_values,
        include_query_maps=args.include_query_maps,
    )
    recorder.install(model.pipe.dit)
    install_call_counter(recorder, model.pipe.dit)

    indices = _parse_indices(args.indices, args.start_index, args.num_cases)
    losses: list[dict[str, Any]] = []

    with torch.no_grad():
        for row_index in indices:
            data = dataset[row_index]
            sample_id = sample_id_from_data(data, row_index)
            inputs = run_units(model, data)
            ranges = token_ranges_from_inputs(inputs, args)
            query_grid = query_grid_from_inputs(inputs, model)
            recorder.set_context(sample_id, ranges, query_grid=query_grid)

            inputs_shared, inputs_posi, _ = inputs
            for repeat_idx in range(int(args.num_repeats)):
                torch.manual_seed(int(args.seed) + int(row_index) * 1000 + repeat_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(args.seed) + int(row_index) * 1000 + repeat_idx)
                loss = FlowMatchSFTLoss(model.pipe, **inputs_shared, **inputs_posi)
                losses.append(
                    {
                        "row_index": int(row_index),
                        "sample_id": sample_id,
                        "repeat": int(repeat_idx),
                        "loss": _as_float(loss),
                        "token_ranges": {name: [int(a), int(b)] for name, (a, b) in ranges.items()},
                    }
                )
                print(
                    f"[attention] row={row_index} repeat={repeat_idx} loss={_as_float(loss):.6f} ranges={ranges}",
                    flush=True,
                )

    summary = recorder.summarize()
    summary["losses"] = losses
    summary["checkpoint"] = args.ckpt_path
    summary["config"] = args.config
    summary["metadata"] = args.dataset_metadata_path
    summary["note"] = (
        "Groups are mean attention mass from video latent query tokens to condition-token key/value ranges. "
        "For the current BWM configs with text disabled, context is usually [physical, action]."
    )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"[attention] wrote {output_json}", flush=True)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_layer_csv(output_csv, summary)
        print(f"[attention] wrote {output_csv}", flush=True)

    for layer_idx, layer in summary["layers"].items():
        groups = layer.get("groups", {})
        print(
            "[attention:layer] "
            f"layer={layer_idx} "
            f"text={groups.get('text', 0.0):.6f} "
            f"physical={groups.get('physical', 0.0):.6f} "
            f"action={groups.get('action', 0.0):.6f} "
            f"other={groups.get('other', 0.0):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
