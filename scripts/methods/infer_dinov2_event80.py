#!/usr/bin/env python3
"""K=1 DINOv2 amortized-context grid inference for Event80."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts.infer import (
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from scripts.methods.train_dinov2_event80 import add_dinov2_config
from wan_video_action.methods.baselines.dinov2_amortized import (
    DINOv2AmortizedContextEncoder,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args


def parse_args():
    parser = add_dinov2_config(
        add_general_config(argparse.ArgumentParser("Event80 DINOv2 inference"))
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--sample_indices", type=str, required=True)
    parser.add_argument("--support_indices", type=str, required=True)
    parser.add_argument("--dinov2_checkpoint_path", type=str, required=True)
    parser.add_argument("--wan_checkpoint_output", type=str, required=True)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def materialize_wan_checkpoint(source_path: str, destination_path: str) -> None:
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("wan."):
                tensors[key[len("wan."):]] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"No wan.* tensors found in {source_path}.")
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    save_file(tensors, str(temporary))
    os.replace(temporary, destination)


def load_support_encoder(args, device: torch.device):
    encoder = DINOv2AmortizedContextEncoder(
        model_path=args.dinov2_model_path,
        sampled_frames=args.dinov2_sampled_frames,
        temporal_stride=args.dinov2_temporal_stride,
        action_dim=args.action_dim,
        hidden_dim=args.dinov2_hidden_dim,
        action_hidden_dim=args.dinov2_action_hidden_dim,
        output_dim=args.dinov2_output_dim,
        temporal_layers=args.dinov2_temporal_layers,
        temporal_heads=args.dinov2_temporal_heads,
    )
    state = {}
    with safe_open(
        args.dinov2_checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        for key in handle.keys():
            if key.startswith("support_encoder."):
                state[key[len("support_encoder."):]] = handle.get_tensor(key)
    incompatible = encoder.load_state_dict(state, strict=False)
    bad_missing = [
        key for key in incompatible.missing_keys if not key.startswith("dino.")
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Invalid DINO head checkpoint: missing={bad_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    encoder.to(device=device)
    encoder.eval()
    return encoder


def main() -> None:
    args = parse_args()
    materialize_wan_checkpoint(
        args.dinov2_checkpoint_path, args.wan_checkpoint_output
    )
    args.ckpt_path = args.wan_checkpoint_output
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    encoder = load_support_encoder(args, pipe.device)

    support_codes = {}
    support_records = []
    with torch.no_grad():
        for support_index in _parse_sample_indices(args.support_indices):
            support = dataset[support_index]
            environment_id = int(support["mu_index"])
            code = encoder(support["video"], support["action"])[0].detach().cpu()
            support_codes[environment_id] = code
            support_records.append(
                {
                    "environment_id": environment_id,
                    "support_index": int(support_index),
                    "context": code.float().tolist(),
                }
            )
    with (Path(args.output_path).parent / "support_codes.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"records": support_records}, handle, indent=2)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for sample_index in _parse_sample_indices(args.sample_indices):
        sample = dataset[sample_index]
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        environment_id = int(sample["mu_index"])
        if environment_id not in support_codes:
            raise KeyError(f"No DINO support code for environment {environment_id}.")
        sample["physical_context"] = support_codes[environment_id].to(
            device=pipe.device, dtype=pipe.torch_dtype
        )
        if args.skip_existing and Path(sample["output_path"]).is_file():
            print(f"[skip] existing prediction {sample['output_path']}")
            continue
        predicted_path = _run_autoregressive(pipe=pipe, sample=sample, args=args)
        print(
            f"[done] sample_index={sample_index} environment={environment_id} "
            f"output={predicted_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
