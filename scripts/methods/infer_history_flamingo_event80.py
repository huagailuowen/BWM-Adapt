#!/usr/bin/env python3
"""K=1 Flamingo support-memory grid inference for Event80."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors import safe_open
import torch

from scripts.infer import (
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from scripts.methods.train_history_conditioned_event80 import (
    add_history_config,
    parse_support_sizes,
)
from wan_video_action.methods.baselines.history_conditioned import (
    FlamingoSupportEncoder,
    install_flamingo_history,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args
from wan_video_action.utils import set_global_seed


def parse_args():
    parser = add_history_config(
        add_general_config(argparse.ArgumentParser("Event80 Flamingo inference"))
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--sample_indices", type=str, required=True)
    parser.add_argument("--support_indices", type=str, required=True)
    parser.add_argument("--flamingo_checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def load_method_state(args, pipe):
    support_encoder = FlamingoSupportEncoder(
        model_path=args.history_dinov2_model_path,
        action_dim=args.action_dim,
        sampled_frames=args.history_sampled_frames,
        memory_dim=args.history_memory_dim,
        num_latents=args.history_num_latents,
        resampler_layers=args.history_resampler_layers,
        heads=args.history_heads,
        max_support_trajectories=max(parse_support_sizes(args.history_support_sizes)),
    )
    installation = install_flamingo_history(
        pipe.dit,
        memory_dim=args.history_memory_dim,
        heads=args.history_heads,
        insertion_frequency=args.history_insertion_frequency,
    )

    support_state = {}
    adapter_state = {}
    adapter_prefix = "wan.pipe.dit.flamingo_icl."
    with safe_open(
        args.flamingo_checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        for key in handle.keys():
            if key.startswith("support_encoder."):
                support_state[key[len("support_encoder."):]] = handle.get_tensor(key)
            elif key.startswith(adapter_prefix):
                adapter_state[key[len(adapter_prefix):]] = handle.get_tensor(key)

    incompatible = support_encoder.load_state_dict(support_state, strict=False)
    bad_missing = [
        key for key in incompatible.missing_keys if not key.startswith("dino.")
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Invalid Flamingo support encoder checkpoint: "
            f"missing={bad_missing}, unexpected={incompatible.unexpected_keys}"
        )
    installation.adapters.load_state_dict(adapter_state, strict=True)
    support_encoder.to(device=pipe.device)
    support_encoder.eval()
    installation.adapters.to(device=pipe.device, dtype=pipe.torch_dtype)
    installation.adapters.eval()
    return support_encoder, installation


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    support_encoder, installation = load_method_state(args, pipe)

    support_memories = {}
    support_records = []
    autocast_enabled = torch.cuda.is_available()
    with torch.no_grad():
        for support_index in _parse_sample_indices(args.support_indices):
            support = dataset[support_index]
            environment_id = int(support["mu_index"])
            with torch.autocast(
                device_type="cuda",
                dtype=pipe.torch_dtype,
                enabled=autocast_enabled,
            ):
                visual, frame_indices, frame_count = (
                    support_encoder.extract_visual_features(support["video"])
                )
                memory = support_encoder.project_supports(
                    visual_features=[visual],
                    actions=[support["action"]],
                    frame_indices=[frame_indices],
                    frame_counts=[frame_count],
                )
            memory = memory.to(dtype=pipe.torch_dtype).detach().cpu()
            support_memories[environment_id] = memory
            support_records.append(
                {
                    "environment_id": environment_id,
                    "support_index": int(support_index),
                    "memory_shape": list(memory.shape),
                    "memory_l2": float(memory.float().norm().item()),
                }
            )

    del support_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    query_indices = _parse_sample_indices(args.sample_indices)
    queries_by_environment = defaultdict(list)
    for sample_index in query_indices:
        queries_by_environment[int(dataset[sample_index]["mu_index"])].append(
            int(sample_index)
        )

    try:
        for environment_id, environment_queries in queries_by_environment.items():
            if environment_id not in support_memories:
                raise KeyError(
                    f"No Flamingo support memory for environment {environment_id}."
                )
            installation.controller.set(
                support_memories[environment_id].to(
                    device=pipe.device, dtype=pipe.torch_dtype
                )
            )
            try:
                with torch.no_grad():
                    for sample_index in environment_queries:
                        sample = prepare_sample_for_rollout(
                            dataset[sample_index], sample_index, pipe, args
                        )
                        if args.skip_existing and Path(sample["output_path"]).is_file():
                            print(f"[skip] existing prediction {sample['output_path']}")
                            continue
                        predicted_path = _run_autoregressive(
                            pipe=pipe, sample=sample, args=args
                        )
                        print(
                            f"[done] sample_index={sample_index} "
                            f"environment={environment_id} output={predicted_path}",
                            flush=True,
                        )
            finally:
                installation.clear_memory()
    finally:
        installation.clear_memory()

    protocol = {
        "method": "history_conditioned_wm",
        "support_size": 1,
        "query_state": "read_only",
        "support_query_disjoint": True,
        "checkpoint": args.flamingo_checkpoint_path,
        "records": support_records,
    }
    with (Path(args.output_path).parent / "protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, indent=2)


if __name__ == "__main__":
    main()
