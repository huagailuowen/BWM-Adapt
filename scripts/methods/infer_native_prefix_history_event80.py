#!/usr/bin/env python3
"""Training-faithful native-prefix support/query inference for Event80."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer import (  # noqa: E402
    _parse_sample_indices,
    build_infer_dataset,
    build_pipeline,
    save_video,
)
from scripts.methods.train_native_prefix_history_event80 import (  # noqa: E402
    add_prefix_config,
)
from wan_video_action.methods.baselines.native_prefix_history import (  # noqa: E402
    build_event80_prefix_pair,
    install_prefix_segments,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


SEGMENT_KEY = "wan.native_prefix_segments.embedding"


def parse_args() -> argparse.Namespace:
    parser = add_prefix_config(
        add_general_config(argparse.ArgumentParser(description=__doc__))
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--sample_indices", type=str, required=True)
    parser.add_argument("--support_indices", type=str, required=True)
    parser.add_argument("--prefix_checkpoint_path", type=str, required=True)
    parser.add_argument("--wan_checkpoint_output", type=str, required=True)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def materialize_wan_checkpoint(source: str, destination: str) -> None:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("wan.") and key != SEGMENT_KEY:
                tensors[key[len("wan."):]] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"No wan.* tensors found in {source}.")
    temporary = output.with_name(f".{output.name}.partial")
    temporary.unlink(missing_ok=True)
    save_file(tensors, str(temporary))
    os.replace(temporary, output)


def load_segment_embedding(path: str, installation) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        if SEGMENT_KEY not in handle.keys():
            raise KeyError(f"Missing {SEGMENT_KEY} in {path}")
        value = handle.get_tensor(SEGMENT_KEY)
    with torch.no_grad():
        installation.module.embedding.copy_(
            value.to(
                device=installation.module.embedding.device,
                dtype=installation.module.embedding.dtype,
            )
        )
    installation.module.eval()


def output_name(index: int, row: dict) -> str:
    return (
        f"sample{index:04d}_episode{int(row['episode_index']):06d}_"
        f"frames{int(row['start_frame']):04d}-{int(row['end_frame']):04d}.mp4"
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    materialize_wan_checkpoint(
        args.prefix_checkpoint_path, args.wan_checkpoint_output
    )
    args.ckpt_path = args.wan_checkpoint_output
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    trained_num_frames = int(args.num_frames)
    trained_history_frames = int(args.num_history_frames)
    if trained_num_frames != 85 or trained_history_frames != 45:
        raise ValueError(
            "Native-prefix checkpoint requires num_frames=85 and num_history_frames=45."
        )
    args.num_frames = 41
    dataset = build_infer_dataset(args)
    args.num_frames = trained_num_frames
    pipe = build_pipeline(args)
    installation = install_prefix_segments(pipe.dit)
    load_segment_embedding(args.prefix_checkpoint_path, installation)

    support_by_environment = {}
    for support_index in _parse_sample_indices(args.support_indices):
        environment = int(dataset[support_index]["mu_index"])
        if environment in support_by_environment:
            raise ValueError(f"Multiple supports for environment {environment}")
        support_by_environment[environment] = int(support_index)

    queries_by_environment = defaultdict(list)
    for query_index in _parse_sample_indices(args.sample_indices):
        environment = int(dataset[query_index]["mu_index"])
        queries_by_environment[environment].append(int(query_index))

    records = []
    with torch.no_grad():
        for environment, query_indices in queries_by_environment.items():
            if environment not in support_by_environment:
                raise KeyError(f"No support for environment {environment}")
            support_index = support_by_environment[environment]
            support = dataset[support_index]
            for query_index in query_indices:
                query_row = dataset.data[query_index]
                output_path = output_root / output_name(query_index, query_row)
                if args.skip_existing and output_path.is_file():
                    print(f"[skip] existing prediction {output_path}", flush=True)
                    continue
                pair = build_event80_prefix_pair(support, dataset[query_index])
                history_video = pair["video"][:, :, :45]
                action = torch.as_tensor(
                    pair["action"], dtype=pipe.torch_dtype, device=pipe.device
                )
                generated = pipe(
                    input_video=history_video,
                    action=action,
                    physical_context=None,
                    seed=int(args.seed),
                    rand_device="cpu",
                    tiled=False,
                    height=int(args.height) * int(history_video.shape[0]),
                    width=int(args.width),
                    num_frames=85,
                    num_history_frames=45,
                    cfg_scale=float(args.cfg_scale),
                    num_inference_steps=int(args.num_inference_steps),
                    use_history_condition_noise_in_inference=True,
                    progress_bar_cmd=lambda iterable, *unused_args, **unused_kwargs: iterable,
                    output_type="floatpoint",
                )
                query_video = generated[:, :, 44:85].detach().cpu()
                if query_video.shape[2] != 41:
                    raise RuntimeError(
                        f"Expected 41 query frames, got {query_video.shape[2]}"
                    )
                temporary = output_path.with_name(
                    f".{output_path.stem}.partial{output_path.suffix}"
                )
                temporary.unlink(missing_ok=True)
                save_video(
                    query_video,
                    output_path=str(temporary),
                    fps=int(args.fps),
                    quality=int(args.quality),
                )
                temporary.replace(output_path)
                records.append(
                    {
                        "environment_id": environment,
                        "support_index": support_index,
                        "query_index": query_index,
                        "prediction": str(output_path),
                        "known_prefix_frames": 45,
                        "returned_query_frames": 41,
                    }
                )
                print(
                    f"[done] environment={environment} support={support_index} "
                    f"query={query_index} output={output_path}",
                    flush=True,
                )

    protocol = {
        "method": "native_prefix_history",
        "checkpoint": args.prefix_checkpoint_path,
        "support_size": 1,
        "support_query_disjoint": True,
        "layout": "41_support+4_query_anchor_reset+40_generated_query_future",
        "query_future_ground_truth_visible": False,
        "records": records,
    }
    protocol_path = output_root.parent / "protocol.json"
    temporary = protocol_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    temporary.replace(protocol_path)


if __name__ == "__main__":
    main()
