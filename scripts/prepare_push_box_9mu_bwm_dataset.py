#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PROMPT = (
    "observe how the cream cheese box slides after a short robot push on the table; "
    "no target is shown"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def relative_episode_path(subset_name: str, kind: str, episode_index: int, video_key: str | None = None) -> str:
    if kind == "data":
        return f"{subset_name}/data/chunk-000/episode_{episode_index:06d}.parquet"
    if kind == "video" and video_key is not None:
        return f"{subset_name}/videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
    raise ValueError(f"unsupported path request kind={kind!r}, video_key={video_key!r}")


def push_bounds(meta: dict) -> tuple[int, int]:
    if meta.get("push_start") is not None and meta.get("push_end") is not None:
        return int(meta["push_start"]), int(meta["push_end"])
    counts = meta.get("phase_counts") or {}
    push_start = int(counts.get("approach", 0)) + int(counts.get("descend", 0))
    push_end = push_start + int(counts.get("push", 0))
    return push_start, push_end


def sample_start(
    rng: random.Random,
    *,
    total_frames: int,
    min_real_frames: int,
    push_focus_ratio: float,
    push_start_min: int,
    push_start_max: int,
) -> tuple[int, str]:
    latest_with_min_real = max(0, int(total_frames) - int(min_real_frames))
    if rng.random() < push_focus_ratio:
        lo = min(int(push_start_min), latest_with_min_real)
        hi = min(int(push_start_max), latest_with_min_real)
        if hi < lo:
            lo = hi
        return rng.randint(lo, hi), "push_focus"
    max_random_start = max(0, min(latest_with_min_real, int(total_frames) - 1))
    return rng.randint(0, max_random_start), "random"


def make_chunk_row(
    *,
    source_root: Path,
    subset_name: str,
    meta: dict,
    chunk_index: int,
    start_frame: int,
    num_frames: int,
    chunk_type: str,
) -> dict:
    total_frames = int(meta["steps"])
    valid_frames = max(0, min(int(num_frames), total_frames - int(start_frame)))
    push_start, push_end = push_bounds(meta)
    episode_index = int(meta["episode_index"])
    video_paths = [
        relative_episode_path(subset_name, "video", episode_index, video_key)
        for video_key in VIDEO_KEYS
    ]
    for rel_path in video_paths:
        if not (source_root / rel_path).exists():
            raise FileNotFoundError(source_root / rel_path)
    action_path = relative_episode_path(subset_name, "data", episode_index)
    if not (source_root / action_path).exists():
        raise FileNotFoundError(source_root / action_path)

    return {
        "sample_id": f"{subset_name}:ep{episode_index:06d}:chunk{chunk_index:02d}",
        "episode_index": episode_index,
        "source_dataset": subset_name,
        "source_split": meta["split"],
        "pair_id": meta["pair_id"],
        "case_id": meta["case_id"],
        "friction_mu": float(meta["friction_mu"]),
        "angle_deg": float(meta["angle_deg"]),
        "push_distance_bucket": meta.get("push_distance_bucket"),
        "push_start": int(push_start),
        "push_end": int(push_end),
        "push_steps": int(meta.get("push_steps", (meta.get("phase_counts") or {}).get("push", 0))),
        "chunk_type": chunk_type,
        "start_frame": int(start_frame),
        "end_frame": int(start_frame) + int(num_frames) - 1,
        "length": int(num_frames),
        "valid_frames": int(valid_frames),
        "total_frames": int(total_frames),
        "pad_short": valid_frames < int(num_frames),
        "video": video_paths,
        "action": action_path,
        "prompt": PROMPT,
        "task": "libero_push_box_9mu_physical_observation",
    }


def read_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    return np.asarray(table.to_pydict()["action"], dtype=np.float32)


def compute_action_stats(source_root: Path, train_episode_rows: list[dict]) -> dict:
    arrays = []
    for meta in train_episode_rows:
        parquet_path = source_root / relative_episode_path("train_lerobot", "data", int(meta["episode_index"]))
        arrays.append(read_actions(parquet_path))
    action = np.concatenate(arrays, axis=0)
    min_vals = action.min(axis=0)
    max_vals = action.max(axis=0)
    stat = {
        "shape": [int(action.shape[1])],
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
        "p01": min_vals.tolist(),
        "p99": max_vals.tolist(),
        "mean": action.mean(axis=0).tolist(),
        "std": action.std(axis=0).tolist(),
    }
    return {"action_pose": stat, "eef_delta": stat}


def build_subset_rows(
    *,
    source_root: Path,
    subset_name: str,
    metas: list[dict],
    chunks_per_episode: int,
    rng: random.Random,
    args: argparse.Namespace,
    force_push_focus: bool,
) -> list[dict]:
    rows = []
    for meta in metas:
        for chunk_idx in range(int(chunks_per_episode)):
            start, chunk_type = sample_start(
                rng,
                total_frames=int(meta["steps"]),
                min_real_frames=args.min_real_frames,
                push_focus_ratio=1.0 if force_push_focus else float(args.push_focus_ratio),
                push_start_min=args.push_start_min,
                push_start_max=args.push_start_max,
            )
            rows.append(
                make_chunk_row(
                    source_root=source_root,
                    subset_name=subset_name,
                    meta=meta,
                    chunk_index=chunk_idx,
                    start_frame=start,
                    num_frames=args.num_frames,
                    chunk_type=chunk_type,
                )
            )
    return rows


def summarize(name: str, metas: list[dict], chunks: list[dict]) -> dict:
    group_counts = Counter((row["source_split"], float(row["friction_mu"])) for row in chunks)
    episode_group_counts = Counter((row["split"], float(row["friction_mu"])) for row in metas)
    bucket_counts = Counter(row.get("push_distance_bucket") for row in metas)
    return {
        "episodes": len(metas),
        "chunks": len(chunks),
        "episode_groups": {str(key): value for key, value in sorted(episode_group_counts.items(), key=lambda item: str(item[0]))},
        "chunk_groups": {str(key): value for key, value in sorted(group_counts.items(), key=lambda item: str(item[0]))},
        "push_distance_buckets": dict(sorted(bucket_counts.items())),
        "steps_min": min(int(row["steps"]) for row in metas),
        "steps_mean": sum(int(row["steps"]) for row in metas) / max(1, len(metas)),
        "steps_max": max(int(row["steps"]) for row in metas),
        "pad_short_chunks": sum(1 for row in chunks if row["pad_short"]),
        "push_focus_chunks": sum(1 for row in chunks if row["chunk_type"] == "push_focus"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BWM manifests for the 9-friction push-box dataset.")
    parser.add_argument(
        "--source-root",
        default="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/data/libero_push_box_friction_9mu_450",
    )
    parser.add_argument("--output-dir", default="data/push_box_bwm_friction_9mu_450")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--train-chunks-per-episode", type=int, default=4)
    parser.add_argument("--test-chunks-per-episode", type=int, default=2)
    parser.add_argument("--push-focus-ratio", type=float, default=0.8)
    parser.add_argument("--push-start-min", type=int, default=50)
    parser.add_argument("--push-start-max", type=int, default=75)
    parser.add_argument("--min-real-frames", type=int, default=24)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    rng = random.Random(args.seed)
    train_metas = read_jsonl(source_root / "train_lerobot" / "meta" / "push_box_episode_metadata.jsonl")
    test_metas = read_jsonl(source_root / "test_lerobot" / "meta" / "push_box_episode_metadata.jsonl")

    train_rows = build_subset_rows(
        source_root=source_root,
        subset_name="train_lerobot",
        metas=train_metas,
        chunks_per_episode=args.train_chunks_per_episode,
        rng=rng,
        args=args,
        force_push_focus=False,
    )
    test_rows = build_subset_rows(
        source_root=source_root,
        subset_name="test_lerobot",
        metas=test_metas,
        chunks_per_episode=args.test_chunks_per_episode,
        rng=rng,
        args=args,
        force_push_focus=True,
    )
    payload = {
        "source_root": str(source_root),
        "num_frames": int(args.num_frames),
        "video_keys": list(VIDEO_KEYS),
        "prompt": PROMPT,
        "train": summarize("train", train_metas, train_rows),
        "test": summarize("test", test_metas, test_rows),
    }
    stats = compute_action_stats(source_root, train_metas)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    with (output_dir / "action_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
