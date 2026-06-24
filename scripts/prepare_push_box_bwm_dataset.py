#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


HIDDEN_DATASETS = (
    "libero_push_box_calibrated_v2_100pairs_hidden_straight_lerobot",
    "libero_push_box_calibrated_v2_100pairs_hidden_angled_lerobot",
)
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
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def relative_episode_path(dataset_name: str, kind: str, episode_index: int, video_key: str | None = None) -> str:
    if kind == "data":
        return f"{dataset_name}/data/chunk-000/episode_{episode_index:06d}.parquet"
    if kind == "video" and video_key is not None:
        return f"{dataset_name}/videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
    raise ValueError(f"unsupported path request kind={kind!r}, video_key={video_key!r}")


def push_bounds(meta: dict) -> tuple[int, int]:
    counts = meta.get("phase_counts") or {}
    push_start = int(counts.get("approach", 0)) + int(counts.get("descend", 0))
    push_end = push_start + int(counts.get("push", 0))
    return push_start, push_end


def sample_start(
    rng: random.Random,
    *,
    total_frames: int,
    num_frames: int,
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
    dataset_name: str,
    meta: dict,
    chunk_index: int,
    start_frame: int,
    num_frames: int,
    chunk_type: str,
) -> dict:
    total_frames = int(meta["steps"])
    valid_frames = max(0, min(num_frames, total_frames - int(start_frame)))
    push_start, push_end = push_bounds(meta)
    episode_index = int(meta["episode_index"])
    video_paths = [
        relative_episode_path(dataset_name, "video", episode_index, video_key)
        for video_key in VIDEO_KEYS
    ]
    for rel_path in video_paths:
        if not (source_root / rel_path).exists():
            raise FileNotFoundError(source_root / rel_path)
    action_path = relative_episode_path(dataset_name, "data", episode_index)
    if not (source_root / action_path).exists():
        raise FileNotFoundError(source_root / action_path)

    return {
        "sample_id": f"{dataset_name}:ep{episode_index:06d}:chunk{chunk_index:02d}",
        "episode_index": episode_index,
        "source_dataset": dataset_name,
        "source_split": meta["split"],
        "pair_id": meta["pair_id"],
        "case_id": meta["case_id"],
        "friction_mu": float(meta["friction_mu"]),
        "angle_deg": float(meta["angle_deg"]),
        "push_start": push_start,
        "push_end": push_end,
        "push_steps": int((meta.get("phase_counts") or {}).get("push", 0)),
        "chunk_type": chunk_type,
        "start_frame": int(start_frame),
        "end_frame": int(start_frame) + int(num_frames) - 1,
        "length": int(num_frames),
        "valid_frames": int(valid_frames),
        "total_frames": total_frames,
        "pad_short": valid_frames < int(num_frames),
        "video": video_paths,
        "action": action_path,
        "prompt": PROMPT,
        "task": "libero_push_box_hidden_physical_observation",
    }


def split_dataset(rows: list[dict], train_fraction: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_fraction))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def read_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    return np.asarray(table.to_pydict()["action"], dtype=np.float32)


def compute_action_stats(source_root: Path, train_episode_rows: list[tuple[str, dict]]) -> dict:
    arrays = []
    for dataset_name, meta in train_episode_rows:
        parquet_path = source_root / relative_episode_path(dataset_name, "data", int(meta["episode_index"]))
        arrays.append(read_actions(parquet_path))
    action = np.concatenate(arrays, axis=0)
    min_vals = action.min(axis=0)
    max_vals = action.max(axis=0)
    # Keep percentile bounds equal to min/max for this small impulse-push dataset.
    # Otherwise rare but important push actions can be clipped away.
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


def build_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    source_root = Path(args.source_root).resolve()
    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    train_episode_rows: list[tuple[str, dict]] = []
    summary = {"datasets": {}, "num_frames": args.num_frames}

    for dataset_name in HIDDEN_DATASETS:
        meta_path = source_root / dataset_name / "meta" / "push_box_episode_metadata.jsonl"
        episode_rows = read_jsonl(meta_path)
        train_eps, test_eps = split_dataset(episode_rows, args.train_fraction, rng)
        train_episode_rows.extend((dataset_name, row) for row in train_eps)
        summary["datasets"][dataset_name] = {
            "episodes": len(episode_rows),
            "train_episodes": len(train_eps),
            "test_episodes": len(test_eps),
        }

        for split_name, episodes, chunks_per_episode, out_rows in (
            ("train", train_eps, args.train_chunks_per_episode, train_rows),
            ("test", test_eps, args.test_chunks_per_episode, test_rows),
        ):
            for meta in episodes:
                for chunk_idx in range(int(chunks_per_episode)):
                    start, chunk_type = sample_start(
                        rng,
                        total_frames=int(meta["steps"]),
                        num_frames=args.num_frames,
                        min_real_frames=args.min_real_frames,
                        push_focus_ratio=args.push_focus_ratio if split_name == "train" else 1.0,
                        push_start_min=args.push_start_min,
                        push_start_max=args.push_start_max,
                    )
                    out_rows.append(
                        make_chunk_row(
                            source_root=source_root,
                            dataset_name=dataset_name,
                            meta=meta,
                            chunk_index=chunk_idx,
                            start_frame=start,
                            num_frames=args.num_frames,
                            chunk_type=chunk_type,
                        )
                    )

    stats = compute_action_stats(source_root, train_episode_rows)
    summary["train_samples"] = len(train_rows)
    summary["test_samples"] = len(test_rows)
    summary["video_keys"] = list(VIDEO_KEYS)
    summary["prompt"] = PROMPT
    return train_rows, test_rows, {"summary": summary, "action_stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BWM push-box hidden-video manifests.")
    parser.add_argument(
        "--source-root",
        default="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos/FastWAM/data/libero_push_box_calibrated_v2_100pairs",
    )
    parser.add_argument("--output-dir", default="data/push_box_bwm_calibrated_v2_100pairs")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--train-chunks-per-episode", type=int, default=4)
    parser.add_argument("--test-chunks-per-episode", type=int, default=2)
    parser.add_argument("--push-focus-ratio", type=float, default=0.8)
    parser.add_argument("--push-start-min", type=int, default=50)
    parser.add_argument("--push-start-max", type=int, default=75)
    parser.add_argument("--min-real-frames", type=int, default=24)
    args = parser.parse_args()

    train_rows, test_rows, payload = build_rows(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    with (output_dir / "action_stats.json").open("w", encoding="utf-8") as f:
        json.dump(payload["action_stats"], f, indent=2, sort_keys=True)
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload["summary"], f, indent=2, sort_keys=True)

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
