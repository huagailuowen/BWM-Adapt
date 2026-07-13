#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SUBSET_NAME = "hidden_straight_lerobot"
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PROMPT = (
    "observe how the cream cheese box slides after a short robot push on the table; "
    "no target is shown"
)
PHYSICAL_CONTEXT_NORMALIZATION = "C = friction_mu / 0.25"
PUSH_ACTION_PEAK_NORMALIZATION = "peak_push_action_x_norm = max(action_x during push) / 0.5"


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


def relative_episode_path(kind: str, episode_index: int, video_key: str | None = None) -> str:
    if kind == "data":
        return f"{SUBSET_NAME}/data/chunk-000/episode_{episode_index:06d}.parquet"
    if kind == "video" and video_key is not None:
        return f"{SUBSET_NAME}/videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
    raise ValueError(f"unsupported path request kind={kind!r}, video_key={video_key!r}")


def normalize_friction_mu(mu: float) -> float:
    # Fixed convention for oracle latent C: map mu in [0, 0.25] to C in [0, 1].
    return float(mu) / 0.25


def normalize_push_action_peak_x(value: float) -> float:
    # The event-tap A500 dataset uses commanded push amplitudes up to about 0.5.
    # Keep this second C dimension in the same nominal [0, 1] range as friction C.
    return float(value) / 0.5


def friction_mu(meta: dict) -> float:
    if meta.get("friction_mu") is not None:
        return float(meta["friction_mu"])
    return float(meta["mu"])


def steps(meta: dict) -> int:
    if meta.get("steps") is not None:
        return int(meta["steps"])
    return int((meta.get("metrics") or {})["steps"])


def push_bounds(meta: dict) -> tuple[int, int]:
    if meta.get("push_start") is not None and meta.get("push_end") is not None:
        return int(meta["push_start"]), int(meta["push_end"])
    metrics = meta.get("metrics") or {}
    counts = metrics.get("phase_counts") or meta.get("phase_counts") or {}
    push_start = int(counts.get("approach", 0)) + int(counts.get("descend", 0))
    push_steps = int(meta.get("push_steps", counts.get("push", 0)))
    return push_start, push_start + push_steps


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
    meta: dict,
    chunk_index: int,
    start_frame: int,
    num_frames: int,
    chunk_type: str,
    source_split_default: str,
    task_name: str,
) -> dict:
    total_frames = steps(meta)
    valid_frames = max(0, min(int(num_frames), total_frames - int(start_frame)))
    push_start, push_end = push_bounds(meta)
    episode_index = int(meta["episode_index"])
    video_paths = [
        relative_episode_path("video", episode_index, video_key)
        for video_key in VIDEO_KEYS
    ]
    for rel_path in video_paths:
        if not (source_root / rel_path).exists():
            raise FileNotFoundError(source_root / rel_path)
    action_path = relative_episode_path("data", episode_index)
    if not (source_root / action_path).exists():
        raise FileNotFoundError(source_root / action_path)

    mu = friction_mu(meta)
    mu_index = int(meta.get("mu_index", round(mu * 10000)))
    action_id = int(meta.get("action_id", 0))
    pair_id = meta.get("pair_id") or f"m{mu_index:02d}_a{action_id:02d}"
    source_split = meta.get("split") or source_split_default

    return {
        "sample_id": f"{SUBSET_NAME}:ep{episode_index:06d}:chunk{chunk_index:02d}",
        "episode_index": episode_index,
        "source_dataset": SUBSET_NAME,
        "source_split": source_split,
        "pair_id": pair_id,
        "case_id": meta["case_id"],
        "friction_mu": mu,
        "mu_index": mu_index,
        "mu_tag": meta.get("mu_tag"),
        "action_id": action_id,
        "angle_deg": float(meta.get("angle_deg", action_id)),
        "push_amplitude": float(meta.get("A", 0.0)),
        "profile_area": float(meta.get("profile_area", 0.0)),
        "push_start": int(push_start),
        "push_end": int(push_end),
        "push_steps": int(meta.get("push_steps", push_end - push_start)),
        "push_action_peak_x": float(meta.get("push_action_peak_x", 0.0)),
        "push_action_peak_x_normalized": normalize_push_action_peak_x(float(meta.get("push_action_peak_x", 0.0))),
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
        "task": task_name,
    }


def with_oracle_context(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["physical_context"] = [normalize_friction_mu(float(row["friction_mu"]))]
        item["physical_context_source"] = "oracle_friction_mu"
        item["physical_context_normalization"] = PHYSICAL_CONTEXT_NORMALIZATION
        out.append(item)
    return out


def with_oracle_context_friction_and_peak_action(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["physical_context"] = [
            normalize_friction_mu(float(row["friction_mu"])),
            normalize_push_action_peak_x(float(row["push_action_peak_x"])),
        ]
        item["physical_context_source"] = "oracle_friction_mu_and_push_action_peak_x"
        item["physical_context_normalization"] = (
            f"{PHYSICAL_CONTEXT_NORMALIZATION}; {PUSH_ACTION_PEAK_NORMALIZATION}"
        )
        out.append(item)
    return out


def read_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    return np.asarray(table.to_pydict()["action"], dtype=np.float32)


def compute_action_stats(source_root: Path, metas: list[dict]) -> dict:
    arrays = []
    for meta in metas:
        parquet_path = source_root / relative_episode_path("data", int(meta["episode_index"]))
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


def compute_push_action_peak_x(source_root: Path, meta: dict) -> float:
    push_start, push_end = push_bounds(meta)
    episode_index = int(meta["episode_index"])
    parquet_path = source_root / relative_episode_path("data", episode_index)
    action = read_actions(parquet_path)
    lo = max(0, min(int(push_start), int(action.shape[0]) - 1))
    hi = max(lo + 1, min(int(push_end), int(action.shape[0])))
    return float(np.max(action[lo:hi, 0]))


def clamp_chunk_start(start: int, *, total_frames: int, num_frames: int) -> int:
    latest_start = max(0, int(total_frames) - int(num_frames))
    return int(max(0, min(latest_start, int(start))))


def push_full_start(meta: dict, *, chunk_index: int, num_frames: int, pre_offsets: list[int]) -> int:
    total_frames = steps(meta)
    push_start, push_end = push_bounds(meta)
    offset = int(pre_offsets[int(chunk_index) % len(pre_offsets)])
    start = clamp_chunk_start(int(push_start) - offset, total_frames=total_frames, num_frames=num_frames)
    if start + int(num_frames) - 1 < int(push_end):
        start = clamp_chunk_start(int(push_end) - int(num_frames) + 1, total_frames=total_frames, num_frames=num_frames)
    return int(start)


def push_mixed_start(
    meta: dict,
    *,
    chunk_index: int,
    num_frames: int,
    prepush_chunks: int,
    prepush_offsets: list[int],
    push_core_offsets: list[int],
) -> tuple[int, str]:
    total_frames = steps(meta)
    push_start, push_end = push_bounds(meta)
    push_core_start = int(push_start)
    push_core_end = int(push_end)

    if int(chunk_index) < int(prepush_chunks):
        offset = int(prepush_offsets[int(chunk_index) % len(prepush_offsets)])
        start = clamp_chunk_start(int(push_start) - offset, total_frames=total_frames, num_frames=num_frames)
        return start, "pre_push"

    push_index = int(chunk_index) - int(prepush_chunks)
    offset = int(push_core_offsets[push_index % len(push_core_offsets)])
    start = clamp_chunk_start(int(push_core_start) - offset, total_frames=total_frames, num_frames=num_frames)
    if start > push_core_start:
        start = clamp_chunk_start(push_core_start, total_frames=total_frames, num_frames=num_frames)
    if start + int(num_frames) - 1 < push_core_end:
        start = clamp_chunk_start(push_core_end - int(num_frames) + 1, total_frames=total_frames, num_frames=num_frames)
    return start, "push_core"


def build_rows(*, source_root: Path, metas: list[dict], rng: random.Random, args: argparse.Namespace) -> list[dict]:
    rows = []
    for meta in metas:
        meta = dict(meta)
        meta["push_action_peak_x"] = compute_push_action_peak_x(source_root, meta)
        for chunk_idx in range(int(args.train_chunks_per_episode)):
            if str(args.chunk_start_mode).lower() == "push_full":
                pre_offsets = [int(part.strip()) for part in str(args.push_full_pre_offsets).split(",") if part.strip()]
                if not pre_offsets:
                    raise ValueError("push_full_pre_offsets must contain at least one integer.")
                start = push_full_start(
                    meta,
                    chunk_index=chunk_idx,
                    num_frames=args.num_frames,
                    pre_offsets=pre_offsets,
                )
                chunk_type = "push_full"
            elif str(args.chunk_start_mode).lower() == "push_mixed":
                prepush_offsets = [
                    int(part.strip()) for part in str(args.prepush_start_offsets).split(",") if part.strip()
                ]
                push_core_offsets = [
                    int(part.strip()) for part in str(args.push_core_start_offsets).split(",") if part.strip()
                ]
                if not prepush_offsets:
                    raise ValueError("prepush_start_offsets must contain at least one integer.")
                if not push_core_offsets:
                    raise ValueError("push_core_start_offsets must contain at least one integer.")
                start, chunk_type = push_mixed_start(
                    meta,
                    chunk_index=chunk_idx,
                    num_frames=args.num_frames,
                    prepush_chunks=args.prepush_chunks_per_episode,
                    prepush_offsets=prepush_offsets,
                    push_core_offsets=push_core_offsets,
                )
            else:
                start, chunk_type = sample_start(
                    rng,
                    total_frames=steps(meta),
                    min_real_frames=args.min_real_frames,
                    push_focus_ratio=float(args.push_focus_ratio),
                    push_start_min=args.push_start_min,
                    push_start_max=args.push_start_max,
                )
            rows.append(
                make_chunk_row(
                    source_root=source_root,
                    meta=meta,
                    chunk_index=chunk_idx,
                    start_frame=start,
                    num_frames=args.num_frames,
                    chunk_type=chunk_type,
                    source_split_default=args.source_split,
                    task_name=args.task_name,
                )
            )
    return rows


def summarize(metas: list[dict], rows: list[dict]) -> dict:
    friction_counts = Counter(float(row["friction_mu"]) for row in rows)
    episode_friction_counts = Counter(friction_mu(row) for row in metas)
    action_counts = Counter(int(row.get("action_id", 0)) for row in metas)
    context_values = [normalize_friction_mu(float(row["friction_mu"])) for row in rows]
    peak_values = [float(row["push_action_peak_x"]) for row in rows]
    peak_context_values = [normalize_push_action_peak_x(value) for value in peak_values]
    frame_counts = [steps(row) for row in metas]
    return {
        "source_subset": SUBSET_NAME,
        "episodes": len(metas),
        "train_samples": len(rows),
        "num_friction_values": len(episode_friction_counts),
        "episode_counts_per_friction": sorted(set(episode_friction_counts.values())),
        "chunk_counts_per_friction": sorted(set(friction_counts.values())),
        "num_action_ids": len(action_counts),
        "action_counts": dict(sorted(action_counts.items())),
        "friction_mu_min": min(episode_friction_counts),
        "friction_mu_max": max(episode_friction_counts),
        "physical_context_min": min(context_values),
        "physical_context_max": max(context_values),
        "physical_context_normalization": PHYSICAL_CONTEXT_NORMALIZATION,
        "push_action_peak_x_min": min(peak_values),
        "push_action_peak_x_max": max(peak_values),
        "push_action_peak_x_normalized_min": min(peak_context_values),
        "push_action_peak_x_normalized_max": max(peak_context_values),
        "push_action_peak_x_normalization": PUSH_ACTION_PEAK_NORMALIZATION,
        "steps_min": min(frame_counts),
        "steps_mean": sum(frame_counts) / max(1, len(frame_counts)),
        "steps_max": max(frame_counts),
        "pad_short_chunks": sum(1 for row in rows if row["pad_short"]),
        "pre_push_chunks": sum(1 for row in rows if row["chunk_type"] == "pre_push"),
        "push_core_chunks": sum(1 for row in rows if row["chunk_type"] == "push_core"),
        "push_focus_chunks": sum(1 for row in rows if row["chunk_type"] == "push_focus"),
        "push_full_chunks": sum(1 for row in rows if row["chunk_type"] == "push_full"),
        "random_chunks": sum(1 for row in rows if row["chunk_type"] == "random"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BWM manifests for the 70-friction fixed-scene push-box dataset.")
    parser.add_argument(
        "--source-root",
        default="/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox/libero_push_box_70fric_9action_fixed_scene_hidden_lerobot_rescanned_hai-machine_2026-07-02",
    )
    parser.add_argument("--output-dir", default="data/push_box_bwm_70fric_9action_fixed_scene_hidden_20260702")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--train-chunks-per-episode", type=int, default=4)
    parser.add_argument("--push-focus-ratio", type=float, default=0.8)
    parser.add_argument("--push-start-min", type=int, default=50)
    parser.add_argument("--push-start-max", type=int, default=75)
    parser.add_argument("--min-real-frames", type=int, default=24)
    parser.add_argument("--source-split", default="fixed_scene_hidden_straight")
    parser.add_argument("--task-name", default="libero_push_box_70fric_fixed_scene_physical_observation")
    parser.add_argument("--chunk-start-mode", default="random", choices=["random", "push_full", "push_mixed"])
    parser.add_argument("--push-full-pre-offsets", default="35,25,15,5")
    parser.add_argument("--prepush-chunks-per-episode", type=int, default=5)
    parser.add_argument("--prepush-start-offsets", default="70,55,40,25,10")
    parser.add_argument("--push-core-start-offsets", default="10,8,5,3,0")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    rng = random.Random(args.seed)
    metas = read_jsonl(source_root / SUBSET_NAME / "meta" / "push_box_episode_metadata.jsonl")
    train_rows = build_rows(source_root=source_root, metas=metas, rng=rng, args=args)
    oracle_rows = with_oracle_context(train_rows)
    oracle_peak_rows = with_oracle_context_friction_and_peak_action(train_rows)
    stats = compute_action_stats(source_root, metas)
    summary = {
        "source_root": str(source_root),
        "num_frames": int(args.num_frames),
        "video_keys": list(VIDEO_KEYS),
        "prompt": PROMPT,
        "train": summarize(metas, train_rows),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "train_oracle_mu025_c1.jsonl", oracle_rows)
    write_jsonl(output_dir / "train_oracle_mu025_peakx050_c2.jsonl", oracle_peak_rows)
    write_jsonl(output_dir / "test.jsonl", [])
    with (output_dir / "action_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
