#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOURCE = (
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pnpDynamics/"
    "dynamic_carrier_physical_grasp_piecewise_formal_200eps_crf18_2026-07-06_hai-machine"
)
DEFAULT_OUTPUT = "data/pnp_dynamics_payloadxy_timestep_c2_40f_8chunk_20260706"
DEFAULT_PROMPT = (
    "predict the moving payload trajectory while the robot approaches and grasps it"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0])


def normalize_payload_xy(payload_xy: np.ndarray) -> list[float]:
    """Map payload x/y from workspace coordinates to model-friendly [0, 1].

    The fixed convention for this experiment is:
        C_xy = (payload_xy + 0.30) / 0.60

    This maps -0.30 -> 0.0, 0.0 -> 0.5, and 0.30 -> 1.0.
    Values are not clipped, so out-of-range coordinates remain visible.
    """
    norm = (np.asarray(payload_xy, dtype=np.float32) + 0.30) / 0.60
    return [float(norm[0]), float(norm[1])]


def stats_for_array(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(values.shape[1:]),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "p01": np.percentile(values, 1, axis=0).astype(float).tolist(),
        "p99": np.percentile(values, 99, axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).astype(float).tolist(),
    }


def uniform_starts(total_frames: int, num_frames: int, chunks_per_episode: int) -> list[int]:
    max_start = max(0, int(total_frames) - int(num_frames))
    if chunks_per_episode <= 1:
        return [0]
    starts = [
        int(round(i * max_start / float(chunks_per_episode - 1)))
        for i in range(chunks_per_episode)
    ]
    return sorted(dict.fromkeys(starts))


def task_prompt_map(source: Path) -> dict[int, str]:
    tasks = {}
    for row in read_jsonl(source / "meta" / "tasks.jsonl"):
        idx = row.get("task_index", row.get("index"))
        text = row.get("task", row.get("description", row.get("prompt")))
        if idx is not None and text:
            tasks[int(idx)] = str(text)
    return tasks


def episode_files(source: Path) -> list[Path]:
    files = sorted((source / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {source / 'data'}")
    return files


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = Path(args.source).resolve()
    output = Path(args.output)
    token_count = ((int(args.num_frames) - 1) // int(args.time_division_factor)) + 1
    prompts = task_prompt_map(source)

    oracle_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    all_actions: list[np.ndarray] = []
    episode_count = 0
    total_video_frames = 0

    for parquet_path in episode_files(source):
        episode_index = int(parquet_path.stem.split("_")[-1])
        chunk_name = parquet_path.parent.name
        df = pd.read_parquet(parquet_path)
        total_frames = int(len(df))
        if total_frames <= 0:
            continue
        episode_count += 1
        total_video_frames += total_frames

        actions = np.stack([np.asarray(item, dtype=np.float32) for item in df["action"].to_list()])
        payload_states = np.stack(
            [np.asarray(item, dtype=np.float32) for item in df["observation.payload_state"].to_list()]
        )
        all_actions.append(actions)

        task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else -1
        prompt = prompts.get(task_index, DEFAULT_PROMPT)
        starts = uniform_starts(total_frames, args.num_frames, args.chunks_per_episode)
        for chunk_id, start_frame in enumerate(starts):
            frame_indices = [
                min(total_frames - 1, start_frame + g * int(args.time_division_factor))
                for g in range(token_count)
            ]
            physical_context = [
                normalize_payload_xy(payload_states[frame_index, :2])
                for frame_index in frame_indices
            ]
            end_frame = start_frame + int(args.num_frames) - 1
            valid_frames = max(0, min(int(args.num_frames), total_frames - start_frame))
            window_frame_indices = [
                min(total_frames - 1, start_frame + offset)
                for offset in range(int(args.num_frames))
            ]
            target_payload_xy_norm_window = [
                normalize_payload_xy(payload_states[frame_index, :2])
                for frame_index in window_frame_indices
            ]
            base_row = {
                "action": f"data/{chunk_name}/episode_{episode_index:06d}.parquet",
                "chunk_id": chunk_id,
                "end_frame": end_frame,
                "episode_index": episode_index,
                "length": int(args.num_frames),
                "pad_short": valid_frames < int(args.num_frames),
                "physical_context_dim": 2,
                "physical_context_frame_indices": frame_indices,
                "physical_context_stride": int(args.time_division_factor),
                "physical_context_tokens": token_count,
                "prompt": prompt,
                "sample_id": f"pnp_dynamic:ep{episode_index:06d}:chunk{chunk_id:02d}",
                "source_dataset": source.name,
                "source_split": "pnp_dynamics_train",
                "start_frame": start_frame,
                "target_payload_xy_frame_indices": window_frame_indices,
                "target_payload_xy_norm_window": target_payload_xy_norm_window,
                "target_physical_context": physical_context,
                "target_physical_context_normalization": "C_xy = (payload_xy + 0.30) / 0.60",
                "task": "pnp_dynamic_payload_trajectory_prediction",
                "task_index": task_index,
                "total_frames": total_frames,
                "valid_frames": valid_frames,
                "video": [
                    f"videos/{chunk_name}/observation.images.image/episode_{episode_index:06d}.mp4",
                    f"videos/{chunk_name}/observation.images.wrist_image/episode_{episode_index:06d}.mp4",
                ],
            }
            oracle_row = dict(base_row)
            oracle_row.update(
                {
                    "physical_context": physical_context,
                    "physical_context_normalization": "C_xy = (payload_xy + 0.30) / 0.60",
                    "physical_context_source": "oracle_payload_xy_per_time_token",
                }
            )
            oracle_rows.append(oracle_row)
            stage2_rows.append(base_row)

    action_values = np.concatenate(all_actions, axis=0)
    stats = {
        "action_pose": stats_for_array(action_values),
        "eef_delta": stats_for_array(action_values),
    }
    summary = {
        "source": str(source),
        "output": str(output),
        "episodes": episode_count,
        "rows": len(stage2_rows),
        "total_video_frames": total_video_frames,
        "num_frames": int(args.num_frames),
        "future_frames": int(args.num_frames) - 1,
        "chunks_per_episode": int(args.chunks_per_episode),
        "time_division_factor": int(args.time_division_factor),
        "physical_context_dim": 2,
        "physical_context_tokens": token_count,
        "physical_context_normalization": "C_xy = (payload_xy + 0.30) / 0.60",
    }
    return oracle_rows, stage2_rows, {"stats": stats, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--chunks_per_episode", type=int, default=8)
    parser.add_argument("--time_division_factor", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oracle_rows, stage2_rows, extra = build_rows(args)
    write_jsonl(output / "train_oracle_payloadxy_c2.jsonl", oracle_rows)
    write_jsonl(output / "train.jsonl", stage2_rows)
    with (output / "action_stats.json").open("w", encoding="utf-8") as f:
        json.dump(extra["stats"], f, indent=2, sort_keys=True)
    with (output / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(extra["summary"], f, indent=2, sort_keys=True)
    print(json.dumps(extra["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
