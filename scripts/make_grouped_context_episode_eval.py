#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_gt_pred_comparison import (  # noqa: E402
    _default_pred_name,
    _read_gt_video,
    _read_pred_video,
    _resize_rgb,
)


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: str | Path, payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _select_window_stride(rows: list[dict], indices: list[int], stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError(f"Window stride must be positive, got {stride}.")
    ordered = sorted(indices, key=lambda index: (int(rows[index]["start_frame"]), index))
    if not ordered:
        return []
    selected = [ordered[0]]
    last_start = int(rows[ordered[0]]["start_frame"])
    for index in ordered[1:]:
        start = int(rows[index]["start_frame"])
        if start - last_start >= stride:
            selected.append(index)
            last_start = start

    # Preserve the final tail-aligned window if the regular stride misses the
    # end of the episode. This keeps complete temporal coverage.
    tail = ordered[-1]
    if int(rows[tail]["end_frame"]) > int(rows[selected[-1]]["end_frame"]):
        if tail != selected[-1]:
            selected.append(tail)
    return selected


def make_plan(args) -> None:
    rows = _read_jsonl(args.metadata_path)
    requested = _parse_floats(args.frictions)
    grouped: dict[float, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        mu = float(row["friction_mu"])
        if requested and min(abs(mu - value) for value in requested) > 1e-8:
            continue
        grouped[mu][int(row["episode_index"])].append(index)

    environments = []
    for split_index, mu in enumerate(sorted(grouped)):
        episodes = []
        for episode, indices in grouped[mu].items():
            ordered = _select_window_stride(rows, indices, int(args.chunk_stride))
            episodes.append((episode, ordered))
        episodes.sort(key=lambda item: (-len(item[1]), item[0]))
        if len(episodes) < 2:
            raise ValueError(f"Need two episodes for mu={mu:g}; found {len(episodes)}.")
        support_episode, support_indices = episodes[0]
        transfer_episode, transfer_indices = episodes[1]
        environments.append(
            {
                "friction_mu": mu,
                "split_index": split_index % int(args.num_splits),
                "source_index": int(support_indices[0]),
                "support_episode": int(support_episode),
                "support_indices": support_indices,
                "support_start_frame": int(rows[support_indices[0]]["start_frame"]),
                "support_end_frame": int(rows[support_indices[-1]]["end_frame"]),
                "transfer_episode": int(transfer_episode),
                "transfer_indices": transfer_indices,
                "transfer_start_frame": int(rows[transfer_indices[0]]["start_frame"]),
                "transfer_end_frame": int(rows[transfer_indices[-1]]["end_frame"]),
            }
        )

    plan = {
        "metadata_path": str(Path(args.metadata_path).resolve()),
        "window_policy": "fixed-stride metadata windows plus one tail-aligned window when needed",
        "chunk_length": 41,
        "chunk_stride": int(args.chunk_stride),
        "episode_selection": "two longest episodes per friction; lower episode id breaks ties",
        "num_splits": int(args.num_splits),
        "environments": environments,
    }
    _write_json(args.plan_path, plan)
    for item in environments:
        print(
            f"[plan] mu={item['friction_mu']:g} split={item['split_index']} "
            f"support_ep={item['support_episode']} windows={len(item['support_indices'])} "
            f"transfer_ep={item['transfer_episode']} windows={len(item['transfer_indices'])} "
            f"source={item['source_index']}",
            flush=True,
        )
    print(f"[done] plan={args.plan_path}", flush=True)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = min(image.width, max(130, 8 * len(text) + 16))
    draw.rectangle((0, 0, width, 25), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _read_chunk(
    *,
    row: dict,
    sample_index: int,
    mode: str,
    dataset_base_path: Path,
    prediction_dir: Path | None,
    width: int,
    height: int,
) -> list[np.ndarray]:
    total_frames = int(row.get("length", int(row["end_frame"]) - int(row["start_frame"]) + 1))
    target_width = int(width) * len(row["video"])
    if mode == "gt":
        frames = _read_gt_video(row, dataset_base_path, int(width), int(height), total_frames)
    else:
        if prediction_dir is None:
            raise ValueError(f"prediction_dir is required for mode={mode}.")
        path = prediction_dir / _default_pred_name(sample_index, row)
        if not path.exists():
            raise FileNotFoundError(f"Missing {mode} prediction: {path}")
        frames = _read_pred_video(path, target_width, int(height))
    return [_resize_rgb(frame, target_width, int(height)) for frame in frames]


def _stitch_episode(
    *,
    rows: list[dict],
    indices: list[int],
    mode: str,
    dataset_base_path: Path,
    prediction_dir: Path | None,
    width: int,
    height: int,
) -> list[np.ndarray]:
    stitched: list[np.ndarray] = []
    covered_end: int | None = None
    for sample_index in sorted(indices, key=lambda index: (int(rows[index]["start_frame"]), index)):
        row = rows[sample_index]
        frames = _read_chunk(
            row=row,
            sample_index=sample_index,
            mode=mode,
            dataset_base_path=dataset_base_path,
            prediction_dir=prediction_dir,
            width=width,
            height=height,
        )
        start = int(row["start_frame"])
        frame_stride = int(row.get("frame_stride", 1))
        if covered_end is None:
            offset = 0
        else:
            offset = max(0, (covered_end - start) // frame_stride + 1)
        if offset < len(frames):
            stitched.extend(frames[offset:])
        if frames:
            covered_end = max(
                covered_end if covered_end is not None else start - frame_stride,
                start + (len(frames) - 1) * frame_stride,
            )
    if not stitched:
        raise ValueError(f"No frames stitched for mode={mode}, indices={indices}.")
    return stitched


def _write_rows_video(
    path: Path,
    variants: list[tuple[str, list[np.ndarray]]],
    *,
    fps: int,
    quality: int,
) -> None:
    frame_count = min(len(frames) for _, frames in variants)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=int(quality)) as writer:
        for frame_index in range(frame_count):
            writer.append_data(
                np.concatenate(
                    [_label(frames[frame_index], label) for label, frames in variants],
                    axis=0,
                )
            )


def _write_environment_grid(
    path: Path,
    environments: list[dict],
    videos: dict[tuple[float, str], list[np.ndarray]],
    *,
    role: str,
    fps: int,
    quality: int,
) -> None:
    variants = ("GT", "Stage1", "Stage2_TTT")
    frame_count = max(len(videos[(float(item["friction_mu"]), variant)]) for item in environments for variant in variants)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=int(quality)) as writer:
        for frame_index in range(frame_count):
            rows = []
            for variant in variants:
                cells = []
                for item in environments:
                    mu = float(item["friction_mu"])
                    frames = videos[(mu, variant)]
                    frame = frames[min(frame_index, len(frames) - 1)]
                    episode = int(item[f"{role}_episode"])
                    cells.append(_label(frame, f"{variant} | mu={mu:g} | ep={episode}"))
                rows.append(np.concatenate(cells, axis=1))
            writer.append_data(np.concatenate(rows, axis=0))


def render(args) -> None:
    rows = _read_jsonl(args.metadata_path)
    plan = json.loads(Path(args.plan_path).read_text(encoding="utf-8"))
    environments = sorted(plan["environments"], key=lambda item: float(item["friction_mu"]))
    dataset_base_path = Path(args.dataset_base_path)
    stage1_dir = Path(args.stage1_pred_dir)
    stage2_dir = Path(args.stage2_pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"plan_path": str(Path(args.plan_path).resolve()), "roles": {}}

    for role in ("support", "transfer"):
        role_dir = output_dir / f"{role}_episodes"
        role_dir.mkdir(parents=True, exist_ok=True)
        grid_videos: dict[tuple[float, str], list[np.ndarray]] = {}
        role_summary = []
        for item in environments:
            mu = float(item["friction_mu"])
            indices = [int(index) for index in item[f"{role}_indices"]]
            variants = {
                "GT": _stitch_episode(
                    rows=rows,
                    indices=indices,
                    mode="gt",
                    dataset_base_path=dataset_base_path,
                    prediction_dir=None,
                    width=int(args.width),
                    height=int(args.height),
                ),
                "Stage1": _stitch_episode(
                    rows=rows,
                    indices=indices,
                    mode="stage1",
                    dataset_base_path=dataset_base_path,
                    prediction_dir=stage1_dir,
                    width=int(args.width),
                    height=int(args.height),
                ),
                "Stage2_TTT": _stitch_episode(
                    rows=rows,
                    indices=indices,
                    mode="stage2",
                    dataset_base_path=dataset_base_path,
                    prediction_dir=stage2_dir,
                    width=int(args.width),
                    height=int(args.height),
                ),
            }
            for variant, frames in variants.items():
                grid_videos[(mu, variant)] = frames
            episode = int(item[f"{role}_episode"])
            video_path = role_dir / f"mu{mu:.4f}_episode{episode:06d}_gt_stage1_stage2.mp4"
            _write_rows_video(
                video_path,
                list(variants.items()),
                fps=int(args.fps),
                quality=int(args.quality),
            )
            role_summary.append(
                {
                    "friction_mu": mu,
                    "episode_index": episode,
                    "window_count": len(indices),
                    "stitched_frame_counts": {name: len(frames) for name, frames in variants.items()},
                    "video": str(video_path),
                }
            )
            print(f"[video] role={role} mu={mu:g} episode={episode} output={video_path}", flush=True)

        grid_path = output_dir / f"{role}_all4fric_gt_stage1_stage2_grid.mp4"
        _write_environment_grid(
            grid_path,
            environments,
            grid_videos,
            role=role,
            fps=int(args.fps),
            quality=int(args.quality),
        )
        summary["roles"][role] = {"episodes": role_summary, "grid": str(grid_path)}
        print(f"[grid] role={role} output={grid_path}", flush=True)

    _write_json(output_dir / "episode_eval_summary.json", summary)
    print(f"[done] output={output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser("Plan or render full-episode grouped-context evaluation.")
    parser.add_argument("--mode", required=True, choices=("plan", "render"))
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--frictions", default="")
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--chunk-stride", type=int, default=20)
    parser.add_argument("--dataset-base-path")
    parser.add_argument("--stage1-pred-dir")
    parser.add_argument("--stage2-pred-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "plan":
        make_plan(args)
        return
    required = (args.dataset_base_path, args.stage1_pred_dir, args.stage2_pred_dir, args.output_dir)
    if any(value is None for value in required):
        raise ValueError("Render mode requires dataset base, both prediction directories, and output directory.")
    render(args)


if __name__ == "__main__":
    main()
