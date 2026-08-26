#!/usr/bin/env python3
"""Compose compact Event80 GT/prediction grids from existing comparisons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg


ENVIRONMENTS = {
    "id": {
        5: (0, 1, 2, 3, 4, 6, 7, 8, 9),
        105: (100, 101, 102, 103, 104, 106, 107, 108, 109),
        205: (200, 201, 202, 203, 204, 206, 207, 208, 209),
        305: (300, 301, 302, 303, 304, 306, 307, 308, 309),
        355: (350, 351, 352, 353, 354, 356, 357, 358, 359),
    },
    "ood": {
        15: (10, 11, 12, 13, 14, 16, 17, 18, 19),
        95: (90, 91, 92, 93, 94, 96, 97, 98, 99),
        175: (170, 171, 172, 173, 174, 176, 177, 178, 179),
        265: (260, 261, 262, 263, 264, 266, 267, 268, 269),
        345: (340, 341, 342, 343, 344, 346, 347, 348, 349),
    },
}

OUTPUT_SUFFIX = {
    "all_active_joint_no_curriculum": "all_active_joint",
    "dinov2_amortized_context": "dinov2",
    "history_conditioned_wm": "history",
    "joint_model_latent_training": "joint_model_latent",
    "lora_tta": "lora",
    "standard_pooled_wm": "pooled",
    "ttt_kqv": "ttt_kqv",
}

DEFAULT_DATASET_BASE_PATH = Path(
    "/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets/pushbox/"
    "libero_push_box_event_tap_segmented80_10action_hidden_lerobot_"
    "A500_offset160_stop_2026-07-05_hai-machine/hidden_straight_lerobot"
)


def load_environments(path: Path | None) -> dict[str, dict[int, tuple[int, ...]]]:
    if path is None:
        return ENVIRONMENTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    environments: dict[str, dict[int, tuple[int, ...]]] = {"id": {}, "ood": {}}
    for record in payload.get("environments", []):
        split = str(record["domain"])
        if split not in environments:
            raise ValueError(f"Unsupported domain {split!r} in {path}.")
        supports = tuple(int(value) for value in record["support_indices"])
        queries = tuple(int(value) for value in record["query_indices"])
        if len(supports) != 1:
            raise ValueError(
                f"Grid composition requires one support per environment, got {supports}."
            )
        if len(queries) != 9:
            raise ValueError(
                f"Grid composition requires nine queries per environment, got {queries}."
            )
        environments[split][supports[0]] = queries
    if sum(len(values) for values in environments.values()) != 10:
        raise ValueError(f"Expected ten environments in {path}, got {environments}.")
    return environments


def find_visualization_root(method_dir: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in method_dir.rglob("visualizations/id")
        if (path.parent / "ood").is_dir()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one visualization root below {method_dir}, found {candidates}."
        )
    return candidates[0]


def find_comparison(visualization_root: Path, split: str, sample_index: int) -> Path:
    matches = sorted(
        (visualization_root / split).glob(
            f"sample{sample_index:04d}_*_gt_*.mp4"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one comparison for {split} sample {sample_index}, found {matches}."
        )
    return matches[0]


def compose_grid(
    *,
    ffmpeg: str,
    method: str,
    visualization_root: Path,
    split: str,
    support_index: int,
    query_indices: tuple[int, ...],
    dataset_base_path: Path,
    start_frame: int,
    end_frame: int,
    overwrite: bool,
) -> tuple[Path, bool]:
    if len(query_indices) != 9:
        raise ValueError(f"Expected nine queries, got {len(query_indices)}.")
    suffix = OUTPUT_SUFFIX.get(method, method)
    output_dir = visualization_root / "grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"grid_{split}_support_sample{support_index:04d}_"
        f"support_plus_9queries_2x5_gt_pred_{suffix}.mp4"
    )
    if output_path.is_file() and not overwrite:
        return output_path, False

    inputs = [
        find_comparison(visualization_root, split, sample_index)
        for sample_index in query_indices
    ]
    support_main = (
        dataset_base_path
        / "videos/chunk-000/observation.images.image"
        / f"episode_{support_index:06d}.mp4"
    )
    support_wrist = (
        dataset_base_path
        / "videos/chunk-000/observation.images.wrist_image"
        / f"episode_{support_index:06d}.mp4"
    )
    for path in (support_main, support_wrist):
        if not path.is_file():
            raise FileNotFoundError(path)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    command.extend(["-i", str(support_main), "-i", str(support_wrist)])
    for path in inputs:
        command.extend(["-i", str(path)])

    filters = [
        f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame + 1},"
        "setpts=PTS-STARTPTS,scale=192:192[support_main]",
        f"[1:v]trim=start_frame={start_frame}:end_frame={end_frame + 1},"
        "setpts=PTS-STARTPTS,scale=192:192[support_wrist]",
        "[support_main][support_wrist]hstack=inputs=2[support_row]",
        "[support_row]split=2[support_top][support_bottom]",
        "[support_top][support_bottom]vstack=inputs=2,"
        "drawbox=x=0:y=0:w=iw:h=ih:color=yellow:t=6[cell0]",
    ]
    for index in range(9):
        filters.append(
            f"[{index + 2}:v]scale=384:384:force_original_aspect_ratio=decrease,"
            f"pad=384:384:(ow-iw)/2:(oh-ih)/2:black[cell{index + 1}]"
        )
    layout = "|".join(
        f"{384 * (index % 5)}_{384 * (index // 5)}" for index in range(10)
    )
    filters.append(
        "".join(f"[cell{index}]" for index in range(10))
        + f"xstack=inputs=10:layout={layout}:fill=black[out]"
    )
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-r",
            "20",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(temporary_path),
        ]
    )
    try:
        subprocess.run(command, check=True)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--dataset-base-path", type=Path, default=DEFAULT_DATASET_BASE_PATH
    )
    parser.add_argument("--start-frame", type=int, default=65)
    parser.add_argument("--end-frame", type=int, default=105)
    parser.add_argument(
        "--support-query-manifest",
        type=Path,
        help="Optional variable-support manifest; omitted preserves the legacy action-5 grid.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    methods_root = args.benchmark_root / "methods"
    if args.methods == "all":
        methods = sorted(path.name for path in methods_root.iterdir() if path.is_dir())
    else:
        methods = [value.strip() for value in args.methods.split(",") if value.strip()]

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    environments_by_split = load_environments(args.support_query_manifest)
    jobs = []
    for method in methods:
        visualization_root = find_visualization_root(methods_root / method)
        for split, environments in environments_by_split.items():
            for support_index, query_indices in environments.items():
                jobs.append(
                    {
                        "ffmpeg": ffmpeg,
                        "method": method,
                        "visualization_root": visualization_root,
                        "split": split,
                        "support_index": support_index,
                        "query_indices": query_indices,
                        "dataset_base_path": args.dataset_base_path,
                        "start_frame": int(args.start_frame),
                        "end_frame": int(args.end_frame),
                        "overwrite": bool(args.overwrite),
                    }
                )

    created = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(compose_grid, **job) for job in jobs]
        for future in as_completed(futures):
            path, was_created = future.result()
            if was_created:
                created += 1
                print(f"[grid] {path}", flush=True)
            else:
                skipped += 1
                print(f"[skip] {path}", flush=True)
    print(f"[done] created={created} skipped={skipped} total={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
