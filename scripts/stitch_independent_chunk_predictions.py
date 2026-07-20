#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio

from make_gt_pred_comparison import _default_pred_name


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_video(path: Path):
    reader = imageio.get_reader(str(path))
    try:
        return [frame[:, :, :3] for frame in reader]
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        "Stitch independently generated fixed-length chunks onto the source timeline."
    )
    parser.add_argument("--chunk-metadata-path", required=True)
    parser.add_argument("--episode-metadata-path", required=True)
    parser.add_argument("--chunk-prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=6)
    args = parser.parse_args()

    chunk_rows = _read_jsonl(Path(args.chunk_metadata_path))
    episode_rows = _read_jsonl(Path(args.episode_metadata_path))
    chunk_prediction_dir = Path(args.chunk_prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_episode = defaultdict(list)
    for sample_index, row in enumerate(chunk_rows):
        chunks_by_episode[int(row["episode_index"])].append((sample_index, row))

    for episode_sample_index, episode_row in enumerate(episode_rows):
        episode_index = int(episode_row["episode_index"])
        target_length = int(episode_row["length"])
        timeline = [None] * target_length
        sources = [None] * target_length
        chunks = sorted(
            chunks_by_episode[episode_index], key=lambda item: int(item[1]["start_frame"])
        )
        if not chunks:
            raise RuntimeError(f"No chunks found for episode {episode_index}")

        for chunk_sample_index, chunk_row in chunks:
            prediction_name = _default_pred_name(chunk_sample_index, chunk_row)
            prediction_path = chunk_prediction_dir / prediction_name
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            frames = _read_video(prediction_path)
            start_frame = int(chunk_row["start_frame"])
            expected = int(chunk_row["length"])
            if len(frames) < expected:
                raise RuntimeError(
                    f"Prediction {prediction_path} has {len(frames)} frames; expected {expected}"
                )
            for local_index in range(expected):
                global_index = start_frame + local_index
                if global_index >= target_length:
                    break
                # Earlier non-overlapping chunks own existing frames. A final
                # overlapping tail chunk only fills frames not covered yet.
                if timeline[global_index] is None:
                    timeline[global_index] = frames[local_index]
                    sources[global_index] = {
                        "chunk_sample_index": chunk_sample_index,
                        "chunk_start_frame": start_frame,
                        "chunk_local_frame": local_index,
                    }

        missing = [index for index, frame in enumerate(timeline) if frame is None]
        if missing:
            raise RuntimeError(
                f"Episode {episode_index} has unfilled frames: {missing[:20]}"
            )

        output_name = _default_pred_name(episode_sample_index, episode_row)
        output_path = output_dir / output_name
        with imageio.get_writer(
            str(output_path), fps=int(args.fps), codec="libx264", quality=int(args.quality)
        ) as writer:
            for frame in timeline:
                writer.append_data(frame)
        (output_dir / output_name.replace(".mp4", ".stitch_map.json")).write_text(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "target_length": target_length,
                    "chunk_count": len(chunks),
                    "frame_sources": sources,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"[stitch] episode={episode_index} chunks={len(chunks)} "
            f"frames={target_length} output={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
