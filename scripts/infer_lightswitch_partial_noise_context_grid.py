#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import (  # noqa: E402
    build_pipeline,
    prepare_sample_for_rollout,
    save_video,
)
from infer_lightswitch_counterfactual_contexts import (  # noqa: E402
    _read_context_table,
    _read_jsonl,
)
from make_gt_pred_comparison import _default_pred_name  # noqa: E402
from wan_video_action.data import RoboTwinUnifiedDataset  # noqa: E402
from wan_video_action.data.operators import (  # noqa: E402
    LoadCobotAction,
    create_video_operator,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


CONTEXT_LABELS = {
    0: "neither",
    1: "red_only",
    2: "blue_only",
    3: "both",
}
CONTEXT_COLORS = {
    0: (185, 185, 185),
    1: (235, 92, 92),
    2: (88, 150, 245),
    3: (235, 184, 67),
}


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in str(raw).split(",") if value.strip()]
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError(f"Expected exactly four distinct sample indices, got {values}.")
    return values


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError(f"Noise sigmas must lie in (0,1), got {values}.")
    return values


def _strength_for_shifted_sigma(sigma: float, shift: float) -> float:
    denominator = float(shift) - (float(shift) - 1.0) * float(sigma)
    if denominator <= 0.0:
        raise ValueError(f"Cannot invert shifted sigma={sigma} with shift={shift}.")
    return float(sigma) / denominator


def _shifted_sigma(strength: float, shift: float) -> float:
    return float(shift) * float(strength) / (
        1.0 + (float(shift) - 1.0) * float(strength)
    )


def _build_full_chunk_dataset(args) -> RoboTwinUnifiedDataset:
    with open(args.action_stat_path, "r", encoding="utf-8") as handle:
        action_stat = json.load(handle)
    return RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=("video", "action"),
        main_data_operator=create_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=args.spatial_division_factor,
            width_division_factor=args.spatial_division_factor,
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            resize_mode=args.resize_mode,
            pad_short=True,
            frame_stride=int(args.frame_stride),
        ),
        special_operator_map={
            "action": LoadCobotAction(
                base_path=args.dataset_base_path,
                action_type=args.action_type,
                stat=action_stat,
                num_frames=None,
                align_num_frames=False,
                time_division_factor=args.time_division_factor,
                time_division_remainder=args.time_division_remainder,
                pad_short=True,
                output_dim=args.action_dim,
                frame_stride=int(args.frame_stride),
            )
        },
    )


def _frame_with_label(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8), mode="RGB")
    panel = Image.new("RGB", (image.width, image.height + 28), (18, 18, 18))
    panel.paste(image, (0, 28))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 27), fill=color)
    draw.text((7, 7), label, fill=(0, 0, 0))
    return panel


def _write_2x2_context_grid(
    paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    title: str,
    fps: int,
    quality: int,
) -> int:
    if len(paths) != 4 or len(labels) != 4:
        raise ValueError("A context grid requires exactly four paths and labels.")
    readers = [imageio.get_reader(str(path)) for path in paths]
    iterators = [iter(reader) for reader in readers]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    try:
        with imageio.get_writer(
            str(output_path),
            fps=int(fps),
            codec="libx264",
            quality=int(quality),
        ) as writer:
            while True:
                try:
                    frames = [next(iterator) for iterator in iterators]
                except StopIteration:
                    break
                panels = [
                    _frame_with_label(
                        frame,
                        labels[index],
                        CONTEXT_COLORS[index],
                    )
                    for index, frame in enumerate(frames)
                ]
                panel_width = max(panel.width for panel in panels)
                panel_height = max(panel.height for panel in panels)
                canvas = Image.new(
                    "RGB",
                    (2 * panel_width, 2 * panel_height + 32),
                    (8, 8, 8),
                )
                draw = ImageDraw.Draw(canvas)
                draw.text((7, 9), title, fill=(245, 245, 245))
                for index, panel in enumerate(panels):
                    if panel.size != (panel_width, panel_height):
                        panel = panel.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
                    x = (index % 2) * panel_width
                    y = 32 + (index // 2) * panel_height
                    canvas.paste(panel, (x, y))
                writer.append_data(np.asarray(canvas))
                frame_count += 1
    finally:
        for reader in readers:
            reader.close()
    if frame_count <= 0:
        raise RuntimeError(f"No frames were written to {output_path}.")
    return frame_count


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(
        "Denoise real LightSwitch chunks from fixed latent noise under all four endpoint contexts."
    )
    parser = add_general_config(parser)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--physical_context_table_path", required=True)
    parser.add_argument("--source_group", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--sample_indices", required=True)
    parser.add_argument("--noise_sigmas", default="0.65,0.90")
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--experiment_output_path", required=True)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))
    sample_indices = _parse_csv_ints(args.sample_indices)
    noise_sigmas = _parse_csv_floats(args.noise_sigmas)
    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    contexts = _read_context_table(args.physical_context_table_path)

    source_group = int(args.source_group)
    for sample_index in sample_indices:
        if not 0 <= sample_index < len(metadata_rows):
            raise IndexError(f"Sample index out of range: {sample_index}.")
        actual_group = int(metadata_rows[sample_index]["context_group_id"])
        if actual_group != source_group:
            raise ValueError(
                f"Sample {sample_index} belongs to environment {actual_group}, "
                f"not requested environment {source_group}."
            )

    output_root = Path(args.experiment_output_path)
    raw_root = output_root / "raw" / f"true_env{source_group}_{CONTEXT_LABELS[source_group]}"
    grid_root = output_root / "grids" / f"true_env{source_group}_{CONTEXT_LABELS[source_group]}"
    raw_root.mkdir(parents=True, exist_ok=True)
    grid_root.mkdir(parents=True, exist_ok=True)

    dataset = _build_full_chunk_dataset(args)
    pipe = build_pipeline(args)
    records = []

    for sample_index in sample_indices:
        row = metadata_rows[sample_index]
        sample = dataset[sample_index]
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        if int(sample["video"].shape[2]) != int(sample["total_frames"]):
            raise ValueError(
                f"Full-video loader mismatch for sample {sample_index}: "
                f"video_frames={sample['video'].shape[2]} total_frames={sample['total_frames']}."
            )
        pred_name = _default_pred_name(sample_index, row)
        stem = Path(pred_name).stem

        for requested_sigma in noise_sigmas:
            strength = _strength_for_shifted_sigma(requested_sigma, args.sigma_shift)
            actual_sigma = _shifted_sigma(strength, args.sigma_shift)
            if not math.isclose(actual_sigma, requested_sigma, abs_tol=1e-7):
                raise RuntimeError(
                    f"Sigma inversion failed: requested={requested_sigma} actual={actual_sigma}."
                )
            sigma_tag = f"sigma{int(round(requested_sigma * 100)):02d}"
            shared_seed = (
                int(args.seed)
                + int(sample_index) * 1009
                + int(round(requested_sigma * 1000)) * 17
            )
            context_paths = []

            for target_group in range(4):
                context_dir = raw_root / stem / sigma_tag
                output_path = (
                    context_dir
                    / f"context_env{target_group}_{CONTEXT_LABELS[target_group]}.mp4"
                )
                context_paths.append(output_path)
                if output_path.exists() and args.skip_existing:
                    print(
                        f"[skip] sample={sample_index} true_env={source_group} "
                        f"sigma={requested_sigma:.2f} context={target_group} output={output_path}",
                        flush=True,
                    )
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                set_global_seed(shared_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(shared_seed)
                print(
                    f"[infer] sample={sample_index} true_env={source_group} "
                    f"sigma={requested_sigma:.2f} strength={strength:.8f} "
                    f"context={target_group} seed={shared_seed} output={output_path}",
                    flush=True,
                )
                video = pipe(
                    input_video=sample["video"],
                    denoising_strength=strength,
                    action=sample["action"],
                    physical_context=torch.tensor(
                        contexts[target_group],
                        dtype=pipe.torch_dtype,
                        device=pipe.device,
                    ),
                    seed=shared_seed,
                    rand_device="cpu",
                    tiled=False,
                    height=int(args.height) * int(sample["video"].shape[0]),
                    width=int(args.width),
                    num_frames=int(sample["total_frames"]),
                    num_history_frames=int(args.num_history_frames),
                    cfg_scale=float(args.cfg_scale),
                    num_inference_steps=int(args.num_inference_steps),
                    sigma_shift=float(args.sigma_shift),
                    use_history_condition_noise_in_inference=True,
                    progress_bar_cmd=lambda iterable, *unused_args, **unused_kwargs: iterable,
                    output_type="floatpoint",
                )
                observed_sigma = float(pipe.scheduler.sigmas[0])
                if not math.isclose(observed_sigma, requested_sigma, abs_tol=2e-4):
                    raise RuntimeError(
                        f"Pipeline started at sigma={observed_sigma}, "
                        f"requested={requested_sigma}."
                    )
                save_video(
                    video.detach().cpu(),
                    output_path=str(output_path),
                    fps=int(args.fps),
                    quality=int(args.quality),
                )
                del video
                torch.cuda.empty_cache()

            grid_path = grid_root / f"{stem}_{sigma_tag}_contexts2x2.mp4"
            if not (grid_path.exists() and args.skip_existing):
                _write_2x2_context_grid(
                    context_paths,
                    [f"C{group}: {CONTEXT_LABELS[group]}" for group in range(4)],
                    grid_path,
                    title=(
                        f"sample={sample_index} true={CONTEXT_LABELS[source_group]} "
                        f"latent_noise_sigma={requested_sigma:.2f}"
                    ),
                    fps=int(args.fps),
                    quality=int(args.quality),
                )
            records.append(
                {
                    "source_group": source_group,
                    "source_label": CONTEXT_LABELS[source_group],
                    "sample_index": int(sample_index),
                    "episode_index": int(row["episode_index"]),
                    "button_colors": row.get("covered_button_colors", []),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "requested_noise_sigma": float(requested_sigma),
                    "scheduler_denoising_strength": float(strength),
                    "actual_initial_sigma": float(actual_sigma),
                    "shared_seed_across_contexts": int(shared_seed),
                    "context_outputs": [str(path) for path in context_paths],
                    "grid_output": str(grid_path),
                }
            )

    manifest = {
        "checkpoint": str(args.ckpt_path),
        "context_table": str(args.physical_context_table_path),
        "dataset_metadata": str(args.dataset_metadata_path),
        "source_group": source_group,
        "source_label": CONTEXT_LABELS[source_group],
        "sample_indices": sample_indices,
        "noise_sigmas": noise_sigmas,
        "num_contexts": 4,
        "num_raw_predictions": len(sample_indices) * len(noise_sigmas) * 4,
        "num_grid_videos": len(sample_indices) * len(noise_sigmas),
        "same_noise_across_four_contexts": True,
        "history_condition_is_preserved": True,
        "records": records,
    }
    _write_json(output_root / f"manifest_env{source_group}.json", manifest)
    _write_json(
        output_root / f"complete_env{source_group}.json",
        {
            "complete": True,
            "source_group": source_group,
            "raw_predictions": manifest["num_raw_predictions"],
            "grid_videos": manifest["num_grid_videos"],
        },
    )
    print(
        f"[done] source_group={source_group} raw={manifest['num_raw_predictions']} "
        f"grids={manifest['num_grid_videos']} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
