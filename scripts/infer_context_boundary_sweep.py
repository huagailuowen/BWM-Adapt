#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    _run_autoregressive,
    build_infer_dataset,
    build_pipeline,
    prepare_sample_for_rollout,
)
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402
from wan_video_action.utils import set_global_seed  # noqa: E402


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_target_line_specs(value: str | None) -> dict[float, list[float]]:
    specs = {}
    if not value:
        return specs
    for item in value.split(";"):
        group_text, alphas_text = item.split(":", maxsplit=1)
        specs[float(group_text.strip())] = _parse_float_list(alphas_text)
    return specs


def _read_table(path: str | Path) -> dict[float, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = {}
    for record in payload["records"]:
        table[float(record["friction_mu"])] = np.asarray(record["context"], dtype=np.float32).reshape(-1)
    return table


def _read_failed_endpoint(path: str | Path, sample_index: int) -> np.ndarray:
    latest = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("sample_index", -1)) != int(sample_index):
                continue
            if latest is None or int(row["inner_step"]) > int(latest["inner_step"]):
                latest = row
    if latest is None:
        raise ValueError(f"No trajectory row for sample_index={sample_index}: {path}")
    return np.asarray(latest["context_flat"], dtype=np.float32).reshape(-1)


def _point_id(prefix: str, value: float) -> str:
    return f"{prefix}_{value:.3f}".replace(".", "p")


def _make_points(args, table: dict[float, np.ndarray]) -> list[dict]:
    oracle = table[float(args.oracle_group)]
    points: list[dict] = []

    def add(point_id: str, family: str, context: np.ndarray, **metadata) -> None:
        context = np.clip(np.asarray(context, dtype=np.float32), args.context_min, args.context_max)
        nearest_group, nearest_distance = min(
            ((group, float(np.linalg.norm(context - value))) for group, value in table.items()),
            key=lambda item: item[1],
        )
        points.append(
            {
                "point_id": point_id,
                "family": family,
                "context": context,
                "oracle_distance": float(np.linalg.norm(context - oracle)),
                "nearest_group": float(nearest_group),
                "nearest_group_distance": float(nearest_distance),
                **metadata,
            }
        )

    add("oracle_env2", "oracle", oracle, alpha=0.0, target="oracle")
    alphas = _parse_float_list(args.line_alphas)
    target_line_specs = _parse_target_line_specs(args.target_line_specs)
    for group in args.target_groups:
        target = table[float(group)]
        for alpha in target_line_specs.get(float(group), alphas):
            add(
                _point_id(f"toward_env{int(group)}_a", alpha),
                f"toward_env{int(group)}",
                oracle + alpha * (target - oracle),
                alpha=float(alpha),
                target=f"env{int(group)}",
            )

    if args.failed_trajectory_path:
        failed = _read_failed_endpoint(args.failed_trajectory_path, args.sample_index)
        failed_alphas = _parse_float_list(args.failed_alphas) if args.failed_alphas else alphas
        for alpha in failed_alphas:
            add(
                _point_id("toward_failed_stage2_a", alpha),
                "toward_failed_stage2",
                oracle + alpha * (failed - oracle),
                alpha=float(alpha),
                target="failed_stage2",
            )

    rng = np.random.default_rng(int(args.radial_seed))
    for direction_index in range(int(args.radial_directions)):
        direction = rng.normal(size=oracle.shape).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        for radius in _parse_float_list(args.radial_radii):
            add(
                _point_id(f"radial_d{direction_index:02d}_r", radius),
                f"radial_d{direction_index:02d}",
                oracle + float(radius) * direction,
                requested_radius=float(radius),
                target=f"radial_d{direction_index:02d}",
            )
    return points


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _read_video_summary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.stack(imageio.mimread(path, memtest=False), axis=0).astype(np.uint8)
    if len(frames) < 2:
        raise ValueError(f"Video has too few frames: {path}")
    timeline_ids = np.linspace(0, len(frames) - 1, 5).round().astype(int)
    timeline = frames[timeline_ids]
    early = frames[: min(5, len(frames))].astype(np.float32).mean(axis=0)
    late = frames[-min(8, len(frames)) :].astype(np.float32).mean(axis=0)
    return timeline, frames[-1], late - early


def _write_sheet(rows: list[dict], timeline: dict[str, np.ndarray], output_path: Path) -> None:
    if not rows:
        return
    first = timeline[rows[0]["point_id"]][0]
    height, width = first.shape[:2]
    canvas = Image.new("RGB", (width * 5, height * len(rows)), color=(0, 0, 0))
    for row_index, row in enumerate(rows):
        label = f"{row['point_id']} d={row['oracle_distance']:.3f}"
        for column, frame in enumerate(timeline[row["point_id"]]):
            tile = _label_frame(frame, label if column == 0 else f"t{column}/4")
            canvas.paste(Image.fromarray(tile), (column * width, row_index * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_final_overview(rows: list[dict], final_frames: dict[str, np.ndarray], output_path: Path) -> None:
    columns = 4
    first = final_frames[rows[0]["point_id"]]
    height, width = first.shape[:2]
    num_rows = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * num_rows), color=(0, 0, 0))
    for index, row in enumerate(rows):
        label = f"{row['point_id']} d={row['oracle_distance']:.3f}"
        tile = _label_frame(final_frames[row["point_id"]], label)
        canvas.paste(Image.fromarray(tile), ((index % columns) * width, (index // columns) * height))
    canvas.save(output_path)


def _write_metrics(rows: list[dict], signatures: dict[str, np.ndarray], output_root: Path) -> None:
    by_id = {row["point_id"]: row for row in rows}
    required = {
        "oracle_env2",
        "toward_env0_a_1p000",
        "toward_env1_a_1p000",
        "toward_env3_a_1p000",
    }
    missing = sorted(required - signatures.keys())
    if missing:
        raise ValueError(f"Missing reference sweep points: {missing}")

    on_reference = 0.5 * (
        signatures["oracle_env2"] + signatures["toward_env3_a_1p000"]
    )
    off_reference = 0.5 * (
        signatures["toward_env0_a_1p000"] + signatures["toward_env1_a_1p000"]
    )
    spatial_difference = np.abs(on_reference - off_reference).mean(axis=-1)
    threshold = float(np.quantile(spatial_difference, 0.98))
    mask = spatial_difference >= threshold
    if not np.any(mask):
        mask = np.ones_like(spatial_difference, dtype=bool)

    mask_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    normalized = spatial_difference / max(float(spatial_difference.max()), 1e-12)
    mask_image[..., 0] = np.round(normalized * 255).astype(np.uint8)
    mask_image[..., 1] = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(mask_image).save(output_root / "automatic_lamp_discriminative_mask.png")

    fieldnames = [
        "point_id",
        "family",
        "oracle_distance",
        "nearest_group",
        "nearest_group_distance",
        "mse_to_on_reference",
        "mse_to_off_reference",
        "on_score",
        "automatic_state",
    ]
    with (output_root / "boundary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            signature = signatures[row["point_id"]]
            mse_on = float(np.mean((signature[mask] - on_reference[mask]) ** 2))
            mse_off = float(np.mean((signature[mask] - off_reference[mask]) ** 2))
            score = (mse_off - mse_on) / max(mse_off + mse_on, 1e-12)
            writer.writerow(
                {
                    "point_id": row["point_id"],
                    "family": row["family"],
                    "oracle_distance": row["oracle_distance"],
                    "nearest_group": row["nearest_group"],
                    "nearest_group_distance": row["nearest_group_distance"],
                    "mse_to_on_reference": mse_on,
                    "mse_to_off_reference": mse_off,
                    "on_score": score,
                    "automatic_state": "on" if score >= 0.0 else "off",
                }
            )


def parse_args():
    parser = argparse.ArgumentParser("Fixed-noise latent-context boundary sweep for one inference sample.")
    parser = add_general_config(parser)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--context_table_path", required=True)
    parser.add_argument("--failed_trajectory_path", default=None)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--oracle_group", type=float, required=True)
    parser.add_argument("--target_groups", type=float, nargs="+", default=(0.0, 1.0, 3.0))
    parser.add_argument("--line_alphas", default="0.05,0.1,0.2,0.3,0.4,0.55,0.7,0.85,1.0")
    parser.add_argument("--target_line_specs", default=None)
    parser.add_argument("--failed_alphas", default=None)
    parser.add_argument("--radial_radii", default="0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--radial_directions", type=int, default=3)
    parser.add_argument("--radial_seed", type=int, default=20260717)
    parser.add_argument("--context_min", type=float, default=0.0)
    parser.add_argument("--context_max", type=float, default=1.0)
    parser.add_argument("--sweep_output_path", required=True)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.sweep_output_path)
    raw_root = output_root / "raw"
    sheet_root = output_root / "sheets"
    raw_root.mkdir(parents=True, exist_ok=True)
    sheet_root.mkdir(parents=True, exist_ok=True)

    table = _read_table(args.context_table_path)
    points = _make_points(args, table)
    manifest = []
    for point in points:
        serializable = {key: value for key, value in point.items() if key != "context"}
        serializable["context"] = point["context"].tolist()
        serializable["output_path"] = str(raw_root / f"{point['point_id']}.mp4")
        manifest.append(serializable)
    (output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    for index, point in enumerate(points, start=1):
        output_path = raw_root / f"{point['point_id']}.mp4"
        if output_path.exists() and args.skip_existing:
            print(f"[skip] {index}/{len(points)} {output_path}", flush=True)
            continue
        set_global_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        sample = dataset[int(args.sample_index)]
        sample = prepare_sample_for_rollout(sample, int(args.sample_index), pipe, args)
        sample["physical_context"] = torch.tensor(
            point["context"], dtype=pipe.torch_dtype, device=pipe.device
        )
        sample["output_path"] = str(output_path)
        print(
            f"[infer] {index}/{len(points)} id={point['point_id']} "
            f"family={point['family']} oracle_l2={point['oracle_distance']:.6f}",
            flush=True,
        )
        _run_autoregressive(pipe=pipe, sample=sample, args=args)
        torch.cuda.empty_cache()

    timeline: dict[str, np.ndarray] = {}
    final_frames: dict[str, np.ndarray] = {}
    signatures: dict[str, np.ndarray] = {}
    for point in points:
        point_id = point["point_id"]
        timeline[point_id], final_frames[point_id], signatures[point_id] = _read_video_summary(
            raw_root / f"{point_id}.mp4"
        )

    _write_final_overview(points, final_frames, sheet_root / "all_points_final_frames.png")
    families = sorted({point["family"] for point in points if point["family"] != "oracle"})
    oracle = next(point for point in points if point["family"] == "oracle")
    for family in families:
        rows = [oracle] + [point for point in points if point["family"] == family]
        _write_sheet(rows, timeline, sheet_root / f"{family}_timeline.png")
    _write_metrics(points, signatures, output_root)
    print(f"[done] points={len(points)} output={output_root}", flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
