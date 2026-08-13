#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACTION_AMPLITUDES = np.asarray([0.38, 0.42, 0.46, 0.56, 0.60, 0.68, 0.85, 0.90, 1.00])
VALIDATION_ENVS = {0, 18, 21, 39, 48, 86}
VALIDATION_ACTIONS = {0, 4, 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure two-box final displacement and derive smooth per-environment action weights."
    )
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--interval-normalization-power",
        type=float,
        default=0.5,
        help="0 uses adjacent observed displacement change; 1 uses derivative per action amplitude.",
    )
    parser.add_argument("--contrast-gamma", type=float, default=1.0)
    parser.add_argument("--primary-floor-fraction", type=float, default=0.6)
    parser.add_argument("--max-uniform-multiple", type=float, default=2.0)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def color_centroid(frame: np.ndarray, color: str) -> tuple[float, float, int] | None:
    hsv = np.asarray(Image.fromarray(frame).convert("HSV"))
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if color == "red":
        mask = (hue >= 4) & (hue <= 18) & (saturation >= 120) & (value >= 65)
    elif color == "blue":
        mask = (hue >= 150) & (hue <= 166) & (saturation >= 110) & (value >= 65)
    else:
        raise ValueError(color)
    mask[:65] = False
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    return float(np.median(xs)), float(np.median(ys)), int(len(xs))


def track_color(points: list[tuple[float, float, int] | None], height: int) -> dict:
    initial_candidates = [point for point in points[:5] if point is not None]
    if not initial_candidates:
        return {
            "initial_x": math.nan,
            "initial_y": math.nan,
            "final_x": math.nan,
            "final_y": math.nan,
            "normalized_displacement": math.nan,
            "offscreen": False,
            "last_visible_frame": -1,
            "visible_fraction": 0.0,
            "valid": False,
        }
    initial_x = float(np.median([point[0] for point in initial_candidates]))
    initial_y = float(np.median([point[1] for point in initial_candidates]))
    visible_indices = [index for index, point in enumerate(points) if point is not None]
    last_visible = visible_indices[-1]
    tail = [point for point in points[-5:] if point is not None]
    visible_fraction = len(visible_indices) / len(points)
    if tail:
        final_x = float(np.median([point[0] for point in tail]))
        final_y = float(np.median([point[1] for point in tail]))
        denominator = max((height - 1) - initial_y, 1.0)
        displacement = float(np.clip((final_y - initial_y) / denominator, 0.0, 1.0))
        offscreen = False
    else:
        last_point = points[last_visible]
        assert last_point is not None
        final_x, final_y = float(last_point[0]), float(last_point[1])
        moved_down = final_y - initial_y >= 8.0
        near_bottom = final_y >= 0.72 * height
        disappeared_after_collision = last_visible >= 5 and last_visible < len(points) - 5
        offscreen = bool(moved_down and near_bottom and disappeared_after_collision)
        displacement = 1.0 if offscreen else math.nan
    return {
        "initial_x": initial_x,
        "initial_y": initial_y,
        "final_x": final_x,
        "final_y": final_y,
        "normalized_displacement": displacement,
        "offscreen": offscreen,
        "last_visible_frame": last_visible,
        "visible_fraction": visible_fraction,
        "valid": bool(np.isfinite(displacement)),
    }


def save_validation_image(
    output_path: Path,
    first_frame: np.ndarray,
    last_frame: np.ndarray,
    red: dict,
    blue: dict,
    label: str,
) -> None:
    frames = [Image.fromarray(first_frame), Image.fromarray(last_frame)]
    width, height = frames[0].size
    canvas = Image.new("RGB", (2 * width, height + 54), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, font=font(16, True), fill="#0f172a")
    draw.text(
        (8, 30),
        f"red={red['normalized_displacement']:.3f} off={int(red['offscreen'])} | "
        f"blue={blue['normalized_displacement']:.3f} off={int(blue['offscreen'])}",
        font=font(13),
        fill="#334155",
    )
    for index, frame in enumerate(frames):
        x_offset = index * width
        canvas.paste(frame, (x_offset, 54))
        draw.text((x_offset + 5, 58), "initial" if index == 0 else "final", font=font(12, True), fill="white", stroke_width=2, stroke_fill="black")
    for color_name, result, rgb in (("red", red, "#ff2d20"), ("blue", blue, "#1683ff")):
        for index, keys in enumerate((("initial_x", "initial_y"), ("final_x", "final_y"))):
            x_value, y_value = result[keys[0]], result[keys[1]]
            if np.isfinite(x_value) and np.isfinite(y_value):
                x = index * width + x_value
                y = 54 + y_value
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=rgb, width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")


def analyze_episode(task: dict) -> dict:
    video_path = Path(task["dataset_base_path"]) / task["video_path"]
    reader = imageio.get_reader(video_path)
    red_points: list[tuple[float, float, int] | None] = []
    blue_points: list[tuple[float, float, int] | None] = []
    first_frame = None
    last_frame = None
    try:
        for frame_index, frame in enumerate(reader):
            if first_frame is None:
                first_frame = frame.copy()
            last_frame = frame.copy()
            red_points.append(color_centroid(frame, "red"))
            blue_points.append(color_centroid(frame, "blue"))
    finally:
        reader.close()
    if first_frame is None or last_frame is None:
        raise RuntimeError(f"No video frames: {video_path}")
    height = int(first_frame.shape[0])
    red = track_color(red_points, height)
    blue = track_color(blue_points, height)
    result = {
        "episode_index": task["episode_index"],
        "environment_group_id": task["environment_group_id"],
        "action_id": task["action_id"],
        "action_amplitude": task["action_amplitude"],
        "target_mass_kg": task["target_mass_kg"],
        "target_table_friction_mu": task["target_table_friction_mu"],
    }
    for prefix, tracked in (("red", red), ("blue", blue)):
        for key, value in tracked.items():
            result[f"{prefix}_{key}"] = value
    if task["save_validation"]:
        save_validation_image(
            Path(task["validation_path"]),
            first_frame,
            last_frame,
            red,
            blue,
            f"env={task['environment_group_id']} action={task['action_id']} amp={task['action_amplitude']:.2f}",
        )
    return result


def isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    blocks = [[float(value), 1, index, index] for index, value in enumerate(values)]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][0] <= blocks[index + 1][0] + 1e-12:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        count = left[1] + right[1]
        merged = [
            (left[0] * left[1] + right[0] * right[1]) / count,
            count,
            left[2],
            right[3],
        ]
        blocks[index : index + 2] = [merged]
        index = max(index - 1, 0)
    output = np.empty_like(values, dtype=np.float64)
    for value, _, start, end in blocks:
        output[start : end + 1] = value
    return output


def bounded_probability(scores: np.ndarray, floor_fraction: float, max_multiple: float, gamma: float) -> np.ndarray:
    count = len(scores)
    uniform = 1.0 / count
    floor = floor_fraction * uniform
    ceiling = max_multiple * uniform
    positive = np.maximum(scores, 0.0)
    if float(positive.sum()) <= 1e-12:
        return np.full(count, uniform)
    preference = np.power(positive + 1e-8, gamma)
    preference /= preference.sum()
    probability = floor + (1.0 - count * floor) * preference
    for _ in range(20):
        above = probability > ceiling
        if not np.any(above):
            break
        overflow = float(np.sum(probability[above] - ceiling))
        probability[above] = ceiling
        free = ~above
        if not np.any(free):
            break
        weights = preference[free]
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(np.count_nonzero(free))
        probability[free] += overflow * weights / weights.sum()
    probability /= probability.sum()
    return probability


def draw_validation_montage(paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in sorted(paths)]
    if not images:
        return
    columns = 3
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "#e2e8f0")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % columns) * cell_w, (index // columns) * cell_h))
    canvas.save(output_path, "PNG")


def heat_color(value: float, minimum: float, maximum: float) -> tuple[int, int, int]:
    ratio = float(np.clip((value - minimum) / max(maximum - minimum, 1e-12), 0.0, 1.0))
    anchors = ((35, 81, 180), (42, 205, 190), (241, 226, 38), (220, 38, 38))
    scaled = ratio * (len(anchors) - 1)
    index = min(int(scaled), len(anchors) - 2)
    local = scaled - index
    return tuple(round(anchors[index][channel] * (1 - local) + anchors[index + 1][channel] * local) for channel in range(3))


def draw_heatmaps(group_rows: list[dict], output_path: Path) -> None:
    environments = sorted(
        {int(row["environment_group_id"]) for row in group_rows},
        key=lambda env: (
            next(float(row["target_mass_kg"]) for row in group_rows if int(row["environment_group_id"]) == env),
            next(float(row["target_table_friction_mu"]) for row in group_rows if int(row["environment_group_id"]) == env),
        ),
    )
    by_key = {(int(row["environment_group_id"]), int(row["action_id"])): row for row in group_rows}
    panels = (
        ("Red final displacement", "red_displacement_isotonic", 0.0, 1.0),
        ("Blue final displacement", "blue_displacement_isotonic", 0.0, 1.0),
        ("Local response importance", "local_response_score", 0.0, max(float(row["local_response_score"]) for row in group_rows)),
        ("Sampling probability / uniform", "primary_relative_to_uniform", 0.6, 2.0),
    )
    cell_w, cell_h = 34, 6
    left, top, panel_gap = 145, 115, 62
    panel_w = len(ACTION_AMPLITUDES) * cell_w
    width = left + len(panels) * panel_w + (len(panels) - 1) * panel_gap + 35
    height = top + len(environments) * cell_h + 70
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 24), "Empirical action response and bounded sampling weights", font=font(27, True), fill="#0f172a")
    draw.text((left, 61), "100 environments x 9 measured GT episodes | rows sorted by target mass then friction", font=font(15), fill="#64748b")
    for panel_index, (title, key, minimum, maximum) in enumerate(panels):
        x0 = left + panel_index * (panel_w + panel_gap)
        draw.text((x0, top - 54), title, font=font(15, True), fill="#1e293b")
        for action_id, amplitude in enumerate(ACTION_AMPLITUDES):
            draw.text((x0 + action_id * cell_w + cell_w / 2, top - 18), f"{amplitude:.2f}", font=font(10), fill="#475569", anchor="ms")
        for row_index, env in enumerate(environments):
            for action_id in range(9):
                value = float(by_key[(env, action_id)][key])
                color = heat_color(value, minimum, maximum)
                x = x0 + action_id * cell_w
                y = top + row_index * cell_h
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=color)
    for marker in (0, 24, 49, 74, 99):
        env = environments[marker]
        row = by_key[(env, 0)]
        y = top + marker * cell_h
        draw.text((left - 10, y + 4), f"m={float(row['target_mass_kg']):.3g} mu={float(row['target_table_friction_mu']):.3g}", font=font(10), fill="#475569", anchor="rs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    dataset_base = Path(args.dataset_base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = output_dir / "tracker_validation_cases"
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    tasks = []
    for row in rows:
        env = int(row["environment_group_id"])
        action_id = int(row["action_id"])
        save_validation = env in VALIDATION_ENVS and action_id in VALIDATION_ACTIONS
        tasks.append(
            {
                "dataset_base_path": str(dataset_base),
                "video_path": row["video"][0],
                "episode_index": int(row["episode_index"]),
                "environment_group_id": env,
                "action_id": action_id,
                "action_amplitude": float(row["action_amplitude"]),
                "target_mass_kg": float(row["target_mass_kg"]),
                "target_table_friction_mu": float(row["target_table_friction_mu"]),
                "save_validation": save_validation,
                "validation_path": str(validation_dir / f"env{env:03d}_action{action_id}.png"),
            }
        )

    episode_results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_episode, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                episode_results.append(future.result())
            except Exception as error:
                errors.append({"episode_index": task["episode_index"], "error": repr(error)})
            if completed % 100 == 0:
                print(f"[progress] {completed}/{len(tasks)} errors={len(errors)}", flush=True)
    episode_results.sort(key=lambda row: (row["environment_group_id"], row["action_id"]))

    episode_csv = output_dir / "episode_displacement_metrics.csv"
    with episode_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_results[0]))
        writer.writeheader()
        writer.writerows(episode_results)

    grouped = {}
    for row in episode_results:
        grouped.setdefault(int(row["environment_group_id"]), []).append(row)
    group_rows = []
    for env, values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["action_id"]))
        if len(values) != 9:
            continue
        red_raw = np.asarray([float(row["red_normalized_displacement"]) for row in values])
        blue_raw = np.asarray([float(row["blue_normalized_displacement"]) for row in values])
        if not np.all(np.isfinite(red_raw)) or not np.all(np.isfinite(blue_raw)):
            continue
        red = isotonic_non_decreasing(red_raw)
        blue = isotonic_non_decreasing(blue_raw)
        vectors = np.stack([red, blue], axis=1)
        interval_change = np.linalg.norm(np.diff(vectors, axis=0), axis=1) / math.sqrt(2.0)
        delta_action = np.diff(ACTION_AMPLITUDES)
        interval_score = interval_change / np.power(delta_action, args.interval_normalization_power)
        point_score = np.empty(9, dtype=np.float64)
        point_score[0], point_score[-1] = interval_score[0], interval_score[-1]
        point_score[1:-1] = 0.5 * (interval_score[:-1] + interval_score[1:])
        point_score = np.convolve(np.pad(point_score, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        probabilities = {
            floor: bounded_probability(point_score, floor, args.max_uniform_multiple, args.contrast_gamma)
            for floor in (0.4, 0.6, 0.8)
        }
        primary = bounded_probability(
            point_score,
            args.primary_floor_fraction,
            args.max_uniform_multiple,
            args.contrast_gamma,
        )
        for index, row in enumerate(values):
            group_rows.append(
                {
                    "environment_group_id": env,
                    "target_mass_kg": row["target_mass_kg"],
                    "target_table_friction_mu": row["target_table_friction_mu"],
                    "action_id": index,
                    "action_amplitude": ACTION_AMPLITUDES[index],
                    "red_displacement_raw": red_raw[index],
                    "blue_displacement_raw": blue_raw[index],
                    "red_displacement_isotonic": red[index],
                    "blue_displacement_isotonic": blue[index],
                    "red_offscreen": row["red_offscreen"],
                    "blue_offscreen": row["blue_offscreen"],
                    "local_response_score": point_score[index],
                    "weight_floor40": probabilities[0.4][index],
                    "weight_floor60": probabilities[0.6][index],
                    "weight_floor80": probabilities[0.8][index],
                    "primary_probability": primary[index],
                    "primary_relative_to_uniform": primary[index] * 9.0,
                }
            )

    group_csv = output_dir / "environment_action_sampling_weights.csv"
    with group_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    by_action = {}
    for action_id in range(9):
        subset = [row for row in group_rows if int(row["action_id"]) == action_id]
        by_action[action_id] = {
            "action_amplitude": float(ACTION_AMPLITUDES[action_id]),
            "mean_red_displacement": float(np.mean([float(row["red_displacement_raw"]) for row in subset])),
            "mean_blue_displacement": float(np.mean([float(row["blue_displacement_raw"]) for row in subset])),
            "red_offscreen_rate": float(np.mean([bool(row["red_offscreen"]) for row in subset])),
            "blue_offscreen_rate": float(np.mean([bool(row["blue_offscreen"]) for row in subset])),
            "mean_primary_probability": float(np.mean([float(row["primary_probability"]) for row in subset])),
            "mean_relative_to_uniform": float(np.mean([float(row["primary_relative_to_uniform"]) for row in subset])),
        }
    peak_counts = {action_id: 0 for action_id in range(9)}
    for env in sorted(grouped):
        subset = [row for row in group_rows if int(row["environment_group_id"]) == env]
        if subset:
            peak = max(subset, key=lambda row: float(row["local_response_score"]))
            peak_counts[int(peak["action_id"])] += 1
    valid_red = sum(bool(row["red_valid"]) for row in episode_results)
    valid_blue = sum(bool(row["blue_valid"]) for row in episode_results)
    summary = {
        "episodes_requested": len(tasks),
        "episodes_analyzed": len(episode_results),
        "errors": errors,
        "valid_red_tracks": valid_red,
        "valid_blue_tracks": valid_blue,
        "environments_with_complete_curves": len({int(row["environment_group_id"]) for row in group_rows}),
        "parameters": {
            "interval_normalization_power": args.interval_normalization_power,
            "contrast_gamma": args.contrast_gamma,
            "primary_floor_fraction": args.primary_floor_fraction,
            "max_uniform_multiple": args.max_uniform_multiple,
            "primary_min_probability": args.primary_floor_fraction / 9.0,
            "primary_max_probability": args.max_uniform_multiple / 9.0,
        },
        "by_action": by_action,
        "peak_local_response_action_counts": peak_counts,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_validation_montage(list(validation_dir.glob("*.png")), output_dir / "tracker_validation_montage.png")
    draw_heatmaps(group_rows, output_dir / "empirical_displacement_and_sampling_heatmaps.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
