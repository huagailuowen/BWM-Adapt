#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import build_infer_dataset, build_pipeline  # noqa: E402
from infer_stage2_ttt import _prepare_loss_inputs, _shared_inputs  # noqa: E402
from wan_video_action.parsers import add_general_config, merge_yaml_and_args  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        "Measure spatial flow-matching error and prediction-gradient concentration."
    )
    parser = add_general_config(parser)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--context_table_path", type=str, required=True)
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--samples_per_environment", type=int, default=2)
    parser.add_argument("--timestep_indices", type=str, default="100,500,900")
    parser.add_argument("--analysis_seed", type=int, default=20260819)
    parser.add_argument("--roi_source_width", type=int, default=640)
    parser.add_argument("--roi_source_height", type=int, default=480)
    parser.add_argument(
        "--roi_source_polygon",
        type=str,
        default="0,185;640,135;640,315;0,365",
        help="Semicolon-separated x,y vertices in the uncropped source frame.",
    )
    parser.add_argument(
        "--roi_weight",
        type=float,
        default=4.0,
        help="Candidate ROI/background loss-weight ratio reported by the audit.",
    )
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if not args.context_table_path:
        raise ValueError("--context_table_path is required.")
    return args


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    result = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(token))
    return result


def _select_balanced_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("samples_per_environment must be positive.")
    by_environment: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if str(row.get("sampling_kind", "")) != "impact":
            continue
        environment = int(row.get("environment_index", row.get("friction_mu", 0)))
        level = int(row.get("skill_gear", row.get("action_id", 0)))
        by_environment[environment][level].append(index)

    rng = np.random.default_rng(seed)
    selected = []
    for environment in sorted(by_environment):
        levels = sorted(by_environment[environment])
        if len(levels) < count:
            raise ValueError(
                f"Environment {environment} has only {len(levels)} impact skill levels; "
                f"requested {count}."
            )
        positions = np.linspace(0, len(levels) - 1, count + 2)[1:-1]
        chosen_positions = []
        for position in positions:
            candidate = int(round(float(position)))
            while candidate in chosen_positions:
                candidate = min(len(levels) - 1, candidate + 1)
            chosen_positions.append(candidate)
        for position in chosen_positions:
            candidates = by_environment[environment][levels[position]]
            selected.append(int(rng.choice(candidates)))
    return selected


def _load_context_table(path: str | Path) -> dict[float, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    table = {
        round(float(record["friction_mu"]), 8): np.asarray(record["context"], dtype=np.float32)
        for record in payload.get("records", [])
    }
    if not table:
        raise ValueError(f"No context records in {path}.")
    return table


def _lookup_context(table: dict[float, np.ndarray], row: dict) -> np.ndarray:
    value = round(float(row.get("friction_mu", row.get("environment_index", 0))), 8)
    if value in table:
        return table[value]
    nearest = min(table, key=lambda candidate: abs(candidate - value))
    if abs(nearest - value) > 1e-6:
        raise KeyError(f"No context-table entry for environment value {value}; nearest={nearest}.")
    return table[nearest]


def _parse_polygon(text: str) -> list[tuple[float, float]]:
    points = []
    for vertex in text.split(";"):
        x_text, y_text = vertex.split(",", 1)
        points.append((float(x_text), float(y_text)))
    if len(points) < 3:
        raise ValueError("ROI polygon needs at least three vertices.")
    return points


def _letterbox_polygon(
    polygon: list[tuple[float, float]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[tuple[float, float]]:
    scale = min(target_width / source_width, target_height / source_height)
    pad_x = (target_width - source_width * scale) / 2.0
    pad_y = (target_height - source_height * scale) / 2.0
    return [(x * scale + pad_x, y * scale + pad_y) for x, y in polygon]


def _polygon_mask(
    polygon: list[tuple[float, float]], width: int, height: int
) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def _video_reference(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 5:
        raise ValueError(f"Expected video (V,C,T,H,W), got {tuple(video.shape)}.")
    frame = video[0, :, video.shape[2] // 2].detach().float().cpu()
    frame = frame.permute(1, 2, 0).numpy()
    if float(frame.min()) < -0.05:
        frame = (frame + 1.0) / 2.0
    return np.clip(frame, 0.0, 1.0)


def _resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(values).float()[None, None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def _region_metrics(values: np.ndarray, roi_mask: np.ndarray, roi_weight: float) -> dict:
    epsilon = 1e-12
    roi_mass = float(np.sum(values * roi_mask))
    background_mass = float(np.sum(values * (1.0 - roi_mask)))
    roi_area = float(np.sum(roi_mask))
    background_area = float(np.sum(1.0 - roi_mask))
    roi_mean = roi_mass / max(roi_area, epsilon)
    background_mean = background_mass / max(background_area, epsilon)
    return {
        "roi_fraction_of_total": roi_mass / max(roi_mass + background_mass, epsilon),
        "roi_mean": roi_mean,
        "background_mean": background_mean,
        "roi_to_background_mean_ratio": roi_mean / max(background_mean, epsilon),
        "candidate_weighted_roi_fraction": (
            roi_weight * roi_mass
            / max(roi_weight * roi_mass + background_mass, epsilon)
        ),
    }


def _heatmap_rgb(values: np.ndarray, reference_mean: float | None = None):
    denominator = max(
        float(values.mean()) if reference_mean is None else float(reference_mean),
        1e-12,
    )
    relative = values / denominator
    vmax = max(1.0, float(np.percentile(relative, 99.0)))
    scaled = np.clip(relative / vmax, 0.0, 1.0)
    anchors = np.asarray([0.0, 0.2, 0.45, 0.7, 0.9, 1.0])
    colors = np.asarray(
        [
            [0, 0, 4],
            [45, 17, 95],
            [132, 32, 107],
            [211, 59, 72],
            [249, 142, 9],
            [252, 253, 191],
        ],
        dtype=np.float32,
    )
    channels = [np.interp(scaled, anchors, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8), vmax


def _panel(image: Image.Image, label: str, polygon=None) -> Image.Image:
    image = image.convert("RGB")
    if polygon is not None:
        draw = ImageDraw.Draw(image)
        points = [(int(round(x)), int(round(y))) for x, y in polygon]
        draw.line(points + [points[0]], fill=(0, 229, 255), width=2)
    panel = Image.new("RGB", (image.width, image.height + 28), (0, 0, 0))
    panel.paste(image, (0, 28))
    ImageDraw.Draw(panel).text((7, 8), label, fill=(255, 255, 255))
    return panel


def _plot_heatmap(
    output_dir: Path,
    stem: str,
    title: str,
    reference: np.ndarray,
    values: np.ndarray,
    polygon: list[tuple[float, float]],
) -> None:
    reference_uint8 = (np.clip(reference, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap, vmax = _heatmap_rgb(values)
    reference_image = Image.fromarray(reference_uint8)
    heatmap_image = Image.fromarray(heatmap)
    overlay_image = Image.blend(reference_image, heatmap_image, alpha=0.62)
    panels = [
        _panel(reference_image, "Reference frame", polygon),
        _panel(heatmap_image, f"Relative heatmap (max={vmax:.2f}x)", polygon),
        _panel(overlay_image, "Heatmap overlay + fixed ROI", polygon),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        (0, 0, 0),
    )
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    canvas.save(output_dir / f"{stem}.png")


def _plot_group_maps(
    output_dir: Path,
    stem: str,
    title: str,
    maps: dict[int, np.ndarray],
    target_height: int,
    target_width: int,
    polygon: list[tuple[float, float]],
) -> None:
    groups = sorted(maps)
    cols = 4
    rows = math.ceil(len(groups) / cols)
    resized = {
        group: _resize_map(values, target_height, target_width)
        for group, values in maps.items()
    }
    all_values = np.concatenate([value.reshape(-1) for value in resized.values()])
    global_mean = max(float(all_values.mean()), 1e-12)
    panels = []
    for group in groups:
        heatmap, vmax = _heatmap_rgb(resized[group], reference_mean=global_mean)
        label = f"Group {group} (shared max={vmax:.2f}x)"
        panels.append(_panel(Image.fromarray(heatmap), label, polygon))
    panel_width = target_width
    panel_height = target_height + 28
    canvas = Image.new(
        "RGB",
        (cols * panel_width, rows * panel_height),
        (0, 0, 0),
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % cols) * panel_width, (index // cols) * panel_height))
    canvas.save(output_dir / f"{stem}.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = _read_jsonl(args.dataset_metadata_path)
    selected_indices = _parse_indices(args.sample_indices)
    if selected_indices is None:
        selected_indices = _select_balanced_indices(
            metadata_rows,
            int(args.samples_per_environment),
            int(args.analysis_seed),
        )
    timestep_indices = [
        int(value.strip()) for value in str(args.timestep_indices).split(",") if value.strip()
    ]
    if not timestep_indices:
        raise ValueError("At least one timestep index is required.")

    source_polygon = _parse_polygon(args.roi_source_polygon)
    model_polygon = _letterbox_polygon(
        source_polygon,
        int(args.roi_source_width),
        int(args.roi_source_height),
        int(args.width),
        int(args.height),
    )
    pixel_roi_mask = _polygon_mask(model_polygon, int(args.width), int(args.height))
    context_table = _load_context_table(args.context_table_path)

    selection_path = output_dir / "selected_samples.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for index in selected_indices:
            row = dict(metadata_rows[index])
            row["sample_index"] = int(index)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        f"[analysis] samples={len(selected_indices)} timesteps={timestep_indices} "
        f"roi_source={source_polygon} roi_model={model_polygon}",
        flush=True,
    )
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    pipe.scheduler.set_timesteps(1000, training=True)
    for model_name in pipe.in_iteration_models:
        model = getattr(pipe, model_name)
        if model is None:
            continue
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    aggregate_error = None
    aggregate_gradient = None
    by_environment_error: dict[int, np.ndarray] = {}
    by_environment_gradient: dict[int, np.ndarray] = {}
    by_environment_count: dict[int, int] = defaultdict(int)
    by_timestep_error: dict[int, np.ndarray] = {}
    by_timestep_gradient: dict[int, np.ndarray] = {}
    by_timestep_count: dict[int, int] = defaultdict(int)
    reference = None
    sample_metrics = []
    map_count = 0
    latent_shape = None

    for sample_order, sample_index in enumerate(selected_indices):
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample_index={sample_index} outside dataset length {len(dataset)}.")
        row = metadata_rows[sample_index]
        environment = int(row.get("environment_index", row.get("friction_mu", 0)))
        data = dataset[sample_index]
        if reference is None:
            reference = _video_reference(data["video"])
        context = torch.tensor(
            _lookup_context(context_table, row),
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        with torch.no_grad():
            prepared = _prepare_loss_inputs(pipe, data, context, args)
        base_inputs = dict(_shared_inputs(prepared))

        for timestep_index in timestep_indices:
            if timestep_index < 0 or timestep_index >= len(pipe.scheduler.timesteps):
                raise IndexError(
                    f"timestep_index={timestep_index} outside [0,{len(pipe.scheduler.timesteps)})."
                )
            seed = int(args.analysis_seed) + sample_order * 1009 + timestep_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            timestep = pipe.scheduler.timesteps[
                torch.tensor([timestep_index], dtype=torch.long)
            ].to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs = dict(base_inputs)
            with torch.no_grad():
                noise = torch.randn_like(inputs["input_latents"])
                inputs["latents"] = pipe.scheduler.add_noise(
                    inputs["input_latents"], noise, timestep
                )
                target = pipe.scheduler.training_target(
                    inputs["input_latents"], noise, timestep
                )
                if "first_frame_latents" in inputs:
                    inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
                if "first_frame_latents" in inputs:
                    prediction = prediction[:, :, 1:]
                    target = target[:, :, 1:]
                training_weight = pipe.scheduler.training_weight(timestep).float().mean()
                squared_error = (prediction.float() - target.float()).square() * training_weight
                error_map = squared_error.mean(dim=(0, 1, 2)).cpu().numpy()

            prediction_probe = prediction.detach().float().requires_grad_(True)
            probe_loss = F.mse_loss(prediction_probe, target.detach().float()) * training_weight
            prediction_gradient = torch.autograd.grad(probe_loss, prediction_probe)[0]
            gradient_map = prediction_gradient.abs().mean(dim=(0, 1, 2)).cpu().numpy()

            if latent_shape is None:
                latent_shape = list(prediction.shape)
            if aggregate_error is None:
                aggregate_error = np.zeros_like(error_map, dtype=np.float64)
                aggregate_gradient = np.zeros_like(gradient_map, dtype=np.float64)
            aggregate_error += error_map
            aggregate_gradient += gradient_map
            by_environment_error.setdefault(environment, np.zeros_like(error_map, dtype=np.float64))
            by_environment_gradient.setdefault(
                environment, np.zeros_like(gradient_map, dtype=np.float64)
            )
            by_environment_error[environment] += error_map
            by_environment_gradient[environment] += gradient_map
            by_environment_count[environment] += 1
            by_timestep_error.setdefault(timestep_index, np.zeros_like(error_map, dtype=np.float64))
            by_timestep_gradient.setdefault(
                timestep_index, np.zeros_like(gradient_map, dtype=np.float64)
            )
            by_timestep_error[timestep_index] += error_map
            by_timestep_gradient[timestep_index] += gradient_map
            by_timestep_count[timestep_index] += 1
            map_count += 1

            latent_roi = F.interpolate(
                torch.from_numpy(pixel_roi_mask)[None, None],
                size=error_map.shape,
                mode="area",
            )[0, 0].numpy()
            sample_metrics.append(
                {
                    "sample_index": int(sample_index),
                    "environment_index": environment,
                    "skill_gear": int(row.get("skill_gear", row.get("action_id", -1))),
                    "episode_index": int(row.get("episode_index", -1)),
                    "start_frame": int(row.get("start_frame", 0)),
                    "end_frame": int(row.get("end_frame", 0)),
                    "timestep_index": int(timestep_index),
                    "scheduler_timestep": float(timestep.float().item()),
                    "flow_loss": float(probe_loss.detach().cpu()),
                    "error": _region_metrics(error_map, latent_roi, float(args.roi_weight)),
                    "prediction_gradient": _region_metrics(
                        gradient_map, latent_roi, float(args.roi_weight)
                    ),
                }
            )
            print(
                f"[sample] index={sample_index} ball={environment} "
                f"level={row.get('skill_gear')} t_index={timestep_index} "
                f"loss={float(probe_loss.detach().cpu()):.6f}",
                flush=True,
            )
            del prediction_probe, prediction_gradient, prediction, target, squared_error

        del prepared, base_inputs, data, context
        torch.cuda.empty_cache()

    if map_count <= 0 or aggregate_error is None or aggregate_gradient is None or reference is None:
        raise RuntimeError("No analysis maps were produced.")

    aggregate_error /= map_count
    aggregate_gradient /= map_count
    for environment in by_environment_error:
        by_environment_error[environment] /= by_environment_count[environment]
        by_environment_gradient[environment] /= by_environment_count[environment]
    for timestep_index in by_timestep_error:
        by_timestep_error[timestep_index] /= by_timestep_count[timestep_index]
        by_timestep_gradient[timestep_index] /= by_timestep_count[timestep_index]

    latent_roi = F.interpolate(
        torch.from_numpy(pixel_roi_mask)[None, None],
        size=aggregate_error.shape,
        mode="area",
    )[0, 0].numpy()
    roi_area_fraction = float(pixel_roi_mask.mean())
    weight_normalizer = 1.0 + (float(args.roi_weight) - 1.0) * roi_area_fraction
    summary = {
        "checkpoint": str(args.ckpt_path),
        "context_table": str(args.context_table_path),
        "dataset_metadata": str(args.dataset_metadata_path),
        "sample_count": len(selected_indices),
        "forward_pass_count": map_count,
        "selected_indices": selected_indices,
        "timestep_indices": timestep_indices,
        "latent_prediction_shape": latent_shape,
        "roi_source_polygon": source_polygon,
        "roi_model_polygon": model_polygon,
        "roi_pixel_area_fraction": roi_area_fraction,
        "candidate_roi_weight": float(args.roi_weight),
        "candidate_weight_normalizer": weight_normalizer,
        "candidate_normalized_roi_weight": float(args.roi_weight) / weight_normalizer,
        "candidate_normalized_background_weight": 1.0 / weight_normalizer,
        "aggregate_error": _region_metrics(
            aggregate_error, latent_roi, float(args.roi_weight)
        ),
        "aggregate_prediction_gradient": _region_metrics(
            aggregate_gradient, latent_roi, float(args.roi_weight)
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in sample_metrics:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "spatial_maps_latent.npz",
        aggregate_error=aggregate_error,
        aggregate_prediction_gradient=aggregate_gradient,
        roi_mask=latent_roi,
        **{
            f"ball_{environment}_error": values
            for environment, values in by_environment_error.items()
        },
        **{
            f"ball_{environment}_prediction_gradient": values
            for environment, values in by_environment_gradient.items()
        },
        **{
            f"timestep_{timestep_index}_error": values
            for timestep_index, values in by_timestep_error.items()
        },
    )

    error_pixel = _resize_map(aggregate_error, int(args.height), int(args.width))
    gradient_pixel = _resize_map(aggregate_gradient, int(args.height), int(args.width))
    _plot_heatmap(
        output_dir,
        "aggregate_flow_error_heatmap",
        "Weighted flow-matching squared error",
        reference,
        error_pixel,
        model_polygon,
    )
    _plot_heatmap(
        output_dir,
        "aggregate_prediction_gradient_heatmap",
        "Absolute gradient with respect to predicted latent",
        reference,
        gradient_pixel,
        model_polygon,
    )
    _plot_group_maps(
        output_dir,
        "per_ball_flow_error_heatmaps",
        "Flow-matching error by ball",
        by_environment_error,
        int(args.height),
        int(args.width),
        model_polygon,
    )
    _plot_group_maps(
        output_dir,
        "per_timestep_flow_error_heatmaps",
        "Flow-matching error by scheduler timestep index",
        by_timestep_error,
        int(args.height),
        int(args.width),
        model_polygon,
    )

    reference_image = Image.fromarray(
        (np.clip(reference, 0.0, 1.0) * 255.0).astype(np.uint8)
    ).convert("RGB")
    tint = Image.new("RGBA", reference_image.size, (0, 0, 0, 0))
    points = [(int(round(x)), int(round(y))) for x, y in model_polygon]
    ImageDraw.Draw(tint).polygon(
        points,
        fill=(255, 122, 0, 72),
        outline=(0, 229, 255, 255),
        width=2,
    )
    roi_image = Image.alpha_composite(reference_image.convert("RGBA"), tint).convert("RGB")
    _panel(
        roi_image,
        "Fixed trough ROI in 224x224 letterboxed input",
    ).save(output_dir / "fixed_roi_definition.png")
    print(f"[done] output={output_dir} summary={summary}", flush=True)


if __name__ == "__main__":
    main()
