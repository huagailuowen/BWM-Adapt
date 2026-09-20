#!/usr/bin/env python3
"""Export the multibackground low/mid/high demo and active-latent PCA."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio_ffmpeg


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parents[1]
TMP = PROJECT / "tmp"
BENCHMARK = (
    REPO
    / "results/pushbox_multibackground/"
    "misresume_step4400_id5_ood5_k1_oracle_informative_support25_60_v1"
)
DATASET = (
    PROJECT
    / "datasets/pushbox_various_env/"
    "libero_plus_push_box_event80_matched_physics_5randombackground_"
    "30friction_10action_1500eps_adaptive_end_2026-08-19_hai-machine"
)
METADATA = (
    REPO
    / "data/push_box_bwm_matchedphysics5bg30fric10action_"
    "65_105_shared_friction30_20260819/train.jsonl"
)
CONTEXT_TABLE = (
    REPO
    / "outputs/push_box_matchedphysics5bg30fric_shared_c32_random_"
    "roi10x_agent_resume4000_stage1_8700_104188/step-4400.context_table.json"
)
PROTOCOL = BENCHMARK / "protocol/support_query_manifest.json"
TRAJECTORY = BENCHMARK / "context_trajectory.jsonl"
PREDICTIONS = BENCHMARK / "methods/ours/step_4400/seed_20260825/transfer/source0134/raw"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-output",
        type=Path,
        default=TMP / "pushbox_multibackground_grid0134_level3_level6_level10_frames",
    )
    parser.add_argument(
        "--pca-output",
        type=Path,
        default=TMP / "pushbox_multibackground_5bg_PCA",
    )
    parser.add_argument("--skip-demo", action="store_true")
    return parser.parse_args()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def load_metadata() -> dict[int, dict]:
    records = {}
    with METADATA.open() as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            records[index] = record
    return records


def flatten_context(record: dict) -> np.ndarray:
    value = np.asarray(record["context"], dtype=np.float64)
    return value.reshape(-1)


def fit_and_export_pca(output: Path, metadata: dict[int, dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    table = json.loads(CONTEXT_TABLE.read_text())
    active_ids = [int(value) for value in protocol["active_context_group_ids"]]

    physical_mu = {}
    for record in metadata.values():
        group = int(record["context_group_id"])
        physical_mu.setdefault(group, float(record["physical_friction_mu"]))

    table_by_group = {
        int(round(float(record["friction_mu"]))): flatten_context(record)
        for record in table["records"]
    }
    train = np.stack([table_by_group[group] for group in active_ids])
    mean = train.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(train - mean, full_matrices=False)
    components = vh[:2]
    train_pc = (train - mean) @ components.T
    explained = singular_values**2
    explained_ratio = explained / explained.sum()

    wanted_groups = {
        int(env["context_group_id"]): env for env in protocol["environments"]
    }
    final_by_group: dict[int, dict] = {}
    with TRAJECTORY.open() as handle:
        for line in handle:
            row = json.loads(line)
            group = int(row["group_key"][0])
            if group not in wanted_groups:
                continue
            previous = final_by_group.get(group)
            if previous is None or int(row["inner_step"]) >= int(previous["inner_step"]):
                final_by_group[group] = row

    test_rows = []
    for group, env in wanted_groups.items():
        row = final_by_group[group]
        support_index = int(env["support_indices"][0])
        latent = np.asarray(row["context_flat"], dtype=np.float64)
        pc = (latent - mean) @ components.T
        test_rows.append(
            {
                "domain": env["domain"],
                "context_group_id": int(env["context_group_id"]),
                "physical_friction_mu": float(env["friction_mu"]),
                "support_index": support_index,
                "inner_step": int(row["inner_step"]),
                "pc": pc,
                "latent": latent,
            }
        )

    latent_columns = [f"latent_{index:02d}" for index in range(1, train.shape[1] + 1)]
    with (output / "training_time.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["context_group_id", "physical_friction_mu", "pc1", "pc2", *latent_columns])
        for group, pc, latent in zip(active_ids, train_pc, train):
            writer.writerow([group, physical_mu[group], pc[0], pc[1], *latent])

    with (output / "test_time.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "domain",
                "context_group_id",
                "physical_friction_mu",
                "support_index",
                "inner_step",
                "pc1",
                "pc2",
                *latent_columns,
            ]
        )
        for row in test_rows:
            writer.writerow(
                [
                    row["domain"],
                    row["context_group_id"],
                    row["physical_friction_mu"],
                    row["support_index"],
                    row["inner_step"],
                    row["pc"][0],
                    row["pc"][1],
                    *row["latent"],
                ]
            )

    pca_info = {
        "fit_scope": "active training-time context codes only",
        "active_context_group_ids": active_ids,
        "mean32": mean.tolist(),
        "pc1": components[0].tolist(),
        "pc2": components[1].tolist(),
        "singular_values": singular_values.tolist(),
        "explained_variance_ratio": explained_ratio[:2].tolist(),
        "inference_projection": "final 40-step TTT codes; excluded from PCA fit",
    }
    (output / "pca_info.json").write_text(json.dumps(pca_info, indent=2) + "\n")

    all_mu = [physical_mu[group] for group in active_ids] + [
        row["physical_friction_mu"] for row in test_rows
    ]
    norm = plt.Normalize(min(all_mu), max(all_mu))
    cmap = plt.get_cmap("coolwarm")
    order = np.argsort([physical_mu[group] for group in active_ids])

    fig, ax = plt.subplots(figsize=(9.2, 7.2), facecolor="white")
    ax.plot(
        train_pc[order, 0],
        train_pc[order, 1],
        color="#777777",
        linewidth=1.1,
        linestyle="--",
        alpha=0.65,
        zorder=1,
    )
    scatter = ax.scatter(
        train_pc[:, 0],
        train_pc[:, 1],
        c=[physical_mu[group] for group in active_ids],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=82,
        edgecolors="white",
        linewidths=0.9,
        label="Training-time Z",
        zorder=3,
    )
    for domain, marker_size in (("id", 112), ("ood", 130)):
        selected = [row for row in test_rows if row["domain"] == domain]
        ax.scatter(
            [row["pc"][0] for row in selected],
            [row["pc"][1] for row in selected],
            c=[row["physical_friction_mu"] for row in selected],
            cmap=cmap,
            norm=norm,
            marker="^",
            s=marker_size,
            edgecolors="black",
            linewidths=1.15,
            label=f"Inference-time Z ({domain.upper()})",
            zorder=4,
        )
    for group, pc in zip(active_ids, train_pc):
        ax.annotate(str(group), pc, xytext=(4, 4), textcoords="offset points", fontsize=7.5)

    ax.set_title("5-background push-box latent space", fontsize=15, weight="semibold")
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% variance)")
    ax.grid(True, linewidth=0.6, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Physical friction coefficient")
    fig.tight_layout()
    fig.savefig(output / "active_training_and_inference_latent_pca.svg", bbox_inches="tight")
    fig.savefig(output / "active_training_and_inference_latent_pca.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def extract_frame(video: Path, frame: int, output: Path) -> None:
    run(
        FFMPEG,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-vsync",
        "0",
        str(output),
    )


def export_demo(output: Path, metadata: dict[int, dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("level3_action1_low", 431, 1, 3, 0.11),
        ("level6_action5_middle", 1335, 5, 6, 0.26),
        ("level10_action9_high", 1339, 9, 10, 0.50),
    ]
    manifest = {
        "source_support_index": 134,
        "source_physical_friction_mu": 0.06333333333333334,
        "middle_selection": "action5 chosen over action6/action7: GT ends near image middle and prediction remains close to GT",
        "cases": [],
    }
    case_videos = []
    final_images = []
    for name, sample, action, grid_level, amplitude in cases:
        record = metadata[sample]
        case_dir = output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        gt_source = DATASET / record["video"][0]
        prediction = PREDICTIONS / f"sample{sample:04d}_episode{sample:06d}_frames0065-0105.mp4"
        gt_video = case_dir / "GT.mp4"
        pred_video = case_dir / "Ours_prediction.mp4"
        run(
            FFMPEG,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(gt_source),
            "-vf",
            "select=between(n\\,65\\,105),setpts=N/FRAME_RATE/TB",
            "-an",
            str(gt_video),
        )
        shutil.copy2(prediction, pred_video)
        extract_frame(gt_source, 65, case_dir / "GT_input.png")
        for index, (gt_frame, pred_frame) in enumerate(
            zip((70, 75, 80, 85, 105), (5, 10, 15, 20, 40)), start=1
        ):
            extract_frame(gt_source, gt_frame, case_dir / f"GT_{index}.png")
            extract_frame(prediction, pred_frame, case_dir / f"Ours_prediction_{index}.png")
        case_videos.append((name, gt_video, pred_video))
        final_images.append((name, case_dir / "GT_5.png", case_dir / "Ours_prediction_5.png"))
        manifest["cases"].append(
            {
                "name": name,
                "grid_level": grid_level,
                "sample_index": sample,
                "action_id": action,
                "action_amplitude": amplitude,
                "physical_friction_mu": record["physical_friction_mu"],
                "video": record["video"][0],
            }
        )

    command = [FFMPEG, "-loglevel", "error", "-y"]
    for _, gt_video, pred_video in case_videos:
        command.extend(["-i", str(gt_video), "-i", str(pred_video)])
    filters = []
    labels = []
    for index in range(3):
        gt_input = index * 2
        pred_input = gt_input + 1
        filters.append(f"[{gt_input}:v]fps=16,scale=256:256[gt{index}]")
        filters.append(f"[{pred_input}:v]fps=16,scale=256:256[pred{index}]")
        labels.extend([f"[gt{index}]", f"[pred{index}]"])
    filters.append(
        "[gt0][gt1][gt2][pred0][pred1][pred2]"
        "xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w3_h0|w3+w4_h0:fill=white[v]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output / "gt_vs_ours_low_middle_high_2x3.mp4"),
        ]
    )
    run(*command)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.0), facecolor="white")
    for column, (name, gt_image, pred_image) in enumerate(final_images):
        axes[0, column].imshow(plt.imread(gt_image))
        axes[1, column].imshow(plt.imread(pred_image))
        axes[0, column].set_title(name.replace("_", " "), fontsize=10.5)
    axes[0, 0].set_ylabel("Ground truth", fontsize=11)
    axes[1, 0].set_ylabel("Ours", fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Low / middle / high push outcomes", fontsize=15, weight="semibold")
    fig.tight_layout()
    fig.savefig(output / "final_frame_comparison_2x3.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "final_frame_comparison_2x3.svg", bbox_inches="tight")
    plt.close(fig)
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    metadata = load_metadata()
    if not args.skip_demo:
        export_demo(args.demo_output, metadata)
    fit_and_export_pca(args.pca_output, metadata)
    print(json.dumps({"demo_output": str(args.demo_output), "pca_output": str(args.pca_output)}))


if __name__ == "__main__":
    main()
