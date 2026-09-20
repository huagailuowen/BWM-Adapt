#!/usr/bin/env python3
"""Build a six-background, six-action GT/Ours push-box demo."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

CURRENT_DATASET = (
    PROJECT
    / "datasets/pushbox_various_env/"
    "libero_plus_push_box_event80_matched_physics_5randombackground_"
    "30friction_10action_1500eps_adaptive_end_2026-08-19_hai-machine"
)
CURRENT_METADATA = (
    REPO
    / "data/push_box_bwm_matchedphysics5bg30fric10action_"
    "65_105_shared_friction30_20260819/train.jsonl"
)
CURRENT_INFER = (
    REPO
    / "outputs/infer_push_box_matchedphysics5bg30fric_roi10x_"
    "misresume104188_step4400_gt_stage1_stage2_x5fp32_10cases_grid_pca_105521"
)
CURRENT_TRANSFER = (
    REPO
    / "results/pushbox_multibackground/"
    "misresume_step4400_id5_ood5_k1_oracle_informative_support25_60_v1/"
    "methods/ours/step_4400/seed_20260825/transfer/source0134/raw"
)

EXTRA_DATASET = (
    PROJECT
    / "datasets/pushbox_various_env/"
    "libero_plus_push_box_event_tap_segmented40_10action_3env_hidden_lerobot_"
    "A500_offset160_stop_2026-07-27_hai-machine"
)
EXTRA_METADATA = REPO / "data/push_box_bwm_various_env3x40_10action_65_105_20260727/train.jsonl"
EXTRA_INFER = (
    REPO
    / "outputs/infer_push_box_various_env3x40_fixed21_6x6_step8200_"
    "gt_stage1_stage2_x5fp32_12each_grid_pca_98506"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT
            / "tmp/pushbox_multibackground_6background_"
            "action1_action2_action3_action4_action5_action8_demo"
        ),
    )
    return parser.parse_args()


def metadata_record(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def run(*args: str) -> None:
    subprocess.run(args, check=True)


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


def main() -> None:
    output = parse_args().output
    output.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "background1_action1",
            "background": 1,
            "action": 1,
            "sample": 431,
            "metadata": CURRENT_METADATA,
            "dataset": CURRENT_DATASET,
            "prediction": CURRENT_TRANSFER / "sample0431_episode000431_frames0065-0105.mp4",
        },
        {
            "name": "background2_action2",
            "background": 2,
            "action": 2,
            "sample": 732,
            "metadata": CURRENT_METADATA,
            "dataset": CURRENT_DATASET,
            "prediction": CURRENT_TRANSFER / "sample0732_episode000732_frames0065-0105.mp4",
        },
        {
            "name": "background3_action3",
            "background": 3,
            "action": 3,
            "sample": 1033,
            "metadata": CURRENT_METADATA,
            "dataset": CURRENT_DATASET,
            "prediction": CURRENT_TRANSFER / "sample1033_episode001033_frames0065-0105.mp4",
        },
        {
            "name": "background0_action4",
            "background": 0,
            "action": 4,
            "sample": 4,
            "metadata": CURRENT_METADATA,
            "dataset": CURRENT_DATASET,
            "prediction": CURRENT_INFER / "stage2_raw/sample0004_episode000004_frames0065-0105.mp4",
        },
        {
            "name": "background4_action5",
            "background": 4,
            "action": 5,
            "sample": 1335,
            "metadata": CURRENT_METADATA,
            "dataset": CURRENT_DATASET,
            "prediction": CURRENT_TRANSFER / "sample1335_episode001335_frames0065-0105.mp4",
        },
        {
            "name": "background5_action8",
            "background": 5,
            "action": 8,
            "sample": 548,
            "metadata": EXTRA_METADATA,
            "dataset": EXTRA_DATASET,
            "prediction": EXTRA_INFER / "env1/stage2_raw/sample0548_episode000548_frames0065-0105.mp4",
        },
    ]

    manifest = []
    videos = []
    final_frames = []
    for case in cases:
        record = metadata_record(case["metadata"], case["sample"])
        case_dir = output / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        gt_source = case["dataset"] / record["video"][0]
        gt_video = case_dir / "GT.mp4"
        prediction_video = case_dir / "Ours_prediction.mp4"
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
        shutil.copy2(case["prediction"], prediction_video)
        gt_final = case_dir / "GT_final.png"
        prediction_final = case_dir / "Ours_prediction_final.png"
        extract_frame(gt_source, 105, gt_final)
        extract_frame(case["prediction"], 40, prediction_final)
        videos.append((gt_video, prediction_video))
        final_frames.append((gt_final, prediction_final))
        manifest.append(
            {
                "name": case["name"],
                "background": case["background"],
                "action_id": case["action"],
                "sample_index": case["sample"],
                "physical_friction_mu": record.get("physical_friction_mu"),
            }
        )

    command = [FFMPEG, "-loglevel", "error", "-y"]
    for gt_video, prediction_video in videos:
        command.extend(["-i", str(gt_video), "-i", str(prediction_video)])
    filters = []
    for index in range(6):
        filters.append(f"[{2 * index}:v]fps=16,scale=256:256[g{index}]")
        filters.append(f"[{2 * index + 1}:v]fps=16,scale=256:256[p{index}]")
    stack_order = "[g0][g1][g2][p0][p1][p2][g3][g4][g5][p3][p4][p5]"
    layout = "|".join(
        [
            "0_0",
            "256_0",
            "512_0",
            "0_256",
            "256_256",
            "512_256",
            "0_512",
            "256_512",
            "512_512",
            "0_768",
            "256_768",
            "512_768",
        ]
    )
    filters.append(f"{stack_order}xstack=inputs=12:layout={layout}:fill=white[v]")
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
            str(output / "six_backgrounds_six_actions_gt_vs_ours.mp4"),
        ]
    )
    run(*command)

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.0), facecolor="white")
    for slot, case in enumerate(cases):
        block = slot // 3
        column = slot % 3
        gt_row = block * 2
        pred_row = gt_row + 1
        axes[gt_row, column].imshow(plt.imread(final_frames[slot][0]))
        axes[pred_row, column].imshow(plt.imread(final_frames[slot][1]))
        axes[gt_row, column].set_title(
            f"Background {case['background']} | Action {case['action']}", fontsize=11
        )
    for row in (0, 2):
        axes[row, 0].set_ylabel("Ground truth", fontsize=11)
        axes[row + 1, 0].set_ylabel("Ours", fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Six backgrounds and six push levels", fontsize=16, weight="semibold")
    fig.tight_layout()
    fig.savefig(output / "six_backgrounds_six_actions_final_frames.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "six_backgrounds_six_actions_final_frames.svg", bbox_inches="tight")
    plt.close(fig)
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
