#!/usr/bin/env python3
"""Reassemble existing comparison videos only; no model/data inference."""
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video assembly must run on a CPU compute node")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    runs = (
        "infer_real97_ball_ours_reference_job114762",
        "infer_real97_door_ours_reference_job114764",
    )
    for name in runs:
        output = ROOT / "outputs" / name
        completion = json.loads((output / "inference_complete.json").read_text())
        fps = float(completion["frames_per_second"])
        grids = output / "grids"
        lock = (grids / ".level_regrid.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        grouped = defaultdict(lambda: defaultdict(list))
        for path in (output / "completed").glob("*.json"):
            record = json.loads(path.read_text())
            row = record["row"]
            if not record["paired_gt"] or row["dataset_split"] not in ("train", "test"):
                raise RuntimeError(f"Unexpected non-factual query: {path}")
            video = output / "comparisons" / (path.stem + ".mp4")
            if not video.is_file():
                raise FileNotFoundError(video)
            grouped[row["environment"]][row["dataset_split"]].append({
                "level": int(row["action_level"]),
                "episode": int(row["episode_index"]),
                "query_index": int(record["index"]),
                "source_comparison": str(video),
            })
        if sum(len(rows) for splits in grouped.values() for rows in splits.values()) != int(completion["queries"]):
            raise RuntimeError("Completed query records are incomplete")
        orders = {}
        for environment, splits in sorted(grouped.items()):
            orders[environment] = {}
            for split in ("test", "train"):
                rows = sorted(splits[split], key=lambda row: (row["level"], row["episode"], row["query_index"]))
                if not rows:
                    raise RuntimeError(f"No {split} columns for {environment}")
                directory = grids if split == "test" else grids / "train"
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{environment}_gt_stage1_stage2_{split}_levels.mp4"
                temporary = target.with_name(f".{target.stem}.{os.environ['SLURM_JOB_ID']}.partial.mp4")
                command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
                for row in rows:
                    command.extend(["-threads", "1", "-i", row["source_comparison"]])
                inputs = "".join(f"[{index}:v]" for index in range(len(rows)))
                graph = inputs + (f"hstack=inputs={len(rows)}:shortest=1[v]" if len(rows) > 1 else "null[v]")
                command.extend([
                    "-filter_complex_threads", "1", "-filter_complex", graph,
                    "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-threads", "4", "-r", str(fps),
                    "-movflags", "+faststart", str(temporary),
                ])
                subprocess.run(command, check=True)
                os.replace(temporary, target)
                orders[environment][split] = {"grid": str(target), "columns_left_to_right": rows}
                print(f"[grid] {target} levels={[row['level'] for row in rows]}", flush=True)
            old = grids / f"{environment}_gt_stage1_stage2_train_test.mp4"
            if old.exists():
                archive = grids / "previous_mixed"
                archive.mkdir(exist_ok=True)
                destination = archive / old.name
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite grid backup: {destination}")
                os.replace(old, destination)
        manifest = grids / "level_order.json"
        temporary = manifest.with_name(".level_order.json.partial")
        temporary.write_text(json.dumps({
            "source_run": name, "operation": "reassemble_existing_comparisons_only",
            "rows_top_to_bottom": ["GT", "Stage1", "Stage2"],
            "columns": "ascending_numeric_action_level",
            "test_train_separate": True, "inference_rerun": False,
            "original_raw_predictions_modified": False, "environments": orders,
        }, indent=2) + "\n")
        os.replace(temporary, manifest)
        print(f"[regrid_complete] {name} environments={len(grouped)}", flush=True)
        lock.close()


if __name__ == "__main__":
    main()
