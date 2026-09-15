#!/usr/bin/env python3
"""Opt-in dual-latent presentation around the unchanged unbounded evaluator."""

import argparse
import colorsys
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["door", "ball"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference and video processing require a compute allocation")
    config = json.loads(cli.config.read_text())
    prepared = ROOT / config["tasks"][cli.task]["prepared"]
    plan = json.loads((prepared / "plan.json").read_text())
    all_environments = plan["environments"]
    table = json.loads((prepared / "reference/context_table.json").read_text())
    from scripts.evaluation import infer_real97_ball_door_reference as reference
    from scripts.evaluation import infer_real97_unbounded_context as unbounded
    from scripts.evaluation.prepare_real97_support_override import render_grid, query_key
    from real97_trial_common import read_jsonl, write_json

    def plot(path, current_table, contexts, completed_environments):
        records = current_table["records"]
        values = np.asarray([row["context"] for row in records], dtype=float).reshape(len(records), -1)
        mean = values.mean(0)
        _, singular, axes = np.linalg.svd(values - mean, full_matrices=False)
        training_xy = (values - mean) @ axes[:2].T
        endpoints = [(env["environment"], (np.asarray(contexts[env["environment"]]["context"]).reshape(-1) - mean) @ axes[:2].T)
                     for env in completed_environments]
        points = np.vstack([training_xy, np.zeros((1, 2))] + [xy.reshape(1, 2) for _, xy in endpoints])
        low = points.min(0)
        span = np.maximum(points.max(0) - low, 1e-6)
        low -= .06 * span
        span *= 1.12
        def xy(point):
            return 60 + 690 * (point[0] - low[0]) / span[0], 535 - 450 * (point[1] - low[1]) / span[1]
        colors = {env["environment"]: "#" + "".join(f"{int(v*255):02x}" for v in colorsys.hsv_to_rgb(i / len(all_environments), .7, .8))
                  for i, env in enumerate(all_environments)}
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="630">',
                 '<rect width="1120" height="630" fill="white"/>',
                 '<text x="40" y="30" font-family="sans-serif" font-size="18">Training-time Z: both replicas; inference-time Z: triangles</text>']
        for row, point in zip(records, training_xy):
            x, y = xy(point)
            radius = 6 if row["stage1_selected"] else 4
            title = html.escape(f"{row['environment']} z{row['replica']}; source group {row['source_virtual_group_id']}")
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{colors[row["environment"]]}" stroke="black" stroke-width="0.7"><title>{title}</title></circle>')
        for name, point in endpoints:
            x, y = xy(point)
            parts.append(f'<polygon points="{x},{y-7} {x-6},{y+5} {x+6},{y+5}" fill="{colors[name]}" stroke="black"><title>{html.escape(name)} inference-time Z</title></polygon>')
        x, y = xy(np.zeros(2))
        parts.append(f'<path d="M{x-5},{y}H{x+5} M{x},{y-5}V{y+5}" stroke="black" stroke-width="2"><title>Shared initial Z: mean of all training latents</title></path>')
        for i, env in enumerate(all_environments):
            name = env["environment"]
            parts.append(f'<text x="785" y="{65+i*27}" fill="{colors[name]}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
        total = max(float(np.sum(singular ** 2)), 1e-12)
        parts.append(f'<text x="55" y="587" font-family="sans-serif" font-size="13">PC1 {100*singular[0]**2/total:.1f}%; PC2 {100*singular[1]**2/total:.1f}%; fitted on all {len(records)} training latents.</text>')
        parts.append('<text x="55" y="609" font-family="sans-serif" font-size="12">Same environment: same color. Large circles: selected Stage1 replica. Black cross: common mean initialization.</text></svg>')
        path.write_text("\n".join(parts))

    reference.pca_plot = plot
    sys.argv = ["infer_real97_unbounded_context", "--config", str(cli.config), "--task", cli.task,
                "--prepared", str(prepared), "--output", str(cli.output)]
    unbounded.main()
    queries = read_jsonl(prepared / "query.jsonl")
    groups = {}
    for index, row in enumerate(queries):
        groups.setdefault((row["environment"], row["dataset_split"]), []).append((index, row))
    order = []
    for (environment, split), rows in sorted(groups.items()):
        rows.sort(key=lambda pair: (int(pair[1]["action_level"]), int(pair[1]["episode_index"]), pair[0]))
        folder = cli.output / "grids" if split == "test" else cli.output / "grids/train"
        destination = folder / f"{environment}_gt_stage1_stage2_{split}_levels.mp4"
        render_grid([cli.output / "comparisons" / (query_key(index, row) + ".mp4") for index, row in rows], destination)
        order.append({"environment": environment, "split": split, "video": str(destination),
                      "columns": [{"level": row["action_level"], "episode_index": row["episode_index"], "query_index": index}
                                  for index, row in rows]})
    for path in (cli.output / "grids").glob("*_gt_stage1_stage2_train_test.mp4"):
        destination = cli.output / "grids/previous_mixed" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    write_json(cli.output / "grids/level_order.json", {"row_order": ["GT", "Stage1", "Stage2"], "grids": order})
    write_json(cli.output / "dual_latent_inference_complete.json", {
        "task": cli.task, "model_step": 5500, "table_step": 5500,
        "stage1_replica": config["stage1_replica"], "stage2_initialization": "mean_of_all_trained_latents",
        "mean_latent_count": len(table["records"]), "effective_context_bounds": None,
        "exact_original_support_and_queries": True, "level_sorted_split_grids_complete": True,
    })


if __name__ == "__main__":
    main()
