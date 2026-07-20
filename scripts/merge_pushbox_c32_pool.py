#!/usr/bin/env python3
"""Merge C32 adaptation shards and plot them in the active-table PCA basis."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--context-table", type=Path, required=True)
    parser.add_argument("--active-values", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-pool", type=Path, required=True)
    parser.add_argument("--output-pca", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    trajectory = []
    for path in args.trajectory:
        trajectory.extend(read_jsonl(path))

    final_by_sample = {}
    for row in trajectory:
        sample_index = int(row["sample_index"])
        if sample_index not in final_by_sample or int(row["inner_step"]) > int(final_by_sample[sample_index]["inner_step"]):
            final_by_sample[sample_index] = row
    missing = sorted(set(range(len(manifest))) - set(final_by_sample))
    if missing:
        raise ValueError(f"Missing final contexts for {len(missing)} samples: {missing[:20]}")

    table = json.loads(args.context_table.read_text())
    table_records = table["records"]
    table_mus = np.asarray([float(row["friction_mu"]) for row in table_records])
    table_contexts = np.asarray([row["context"] for row in table_records], dtype=np.float32).reshape(len(table_records), -1)
    active_values = [float(value) for value in args.active_values.split(",") if value.strip()]
    active_indices = [int(np.argmin(np.abs(table_mus - value))) for value in active_values]
    active_indices = sorted(set(active_indices), key=lambda index: table_mus[index])
    active_mus = table_mus[active_indices]
    active_contexts = table_contexts[active_indices]

    center = active_contexts.mean(axis=0)
    _, singular_values, components = np.linalg.svd(active_contexts - center, full_matrices=False)
    components = components[:2]
    explained = singular_values[:2] ** 2 / np.sum(singular_values**2)
    active_xy = (active_contexts - center) @ components.T

    groups = []
    all_final_contexts = []
    for target_mu in sorted(set(float(row["pool_target_mu"]) for row in manifest)):
        indices = [index for index, row in enumerate(manifest) if float(row["pool_target_mu"]) == target_mu]
        contexts = np.asarray([final_by_sample[index]["context_flat"] for index in indices], dtype=np.float32)
        if contexts.shape != (50, 32):
            raise ValueError(f"Expected (50,32) contexts for mu={target_mu}, got {contexts.shape}.")
        records = []
        for index, context in zip(indices, contexts):
            source = manifest[index]
            final = final_by_sample[index]
            records.append(
                {
                    "pool_index": int(source["pool_index"]),
                    "support_episode_index": int(source["episode_index"]),
                    "support_action_id": int(source["action_id"]),
                    "support_displacement_m": float(source["support_displacement_m"]),
                    "initial_context_seed": int(source["initial_context_seed"]),
                    "final_support_loss": final.get("support_loss"),
                    "context": [float(value) for value in context],
                }
            )
        groups.append(
            {
                "mu": target_mu,
                "source_mu": float(manifest[indices[0]]["pool_source_mu"]),
                "contexts": contexts.tolist(),
                "records": records,
            }
        )
        all_final_contexts.append((target_mu, contexts))

    payload = {
        "format_version": 1,
        "latent_dim": 32,
        "contexts_per_group": 50,
        "initialization": "independent_uniform_0_1",
        "inner_lr_schedule": "3.0:10,1.5:10,0.5:10,0.15:10",
        "source_checkpoint": args.checkpoint,
        "source_context_table": str(args.context_table),
        "active_table_values": active_values,
        "groups": groups,
    }
    args.output_pool.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_pool.with_suffix(args.output_pool.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output_pool)

    colors = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, len(groups)))
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#f8fafc")
    ax.set_facecolor("white")
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.plot(active_xy[:, 0], active_xy[:, 1], "--", color="#64748b", linewidth=1.5, alpha=0.8)
    table_scatter = ax.scatter(
        active_xy[:, 0],
        active_xy[:, 1],
        c=active_mus,
        cmap="viridis",
        marker="o",
        s=72,
        edgecolors="white",
        linewidths=1.0,
        label="training-time active C table",
        zorder=3,
    )
    for color, (target_mu, contexts) in zip(colors, all_final_contexts):
        xy = (contexts - center) @ components.T
        mean_xy = xy.mean(axis=0)
        ax.scatter(xy[:, 0], xy[:, 1], s=31, color=color, alpha=0.52, edgecolors="none")
        ax.scatter(
            [mean_xy[0]],
            [mean_xy[1]],
            marker="*",
            s=210,
            color=color,
            edgecolors="#111827",
            linewidths=0.9,
            label=f"mu={target_mu:g}: 50 adapted Z",
            zorder=5,
        )
    colorbar = fig.colorbar(table_scatter, ax=ax, pad=0.02)
    colorbar.set_label("training-table friction")
    ax.set_title("Origin-random C32: active training table and six 50-Z adaptation pools", loc="left", weight="bold")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False)
    fig.tight_layout()
    args.output_pca.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pca, format="svg")
    plt.close(fig)
    np.savez(
        args.output_pca.with_suffix(".basis.npz"),
        center=center,
        components=components,
        explained_variance=explained,
        active_mus=active_mus,
    )
    print(f"pool={args.output_pool} pca={args.output_pca} groups={len(groups)} contexts={len(manifest)}")


if __name__ == "__main__":
    main()
