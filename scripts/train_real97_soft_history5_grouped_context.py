#!/usr/bin/env python3
"""Opt-in Ours Soft training with the same five-frame history as Standard."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_stage1_grouped_context as grouped
from wan_video_action.data.history_prefix import CanonicalTargetEEF, HistoryPrefixDataset
from wan_video_action.methods.baselines.history_loss import history_prefix_flow_loss


_legacy_build_dataset = grouped.build_dataset


def require_history_contract(args):
    expected = (33, 5, 3, "eef_target", 14)
    actual = (
        int(args.num_frames), int(args.num_history_frames), int(args.frame_stride),
        args.action_type, int(args.action_dim),
    )
    if actual != expected:
        raise ValueError(f"Soft history entrypoint requires {expected}, got {actual}")
    if args.task not in ("sft", "sft:train"):
        raise ValueError("History Ours only supports flow-matching SFT")
    if str(getattr(args, "spatial_loss_mode", "none")) != "none":
        raise ValueError("This experiment uses uniform future-only flow loss, not ROI")
    if bool(getattr(args, "grouped_context_bridge_enabled", False)) or bool(
        getattr(args, "grouped_context_self_correction_enabled", False)
    ):
        raise ValueError("Bridge/shared-timestep self-correction is not enabled for history Ours")


def build_history_dataset(args, runtime_config):
    require_history_contract(args)
    dataset = _legacy_build_dataset(args, runtime_config)
    for row in dataset.data:
        if row.get("dataset_split") != "train":
            raise ValueError("History Ours manifest contains a non-training episode")
        if row.get("action_semantics") != "eef_target":
            raise ValueError("History Ours requires recorded target EEF metadata")
    with open(args.action_stat_path, encoding="utf-8") as handle:
        stats = json.load(handle)["eef_target"]
    dataset.special_operator_map["action"] = CanonicalTargetEEF(args, stats)
    return HistoryPrefixDataset(dataset, args)


class HistoryGroupedContextStage1Module(grouped.GroupedContextStage1Module):
    def __init__(self, *args, grouped_args, **kwargs):
        require_history_contract(grouped_args)
        super().__init__(*args, grouped_args=grouped_args, **kwargs)
        self.task_to_loss[self.task] = history_prefix_flow_loss
        print(
            "[history_ours] observed_video_frames=5 future_video_frames=28 "
            "clean_history_latents=2 future_loss_latents=7 action=eef_target "
            "context_and_model_phases=legacy_curriculum",
            flush=True,
        )

    def forward(self, data, inputs=None):
        if data.get("_flow_timestep_index") is not None or data.get(
            "_self_correction_donor_data"
        ) is not None:
            raise ValueError("Do not route five-frame history through the legacy one-frame loss")
        return super().forward(data, inputs=inputs)


def main():
    # Only this explicit entrypoint replaces the two factories in this process.
    # Sampling, alternating freezing, optimizers, and paired checkpoint retention
    # remain the existing Ours implementation. No legacy import is modified on disk.
    grouped.build_dataset = build_history_dataset
    grouped.GroupedContextStage1Module = HistoryGroupedContextStage1Module
    grouped.main()


if __name__ == "__main__":
    main()
