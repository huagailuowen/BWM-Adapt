#!/usr/bin/env python3
"""Process-local light augmentation for paired Standard and near-table Ours."""

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_json(path, data):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise RuntimeError(f"Frozen light experiment input changed: {path}")
    else:
        publish(path, data)


def episode_key(row):
    return f"{row['environment']}/ep{int(row['episode_index']):06d}"


def parameters(key, config):
    digest = hashlib.sha256(f"{config['augmentation_seed']}:{key}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)
    rng = random.Random(seed)
    result = {"episode_key": key, "parameter_seed": seed, "applied": True}
    for name in ("gain", "contrast", "gamma", "offset", "gx", "gy"):
        result[name] = rng.uniform(*config[name + "_range"])
    result["tint"] = [rng.uniform(*config["tint_range"]) for _ in range(3)]
    result["noise_seed"] = int.from_bytes(digest[8:16], "big") % (2 ** 63 - 1)
    result["noise_std"] = float(config["noise_std"])
    return result


class LightingDataset:
    def __init__(self, dataset, params, used):
        self.dataset = dataset
        self.params = params
        self.used = used

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        sample = dict(self.dataset[index])
        row = self.dataset.data[index]
        key = episode_key(row)
        if key not in self.params:
            raise RuntimeError(f"Episode is outside the frozen light manifest: {key}")
        values = self.params[key]
        video = sample["video"]
        if not torch.is_tensor(video) or video.ndim != 5 or video.shape[:2] != (1, 3):
            raise RuntimeError("Expected the unchanged single-view RGB loader tensor [1,3,T,H,W]")
        if video.device.type != "cpu" or float(video.min()) < -1.0001 or float(video.max()) > 1.0001:
            raise RuntimeError("Lighting must be applied to the CPU [-1,1] video before VAE encoding")
        height, width = video.shape[-2:]
        xx = torch.linspace(-1, 1, width, dtype=torch.float32).reshape(1, 1, 1, 1, width)
        yy = torch.linspace(-1, 1, height, dtype=torch.float32).reshape(1, 1, 1, height, 1)
        field = 1 + values["gx"] * xx + values["gy"] * yy
        tint = torch.tensor(values["tint"], dtype=torch.float32).reshape(1, 3, 1, 1, 1)
        generator = torch.Generator(device="cpu").manual_seed(values["noise_seed"])
        noise = torch.randn((1, 3, 1, height, width), generator=generator, dtype=torch.float32) * values["noise_std"]
        x = ((video.float() + 1) * 0.5).clamp(0, 1)
        x = x.pow(values["gamma"])
        x = (x - 0.5) * values["contrast"] + 0.5
        x = x * values["gain"] * tint * field + values["offset"] + noise
        quantized = (x.clamp(0, 1) * 255).round().to(torch.uint8)
        sample["video"] = (quantized.float() / 255 * 2 - 1).to(video.dtype)
        self.used.add(key)
        return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("standard", "ours"), required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    own = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this inference on a compute node")
    config = json.loads(own.config.read_text())
    if config["task"] != "stick" or float(config["apply_probability"]) != 1:
        raise RuntimeError("This branch is the all-episodes Stick lighting ablation")
    pair_root = own.pair_root.resolve()
    output = pair_root / own.method
    if pair_root == ROOT or pair_root == ROOT / "outputs":
        raise RuntimeError("A dedicated paired-experiment directory is required")
    pair_root.mkdir(parents=True, exist_ok=True)
    freeze_json(pair_root / "experiment.json", config)
    prepared = resolve(config["prepared"])
    query_bytes = (prepared / "query.jsonl").read_bytes()
    support_bytes = (prepared / "support.jsonl").read_bytes()
    query = [json.loads(line) for line in query_bytes.decode().splitlines() if line.strip()]
    support = [json.loads(line) for line in support_bytes.decode().splitlines() if line.strip()]
    keys = sorted({episode_key(row) for row in query + support})
    params = {key: parameters(key, config) for key in keys}
    shared = {
        "augmentation_config": config, "episode_parameters": params,
        "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "support_sha256": hashlib.sha256(support_bytes).hexdigest(),
        "model_resolution_wh": [256, 192], "frames": 41, "frame_stride": 3,
        "shared_between_methods": True, "query_future_GT_is_input": False,
    }
    freeze_json(pair_root / "lighting_manifest.json", shared)
    standard_config = json.loads(resolve(config["standard_config"]).read_text())
    ours_config = json.loads(resolve(config["ours_config"]).read_text())
    ours_config.update(reuse_gt_and_stage1=False, input_light_augmentation=True,
                       light_augmentation_manifest=str(pair_root / "lighting_manifest.json"))
    freeze_json(pair_root / "standard_config.json", standard_config)
    freeze_json(pair_root / "ours_config.json", ours_config)
    output.mkdir(parents=True, exist_ok=True)
    used = set()
    sys.path.insert(0, str(ROOT / "scripts"))
    infer_module = importlib.import_module("infer")
    original_builder = infer_module.build_infer_dataset

    def build_augmented_dataset(args):
        if (args.num_frames, args.frame_stride, args.height, args.width, args.action_type) != (41, 3, 192, 256, "eef_target"):
            raise RuntimeError("Both models must retain the original Stick sampling and target-EEF contract")
        return LightingDataset(original_builder(args), params, used)

    infer_module.build_infer_dataset = build_augmented_dataset

    def describe_write(original):
        def write(path, data):
            if Path(path) == output / "provenance.json" and isinstance(data, dict):
                data.update(input_light_augmentation=True, augmentation_probability=1.0,
                            lighting_manifest=str(pair_root / "lighting_manifest.json"),
                            light_augmented_GT=True, query_future_GT_used_for_adaptation=False,
                            legacy_augmentation_flag_is_not_this_process_local_branch=True)
            return original(path, data)
        return write

    if own.method == "standard":
        standard = importlib.import_module("infer_real97_standard_reference")
        standard.write_json = describe_write(standard.write_json)
        sys.argv = ["infer_real97_standard_reference", "--task", "stick", "--config",
                    str(pair_root / "standard_config.json"), "--output", str(output)]
        standard.main()
    else:
        near = importlib.import_module("infer_real97_stick_near_table")
        # Clean source videos are not valid references for an augmented query.
        # Existing videos in THIS output remain resumable augmented results.
        near.copy_reference_videos = lambda source, destination: 0

        def cpu_plot(destination, environments, table, records):
            payload = destination / "lighting_pca_input.json"
            publish(payload, {"output": str(destination), "environments": environments,
                              "table": table, "records": records})
            program = (
                "import json,sys; from pathlib import Path; "
                "sys.path.insert(0,sys.argv[1]); "
                "from infer_real97_stick_near_table import plot_contexts; "
                "p=json.loads(Path(sys.argv[2]).read_text()); "
                "plot_contexts(Path(p['output']),p['environments'],p['table'],p['records'])"
            )
            subprocess.run([config["cpu_plot_python"], "-c", program,
                            str(ROOT / "scripts/evaluation"), str(payload)], check=True)

        near.plot_contexts = cpu_plot
        reference = importlib.import_module("infer_real97_reference_trial")
        reference.write_json = describe_write(reference.write_json)
        # Cover module-level aliases as well as imports inside reference.main.
        if getattr(reference, "build_infer_dataset", None) is original_builder:
            reference.build_infer_dataset = build_augmented_dataset
        sys.argv = ["infer_real97_stick_near_table", "--config", str(pair_root / "ours_config.json"),
                    "--output", str(output)]
        near.main()
    publish(output / "lighting_inference_complete.json", {
        "method": own.method, "job_id": os.environ["SLURM_JOB_ID"], "queries": len(query),
        "used_episode_keys_this_process": sorted(used), "lighting_manifest": str(pair_root / "lighting_manifest.json"),
        "apply_probability": 1.0, "shared_query_augmentation": True,
        "ours_support_augmented": own.method == "ours", "actions_and_generation_seeds_unchanged": True,
        "preserved_unaugmented_outputs": True, "formal_metric_approved": False,
    })
    print(f"[lighting_complete] {own.method}: {output}", flush=True)


if __name__ == "__main__":
    main()
