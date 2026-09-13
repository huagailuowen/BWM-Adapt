#!/usr/bin/env python3
"""Opt-in Real97 sampling and safe saving around the existing TTT causal scan.

The legacy trainer, its loss, and the differentiable write-then-predict memory
implementation are unchanged. Only this process substitutes the sampler/logger.
"""
from __future__ import annotations

import argparse
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from safetensors.torch import save_file

from scripts.methods import train_ttt_kvb_event80 as legacy
from scripts.methods.train_real97_standard_pooled import PooledEpisodeSampler
from wan_video_action.methods.baselines.ttt_kvb.prequential import PrequentialEpisode


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Real97Streams:
    def __init__(self, *, metadata_path, seed, sequence_length, streams_per_environment=1, **kwargs):
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.streams_per_environment = int(streams_per_environment)
        self.pool = PooledEpisodeSampler(SimpleNamespace(
            dataset_metadata_path=metadata_path, seed=seed,
            standard_environments_per_rank=3,
            standard_episodes_per_environment=self.sequence_length * self.streams_per_environment,
            standard_window_kind_field=kwargs["window_kind_field"],
            standard_preferred_window_kind=kwargs["preferred_window_kind"],
            standard_preferred_window_probability=kwargs["preferred_window_probability"],
        ))

    def sample(self, *, step, process_index, num_processes, environments_per_rank):
        if int(num_processes) != 2 or int(environments_per_rank) != 3:
            raise ValueError("Real97 TTT requires two ranks and three environments per rank.")
        selected = self.pool.sample(step, process_index)
        groups = {}
        for index in selected:
            env = int(self.pool.rows[index]["environment_index"])
            groups.setdefault(env, []).append(index)
        rng = random.Random(self.seed + step * 15485863 + process_index * 32452843)
        streams = []
        for env, indices in groups.items():
            rng.shuffle(indices)
            for stream_index in range(self.streams_per_environment):
                start = stream_index * self.sequence_length
                stream_indices = indices[start:start + self.sequence_length]
                if self.streams_per_environment > 1:
                    # First randomly partition eight distinct episodes, then
                    # independently randomize each four-chunk causal stream.
                    rng.shuffle(stream_indices)
                streams.append(PrequentialEpisode(
                    environment_id=env, indices=tuple(stream_indices),
                    action_ids=tuple(int(self.pool.rows[i]["episode_index"]) for i in stream_indices),
                ))
        return tuple(streams)


class AllocationFinished(Exception):
    pass


class SafeTTTLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output = Path(cfg["output_path"])
        self.step = 0
        self.previous_time = time.monotonic()
        self.deadline = float(os.environ["BWM_TTT_DEADLINE_EPOCH"])
        self.stop_flag = None
        self.saved = set()
        self.lock = None

    def save(self, accelerator, model, *, protected=False, reason="periodic"):
        accelerator.wait_for_everyone()
        state = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            directory = self.output / "protected" if protected else self.output
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"step-{self.step}.safetensors"
            marker = directory / f".step-{self.step}.safetensors.complete"
            metadata = directory / f"step-{self.step}.metadata.json"
            if target.exists() or marker.exists() or metadata.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint bundle: {target}")
            exported = accelerator.unwrap_model(model).export_trainable_state_dict(
                state, remove_prefix=self.cfg["remove_prefix_in_ckpt"],
            )
            exported = {k: v.detach().cpu().contiguous() for k, v in exported.items()}
            if not any("ttt_kvb_memory" in k for k in exported):
                raise RuntimeError("Refusing to save a checkpoint without TTT memory parameters.")
            if not any("ttt_kvb_gate" in k for k in exported):
                raise RuntimeError("Refusing to save a checkpoint without TTT gates.")
            temporary = directory / f".step-{self.step}.{os.getpid()}.partial"
            save_file(exported, str(temporary), metadata={
                "step": str(self.step), "method": "ttt_kqv",
                "protocol": "oneminute_write_then_predict", "reason": reason,
            })
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            atomic_json(metadata, {
                "step": self.step, "reason": reason, "protected": protected,
                "method": "ttt_kqv", "config": self.cfg,
                "learned_fast_initialization_in_model": True,
                "ephemeral_environment_fast_state_saved": False,
                "optimizer_resume_state_saved": False,
            })
            atomic_json(marker, {"checkpoint": target.name, "step": self.step})
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            print(f"[checkpoint] saved={target} reason={reason}", flush=True)
            if not protected:
                completed = sorted(
                    (p for p in directory.glob("step-*.safetensors")
                     if p.stem.removeprefix("step-").isdigit()
                     and (directory / ("." + p.name + ".complete")).is_file()),
                    key=lambda p: int(p.stem.removeprefix("step-")),
                )
                for old in completed[:-int(self.cfg["checkpoint_keep_last"])]:
                    old.unlink()
                    (directory / f"{old.stem}.metadata.json").unlink()
                    (directory / ("." + old.name + ".complete")).unlink()
        del state
        self.saved.add(self.step)
        accelerator.wait_for_everyone()

    def on_step_end(self, accelerator, model, save_steps, *, loss):
        self.step += 1
        if self.stop_flag is None:
            self.stop_flag = torch.zeros((), device=accelerator.device, dtype=torch.int32)
            if accelerator.is_main_process:
                self.lock = (self.output / ".training.lock").open("a")
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                atomic_json(self.output / "ttt_run_settings.json", {
                    "config": self.cfg, "deadline_epoch": self.deadline,
                    "sampling": "independent_rank_uniform_env_episode_window_then_shuffle",
                    "per_rank_chunks": (
                        f"3 environments x {self.cfg.get('ttt_streams_per_environment', 1)} "
                        f"independent streams x {self.cfg['ttt_sequence_length']} distinct episodes"
                    ),
                    "physical_environments_per_rank": 3,
                    "streams_per_environment": self.cfg.get("ttt_streams_per_environment", 1),
                    "chunks_per_stream": self.cfg["ttt_sequence_length"],
                    "partition": "disjoint_random_partition_with_independent_stream_shuffle",
                    "global_chunks_per_optimizer_update": 48,
                    "state_reset": "at_each_environment_stream",
                    "flow_timestep": "random_once_per_environment_stream",
                    "noise": "independent_per_chunk",
                    "protocol": "oneminute_write_then_predict",
                })
        # No Slurm commands or shared-filesystem reads in the training hot path.
        if accelerator.is_main_process:
            self.stop_flag.fill_(int(time.time() >= self.deadline))
        torch.distributed.broadcast(self.stop_flag, src=0)
        stopping = bool(self.stop_flag.item())
        if self.step % int(self.cfg["log_steps"]) == 0:
            now = time.monotonic()
            if accelerator.is_main_process:
                record = {
                    "step": self.step, "loss": float(loss.detach().cpu()),
                    "seconds_per_step": (now - self.previous_time) / int(self.cfg["log_steps"]),
                    "timestamp": time.time(), "global_chunks": 48,
                }
                with (self.output / "loss.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print("[real97_ttt] " + json.dumps(record), flush=True)
            self.previous_time = now
        final = self.step == int(self.cfg["ttt_max_updates"])
        if self.step == 2300 or stopping or final:
            reason = "wallclock_23h30" if stopping else ("final" if final else "step2300")
            self.save(accelerator, model, protected=True, reason=reason)
        elif self.step % int(save_steps) == 0:
            self.save(accelerator, model)
        if stopping:
            accelerator.end_training()
            raise AllocationFinished()

    def on_training_end(self, accelerator, model, save_steps):
        if self.step not in self.saved:
            self.save(accelerator, model, protected=True, reason="final")
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    own, _ = parser.parse_known_args()
    cfg = yaml.safe_load(Path(own.config).read_text())["training"]
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("TTT needs exactly two CUDA ranks; no CPU fallback.")
    streams_per_environment = int(cfg.get("ttt_streams_per_environment", 1))
    batch_layout = (cfg["ttt_sequence_length"], cfg["ttt_environments_per_rank"], streams_per_environment)
    if batch_layout not in ((8, 3, 1), (4, 3, 2)):
        raise ValueError("Expected per-rank 3x8 or 3x2x4 configuration.")
    if cfg["ttt_protocol"] != "oneminute_write_then_predict":
        raise ValueError("Use the established write-then-predict protocol.")
    if cfg["spatial_loss_mode"] != "none" or not cfg["video_light_augmentation_enabled"]:
        raise ValueError("This Real97 run requires lighting augmentation and no ROI.")
    legacy.Event80PrequentialSampler = partial(
        Real97Streams, streams_per_environment=streams_per_environment,
    )
    original_add_config = legacy.add_ttt_kvb_config

    def add_real97_config(parser):
        parser = original_add_config(parser)
        parser.add_argument("--ttt_streams_per_environment", type=int, default=1)
        return parser

    legacy.add_ttt_kvb_config = add_real97_config
    logger = SafeTTTLogger(cfg)
    legacy.TimedRetentionModelLogger = lambda *args, **kwargs: logger
    try:
        legacy.main()
    except AllocationFinished:
        print("[real97_ttt] allocation deadline checkpoint published; exiting.", flush=True)


if __name__ == "__main__":
    main()
