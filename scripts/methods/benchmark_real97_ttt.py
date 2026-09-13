#!/usr/bin/env python3
"""Short, low-priority TTT performance/gradient probes; not production training."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from scripts.methods import train_real97_ttt as real97


class BenchmarkLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output = Path(cfg["output_path"])
        self.records = []
        self.expected = json.loads(os.environ.get("BWM_TTT_BENCHMARK_EXPECTED_JSON", "{}"))
        mismatches = {
            key: {"expected": value, "actual": cfg.get(key)}
            for key, value in self.expected.items() if cfg.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Benchmark configuration did not take effect: {mismatches}")
        print("[benchmark_effective_config] " + json.dumps({
            key: cfg.get(key) for key in (
                "ttt_max_updates", "ttt_environments_per_rank", "ttt_streams_per_environment",
                "ttt_sequence_length", "ttt_protocol", "ttt_saved_tensor_policy",
                "ttt_gpu_saved_tensor_budget_gib", "ttt_saved_tensor_cpu_offload",
                "ttt_backward_per_stream", "ttt_sync_per_update",
            )
        }), flush=True)

    def on_step_diagnostics(self, *, accelerator, model, step, loss, position_losses,
                            gradient_norm, started_wall, started_monotonic, storage_statistics):
        torch.cuda.synchronize(accelerator.device)
        finished_wall = time.time()
        seconds = time.monotonic() - started_monotonic
        probes = {}
        unwrapped = accelerator.unwrap_model(model)
        selected_categories = {}
        for name, parameter in unwrapped.named_parameters():
            if parameter.grad is None:
                continue
            category = (
                "gate" if "ttt_kvb_gate" in name else
                "memory_q" if "ttt_kvb_memory.q_proj.weight" in name else
                "action" if "action_encoder" in name else
                "backbone_q" if ".self_attn.q.weight" in name else None
            )
            if category is None or selected_categories.get(category, 0) >= 2:
                continue
            selected_categories[category] = selected_categories.get(category, 0) + 1
            grad = parameter.grad.detach().reshape(-1)
            indices = torch.linspace(0, grad.numel() - 1, min(32, grad.numel()),
                                     device=grad.device).long()
            probes[name] = grad[indices].float().cpu().tolist()
        local = {
            "rank": accelerator.process_index, "step": step,
            "started_epoch": started_wall, "finished_epoch": finished_wall,
            "seconds": seconds, "loss": float(loss.detach().cpu()),
            "position_losses": position_losses.detach().cpu().tolist(),
            "preclip_gradient_norm": None if gradient_norm is None else float(gradient_norm.detach().cpu()),
            "clipped_gradient_probes": probes,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(accelerator.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(accelerator.device),
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "saved_tensor_statistics": storage_statistics,
        }
        gathered = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered, local)
        for entry in gathered:
            scalars = [entry["loss"], *entry["position_losses"]]
            if entry["preclip_gradient_norm"] is not None:
                scalars.append(entry["preclip_gradient_norm"])
            scalars.extend(value for values in entry["clipped_gradient_probes"].values() for value in values)
            if not all(math.isfinite(value) for value in scalars):
                raise FloatingPointError(f"Nonfinite benchmark values at step {step}, rank {entry['rank']}")
        if accelerator.is_main_process:
            record = {"step": step, "max_rank_seconds": max(x["seconds"] for x in gathered), "ranks": gathered}
            self.records.append(record)
            with (self.output / "benchmark_steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("[benchmark_step] " + json.dumps({
                "step": step, "seconds": record["max_rank_seconds"], "loss": local["loss"],
                "peak_gpu_gib": max(x["cuda_peak_allocated_bytes"] for x in gathered) / 1024**3,
            }), flush=True)

    def on_step_end(self, accelerator, model, save_steps, *, loss):
        # Benchmark artifacts are diagnostics, not resumable checkpoints.
        pass

    def on_training_end(self, accelerator, model, save_steps):
        if accelerator.is_main_process:
            if len(self.records) != int(self.cfg["ttt_max_updates"]):
                raise RuntimeError(
                    f"Incomplete benchmark: expected {self.cfg['ttt_max_updates']} updates, "
                    f"got {len(self.records)}"
                )
            payload = {
                "benchmark_only": True, "production_checkpoint_saved": False,
                "job_id": os.environ.get("SLURM_JOB_ID"), "config": self.cfg,
                "optimizer_updates": len(self.records), "records": self.records,
            }
            target = self.output / "benchmark_complete.json"
            temporary = target.with_suffix(".json.partial")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, target)
        accelerator.wait_for_everyone()
        accelerator.end_training()


if __name__ == "__main__":
    real97.SafeTTTLogger = BenchmarkLogger
    real97.main()
