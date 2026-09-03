#!/usr/bin/env python3
"""One-Minute-style TTT-KVB support/query inference on Event80.

For every query denoising timestep, this runner rebuilds the fast state from
the observed support at the same diffusion timestep, then lets the query
perform an update-then-read scan. Each query is an independent branch from
the same support; query fast weights never leak into another query.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.infer import (
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    prepare_sample_for_rollout,
)
from scripts.methods.infer_ttt_kvb_event80 import build_training_faithful_model
from scripts.methods.train_ttt_kvb_event80 import add_ttt_kvb_config
from wan_video_action.parsers import (
    add_general_config,
    merge_yaml_and_args,
)
from wan_video_action.utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = add_ttt_kvb_config(
        add_general_config(
            argparse.ArgumentParser("Event80 One-Minute-style TTT-KVB inference")
        )
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--sample_indices", type=str, required=True)
    parser.add_argument("--support_indices", type=str, required=True)
    parser.add_argument("--ttt_checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--ttt_support_noise_seed_offset",
        type=int,
        default=1_000_003,
        help="Deterministic support-noise seed offset used at every denoising timestep.",
    )
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if args.ttt_protocol != "oneminute_write_then_predict":
        raise ValueError(
            "This runner requires ttt_protocol=oneminute_write_then_predict, "
            f"got {args.ttt_protocol!r}."
        )
    return args


def _mean_state_norm(norms: dict[str, float]) -> float:
    if not norms:
        return 0.0
    return sum(norms.values()) / float(len(norms))


def _update_counts(
    statistics: dict[str, list[dict[str, float]]],
) -> dict[str, int]:
    return {layer_id: len(values) for layer_id, values in statistics.items()}


def _prepare_support_model_inputs(
    model,
    support: dict[str, Any],
    noise_seed: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Encode a support clip once and retain its clean latent and conditions."""

    pipe = model.pipe
    with torch.no_grad():
        inputs = model.get_pipeline_inputs(support)
        inputs = model.transfer_data_to_device(
            inputs, pipe.device, pipe.torch_dtype
        )
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)

    inputs_shared, inputs_posi, _ = inputs
    model_inputs = {**inputs_shared, **inputs_posi}
    input_latents = model_inputs.get("input_latents")
    if not isinstance(input_latents, torch.Tensor):
        raise RuntimeError(
            "Support preprocessing did not produce input_latents; the full observed "
            "support video is required for same-timestep TTT replay."
        )

    generator = torch.Generator(device=input_latents.device)
    generator.manual_seed(int(noise_seed))
    support_noise = torch.randn(
        input_latents.shape,
        dtype=input_latents.dtype,
        device=input_latents.device,
        generator=generator,
    )
    return model_inputs, input_latents.detach(), support_noise


class SameTimestepSupportQueryModelFn:
    """Wrap one Wan model call with support replay and query adaptation."""

    def __init__(
        self,
        *,
        pipe,
        controller,
        original_model_fn,
        support_model_inputs: dict[str, Any],
        support_input_latents: torch.Tensor,
        support_noise: torch.Tensor,
    ) -> None:
        self.pipe = pipe
        self.controller = controller
        self.original_model_fn = original_model_fn
        self.support_model_inputs = support_model_inputs
        self.support_input_latents = support_input_latents
        self.support_noise = support_noise
        self.trace: list[dict[str, Any]] = []

    def _support_call_inputs(self, timestep: torch.Tensor) -> dict[str, Any]:
        support_inputs = dict(self.support_model_inputs)
        support_latents = self.pipe.scheduler.add_noise(
            self.support_input_latents,
            self.support_noise,
            timestep,
        )
        first_frame_latents = support_inputs.get("first_frame_latents")
        if isinstance(first_frame_latents, torch.Tensor):
            support_latents = support_latents.clone()
            support_latents[:, :, 0:1] = first_frame_latents[:, :, 0:1]

        support_inputs["latents"] = support_latents
        support_inputs["noise"] = self.support_noise
        support_inputs["timestep"] = timestep
        support_inputs["use_gradient_checkpointing"] = False
        support_inputs["use_gradient_checkpointing_offload"] = False
        for model_name in self.pipe.in_iteration_models:
            support_inputs[model_name] = getattr(self.pipe, model_name)
        return support_inputs

    def __call__(self, *args, **query_inputs):
        timestep = query_inputs.get("timestep")
        if not isinstance(timestep, torch.Tensor):
            raise RuntimeError("Wan model_fn call did not provide a tensor timestep.")

        # A diffusion timestep is one independent One-Minute sequence:
        # learned initial state -> support scan -> query scan -> discard.
        self.controller.reset(batch_size=1)
        try:
            with self.controller.causal_scan(differentiable=False):
                self.original_model_fn(
                    **self._support_call_inputs(timestep)
                )
            support_statistics = self.controller.write_statistics()
            support_counts = _update_counts(support_statistics)
            support_norms = self.controller.state_norms()

            with self.controller.causal_scan(differentiable=False):
                prediction = self.original_model_fn(*args, **query_inputs)
            complete_statistics = self.controller.write_statistics()
            complete_counts = _update_counts(complete_statistics)
            query_counts = {
                layer_id: complete_counts.get(layer_id, 0)
                - support_counts.get(layer_id, 0)
                for layer_id in complete_counts
            }
            query_norms = self.controller.state_norms()
            self.trace.append(
                {
                    "denoising_call": len(self.trace),
                    "timestep": float(
                        timestep.detach().float().reshape(-1)[0].cpu()
                    ),
                    "support_updates_per_layer": support_counts,
                    "query_updates_per_layer": query_counts,
                    "support_state_norm_mean": _mean_state_norm(support_norms),
                    "query_state_norm_mean": _mean_state_norm(query_norms),
                }
            )
            return prediction
        finally:
            self.controller.clear()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    model, installation = build_training_faithful_model(args)
    pipe = model.pipe
    controller = installation.controller
    original_model_fn = pipe.model_fn

    support_by_environment: dict[int, int] = {}
    for support_index in _parse_sample_indices(args.support_indices):
        environment_id = int(dataset[support_index]["mu_index"])
        if environment_id in support_by_environment:
            raise ValueError(
                "One-Minute K=1 inference received multiple supports for "
                f"environment {environment_id}."
            )
        support_by_environment[environment_id] = int(support_index)

    queries_by_environment: dict[int, list[int]] = defaultdict(list)
    for sample_index in _parse_sample_indices(args.sample_indices):
        environment_id = int(dataset[sample_index]["mu_index"])
        queries_by_environment[environment_id].append(int(sample_index))

    records = []
    for environment_id, query_indices in queries_by_environment.items():
        if environment_id not in support_by_environment:
            raise KeyError(f"No TTT support for environment {environment_id}.")
        support_index = support_by_environment[environment_id]
        noise_seed = (
            int(args.seed)
            + int(args.ttt_support_noise_seed_offset)
            + environment_id
        )
        support_model_inputs, support_input_latents, support_noise = (
            _prepare_support_model_inputs(
                model,
                dataset[support_index],
                noise_seed=noise_seed,
            )
        )
        print(
            f"[support_ready] environment={environment_id} "
            f"support={support_index} noise_seed={noise_seed}",
            flush=True,
        )

        for sample_index in query_indices:
            sample = prepare_sample_for_rollout(
                dataset[sample_index], sample_index, pipe, args
            )
            if args.skip_existing and Path(sample["output_path"]).is_file():
                print(f"[skip] existing prediction {sample['output_path']}")
                continue

            wrapped_model_fn = SameTimestepSupportQueryModelFn(
                pipe=pipe,
                controller=controller,
                original_model_fn=original_model_fn,
                support_model_inputs=support_model_inputs,
                support_input_latents=support_input_latents,
                support_noise=support_noise,
            )
            pipe.model_fn = wrapped_model_fn
            try:
                predicted_path = _run_autoregressive(
                    pipe=pipe,
                    sample=sample,
                    args=args,
                )
            finally:
                pipe.model_fn = original_model_fn
                controller.clear()

            records.append(
                {
                    "environment_id": environment_id,
                    "support_index": support_index,
                    "query_index": sample_index,
                    "support_noise_seed": noise_seed,
                    "prediction": predicted_path,
                    "denoising_trace": wrapped_model_fn.trace,
                }
            )
            print(
                f"[done] sample_index={sample_index} environment={environment_id} "
                f"denoising_calls={len(wrapped_model_fn.trace)} "
                f"output={predicted_path}",
                flush=True,
            )

    protocol = {
        "method": "ttt_kqv_oneminute_forward",
        "implementation": "same_timestep_support_query_branch",
        "support_size": 1,
        "support_query_disjoint": True,
        "diffusion_timestep_policy": "support_and_query_share_each_denoising_timestep",
        "fast_state_lifecycle": (
            "reset_to_learned_initial_then_support_scan_then_query_scan_per_timestep"
        ),
        "query_policy": "update_then_read",
        "cross_query_state": "independent_branch_from_same_support",
        "cross_denoising_timestep_state": "reset_and_rebuild",
        "outer_loss_at_inference": False,
        "checkpoint": args.ttt_checkpoint_path,
        "ttt_layers": list(installation.layer_indices),
        "records": records,
    }
    protocol_path = Path(args.output_path) / "protocol.json"
    temporary_path = protocol_path.with_suffix(".json.partial")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    temporary_path.replace(protocol_path)
    print(f"[protocol] {protocol_path}", flush=True)


if __name__ == "__main__":
    main()
