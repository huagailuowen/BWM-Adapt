#!/usr/bin/env python3
"""Training-faithful TTT-KVB support-write/query-rollout inference on Event80."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors.torch import load_file
import torch

from scripts.infer import (
    _parse_sample_indices,
    _run_autoregressive,
    build_infer_dataset,
    prepare_sample_for_rollout,
)
from scripts.methods.train_ttt_kvb_event80 import add_ttt_kvb_config
from scripts.train import WanTrainingModule
from wan_video_action.methods.baselines.ttt_kvb.wan_adapter import install_ttt_kvb
from wan_video_action.parsers import (
    add_general_config,
    merge_yaml_and_args,
    prepare_runtime_config,
)
from wan_video_action.utils import set_global_seed


_FROZEN_DIT_KEYS = {
    "text_embedding.0.weight",
    "text_embedding.0.bias",
    "text_embedding.2.weight",
    "text_embedding.2.bias",
}
_FROZEN_ACTION_KEYS = {
    "action_embedding.0.weight",
    "action_embedding.0.bias",
    "action_embedding.2.weight",
    "action_embedding.2.bias",
}


def parse_args():
    parser = add_ttt_kvb_config(
        add_general_config(argparse.ArgumentParser("Event80 TTT-KVB inference"))
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--sample_indices", type=str, required=True)
    parser.add_argument("--support_indices", type=str, required=True)
    parser.add_argument("--ttt_checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args


def load_ttt_kvb_checkpoint(model, installation, checkpoint_path: str) -> None:
    """Load the partial training checkpoint without hiding key mismatches."""
    state = load_file(checkpoint_path, device="cpu")
    action_prefix = "pipe.action_encoder."
    action_state = {
        key[len(action_prefix):]: value
        for key, value in state.items()
        if key.startswith(action_prefix)
    }
    unsupported_pipe_keys = sorted(
        key
        for key in state
        if key.startswith("pipe.") and not key.startswith(action_prefix)
    )
    if unsupported_pipe_keys:
        raise RuntimeError(
            "Unsupported pipe-prefixed keys in TTT-KVB checkpoint: "
            f"{unsupported_pipe_keys}"
        )
    if not action_state:
        raise RuntimeError("TTT-KVB checkpoint contains no action-encoder weights.")

    dit_state = {
        key: value for key, value in state.items() if not key.startswith("pipe.")
    }
    checkpoint_layers = sorted(
        int(key.split(".", 2)[1])
        for key in dit_state
        if key.startswith("blocks.") and key.endswith(".self_attn.ttt_kvb_gate")
    )
    installed_layers = sorted(int(index) for index in installation.layer_indices)
    if checkpoint_layers != installed_layers:
        raise RuntimeError(
            "TTT-KVB checkpoint/install layer mismatch: "
            f"checkpoint={checkpoint_layers}, installed={installed_layers}."
        )
    for layer_index in installed_layers:
        prefix = f"blocks.{layer_index}.self_attn.ttt_kvb_memory."
        required = {
            f"blocks.{layer_index}.self_attn.ttt_kvb_gate",
            prefix + "q_proj.weight",
            prefix + "k_proj.weight",
            prefix + "v_proj.weight",
            prefix + "out_proj.weight",
            prefix + "lr_weight",
            prefix + "lr_bias",
            prefix + "initial_w1",
            prefix + "initial_b1",
            prefix + "initial_w2",
            prefix + "initial_b2",
            prefix + "fast_norm_weight",
            prefix + "fast_norm_bias",
        }
        missing = sorted(required.difference(dit_state))
        if missing:
            raise RuntimeError(
                f"TTT-KVB checkpoint is incomplete for layer {layer_index}: {missing}"
            )

    dit_incompatible = model.pipe.dit.load_state_dict(dit_state, strict=False)
    if (
        set(dit_incompatible.missing_keys) != _FROZEN_DIT_KEYS
        or dit_incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Invalid TTT-KVB DiT checkpoint: "
            f"missing={sorted(dit_incompatible.missing_keys)}, "
            f"unexpected={sorted(dit_incompatible.unexpected_keys)}"
        )

    action_incompatible = model.pipe.action_encoder.load_state_dict(
        action_state, strict=False
    )
    if (
        set(action_incompatible.missing_keys) != _FROZEN_ACTION_KEYS
        or action_incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Invalid TTT-KVB action checkpoint: "
            f"missing={sorted(action_incompatible.missing_keys)}, "
            f"unexpected={sorted(action_incompatible.unexpected_keys)}"
        )
    print(
        "[checkpoint] loaded TTT-KVB "
        f"dit={len(dit_state)} action={len(action_state)} "
        f"layers={installed_layers}",
        flush=True,
    )


def build_training_faithful_model(args):
    runtime_config = prepare_runtime_config(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WanTrainingModule(
        model_paths=json.dumps(runtime_config["model_paths_list"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=runtime_config["tokenizer_path"],
        enable_text=args.enable_text,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        modules=runtime_config["modules"],
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device=device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        args=args,
    )
    model.requires_grad_(False)
    installation = install_ttt_kvb(
        model.pipe.dit,
        layer_spec=args.ttt_layers,
        expansion=args.ttt_expansion,
        base_inner_lr=args.ttt_base_inner_lr,
        inner_batch_size=args.ttt_inner_batch_size,
        write_token_budget=args.ttt_write_token_budget,
        gate_init=args.ttt_gate_init,
    )
    load_ttt_kvb_checkpoint(model, installation, args.ttt_checkpoint_path)
    model.use_gradient_checkpointing = False
    model.use_gradient_checkpointing_offload = False
    model.pipe.use_gradient_checkpointing = False
    model.pipe.use_gradient_checkpointing_offload = False
    model.eval()
    model.pipe.eval()
    return model, installation


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    model, installation = build_training_faithful_model(args)
    pipe = model.pipe
    controller = installation.controller

    support_by_environment = {}
    for support_index in _parse_sample_indices(args.support_indices):
        environment_id = int(dataset[support_index]["mu_index"])
        support_by_environment[environment_id] = int(support_index)

    queries_by_environment = defaultdict(list)
    for sample_index in _parse_sample_indices(args.sample_indices):
        environment_id = int(dataset[sample_index]["mu_index"])
        queries_by_environment[environment_id].append(int(sample_index))

    records = []
    for environment_id, query_indices in queries_by_environment.items():
        if environment_id not in support_by_environment:
            raise KeyError(f"No TTT support for environment {environment_id}.")
        support_index = support_by_environment[environment_id]
        support = dataset[support_index]
        controller.reset(batch_size=1)
        try:
            # This is the same observed-support flow-matching forward used by
            # training.  Its DiT tokens write the fast state after the support
            # is observed; the returned flow loss is diagnostic only and no
            # outer backward/optimizer step is performed at inference time.
            torch.manual_seed(int(args.seed) + environment_id)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed) + environment_id)
            with controller.support_write(differentiable=False):
                with torch.no_grad():
                    support_flow_loss = model(support)

            record = {
                "environment_id": environment_id,
                "support_index": support_index,
                "support_flow_loss": float(support_flow_loss.detach().float().cpu()),
                "state_norms": controller.state_norms(),
                "write_statistics": controller.write_statistics(),
                "updates_per_layer": int(args.ttt_write_token_budget)
                // int(args.ttt_inner_batch_size),
                "query_indices": query_indices,
            }
            records.append(record)
            print(
                f"[support_write] environment={environment_id} "
                f"support={support_index} loss={record['support_flow_loss']:.6f} "
                f"state_norms={record['state_norms']}",
                flush=True,
            )

            # The adapted state is frozen for the complete diffusion rollout of
            # every query.  Denoising updates video latents, never fast weights.
            with controller.query_read():
                with torch.no_grad():
                    for sample_index in query_indices:
                        sample = prepare_sample_for_rollout(
                            dataset[sample_index], sample_index, pipe, args
                        )
                        if args.skip_existing and Path(sample["output_path"]).is_file():
                            print(f"[skip] existing prediction {sample['output_path']}")
                            continue
                        predicted_path = _run_autoregressive(
                            pipe=pipe, sample=sample, args=args
                        )
                        print(
                            f"[done] sample_index={sample_index} "
                            f"environment={environment_id} output={predicted_path}",
                            flush=True,
                        )
        finally:
            controller.clear()

    protocol = {
        "method": "ttt_kqv",
        "implementation": "ttt_kvb_prequential6",
        "support_size": 1,
        "support_update": "training_flow_forward_then_fast_weight_write",
        "outer_loss_at_inference": False,
        "query_state": "frozen_read_only",
        "query_updates": "video_latent_denoising_only",
        "support_query_disjoint": True,
        "checkpoint": args.ttt_checkpoint_path,
        "records": records,
    }
    with (Path(args.output_path).parent / "protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, indent=2)


if __name__ == "__main__":
    main()
