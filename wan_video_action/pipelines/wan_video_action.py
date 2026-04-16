import torch
from einops import rearrange, repeat
from typing import Optional, Union

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.core.vram.initialization import skip_model_initialization
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

from ..models.wan_video_action_encoder import WanVideoActionEncoder


def load_checkpoint_weights(pipe, ckpt_path: str):
    print(f"Loading training weights from checkpoint: {ckpt_path}")
    state_dict = load_state_dict(ckpt_path, torch_dtype=pipe.torch_dtype, device="cpu")

    states = {"dit": {}, "action_encoder": {}}
    ignored_count = 0

    for k, v in state_dict.items():
        clean_k = k.removeprefix("pipe.")

        if clean_k.startswith("action_encoder."):
            states["action_encoder"][clean_k.removeprefix("action_encoder.")] = v
        elif clean_k.startswith("dit."):
            states["dit"][clean_k.removeprefix("dit.")] = v
        elif k.startswith("pipe."):
            ignored_count += 1
        else:
            raise KeyError(f"Unknown bare key detected: {k}")

    for name, st in states.items():
        if not st:
            continue
        module = getattr(pipe, name, None)
        if module:
            res = module.load_state_dict(st, strict=False, assign=(name == "action_encoder"))
            print(f"  - Loaded {name} keys: {len(st)} (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")
        else:
            print(f"  - Warning: {name} weights found ({len(st)} keys), but pipeline.{name} is None")

    if ignored_count:
        print(f"  - Ignored {ignored_count} keys with unsupported `pipe.*` prefixes")


def build_wan_video_action_pipeline(
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = get_device_type(),
    model_configs: list[ModelConfig] = [],
    tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    audio_processor_config: ModelConfig = None,
    redirect_common_files: bool = True,
    use_usp: bool = False,
    vram_limit: float = None,
    args = None,
):
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        audio_processor_config=audio_processor_config,
        redirect_common_files=redirect_common_files,
        use_usp=use_usp,
        vram_limit=vram_limit
    )

    pipe.args = args
    pipe.action_encoder = None

    if args.action_mode != "none":
        dim = getattr(pipe.dit, "dim", 1536) if pipe.dit is not None else 1536
        has_ckpt = getattr(args, "ckpt_path", None) is not None

        if has_ckpt:
            with skip_model_initialization():
                pipe.action_encoder = WanVideoActionEncoder(
                    action_dim=args.action_dim,
                    dim=dim,
                    num_action_per_chunk=81 if args.action_mode == "adaln" else None,
                )
            load_checkpoint_weights(pipe, args.ckpt_path)
        else:
            pipe.action_encoder = WanVideoActionEncoder(
                action_dim=args.action_dim,
                dim=dim,
                num_action_per_chunk=81 if args.action_mode == "adaln" else None,
            )

        pipe.action_encoder = pipe.action_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.action_encoder.eval()

    if args.text_mode == "none":
        pipe.units = [u for u in pipe.units if u.__class__.__name__ != "WanVideoUnit_PromptEmbedder"]
        
    if args.action_mode != "none":
        pipe.units.append(WanVideoUnit_ActionEmbedder())

    pipe.model_fn = model_fn_wan_video_action

    return pipe


class WanVideoUnit_ActionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("action", "num_frames"),
            output_params=("action_emb",),
            onload_model_names=("action_encoder",)
        )

    def process(self, pipe, action, num_frames):
        if action is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        action = torch.as_tensor(action, device=pipe.device, dtype=pipe.torch_dtype)

        if pipe.args.action_mode == "noise":
            length = (num_frames - 1) // 4 + 1
            action = torch.concat(
                [torch.repeat_interleave(action[:, 0:1], repeats=4, dim=1), action[:, 1:]],
                dim=1,
            )
            action = action.contiguous().view(action.shape[0], length, 4, action.shape[-1]).mean(dim=2)

        if pipe.args.action_mode == "adaln":
            action = rearrange(action, "b f d -> b (f d)").contiguous()

        action_emb = pipe.action_encoder(action)
        return {"action_emb": action_emb}


def model_fn_wan_video_action(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_injection_mode: str = "none",
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))

    if action_injection_mode == "adaln" and action_emb is not None:
        t = t + action_emb

    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if context is not None:
        context = dit.text_embedding(context)

    x = latents

    if y is not None and dit.has_image_input and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    if clip_feature is not None and dit.has_image_input and dit.require_clip_embedding:
        clip_embedding = dit.img_emb(clip_feature)
        if context is None:
            context = clip_embedding
        else:
            context = torch.cat([clip_embedding, context], dim=1)

    x = dit.patchify(x)
    f, h, w = x.shape[2:]

    if action_injection_mode == "noise" and action_emb is not None:
        action_emb = rearrange(action_emb, "b f d -> b d f 1 1")
        action_emb = repeat(action_emb, "b d f 1 1 -> b d f h w", h=h, w=w)
        x = x + action_emb

    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))

    return x