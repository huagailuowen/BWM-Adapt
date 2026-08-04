#!/usr/bin/env python3
from __future__ import annotations

import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan_video_action.counterfactual_bridge import (
    nonlinear_bridge_alpha,
    sample_nonlinear_bridge_condition,
)


class TinyBridge(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.global_context = nn.Parameter(torch.zeros(dim))
        self.endpoints = nn.Parameter(torch.randn(4, dim) * 0.05, requires_grad=False)
        self.backbone = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def predict(self, noisy: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat((noisy, context.expand_as(noisy)), dim=-1))


def set_phase(model: TinyBridge, warmup: bool) -> None:
    model.global_context.requires_grad_(warmup)
    model.endpoints.requires_grad_(False)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not warmup)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Dynamic smoke must run on a GPU compute node.")
    device = torch.device("cuda")
    torch.manual_seed(9)
    model = TinyBridge().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = random.Random(9)
    stages = (
        ("global_real", None, False),
        ("endpoint_real", "endpoint", False),
        ("near_real", "near_global", False),
        ("near_counterfactual", "near_global", True),
        ("interior_real", "interior", False),
        ("interior_counterfactual", "interior", True),
    )
    reports = []
    for name, kind, counterfactual in stages:
        warmup = kind is None
        set_phase(model, warmup)
        optimizer.zero_grad(set_to_none=True)
        if warmup:
            context = model.global_context
            alpha = 0.0
        else:
            condition = sample_nonlinear_bridge_condition(
                rng,
                forced_kind=kind,
                curve_power=5.0,
            )
            alpha = nonlinear_bridge_alpha(condition.position, 5.0)
            context = (
                (1.0 - alpha) * model.global_context.detach()
                + alpha * model.endpoints[condition.target_index]
            )
        target = torch.randn(16, 32, device=device)
        noise = torch.randn_like(target)
        sigma = 0.8 if counterfactual else 0.45
        if counterfactual:
            source = target.roll(shifts=1, dims=0)
            noisy = (1.0 - sigma) * source + sigma * noise
            velocity_target = (noisy - target) / sigma
        else:
            noisy = (1.0 - sigma) * target + sigma * noise
            velocity_target = noise - target
        prediction = model.predict(noisy, context)
        loss = F.mse_loss(prediction, velocity_target)
        loss.backward()
        global_has_grad = model.global_context.grad is not None
        model_has_grad = any(
            parameter.grad is not None for parameter in model.backbone.parameters()
        )
        if warmup and (not global_has_grad or model_has_grad):
            raise AssertionError("Global warmup freeze policy failed.")
        if not warmup and (global_has_grad or not model_has_grad):
            raise AssertionError("Post-warmup freeze policy failed.")
        optimizer.step()
        reports.append(
            {
                "stage": name,
                "counterfactual": counterfactual,
                "alpha": alpha,
                "loss": float(loss.detach().cpu()),
                "global_grad": global_has_grad,
                "model_grad": model_has_grad,
            }
        )
    print(json.dumps({"status": "ok", "stages": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
