"""Support-write/query-read execution for the TTT-KVB baseline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import torch

from .controller import TTTKVBController


@contextmanager
def _checkpoint_free_support(model: Any) -> Iterator[None]:
    """Disable support checkpoint replay while preserving query checkpointing."""

    changed = []
    for name in ("use_gradient_checkpointing", "use_gradient_checkpointing_offload"):
        if hasattr(model, name):
            changed.append((name, getattr(model, name)))
            setattr(model, name, False)
    try:
        yield
    finally:
        for name, value in changed:
            setattr(model, name, value)


@dataclass(frozen=True)
class TTTKVBStepResult:
    loss: torch.Tensor
    query_losses: tuple[float, ...]
    write_statistics: dict[str, list[dict[str, float]]]
    state_norms: dict[str, float]


class TTTKVBProtocolRunner:
    """Runs one differentiable adaptation episode and backward pass.

    Backward is intentionally executed while ``QUERY_READ`` is active.  This
    is required because non-reentrant activation checkpointing recomputes the
    query forward and must observe the same read-only fast state.
    """

    def __init__(self, model: Any, controller: TTTKVBController) -> None:
        self.model = model
        self.controller = controller

    def outer_step(
        self,
        support_batches: Iterable[Any],
        query_batches: Iterable[Any],
        support_forward: Callable[[Any], Any],
        query_loss_forward: Callable[[Any], torch.Tensor],
        backward: Callable[[torch.Tensor], None],
        environment_batch_size: int,
        query_state_indices: torch.Tensor | None = None,
        differentiable_writes: bool = True,
    ) -> TTTKVBStepResult:
        self.controller.reset(environment_batch_size)
        try:
            # The frozen support backbone is inference-only.  TTTMLPMemory
            # locally re-enables autograd for K/V binding and the meta-gradient.
            with _checkpoint_free_support(self.model):
                with self.controller.support_write(differentiable=differentiable_writes):
                    for batch in support_batches:
                        with torch.no_grad():
                            support_forward(batch)

            with self.controller.query_read(state_indices=query_state_indices):
                losses = tuple(query_loss_forward(batch) for batch in query_batches)
                if not losses:
                    raise ValueError("At least one disjoint query batch is required")
                loss = torch.stack([value.float() for value in losses]).mean()
                backward(loss)

            return TTTKVBStepResult(
                loss=loss.detach(),
                query_losses=tuple(float(value.detach().float().cpu()) for value in losses),
                write_statistics=self.controller.write_statistics(),
                state_norms=self.controller.state_norms(),
            )
        finally:
            self.controller.clear()
