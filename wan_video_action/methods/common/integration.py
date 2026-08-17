"""Explicit boundary between the additive framework and existing Wan code."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


class IntegrationRequiredError(RuntimeError):
    """Raised when a method needs an integration point that is not approved."""


def load_symbol(reference: str) -> Any:
    if ":" not in reference:
        raise ValueError(
            f"Integration reference must use 'module:attribute', got {reference!r}"
        )
    module_name, attribute = reference.split(":", 1)
    module = import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"Integration symbol {attribute!r} is not defined by {module_name!r}."
        ) from exc


def resolve_factory(config: dict[str, Any], phase: str) -> Callable[..., Any]:
    if phase not in {"train", "infer"}:
        raise ValueError(f"Unsupported runtime phase: {phase}")
    reference = config.get("runtime", {}).get(f"{phase}_factory")
    if not reference:
        raise IntegrationRequiredError(
            f"The {phase} integration for method "
            f"{config.get('method', {}).get('slug')!r} is intentionally unset. "
            "A method-local Wan adapter must be approved before execution; "
            "dry-run remains available without changing existing scripts."
        )
    factory = load_symbol(reference)
    if not callable(factory):
        raise TypeError(f"Integration factory is not callable: {reference}")
    return factory
