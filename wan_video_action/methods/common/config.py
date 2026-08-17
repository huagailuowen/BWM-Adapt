"""Configuration composition and validation for additive method runners."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..protocol import MethodSpec
from ..registry import get_method_spec
from .budget import FixedHardwareTimeBudget


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive config inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")

    parents = payload.pop("extends", ())
    legacy_parent = payload.pop("base_config", None)
    if parents and legacy_parent:
        raise ValueError(f"Use extends, not both extends and base_config: {path}")
    if legacy_parent:
        parents = legacy_parent
    if isinstance(parents, str):
        parents = (parents,)
    if not isinstance(parents, (list, tuple)):
        raise TypeError(f"extends must be a path or list of paths: {path}")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, _load_yaml(parent_path, (*stack, path)))
    return _deep_merge(merged, payload)


@dataclass(frozen=True)
class LoadedMethodConfig:
    path: Path
    data: dict[str, Any]
    spec: MethodSpec

    def summary(self) -> dict[str, Any]:
        runtime = self.data.get("runtime", {})
        training = self.data.get("training", {})
        grouped = training.get("grouped_batch", {})
        execution = self.data.get("execution", {})
        budget = FixedHardwareTimeBudget.from_config(self.data)
        return {
            "config_path": str(self.path),
            "experiment_id": self.data.get("experiment", {}).get("id"),
            "method": self.spec.as_dict(),
            "training_budget": budget.as_dict(),
            "grouped_batch": grouped,
            "support_sizes": self.data.get("protocol", {}).get(
                "support_sizes", ()
            ),
            "training_execution": execution.get("training", {}),
            "inference_execution": execution.get("inference", {}),
            "implementation_status": runtime.get("implementation_status"),
            "train_integration_ready": bool(runtime.get("train_factory")),
            "infer_integration_ready": bool(runtime.get("infer_factory")),
        }


def _positive_int(value: Any, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")


def _validate(data: dict[str, Any], spec: MethodSpec) -> None:
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported method config schema: {schema_version!r}")

    experiment_id = data.get("experiment", {}).get("id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment.id is required.")

    method = data.get("method", {})
    configured_family = method.get("family")
    if configured_family and configured_family != spec.family.value:
        raise ValueError(
            f"{spec.slug} belongs to family={spec.family.value!r}, "
            f"got {configured_family!r}."
        )

    protocol = data.get("protocol", {})
    policy = protocol.get("query_state_policy")
    if policy != spec.query_state_policy.value:
        raise ValueError(
            f"{spec.slug} requires query_state_policy="
            f"{spec.query_state_policy.value!r}, got {policy!r}"
        )
    if protocol.get("disjoint_support_query") is not True:
        raise ValueError("disjoint_support_query must be true.")

    support_sizes = protocol.get("support_sizes", ())
    if not support_sizes:
        raise ValueError("At least one test-time support size K is required.")
    for support_size in support_sizes:
        _positive_int(support_size, "protocol.support_sizes item")

    budget = FixedHardwareTimeBudget.from_config(data)

    training = data.get("training", {})
    grouped = training.get("grouped_batch", {})
    for key in (
        "environments_per_rank",
        "trajectories_per_environment",
        "gpu_count",
    ):
        _positive_int(grouped.get(key), f"training.grouped_batch.{key}")
    if grouped["gpu_count"] != budget.gpu_count:
        raise ValueError(
            "training.grouped_batch.gpu_count must match the fixed resource budget."
        )
    global_environments = grouped.get("global_environments")
    expected_environments = grouped["environments_per_rank"] * grouped["gpu_count"]
    if global_environments != expected_environments:
        raise ValueError(
            "training.grouped_batch.global_environments must equal "
            "environments_per_rank * gpu_count."
        )

    matching = training.get("budget_matching", {})
    for key in (
        "match_optimizer_steps",
        "match_generator_gradient_evaluations",
        "match_clip_exposure",
        "match_flops",
        "match_environment_stream",
    ):
        if matching.get(key) is not False:
            raise ValueError(f"training.budget_matching.{key} must be false.")
    if matching.get("match_training_environment_set") is not True:
        raise ValueError(
            "training.budget_matching.match_training_environment_set must be true."
        )

    adaptation = data.get("adaptation", {})
    if adaptation.get("enabled") and adaptation.get("mode") == "staged_optimizer":
        stages = adaptation.get("stages", ())
        if not stages:
            raise ValueError("Enabled adaptation requires at least one LR stage.")
        for index, stage in enumerate(stages):
            _positive_int(stage.get("steps"), f"adaptation.stages[{index}].steps")
            learning_rate = stage.get("learning_rate")
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise ValueError(
                    f"adaptation.stages[{index}].learning_rate must be positive."
                )

    execution = data.get("execution", {})
    training_mode = execution.get("training", {}).get("mode")
    if training_mode not in {"independent", "reuse", "not_applicable"}:
        raise ValueError(
            "execution.training.mode must be independent, reuse, or not_applicable."
        )
    if not data.get("runtime", {}).get("implementation_status"):
        raise ValueError("runtime.implementation_status is required.")


def load_method_config(path: str | Path) -> LoadedMethodConfig:
    config_path = Path(path).expanduser().resolve()
    data = _load_yaml(config_path)
    method = data.get("method", {})
    slug = method.get("slug")
    if not slug:
        raise ValueError(f"method.slug is required: {config_path}")
    spec = get_method_spec(slug)
    _validate(data, spec)
    return LoadedMethodConfig(path=config_path, data=data, spec=spec)
