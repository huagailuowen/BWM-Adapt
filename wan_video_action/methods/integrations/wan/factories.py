"""Config-driven process factories for stable Wan entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ...common.config import LoadedMethodConfig
from ...common.budget import FixedHardwareTimeBudget
from ...common.integration import IntegrationRequiredError
from ...common.runtime import CommandApplication


def _find_repo_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (
            candidate.joinpath("wan_video_action").is_dir()
            and candidate.joinpath("scripts").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Cannot resolve repository root from {config_path}.")


def _resolve_repo_path(root: Path, value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _expand(
    values: list[Any],
    *,
    gpu_count: int,
    python: str,
) -> list[str]:
    replacements = {
        "gpu_count": str(gpu_count),
        "python": python,
    }
    return [
        str(value).format_map(replacements)
        for value in values
    ]


def _build_application(
    loaded: LoadedMethodConfig,
    phase: str,
) -> CommandApplication:
    if phase not in {"train", "infer"}:
        raise ValueError(f"Unsupported phase: {phase}")

    data = loaded.data
    root = _find_repo_root(loaded.path)
    source_key = f"legacy_{phase}_config"
    source_value = data.get("source", {}).get(source_key)
    if not source_value:
        raise IntegrationRequiredError(
            f"{loaded.spec.slug} has no source.{source_key}. "
            "Assign a dedicated new experiment YAML before execution; the "
            "factory will not guess or reuse an unrelated legacy experiment."
        )
    source_config = _resolve_repo_path(
        root, str(source_value), f"source.{source_key}"
    )

    settings = data.get("runtime", {}).get(phase, {})
    entrypoint_value = settings.get("entrypoint")
    if not entrypoint_value:
        raise IntegrationRequiredError(
            f"runtime.{phase}.entrypoint is required for {loaded.spec.slug}."
        )
    entrypoint = _resolve_repo_path(
        root, str(entrypoint_value), f"runtime.{phase}.entrypoint"
    )

    budget = FixedHardwareTimeBudget.from_config(data) if phase == "train" else None
    gpu_count = budget.gpu_count if budget is not None else int(
        data.get("resources", {}).get("inference", {}).get("gpu_count", 1)
    )
    launcher = settings.get("launcher") or ["{python}"]
    command = _expand(
        list(launcher),
        gpu_count=gpu_count,
        python=sys.executable,
    )
    command.append(str(entrypoint))
    config_flag = settings.get("config_flag", "--config")
    if config_flag:
        command.extend((str(config_flag), str(source_config)))
    command.extend(
        _expand(
            list(settings.get("extra_args") or ()),
            gpu_count=gpu_count,
            python=sys.executable,
        )
    )

    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(root)
    )
    for key, value in (settings.get("environment") or {}).items():
        environment[str(key)] = str(value)
    experiment_id = str(data.get("experiment", {}).get("id"))
    environment["BWM_METHOD_CONFIG"] = str(loaded.path)
    environment["BWM_METHOD_EXPERIMENT_ID"] = experiment_id

    run_dir = None
    artifact_root = data.get("runtime", {}).get("artifact_root")
    if phase == "train" and artifact_root:
        root_path = Path(str(artifact_root)).expanduser()
        if not root_path.is_absolute():
            root_path = root / root_path
        seed = int(data.get("resources", {}).get("seed", 0))
        run_dir = root_path.resolve() / experiment_id / f"seed_{seed}"

    return CommandApplication(
        phase=phase,
        method_slug=loaded.spec.slug,
        command=command,
        cwd=root,
        environment=environment,
        budget=budget,
        run_dir=run_dir,
    )


def build_train_application(
    loaded: LoadedMethodConfig,
) -> CommandApplication:
    return _build_application(loaded, "train")


def build_infer_application(
    loaded: LoadedMethodConfig,
) -> CommandApplication:
    return _build_application(loaded, "infer")
