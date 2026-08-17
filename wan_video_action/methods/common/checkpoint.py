"""Atomic metadata sidecars for method-specific checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MethodCheckpointManifest:
    schema_version: int
    method_slug: str
    step: int
    model_checkpoint: str
    method_state: str | None
    config_path: str
    metadata: dict[str, Any]


def write_manifest_atomic(
    path: str | Path, manifest: MethodCheckpointManifest
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
