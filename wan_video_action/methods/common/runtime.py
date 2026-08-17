"""Runtime bundles supplied by future method-local Wan integration factories."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from ..protocol import MethodRunner, SupportQueryEpisode
from .budget import FixedHardwareTimeBudget, GPUTelemetryRecorder, write_json_atomic


TrainingCallback = Callable[[int, Mapping[str, Any]], None]
PredictionWriter = Callable[[SupportQueryEpisode[Any, Any], Any], None]


@dataclass(frozen=True)
class ApplicationResult:
    phase: str
    method_slug: str
    return_code: int
    command: tuple[str, ...]
    elapsed_seconds: float
    actual_gpu_hours: float | None
    resource_summary_path: str | None


class CommandApplication:
    def __init__(
        self,
        *,
        phase: str,
        method_slug: str,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        budget: FixedHardwareTimeBudget | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.phase = phase
        self.method_slug = method_slug
        self.command = tuple(str(item) for item in command)
        self.cwd = cwd
        self.environment = dict(environment)
        self.budget = budget
        self.run_dir = run_dir

    def describe(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "method_slug": self.method_slug,
            "command": list(self.command),
            "cwd": str(self.cwd),
            "budget": self.budget.as_dict() if self.budget else None,
            "run_dir": str(self.run_dir) if self.run_dir else None,
        }

    def run(self) -> ApplicationResult:
        import subprocess

        started_wall = datetime.now(timezone.utc)
        started = time.monotonic()
        recorder = None
        if self.budget is not None and self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                self.run_dir / "resource_run.json",
                {
                    **self.describe(),
                    "status": "running",
                    "started_at_utc": started_wall.isoformat(),
                },
            )
            recorder = GPUTelemetryRecorder(
                destination=self.run_dir / "gpu_telemetry.jsonl",
                budget=self.budget,
                environment=self.environment,
            )
            recorder.start()

        completed = subprocess.run(
            self.command,
            cwd=self.cwd,
            env=self.environment,
            check=False,
        )
        elapsed_seconds = time.monotonic() - started
        telemetry = recorder.stop() if recorder is not None else None
        actual_gpu_hours = (
            self.budget.gpu_count * elapsed_seconds / 3600.0
            if self.budget is not None
            else None
        )
        resource_summary_path = None
        if self.budget is not None and self.run_dir is not None:
            resource_summary_path = str(self.run_dir / "resource_run.json")
            write_json_atomic(
                resource_summary_path,
                {
                    **self.describe(),
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "started_at_utc": started_wall.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "return_code": completed.returncode,
                    "elapsed_seconds": elapsed_seconds,
                    "actual_gpu_hours": actual_gpu_hours,
                    "within_declared_wall_clock": (
                        elapsed_seconds <= self.budget.wall_clock_seconds
                    ),
                    "gpu_telemetry": telemetry,
                },
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.phase} command failed with return code "
                f"{completed.returncode}: {self.command}"
            )
        return ApplicationResult(
            phase=self.phase,
            method_slug=self.method_slug,
            return_code=completed.returncode,
            command=self.command,
            elapsed_seconds=elapsed_seconds,
            actual_gpu_hours=actual_gpu_hours,
            resource_summary_path=resource_summary_path,
        )


@dataclass(frozen=True)
class TrainingBundle:
    runner: MethodRunner[Any, Any, Any, Any]
    batches: Iterable[Any]
    max_steps: int
    callbacks: Sequence[TrainingCallback] = ()

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("TrainingBundle.max_steps must be positive.")


@dataclass(frozen=True)
class InferenceBundle:
    runner: MethodRunner[Any, Any, Any, Any]
    episodes: Iterable[SupportQueryEpisode[Any, Any]]
    prediction_writer: PredictionWriter
