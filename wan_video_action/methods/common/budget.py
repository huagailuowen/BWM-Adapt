"""Fixed hardware-time budgets and lightweight GPU telemetry."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
from typing import Any, Mapping


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


@dataclass(frozen=True)
class FixedHardwareTimeBudget:
    protocol: str
    hardware_block_id: str
    gpu_count: int
    gpu_sku: str
    wall_clock_hours: float
    gpu_hours: float
    timer_boundary: str
    queue_time_included: bool
    staging_time_included: bool
    deadline_enforcement: str
    utilization_target_percent: float
    utilization_sample_seconds: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FixedHardwareTimeBudget":
        values = config.get("resources", {}).get("training", {})
        utilization = values.get("utilization", {})
        budget = cls(
            protocol=str(values.get("protocol", "")),
            hardware_block_id=str(values.get("hardware_block_id", "")),
            gpu_count=int(values.get("gpu_count", 0) or 0),
            gpu_sku=str(values.get("gpu_sku", "")),
            wall_clock_hours=float(values.get("wall_clock_hours", 0.0) or 0.0),
            gpu_hours=float(values.get("gpu_hours", 0.0) or 0.0),
            timer_boundary=str(values.get("timer_boundary", "")),
            queue_time_included=bool(values.get("queue_time_included", False)),
            staging_time_included=bool(values.get("staging_time_included", False)),
            deadline_enforcement=str(values.get("deadline_enforcement", "")),
            utilization_target_percent=float(
                utilization.get("target_percent", 0.0) or 0.0
            ),
            utilization_sample_seconds=float(
                utilization.get("sample_seconds", 0.0) or 0.0
            ),
        )
        budget.validate()
        return budget

    def validate(self) -> None:
        if self.protocol != "fixed_hardware_time":
            raise ValueError(
                "resources.training.protocol must be 'fixed_hardware_time'."
            )
        if not self.hardware_block_id:
            raise ValueError("resources.training.hardware_block_id is required.")
        if self.gpu_count <= 0:
            raise ValueError("resources.training.gpu_count must be positive.")
        if not self.gpu_sku or "_or_" in self.gpu_sku.lower():
            raise ValueError(
                "resources.training.gpu_sku must name one exact SKU, not alternatives."
            )
        if self.wall_clock_hours <= 0:
            raise ValueError("resources.training.wall_clock_hours must be positive.")
        expected_gpu_hours = self.gpu_count * self.wall_clock_hours
        if abs(self.gpu_hours - expected_gpu_hours) > 1e-6:
            raise ValueError(
                "resources.training.gpu_hours must equal gpu_count * wall_clock_hours "
                f"({expected_gpu_hours}), got {self.gpu_hours}."
            )
        if self.timer_boundary != "training_process_start_after_staging":
            raise ValueError(
                "The fixed Event80 timer boundary must be "
                "'training_process_start_after_staging'."
            )
        if self.deadline_enforcement not in {"scheduler", "runner"}:
            raise ValueError(
                "resources.training.deadline_enforcement must be scheduler or runner."
            )
        if not 0 < self.utilization_target_percent <= 100:
            raise ValueError("GPU utilization target must be in (0, 100].")
        if self.utilization_sample_seconds <= 0:
            raise ValueError("GPU utilization sampling interval must be positive.")

    @property
    def wall_clock_seconds(self) -> float:
        return self.wall_clock_hours * 3600.0

    def signature(self) -> tuple[Any, ...]:
        return (
            self.protocol,
            self.hardware_block_id,
            self.gpu_count,
            self.gpu_sku,
            self.wall_clock_hours,
            self.gpu_hours,
            self.timer_boundary,
            self.queue_time_included,
            self.staging_time_included,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUTelemetryRecorder:
    """Sample allocated GPU busy-time counters without adding dependencies."""

    def __init__(
        self,
        *,
        destination: str | Path,
        budget: FixedHardwareTimeBudget,
        environment: Mapping[str, str],
    ) -> None:
        self.destination = Path(destination)
        self.budget = budget
        self.environment = dict(environment)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.budget.utilization_sample_seconds + 10.0)
        valid = [sample for sample in self._samples if "gpus" in sample]
        utilizations = [
            gpu["utilization_gpu_percent"]
            for sample in valid
            for gpu in sample["gpus"]
        ]
        memory_values = [
            gpu["memory_used_mib"]
            for sample in valid
            for gpu in sample["gpus"]
        ]
        names = sorted(
            {
                gpu["name"]
                for sample in valid
                for gpu in sample["gpus"]
            }
        )
        mean_utilization = statistics.fmean(utilizations) if utilizations else None
        observed_counts = [len(sample["gpus"]) for sample in valid]
        return {
            "sample_count": len(valid),
            "error_count": len(self._samples) - len(valid),
            "gpu_names": names,
            "observed_gpu_counts": sorted(set(observed_counts)),
            "expected_gpu_count": self.budget.gpu_count,
            "gpu_count_match": bool(observed_counts)
            and all(value == self.budget.gpu_count for value in observed_counts),
            "sku_match": bool(names)
            and all(self.budget.gpu_sku.lower() in name.lower() for name in names),
            "mean_gpu_utilization_percent": mean_utilization,
            "peak_memory_used_mib": max(memory_values) if memory_values else None,
            "utilization_target_percent": self.budget.utilization_target_percent,
            "utilization_target_met": mean_utilization is not None
            and mean_utilization >= self.budget.utilization_target_percent,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._collect()
            self._samples.append(sample)
            with self.destination.open("a", encoding="utf-8") as handle:
                json.dump(sample, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
            self._stop.wait(self.budget.utilization_sample_seconds)

    def _collect(self) -> dict[str, Any]:
        command = ["nvidia-smi"]
        visible = self.environment.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            command.extend(("--id", visible))
        command.extend(
            (
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            )
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            completed = subprocess.run(
                command,
                env=self.environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"timestamp_utc": timestamp, "error": str(exc)}
        if completed.returncode != 0:
            return {
                "timestamp_utc": timestamp,
                "error": completed.stderr.strip() or f"nvidia-smi rc={completed.returncode}",
            }
        gpus = []
        for row in csv.reader(completed.stdout.splitlines()):
            if len(row) != 6:
                continue
            try:
                gpus.append(
                    {
                        "index": int(row[0].strip()),
                        "uuid": row[1].strip(),
                        "name": row[2].strip(),
                        "utilization_gpu_percent": float(row[3].strip()),
                        "memory_used_mib": float(row[4].strip()),
                        "memory_total_mib": float(row[5].strip()),
                    }
                )
            except ValueError:
                continue
        return {"timestamp_utc": timestamp, "gpus": gpus}
