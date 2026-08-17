"""Planning and consistency checks for a method comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .budget import FixedHardwareTimeBudget
from .config import LoadedMethodConfig, load_method_config


@dataclass(frozen=True)
class PlannedExperiment:
    category: str
    config: LoadedMethodConfig

    def as_dict(self) -> dict[str, Any]:
        data = self.config.data
        execution = data.get("execution", {})
        training = execution.get("training", {})
        inference = execution.get("inference", {})
        config_path = str(self.config.path)
        return {
            "category": self.category,
            "experiment_id": data["experiment"]["id"],
            "method_slug": self.config.spec.slug,
            "config_path": config_path,
            "implementation_status": data["runtime"]["implementation_status"],
            "training": training,
            "inference": inference,
            "dependencies": execution.get("dependencies", []),
            "training_budget": FixedHardwareTimeBudget.from_config(data).as_dict(),
            "commands": {
                "validate": [
                    "python",
                    "scripts/methods/train_method.py",
                    "--config",
                    config_path,
                    "--dry-run",
                ],
                "train_plan": [
                    "python",
                    "scripts/methods/train_method.py",
                    "--config",
                    config_path,
                    "--plan",
                ]
                if training.get("ready") and training.get("mode") == "independent"
                else None,
                "infer_plan": [
                    "python",
                    "scripts/methods/infer_method.py",
                    "--config",
                    config_path,
                    "--plan",
                ]
                if inference.get("ready")
                else None,
            },
        }


@dataclass(frozen=True)
class MethodMatrixPlan:
    path: Path
    benchmark: str
    evaluation_id: str
    experiments: tuple[PlannedExperiment, ...]
    evaluation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        independent = [
            item
            for item in self.experiments
            if item.config.data["execution"]["training"]["mode"] == "independent"
        ]
        budget = FixedHardwareTimeBudget.from_config(independent[0].config.data)
        return {
            "matrix_path": str(self.path),
            "benchmark": self.benchmark,
            "evaluation_id": self.evaluation_id,
            "budget_signature": budget.as_dict(),
            "experiments": [item.as_dict() for item in self.experiments],
            "evaluation": self.evaluation,
        }


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def load_method_matrix(path: str | Path) -> MethodMatrixPlan:
    matrix_path = Path(path).expanduser().resolve()
    with matrix_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported method matrix schema.")
    if payload.get("do_not_submit") is not True:
        raise ValueError("Planning matrices must retain do_not_submit: true.")

    experiments = []
    seen_ids = set()
    for category, values in payload.get("experiments", {}).items():
        for value in values:
            loaded = load_method_config(_resolve(matrix_path.parent, str(value)))
            experiment_id = loaded.data["experiment"]["id"]
            if experiment_id in seen_ids:
                raise ValueError(f"Duplicate experiment.id in matrix: {experiment_id}")
            seen_ids.add(experiment_id)
            experiments.append(PlannedExperiment(str(category), loaded))
    if not experiments:
        raise ValueError("Method matrix contains no experiments.")

    independent = [
        item
        for item in experiments
        if item.config.data["execution"]["training"]["mode"] == "independent"
    ]
    if not independent:
        raise ValueError("Method matrix has no independently trained method.")
    reference = FixedHardwareTimeBudget.from_config(independent[0].config.data)
    for item in independent[1:]:
        candidate = FixedHardwareTimeBudget.from_config(item.config.data)
        if candidate.signature() != reference.signature():
            raise ValueError(
                f"Resource budget mismatch for {item.config.data['experiment']['id']}."
            )

    evaluation = dict(payload.get("evaluation", {}))
    for key in ("global_config", "object_centric_config"):
        if key in evaluation:
            evaluation[key] = str(_resolve(matrix_path.parent, evaluation[key]))
    return MethodMatrixPlan(
        path=matrix_path,
        benchmark=str(payload.get("benchmark")),
        evaluation_id=str(payload.get("evaluation_id")),
        experiments=tuple(experiments),
        evaluation=evaluation,
    )
