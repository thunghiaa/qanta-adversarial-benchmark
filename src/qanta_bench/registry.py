"""Canonical model-cohort registry.

Analysis code must use this registry instead of inferring provenance from a
username, timestamp, or output filename.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_COHORTS = frozenset({"qanta_submitted", "additional_benchmark"})
VALID_TASKS = frozenset({"tossup", "bonus"})


@dataclass(frozen=True)
class Model:
    id: str
    cohort: str
    tasks: tuple[str, ...]
    kind: str = "submission"
    status: str = "complete"
    artifact: str | None = None


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "models.json"


def load_registry(path: str | Path | None = None) -> list[Model]:
    registry_path = Path(path) if path else default_registry_path()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    models = [
        Model(
            id=row["id"],
            cohort=row["cohort"],
            tasks=tuple(row["tasks"]),
            kind=row.get("kind", "submission"),
            status=row.get("status", "complete"),
            artifact=row.get("artifact"),
        )
        for row in payload["models"]
    ]
    validate_registry(models)
    return models


def validate_registry(models: list[Model]) -> None:
    seen: set[str] = set()
    for model in models:
        if model.id in seen:
            raise ValueError(f"Duplicate model id: {model.id}")
        seen.add(model.id)
        if "/" not in model.id:
            raise ValueError(f"Model id must be owner/name: {model.id}")
        if model.cohort not in VALID_COHORTS:
            raise ValueError(f"Unknown cohort for {model.id}: {model.cohort}")
        if not model.tasks or not set(model.tasks) <= VALID_TASKS:
            raise ValueError(f"Invalid tasks for {model.id}: {model.tasks}")


def model_index(path: str | Path | None = None) -> dict[str, Model]:
    return {model.id.casefold(): model for model in load_registry(path)}


def models_for(cohort: str, task: str, path: str | Path | None = None) -> list[Model]:
    if cohort not in VALID_COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}")
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}")
    return [m for m in load_registry(path) if m.cohort == cohort and task in m.tasks]
