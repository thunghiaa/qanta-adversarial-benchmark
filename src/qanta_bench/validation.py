from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qanta_bench.registry import Model, load_registry


@dataclass(frozen=True)
class ValidationReport:
    jsonl_files: int
    jsonl_rows: int
    unknown_output_models: tuple[str, ...]


def _model_from_output_name(filename: str) -> str | None:
    stem = filename.removesuffix(".jsonl")
    fields = stem.split("__")
    if len(fields) >= 5 and fields[1].startswith("packet") and fields[2] == "local":
        return f"local/{fields[-1]}"
    if len(fields) >= 5 and fields[1] == "hf":
        return f"{fields[-2]}/{fields[-1]}"
    if len(fields) >= 4 and fields[1].isdigit() is False:
        return f"{fields[-2]}/{fields[-1]}"
    return None


def validate_jsonl(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if "qid" not in row:
                raise ValueError(f"{path}:{line_number}: missing qid")
            rows += 1
    return rows


def validate_repository(root: Path) -> ValidationReport:
    models: list[Model] = load_registry(root / "configs" / "models.json")
    known = {model.id.casefold() for model in models}
    files = sorted((root / "results").rglob("*.jsonl"))
    row_count = sum(validate_jsonl(path) for path in files)
    unknown: set[str] = set()
    for path in files:
        model = _model_from_output_name(path.name)
        if model and model.casefold() not in known:
            unknown.add(model)
    return ValidationReport(len(files), row_count, tuple(sorted(unknown)))
