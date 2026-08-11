#!/usr/bin/env python3
"""Non-destructively create PEDANT-scored copies of registered model outputs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from qanta_bench.registry import models_for


def clean_text(value: object) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = text.replace("_", " ").replace("*", " ").replace("`", " ")
    return re.sub(r"\s+", " ", text.replace("–", "-").replace("—", "-")).strip()


def split_alternates(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                value = parsed
        except (ValueError, SyntaxError):
            pass
    entries = value if isinstance(value, list) else [value]
    answers: list[str] = []
    for entry in entries:
        for piece in re.split(r"\bor\b|,|;|/", clean_text(entry)):
            if piece.strip():
                answers.append(piece.strip())
    return answers


def _suffix(qid: str) -> str:
    return qid.split("-")[-1]


def load_tossup_gold(path: Path) -> dict[str, dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    gold: dict[str, dict[str, object]] = {}
    for row in rows:
        references = split_alternates(row.get("clean_answers"))
        primary = clean_text(row.get("answer_primary") or row.get("answer") or "")
        if primary and primary not in references:
            references.append(primary)
        gold[_suffix(row["qid"])] = {
            "references": references,
            "question": row.get("question", ""),
        }
    return gold


def load_bonus_gold(path: Path) -> dict[str, list[dict[str, object]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    gold: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        parts: list[dict[str, object]] = []
        for part in row.get("parts", []):
            references = split_alternates(part.get("clean_answers"))
            primary = clean_text(part.get("answer", ""))
            if primary and primary not in references:
                references.append(primary)
            parts.append(
                {
                    "references": references,
                    "question": f"{row.get('leadin', '')} {part.get('question', '')}".strip(),
                }
            )
        gold[_suffix(row["qid"])] = parts
    return gold


class Scorer:
    def __init__(self, threshold: float) -> None:
        try:
            from qa_metrics.pedant import PEDANT
        except ImportError as error:
            raise SystemExit("Install PEDANT support with: pip install -e '.[pedant]'") from error
        self.pedant = PEDANT()
        self.threshold = threshold
        self.cache: dict[tuple[object, ...], int] = {}

    def correct(
        self,
        guess: object,
        references: list[str],
        question: str,
        cache_key: tuple[object, ...],
    ) -> int:
        cleaned = clean_text(guess)
        if not cleaned or cleaned.upper() == "ERROR":
            return 0
        key = (*cache_key, cleaned.casefold())
        if key not in self.cache:
            self.cache[key] = int(
                any(
                    self.pedant.get_score(reference, cleaned, question) > self.threshold
                    for reference in references
                )
            )
        return self.cache[key]


def rescore_tossup(
    source: Path, destination: Path, gold: dict[str, dict[str, object]], scorer: Scorer
) -> int:
    output: list[str] = []
    changed = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            nano = _suffix(row["qid"])
            answer = gold.get(nano)
            if answer:
                for run in row.get("run_outputs", []):
                    previous = int(str(run.get("correct")) in {"1", "True", "true"})
                    current = scorer.correct(
                        run.get("guess", ""),
                        answer["references"],
                        answer["question"],
                        (nano,),
                    )
                    run["correct"] = current
                    changed += int(previous != current)
            output.append(json.dumps(row, ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return changed


def rescore_bonus(
    source: Path,
    destination: Path,
    gold: dict[str, list[dict[str, object]]],
    scorer: Scorer,
) -> int:
    output: list[str] = []
    changed = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            nano = _suffix(row["qid"])
            gold_parts = gold.get(nano, [])
            for index, part in enumerate(row.get("part_outputs", [])):
                if index >= len(gold_parts):
                    continue
                previous = int(str(part.get("correct")) in {"1", "True", "true"})
                answer = gold_parts[index]
                current = scorer.correct(
                    part.get("guess", ""),
                    answer["references"],
                    answer["question"],
                    (nano, index),
                )
                part["correct"] = current
                changed += int(previous != current)
            output.append(json.dumps(row, ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        choices=("qanta_submitted", "additional_benchmark"),
        required=True,
    )
    parser.add_argument("--task", choices=("tossup", "bonus"), required=True)
    parser.add_argument("--packets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    cohort_dir = {
        "qanta_submitted": "qanta_submitted",
        "additional_benchmark": "additional_benchmarks",
    }[args.cohort]
    allowed = {
        model.id
        for model in models_for(
            args.cohort, args.task, args.root / "configs" / "models.json"
        )
    }
    scorer = Scorer(args.threshold)
    raw_name = args.task.capitalize()
    pedant_name = f"{raw_name}_pedant"
    files_written = changes = 0

    for packet in args.packets:
        packet_dirs = sorted((args.root / "results" / cohort_dir).glob(f"Packet {packet} -*"))
        for packet_dir in packet_dirs:
            gold_path = (
                args.root
                / "data"
                / "packets"
                / f"packet{packet}_final"
                / args.task
                / "data.json"
            )
            gold = (
                load_tossup_gold(gold_path)
                if args.task == "tossup"
                else load_bonus_gold(gold_path)
            )
            for source in sorted((packet_dir / raw_name).glob("*.jsonl")):
                if _model_from_filename(source) not in allowed:
                    continue
                destination = packet_dir / pedant_name / source.name
                if destination.exists() and not args.overwrite:
                    continue
                if args.task == "tossup":
                    changes += rescore_tossup(source, destination, gold, scorer)
                else:
                    changes += rescore_bonus(source, destination, gold, scorer)
                files_written += 1
    print(f"files_written={files_written}")
    print(f"correctness_labels_changed={changes}")


def _model_from_filename(path: Path) -> str:
    fields = path.stem.split("__")
    return f"{fields[-2]}/{fields[-1]}"


if __name__ == "__main__":
    main()
