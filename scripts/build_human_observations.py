#!/usr/bin/env python3
"""Create a de-identified human buzz table from the public QANTA game logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def question_max_positions(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["qid"].split("-")[-1]: int(row["question_max_position"])
            for row in csv.DictReader(handle)
            if row["packet"] != "6" and row["question_max_position"]
        }


def build_rows(gamelogs_dir: Path, stats_csv: Path) -> list[dict[str, object]]:
    maximum = question_max_positions(stats_csv)
    rows: list[dict[str, object]] = []
    is_human = lambda player_id: bool(re.fullmatch(r"H\d+", str(player_id)))
    for path in sorted(gamelogs_dir.rglob("cycles.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cycle = json.loads(line)
                plays: dict[tuple[str, object, int], list[dict[str, object]]] = defaultdict(list)
                for response in cycle.get("tossupResponses", []):
                    qid = response.get("qid", "")
                    match = re.search(r"packet(\d+)", qid)
                    packet = int(match.group(1)) if match else 0
                    if packet not in {1, 2, 3, 4, 5}:
                        continue
                    nano = qid.split("-")[-1]
                    plays[(nano, response.get("tossupIndex"), packet)].append(
                        response.get("marker", {})
                    )
                for (nano, _, packet), markers in plays.items():
                    if nano not in maximum:
                        continue
                    human_markers = [
                        marker
                        for marker in markers
                        if is_human(marker.get("player", {}).get("id"))
                        and marker.get("position") is not None
                    ]
                    if human_markers:
                        first = min(human_markers, key=lambda marker: marker["position"])
                        reveal = min(1.0, max(0.0, first["position"] / maximum[nano]))
                        correct, censored = int(bool(first.get("isCorrect"))), 0
                    else:
                        reveal, correct, censored = 1.0, 0, 1
                    rows.append(
                        {
                            "packet": packet,
                            "nano": nano,
                            "reveal_fraction": round(reveal, 8),
                            "correct": correct,
                            "censored": censored,
                        }
                    )
    rows.sort(key=lambda row: (row["packet"], row["nano"], row["reveal_fraction"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamelogs-dir", type=Path, required=True)
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("data/human/question_tossup_adversarialness.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/human/human_buzz_observations.csv")
    )
    args = parser.parse_args()
    rows = build_rows(args.gamelogs_dir, args.stats_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} de-identified plays)")


if __name__ == "__main__":
    main()
