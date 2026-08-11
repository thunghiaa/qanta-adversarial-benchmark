#!/usr/bin/env python3
"""Benchmark a registered frontier model through an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qanta_bench.frontier import (  # noqa: E402
    PACKET_LABELS,
    SYSTEM_PROMPT,
    inject_packet_qid,
    parse_answer_json,
    run_indices_from_tokens,
    strict_correct,
    text_fragment,
)
from qanta_bench.registry import model_index  # noqa: E402


def _post_chat(base_url: str, model: str, prompt: str, timeout: float) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
        # Laguna exposes native reasoning through this official per-request
        # switch. Direct-QA benchmarking disables it so hidden chain-of-thought
        # does not consume the short answer budget or distort latency.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail[:1000]}") from error
    latency = time.perf_counter() - started
    content = body["choices"][0]["message"].get("content") or ""
    telemetry = {"latency_seconds": round(latency, 4), "usage": body.get("usage") or {}}
    return content, telemetry


def _load_rows(packet: int, task: str) -> list[dict[str, Any]]:
    path = ROOT / "data" / "packets" / f"packet{packet}_final" / f"{task}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def _output_path(root: Path, packet: int, task: str, run_id: str, model_id: str) -> Path:
    owner, name = model_id.split("/", 1)
    task_dir = "Tossup" if task == "tossup" else "Bonus"
    filename = f"{task}__frontier__{run_id}__{owner}__{name}.jsonl"
    return root / PACKET_LABELS[packet] / task_dir / filename


def _completed_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["qid"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _tossup_row(
    row: dict[str, Any], packet: int, base_url: str, served_model: str, timeout: float
) -> dict[str, Any]:
    # Preserve the historical QANTA token order so token_position is directly
    # comparable with the submitted outputs and paper plots.
    tokens = sorted(
        row.get("tokens") or row.get("multimodal_tokens") or [],
        key=lambda token: token["position"],
    )
    outputs = []
    for run_number, stop_index in enumerate(run_indices_from_tokens(tokens), start=1):
        prompt = text_fragment(tokens, stop_index)
        content, telemetry = _post_chat(base_url, served_model, prompt, timeout)
        guess, confidence = parse_answer_json(content)
        outputs.append(
            {
                "guess": guess,
                "confidence": confidence,
                "buzz": confidence >= 0.75,
                "run_idx": run_number,
                "correct": strict_correct(guess, row.get("clean_answers") or [row.get("answer", "")]),
                "token_position": stop_index + 1,
                **telemetry,
            }
        )
    return {"qid": inject_packet_qid(row["qid"], packet, "tossup"), "run_outputs": outputs}


def _bonus_row(
    row: dict[str, Any], packet: int, base_url: str, served_model: str, timeout: float
) -> dict[str, Any]:
    outputs = []
    leadin = row.get("leadin") or ""
    for part in row.get("parts") or []:
        prompt = (
            f"Bonus lead-in: {leadin}\nPart {part.get('number')}: {part.get('question', '')}\n"
            "Images in this question are omitted because this is a text-only model track."
        )
        content, telemetry = _post_chat(base_url, served_model, prompt, timeout)
        guess, confidence = parse_answer_json(content)
        outputs.append(
            {
                "guess": guess,
                "confidence": confidence,
                "explanation": "",
                "number": part.get("number"),
                "correct": strict_correct(
                    guess, part.get("clean_answers") or [part.get("answer", "")]
                ),
                **telemetry,
            }
        )
    return {"qid": inject_packet_qid(row["qid"], packet, "bonus"), "part_outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="canonical id in configs/models.json")
    parser.add_argument("--served-model", required=True, help="model name exposed by the server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--task", choices=("tossup", "bonus"), required=True)
    parser.add_argument("--packets", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    registered = model_index().get(args.model.casefold())
    if registered is None or registered.cohort != "additional_benchmark":
        raise SystemExit(f"{args.model!r} is not registered as additional_benchmark")
    if registered.kind != "frontier_open_weight":
        raise SystemExit(f"{args.model!r} is not a frontier_open_weight model")
    if args.task not in registered.tasks:
        raise SystemExit(f"{args.model!r} is not registered for {args.task}")
    if any(packet not in PACKET_LABELS for packet in args.packets):
        raise SystemExit("Packets must be between 1 and 6")

    for packet in args.packets:
        output = _output_path(args.output_root, packet, args.task, args.run_id, args.model)
        completed = _completed_qids(output)
        rows = _load_rows(packet, args.task)
        if args.limit_questions is not None:
            rows = rows[: args.limit_questions]
        for index, row in enumerate(rows, start=1):
            qid = inject_packet_qid(row["qid"], packet, args.task)
            if qid in completed:
                continue
            print(f"[{packet}/{args.task}] {index}/{len(rows)} {qid}", flush=True)
            if args.task == "tossup":
                result = _tossup_row(row, packet, args.base_url, args.served_model, args.timeout)
            else:
                result = _bonus_row(row, packet, args.base_url, args.served_model, args.timeout)
            _append(output, result)
        print(output, flush=True)


if __name__ == "__main__":
    main()
