#!/usr/bin/env python3
"""Registry-aware front door for the vendored QANTA inference runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qanta_bench.registry import models_for  # noqa: E402


def _runner_name(model_id: str, cohort: str) -> str:
    """Return the case-preserving username/model_name stored in a submission spec."""
    owner, model_name = model_id.split("/", 1)
    queue = ROOT / "submission_specs" / cohort / "eval-queue" / owner
    for path in sorted(queue.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = f"{payload['username']}/{payload['model_name']}"
        if candidate.casefold() == model_id.casefold():
            return candidate
    raise SystemExit(
        f"No runnable submission spec for {model_id!r} under {queue}. "
        "Local VLM artifacts must be downloaded before they can be run."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("qanta_submitted", "additional_benchmark"), required=True)
    parser.add_argument("--task", choices=("tossup", "bonus"), required=True)
    parser.add_argument("--packets", nargs="+", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    allowed = {model.id.casefold(): model for model in models_for(args.cohort, args.task)}
    requested = []
    for name in args.models:
        model = allowed.get(name.casefold())
        if model is None:
            raise SystemExit(f"{name!r} is not registered for {args.cohort}/{args.task}")
        requested.append(model)

    packet_dirs = [ROOT / "data" / "packets" / f"packet{packet}_final" for packet in args.packets]
    missing = [path for path in packet_dirs if not path.is_dir()]
    if missing:
        raise SystemExit(f"Missing packet data: {missing[0]} (run scripts/fetch_data.py)")

    runner = ROOT / "inference" / "generate_outputs.py"
    env = os.environ.copy()
    env["HF_HOME"] = str(ROOT / "submission_specs" / args.cohort)
    env["LLM_CACHE_PATH"] = str(ROOT / ".cache")
    if args.dry_run:
        env["HF_HUB_OFFLINE"] = "1"
    backend = ROOT / "inference" / "backend"
    source = ROOT / "inference" / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(backend / "src"), str(backend), env.get("PYTHONPATH", "")]
    )
    for packet, packet_dir in zip(args.packets, packet_dirs):
        output = ROOT / "local_outputs" / args.cohort / f"packet{packet}" / args.task
        for model in requested:
            if model.kind == "local_vlm":
                artifact_dir = ROOT / "model_artifacts" / model.id.split("/", 1)[1]
                if not artifact_dir.is_dir():
                    raise SystemExit(
                        f"Missing {artifact_dir}; run scripts/fetch_model_artifacts.py {model.id}"
                    )
                model_args = ["--hf-model", str(artifact_dir)]
            else:
                model_args = ["--submission", _runner_name(model.id, args.cohort)]

            command = [
                sys.executable,
                str(runner),
                "--local-packets-dir",
                str(packet_dir),
                "--local-output-dir",
                str(output),
                "--competition-type",
                args.task,
                *model_args,
                "--no-sync",
                "--no-upload",
            ]
            if args.debug:
                command.append("--debug")
            if args.dry_run:
                command.append("--dry-run")
            if args.reprocess:
                command.append("--reprocess")
            subprocess.run(command, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
