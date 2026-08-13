#!/usr/bin/env python3
"""Plan, serve, and benchmark the pinned frontier GPU checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "frontier_gpu.json"
sys.path.insert(0, str(ROOT / "src"))

from qanta_bench.preflight import checkpoint_scratch_bytes, scratch_from_env  # noqa: E402


def load_config() -> dict[str, Any]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or len(data.get("models", [])) != 10:
        raise RuntimeError(f"Unexpected frontier GPU registry: {CONFIG_PATH}")
    return data


def model_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for model in config["models"]:
        for alias in (model["slug"], model["canonical_id"], model["served_checkpoint"]):
            index[alias.casefold()] = model
    return index


def resolve_model(config: dict[str, Any], value: str) -> dict[str, Any]:
    model = model_index(config).get(value.casefold())
    if model is None:
        choices = ", ".join(item["slug"] for item in config["models"])
        raise SystemExit(f"Unknown model {value!r}. Choose one of: {choices}")
    return model


def gpu_inventory() -> list[dict[str, Any]]:
    """Return visible NVIDIA GPUs without importing a GPU Python package."""
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return []
    inventory = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        try:
            inventory.append(
                {"index": int(fields[0]), "name": fields[1], "memory_gb": int(fields[2]) / 1024}
            )
        except ValueError:
            continue
    return inventory


def hardware_check(
    model: dict[str, Any],
    inventory: list[dict[str, Any]],
    parallel_gpus: int | None = None,
) -> tuple[bool, list[str]]:
    reasons = []
    if not inventory:
        return False, ["No visible NVIDIA GPUs were detected with nvidia-smi."]
    if len(inventory) < model["minimum_gpus"]:
        reasons.append(f"needs >= {model['minimum_gpus']} GPUs; found {len(inventory)}")
    serving_gpu_count = parallel_gpus or model.get("memory_parallel_gpus", model["default_tp"])
    total_vram = sum(gpu["memory_gb"] for gpu in inventory[:serving_gpu_count])
    if total_vram < model["minimum_total_vram_gb"]:
        reasons.append(
            f"needs >= {model['minimum_total_vram_gb']} GB in its {serving_gpu_count}-GPU "
            f"parallel group; found {total_vram:.0f} GB"
        )
    allowed = model["supported_gpu_families"]
    unsupported = [
        gpu["name"]
        for gpu in inventory
        if not any(family.casefold() in gpu["name"].casefold() for family in allowed)
    ]
    if unsupported:
        reasons.append(f"unverified GPU family: {', '.join(sorted(set(unsupported)))}")
    return not reasons, reasons


def render_command(
    model: dict[str, Any], *, tp: int, port: int, max_model_len: int
) -> list[str]:
    replacements = {
        "{checkpoint}": model["served_checkpoint"],
        "{revision}": model["revision"],
        "{served_model}": model["served_model_name"],
        "{tp}": str(tp),
        "{port}": str(port),
        "{max_model_len}": str(max_model_len),
    }
    return [replacements.get(token, token) for token in model["serve_command"]]


def default_cache_dir() -> Path:
    report = scratch_from_env(ROOT)
    return report.path / "qanta-frontier-hf-cache"


def ensure_external_cache(cache_dir: Path, scratch_root: Path | None = None) -> None:
    resolved = cache_dir.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"Refusing to store model weights inside the Git checkout: {resolved}. "
            "Use QANTA_SCRATCH."
        )
    if scratch_root is not None:
        try:
            resolved.relative_to(scratch_root.resolve())
        except ValueError as error:
            raise SystemExit(
                f"Weight cache must be inside QANTA_SCRATCH ({scratch_root}): {resolved}"
            ) from error


def print_model_plan(
    model: dict[str, Any], inventory: list[dict[str, Any]], parallel_gpus: int | None = None
) -> bool:
    passed, reasons = hardware_check(model, inventory, parallel_gpus)
    print(f"Model:       {model['slug']} ({model['canonical_id']})")
    print(f"Checkpoint:  {model['served_checkpoint']}@{model['revision']}")
    print(f"Provenance:  {model['checkpoint_provenance']} / {model['precision']}")
    print(f"Runtime:     {model['minimum_runtime']}")
    print(f"Container:   {model['container_image']}")
    print(
        f"Hardware:    >= {model['minimum_gpus']} GPUs and >= "
        f"{model['minimum_total_vram_gb']} GB total VRAM ({model['deployment']})"
    )
    print(f"GPU families: {', '.join(model['supported_gpu_families'])}")
    print(f"VRAM basis:  {model['vram_basis']}")
    print(f"Track:       QANTA text-only common track; modalities={','.join(model['modalities'])}")
    print(f"Recipe:      {model['recipe_url']}")
    if inventory:
        total = sum(gpu["memory_gb"] for gpu in inventory)
        print(f"Detected:    {len(inventory)} GPUs / {total:.0f} GB: {inventory[0]['name']}")
    print(f"Preflight:   {'PASS' if passed else 'NOT READY'}")
    for reason in reasons:
        print(f"             - {reason}")
    return passed


def list_models(config: dict[str, Any]) -> None:
    print(f"{'SLUG':<21} {'CHECKPOINT':<42} {'GPU':>4} {'VRAM':>7}  DEPLOYMENT")
    for model in config["models"]:
        print(
            f"{model['slug']:<21} {model['served_checkpoint']:<42} "
            f"{model['minimum_gpus']:>4} {model['minimum_total_vram_gb']:>5}GB  "
            f"{model['deployment']}"
        )


def serve(args: argparse.Namespace, model: dict[str, Any]) -> None:
    try:
        scratch = scratch_from_env(ROOT, checkpoint_scratch_bytes(model))
    except ValueError as error:
        raise SystemExit(f"Scratch preflight failed: {error}") from error
    cache_dir = args.cache_dir or scratch.path / "qanta-frontier-hf-cache"
    ensure_external_cache(cache_dir, scratch.path)
    command = render_command(
        model,
        tp=args.tp or model["default_tp"],
        port=args.port,
        max_model_len=args.max_model_len,
    )
    inventory = gpu_inventory()
    parallel_gpus = model.get("memory_parallel_gpus", args.tp or model["default_tp"])
    passed = print_model_plan(model, inventory, parallel_gpus)
    print(f"Weight cache: {cache_dir.expanduser().resolve()}")
    for key, value in model.get("environment", {}).items():
        print(f"export {key}={shlex.quote(value)}")
    print(f"export HF_HOME={shlex.quote(str(cache_dir.expanduser().resolve()))}")
    print(shlex.join(command))
    if args.dry_run:
        return
    if not passed and not args.skip_hardware_check:
        raise SystemExit(
            "Hardware preflight failed. Use a suitable allocation; do not bypass for real runs."
        )
    if shutil.which(command[0]) is None:
        raise SystemExit(
            f"{command[0]!r} is not installed. Use the model-specific container/runtime above."
        )
    cache_dir.expanduser().mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(model.get("environment", {}))
    environment["HF_HOME"] = str(cache_dir.expanduser().resolve())
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


def server_is_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def benchmark(args: argparse.Namespace, model: dict[str, Any]) -> None:
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_frontier.py"),
        "--model",
        model["canonical_id"],
        "--served-model",
        model["served_model_name"],
        "--base-url",
        args.base_url,
        "--task",
        args.task,
        "--packets",
        *[str(packet) for packet in args.packets],
        "--output-root",
        str(output_root),
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--run-id",
        run_id,
    ]
    if args.limit_questions is not None:
        command.extend(["--limit-questions", str(args.limit_questions)])
    print(f"Track:  QANTA text-only common track ({model['canonical_id']})")
    print(f"Output: {output_root}")
    print(shlex.join(command))
    if args.dry_run:
        return
    if not server_is_ready(args.base_url):
        raise SystemExit(f"Server is not ready at {args.base_url}; start it in terminal A first.")
    subprocess.run(command, check=True, cwd=ROOT)
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    if shutil.which("vllm"):
        runtime_version = subprocess.run(
            ["vllm", "--version"], check=False, text=True, capture_output=True
        ).stdout.strip()
    else:
        runtime_version = "server runtime not visible in client environment"
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "benchmark_track": "qanta_text_only_common",
        "model": model,
        "git_revision": git_revision,
        "runtime_version": runtime_version,
        "visible_gpus": gpu_inventory(),
        "request": {
            "task": args.task,
            "packets": args.packets,
            "workers": args.workers,
            "timeout": args.timeout,
            "limit_questions": args.limit_questions,
            "base_url": args.base_url,
        },
    }
    manifest_dir = output_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_id}__{model['slug']}__{args.task}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list all ten pinned GPU targets")

    plan_parser = subparsers.add_parser("plan", help="show requirements and inspect visible GPUs")
    plan_parser.add_argument("model")

    serve_parser = subparsers.add_parser("serve", help="launch an OpenAI-compatible vLLM server")
    serve_parser.add_argument("model")
    serve_parser.add_argument("--tp", type=int)
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--max-model-len", type=int, default=8192)
    serve_parser.add_argument("--cache-dir", type=Path)
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.add_argument("--skip-hardware-check", action="store_true", help=argparse.SUPPRESS)

    benchmark_parser = subparsers.add_parser("benchmark", help="run QANTA against a live server")
    benchmark_parser.add_argument("model")
    benchmark_parser.add_argument("--task", choices=("tossup", "bonus"), required=True)
    benchmark_parser.add_argument("--packets", nargs="+", type=int, required=True)
    benchmark_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    benchmark_parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results" / "additional_benchmarks"
    )
    benchmark_parser.add_argument("--workers", type=int, default=1)
    benchmark_parser.add_argument("--timeout", type=float, default=600)
    benchmark_parser.add_argument("--limit-questions", type=int)
    benchmark_parser.add_argument("--run-id")
    benchmark_parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    if args.command == "list":
        list_models(config)
        return
    model = resolve_model(config, args.model)
    if args.command == "plan":
        print_model_plan(model, gpu_inventory())
    elif args.command == "serve":
        serve(args, model)
    elif args.command == "benchmark":
        benchmark(args, model)


if __name__ == "__main__":
    main()
