#!/usr/bin/env python3
"""Validate scratch, a fresh GitHub clone, Hub pins, Git auth, and GPU allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from qanta_bench.preflight import GIB, checkpoint_scratch_bytes, scratch_from_env  # noqa: E402
from qanta_bench.preflight import verify_huggingface_revision  # noqa: E402


GITHUB_URL = "https://github.com/thunghiaa/qanta-adversarial-benchmark.git"


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def load_gpu_config(root: Path) -> dict[str, Any]:
    return json.loads((root / "configs" / "frontier_gpu.json").read_text(encoding="utf-8"))


def resolve_model(config: dict[str, Any], slug: str | None) -> dict[str, Any] | None:
    if slug is None:
        return None
    for model in config["models"]:
        if slug.casefold() in {
            model["slug"].casefold(),
            model["canonical_id"].casefold(),
            model["served_checkpoint"].casefold(),
        }:
            return model
    choices = ", ".join(model["slug"] for model in config["models"])
    raise SystemExit(f"Unknown model {slug!r}. Choose one of: {choices}")


def clone_to_scratch(scratch: Path, git_ref: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clone = scratch / f"qanta-adversarial-benchmark-{stamp}-{os.getpid()}"
    run(["git", "clone", "--depth", "1", "--branch", git_ref, GITHUB_URL, str(clone)])
    print(f"PASS clone: {clone}")
    print(f"     commit: {run(['git', 'rev-parse', 'HEAD'], cwd=clone, capture=True)}")
    return clone


def verify_git(clone: Path, check_write: bool) -> None:
    name_result = subprocess.run(
        ["git", "config", "user.name"], cwd=clone, check=False, text=True, capture_output=True
    )
    email_result = subprocess.run(
        ["git", "config", "user.email"], cwd=clone, check=False, text=True, capture_output=True
    )
    name = name_result.stdout.strip()
    email = email_result.stdout.strip()
    if not name or not email:
        raise RuntimeError("Git user.name/user.email are not configured")
    print(f"PASS Git identity: {name} <configured email>")
    if check_write:
        run(["git", "push", "--dry-run", "origin", "HEAD:refs/heads/main"], cwd=clone)
        print("PASS GitHub write authentication: dry-run push to main")


def verify_hub_pins(config: dict[str, Any]) -> None:
    models = config.get("models", [])
    if len(models) != 10:
        raise RuntimeError(f"Expected ten GPU models, found {len(models)}")
    for model in models:
        actual = verify_huggingface_revision(model["served_checkpoint"], model["revision"])
        print(f"PASS Hub pin: {model['served_checkpoint']}@{actual}")


def verify_gpu_allocation(clone: Path, model: dict[str, Any]) -> None:
    sys.path.insert(0, str(clone / "scripts"))
    import frontier_gpu  # type: ignore[import-not-found]  # noqa: PLC0415

    inventory = frontier_gpu.gpu_inventory()
    passed, reasons = frontier_gpu.hardware_check(model, inventory)
    if not passed:
        detail = "; ".join(reasons)
        raise RuntimeError(f"Unsupported or insufficient GPU allocation for {model['slug']}: {detail}")
    total = sum(gpu["memory_gb"] for gpu in inventory)
    print(f"PASS GPU allocation: {len(inventory)} GPUs, {total:.0f} GiB visible")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="also size scratch for this model and verify visible GPUs")
    parser.add_argument("--git-ref", default="main")
    parser.add_argument("--skip-git-write-check", action="store_true")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="validate storage, clone, Hub pins, and Git; defer GPU allocation check",
    )
    args = parser.parse_args()

    source_config = load_gpu_config(SOURCE_ROOT)
    model = resolve_model(source_config, args.model)
    required_bytes = checkpoint_scratch_bytes(model) if model else 5 * GIB
    try:
        report = scratch_from_env(SOURCE_ROOT, required_bytes)
    except ValueError as error:
        raise SystemExit(f"FAIL scratch: {error}") from error
    print(
        f"PASS scratch: {report.path} ({report.free_bytes / GIB:.1f} GiB free; "
        f"{report.required_bytes / GIB:.1f} GiB required)"
    )

    try:
        clone = clone_to_scratch(report.path, args.git_ref)
        cloned_config = load_gpu_config(clone)
        cloned_model = resolve_model(cloned_config, args.model)
        verify_hub_pins(cloned_config)
        verify_git(clone, not args.skip_git_write_check)
        if cloned_model is not None and not args.setup_only:
            verify_gpu_allocation(clone, cloned_model)
        elif model is None and not args.setup_only:
            print("DEFER GPU allocation: rerun with --model SLUG inside a supported allocation")
        else:
            print("DEFER GPU allocation: --setup-only requested")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"FAIL preflight: {error}") from error
    print("PASS frontier preflight")


if __name__ == "__main__":
    main()
