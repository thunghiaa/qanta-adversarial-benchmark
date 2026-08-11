#!/usr/bin/env python3
"""Download registered local-model artifacts without committing model weights."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qanta_bench.registry import load_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="registered ids such as local/pixtral-12b")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("Install inference dependencies: pip install -e '.[inference]'") from error

    registry = {model.id.casefold(): model for model in load_registry()}
    for requested in args.models:
        model = registry.get(requested.casefold())
        if model is None or not model.artifact:
            raise SystemExit(f"{requested!r} has no downloadable artifact in configs/models.json")
        destination = ROOT / "model_artifacts" / model.id.split("/", 1)[1]
        snapshot_download(
            repo_id=model.artifact,
            revision=None,
            local_dir=destination,
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Downloaded {model.artifact} to {destination}")


if __name__ == "__main__":
    main()
