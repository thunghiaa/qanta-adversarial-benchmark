#!/usr/bin/env python3
"""Download the canonical packet dataset into data/packets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=os.getenv("PACKET_DATASET", "qanta-challenge/packet-questions"),
    )
    parser.add_argument("--revision", default=None, help="optional immutable Hub revision")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "packets",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("Install inference dependencies: pip install -e '.[inference]'") from error

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Downloaded {args.repo_id} to {args.output}")


if __name__ == "__main__":
    main()
