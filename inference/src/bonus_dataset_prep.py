"""Prepare AdvVQA bonus records for packet-outputs / ai_workflows inference."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from PIL import Image as PILImage


def _resolve_image_file(base_dir: Path, rel_path: str, *, images_subdir: str) -> Path | None:
    candidate = base_dir / rel_path
    if candidate.is_file():
        return candidate
    fallback = base_dir / images_subdir / "images" / Path(rel_path).name
    if fallback.is_file():
        return fallback
    return None


def _load_image_dict(img: dict[str, Any] | None, base_dir: Path) -> PILImage.Image | None:
    if not img:
        return None
    rel = img.get("path")
    if not rel:
        return None
    resolved = _resolve_image_file(base_dir, rel, images_subdir="bonus")
    if resolved is None:
        return None
    return PILImage.open(resolved).convert("RGB")


def part_image_splits_from_parts(parts: list[dict[str, Any]], num_part_images: int) -> list[int]:
    """Cumulative slice indices into the flat part_images list (per-part images)."""
    splits = [0]
    for i, part in enumerate(parts):
        idx = part.get("image_idx")
        if idx is None:
            splits.append(splits[-1])
            continue
        end = num_part_images
        for j in range(i + 1, len(parts)):
            next_idx = parts[j].get("image_idx")
            if next_idx is not None:
                end = next_idx
                break
        splits.append(end)
    return splits


def prepare_advvqa_bonus(
    record: dict[str, Any],
    base_dir: Path | str,
    *,
    embed_images: bool = False,
) -> dict[str, Any]:
    """
    Add fields required by generate_bonus_outputs / ai_workflows runners:
    leadin_images, part_images (flat), and part_image_splits.
    """
    base = Path(base_dir)
    out = copy.deepcopy(record)

    parts = list(out.get("parts") or [])
    part_image_dicts = list(out.get("part_images") or [])
    out["part_image_splits"] = part_image_splits_from_parts(parts, len(part_image_dicts))

    if embed_images:
        leadin_img = _load_image_dict(out.get("leadin_image"), base)
        out["leadin_images"] = [leadin_img] if leadin_img is not None else []

        flat_part_images: list[PILImage.Image] = []
        for img in part_image_dicts:
            loaded = _load_image_dict(img, base)
            if loaded is not None:
                flat_part_images.append(loaded)
        out["part_images"] = flat_part_images
    return out


def prepare_records_for_json(
    records: list[dict[str, Any]],
    base_dir: Path | str,
) -> list[dict[str, Any]]:
    """Model-ready JSON records (path-based images, with part_image_splits)."""
    return [prepare_advvqa_bonus(r, base_dir, embed_images=False) for r in records]


def prepare_records_for_hub(
    records: list[dict[str, Any]],
    base_dir: Path | str,
) -> list[dict[str, Any]]:
    """Model-ready records with embedded PIL images for HF parquet."""
    return [prepare_advvqa_bonus(r, base_dir, embed_images=True) for r in records]
