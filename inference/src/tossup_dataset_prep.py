"""Prepare AdvVQA tossup records for packet-outputs / ai_workflows inference."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

from PIL import Image as PILImage

TokenLayout = Literal["image_before_text", "text_before_image"]


def _is_end_of_clue_token(token: dict) -> bool:
    if token.get("type") != "text":
        return False
    content = (token.get("content") or "").strip()
    return content.endswith((".", "?", "!"))


def run_indices_from_tokens(
    multimodal_tokens: list[dict],
    *,
    run_length: int = 7,
) -> list[int]:
    """Same progressive-reveal rule as quizbowl-submission load_multimodal_dataset."""
    sorted_tokens = sorted(multimodal_tokens, key=lambda x: x["position"])
    n = len(sorted_tokens)
    if n == 0:
        return []

    run_indices: list[int] = []
    prev_idx = -1

    for i in range(n):
        t = sorted_tokens[i]
        if _is_end_of_clue_token(t):
            run_indices.append(i)
            prev_idx = i
        elif t.get("type") == "delay":
            run_indices.append(i)
            prev_idx = i
        elif i == prev_idx + run_length:
            run_indices.append(i)
            prev_idx = i

    if n > 0 and (n - 1) not in run_indices:
        run_indices.append(n - 1)
    return sorted(run_indices)


def _split_tossup_slides(tokens: list[dict]) -> list[list[dict]]:
    """Split flat tossup tokens into per-slide groups (legacy: text,image,delay)."""
    slides: list[list[dict]] = []
    current: list[dict] = []
    for tok in sorted(tokens, key=lambda t: t["position"]):
        if tok.get("type") == "delay":
            if current:
                slides.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        slides.append(current)
    return slides


def _slide_text_before_image(slide: list[dict]) -> bool:
    text_pos = next((i for i, t in enumerate(slide) if t.get("type") == "text"), None)
    image_pos = next((i for i, t in enumerate(slide) if t.get("type") == "image"), None)
    if text_pos is None or image_pos is None:
        return False
    return text_pos < image_pos


def _renumber_token_positions(tokens: list[dict]) -> list[dict]:
    ordered = sorted(tokens, key=lambda t: t["position"])
    out: list[dict] = []
    for i, tok in enumerate(ordered):
        new_tok = dict(tok)
        new_tok["position"] = i
        out.append(new_tok)
    return out


def _is_advvqa_export_token_order(tokens: list[dict]) -> bool:
    """AdvVQA export_service emits image → delay → text per slot (delay is not a slide boundary)."""
    sorted_tokens = sorted(tokens, key=lambda t: t["position"])
    for i, tok in enumerate(sorted_tokens):
        if tok.get("type") == "delay" and (i == 0 or sorted_tokens[i - 1].get("type") != "image"):
            return False
        if (
            tok.get("type") == "image"
            and i + 1 < len(sorted_tokens)
            and sorted_tokens[i + 1].get("type") == "image"
        ):
            return False
    return bool(sorted_tokens)


def _split_advvqa_export_slots(tokens: list[dict]) -> list[list[dict]]:
    """Group AdvVQA export tokens into per-slot chunks (image → delay → text, or text-only)."""
    sorted_tokens = sorted(tokens, key=lambda t: t["position"])
    slots: list[list[dict]] = []
    i = 0
    n = len(sorted_tokens)
    while i < n:
        slot: list[dict] = []
        if sorted_tokens[i].get("type") == "image":
            slot.append(sorted_tokens[i])
            i += 1
            if i < n and sorted_tokens[i].get("type") == "delay":
                slot.append(sorted_tokens[i])
                i += 1
            while i < n and sorted_tokens[i].get("type") == "text":
                slot.append(sorted_tokens[i])
                i += 1
            slots.append(slot)
        elif sorted_tokens[i].get("type") == "text":
            while i < n and sorted_tokens[i].get("type") == "text":
                slot.append(sorted_tokens[i])
                i += 1
            slots.append(slot)
        else:
            i += 1
    return slots


def _is_legacy_text_before_image_order(tokens: list[dict]) -> bool:
    for slide in _split_tossup_slides(tokens):
        if _slide_text_before_image(slide):
            return True
    return False


def reorder_tossup_tokens_text_before_image(tokens: list[dict]) -> list[dict]:
    """Convert AdvVQA export slots (image → delay → text) to legacy (text → image → delay)."""
    if _is_legacy_text_before_image_order(tokens) and not _is_advvqa_export_token_order(tokens):
        return _renumber_token_positions(tokens)

    export_tokens = (
        _renumber_token_positions(tokens)
        if _is_advvqa_export_token_order(tokens)
        else reorder_tossup_tokens_image_before_text(tokens)
    )
    out: list[dict] = []
    pos = 0
    for slot in _split_advvqa_export_slots(export_tokens):
        texts = [t for t in slot if t.get("type") == "text"]
        images = [t for t in slot if t.get("type") == "image"]
        delays = [t for t in slot if t.get("type") == "delay"]
        ordered = list(texts) + list(images)
        if images:
            if delays:
                ordered.append(delays[0])
            else:
                ordered.append({
                    "content": None,
                    "hash_key": None,
                    "mm_index": None,
                    "position": 0,
                    "type": "delay",
                })
        for tok in ordered:
            new_tok = dict(tok)
            new_tok["position"] = pos
            out.append(new_tok)
            pos += 1
    return out


def _question_text_from_tokens(tokens: list[dict]) -> str:
    parts: list[str] = []
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            parts.append(" ".join(text_buf))
            text_buf.clear()

    for tok in sorted(tokens, key=lambda t: t["position"]):
        if tok.get("type") == "text":
            text_buf.append(tok.get("content") or "")
        elif tok.get("type") == "image":
            flush_text()
            hash_key = tok.get("hash_key") or ""
            parts.append(f'<multimodal type="img" hash="{hash_key}">')
        elif tok.get("type") == "delay":
            flush_text()
            parts.append('<multimodal type="delay">')
    flush_text()
    return " ".join(parts)


def order_tossup_tokens(tokens: list[dict], token_layout: TokenLayout) -> list[dict]:
    if token_layout == "text_before_image":
        return reorder_tossup_tokens_text_before_image(tokens)
    return reorder_tossup_tokens_image_before_text(tokens)


def reorder_tossup_tokens_image_before_text(tokens: list[dict]) -> list[dict]:
    """Per slide: image, delay (if image+text), then text.

    AdvVQA ``export_service`` already emits image → delay → text per slot; only legacy
    text → image → delay slides need reordering (delay marks slide boundaries there).
    """
    if _is_advvqa_export_token_order(tokens):
        return _renumber_token_positions(tokens)

    slides = _split_tossup_slides(tokens)
    if not slides:
        return []

    out: list[dict] = []
    pos = 0
    for slide in slides:
        if not _slide_text_before_image(slide):
            ordered = slide
        else:
            texts = [t for t in slide if t.get("type") == "text"]
            images = [t for t in slide if t.get("type") == "image"]
            ordered = list(images)
            if images and texts:
                ordered.append({
                    "content": None,
                    "hash_key": None,
                    "mm_index": None,
                    "position": 0,
                    "type": "delay",
                })
            ordered.extend(texts)

        for tok in ordered:
            new_tok = dict(tok)
            new_tok["position"] = pos
            out.append(new_tok)
            pos += 1
    return out


def _tokens_to_multimodal_tokens(
    tokens: list[dict],
    image_paths: list[str],
) -> list[dict]:
    img_i = 0
    out: list[dict] = []
    for token in tokens:
        tok = dict(token)
        if tok.get("type") == "image":
            if img_i < len(image_paths):
                tok["path"] = image_paths[img_i]
            img_i += 1
        out.append(tok)
    return out


def _resolve_image_file(base_dir: Path, rel_path: str) -> Path | None:
    candidate = base_dir / rel_path
    if candidate.is_file():
        return candidate
    # Fallback: bare filename under tossup/images/
    fallback = base_dir / "tossup" / "images" / Path(rel_path).name
    if fallback.is_file():
        return fallback
    return None


def load_pil_images_in_token_order(
    multimodal_tokens: list[dict],
    image_paths: list[str],
    base_dir: Path,
) -> list[PILImage.Image]:
    """Load one PIL image per image token, in token order."""
    images: list[PILImage.Image] = []
    path_by_mm_index = list(image_paths)
    img_i = 0
    for token in sorted(multimodal_tokens, key=lambda x: x["position"]):
        if token.get("type") != "image":
            continue
        rel = token.get("path") or (path_by_mm_index[img_i] if img_i < len(path_by_mm_index) else "")
        img_i += 1
        resolved = _resolve_image_file(base_dir, rel) if rel else None
        if resolved is not None:
            images.append(PILImage.open(resolved).convert("RGB"))
    return images


def prepare_advvqa_tossup(
    record: dict[str, Any],
    base_dir: Path | str,
    *,
    run_length: int = 7,
    embed_images: bool = False,
    token_layout: TokenLayout = "image_before_text",
) -> dict[str, Any]:
    """
    Add fields required by generate_tossup_outputs / ai_workflows runners:
    multimodal_tokens, run_indices, and optionally embedded PIL images.
    """
    base = Path(base_dir)
    out = copy.deepcopy(record)

    raw_tokens = out.get("multimodal_tokens") or out.get("tokens") or []
    image_paths = list(out.get("image_paths") or [])
    ordered = order_tossup_tokens(raw_tokens, token_layout)
    multimodal_tokens = _tokens_to_multimodal_tokens(ordered, image_paths)

    out["multimodal_tokens"] = multimodal_tokens
    out["tokens"] = [{k: v for k, v in t.items() if k != "path"} for t in multimodal_tokens]
    out["run_indices"] = run_indices_from_tokens(
        multimodal_tokens,
        run_length=run_length,
    )
    out["question"] = _question_text_from_tokens(multimodal_tokens)

    if embed_images:
        out["images"] = load_pil_images_in_token_order(
            multimodal_tokens,
            image_paths,
            base,
        )
    return out


def prepare_records_for_json(
    records: list[dict[str, Any]],
    base_dir: Path | str,
    *,
    run_length: int = 7,
    token_layout: TokenLayout = "image_before_text",
) -> list[dict[str, Any]]:
    """Model-ready JSON records (paths only, no PIL)."""
    return [
        prepare_advvqa_tossup(
            r,
            base_dir,
            run_length=run_length,
            embed_images=False,
            token_layout=token_layout,
        )
        for r in records
    ]


def prepare_records_for_hub(
    records: list[dict[str, Any]],
    base_dir: Path | str,
    *,
    run_length: int = 7,
    token_layout: TokenLayout = "image_before_text",
) -> list[dict[str, Any]]:
    """Model-ready records with embedded PIL images for HF parquet."""
    return [
        prepare_advvqa_tossup(
            r,
            base_dir,
            run_length=run_length,
            embed_images=True,
            token_layout=token_layout,
        )
        for r in records
    ]
