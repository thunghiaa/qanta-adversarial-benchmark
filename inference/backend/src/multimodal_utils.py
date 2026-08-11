"""Minimal multimodal utilities for backend evaluation.

Provides the functions required by shared/workflows (ai_workflows) when evaluating
multimodal tossups and bonuses. Kept in sync with quizbowl-submission/src/multimodal_utils
for get_text_with_placeholders, resolve_image_path, resolve_image_path_from_token.
"""

import os
from typing import Optional


def get_text_with_placeholders(multimodal_tokens: list[dict]) -> str:
    """
    Reconstruct question text with image and delay placeholders:
    - Image tokens -> <img:hash_key>
    - Delay tokens -> <delay>
    """
    text_parts = []
    for token in sorted(multimodal_tokens, key=lambda x: x["position"]):
        if token["type"] == "text":
            text_parts.append(token["content"])
        elif token["type"] == "image":
            hk = token.get("hash_key") or token.get("hashKey")
            text_parts.append(f"<img:{hk}>" if hk else "")
        elif token["type"] == "delay":
            text_parts.append("<delay>")
    return " ".join(p for p in text_parts if p)


def resolve_image_path(
    path: str,
    base_dir: Optional[str] = None,
    question_id: Optional[str] = None,
) -> str:
    """
    Resolve image path. Priority: absolute URL, absolute file path,
    relative to base_dir, then as-is.
    """
    if not path:
        return path
    if path.startswith(("http://", "https://")):
        return path
    if os.path.isabs(path):
        return path
    if base_dir:
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        candidate = os.path.normpath(os.path.join(base_dir, path))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    if os.path.exists(path):
        return os.path.abspath(path)
    return path


def resolve_image_path_from_token(
    token: dict,
    base_dir: Optional[str] = None,
    question_id: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve image file path for an image token. Uses path if set, else
    {base_dir}/{question_id}/{hash_key}.png|jpeg|jpg.
    """
    path = token.get("path")
    if path:
        return resolve_image_path(path, base_dir, question_id)
    hk = token.get("hash_key") or token.get("hashKey")
    if not hk or not question_id or not base_dir:
        return None
    dir_for_question = os.path.join(base_dir, question_id)
    for ext in ("png", "jpeg", "jpg"):
        candidate = os.path.join(dir_for_question, f"{hk}.{ext}")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(dir_for_question, f"{hk}.png"))
