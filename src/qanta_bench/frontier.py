"""Shared primitives for direct frontier-model QANTA benchmarks.

This module deliberately has no model-runtime dependency.  A local server can
be llama.cpp, vLLM, SGLang, or another implementation exposing the OpenAI chat
completions protocol.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from typing import Any

PACKET_LABELS = {
    1: "Packet 1 - Jordan",
    2: "Packet 2 - Kurtis",
    3: "Packet 3 - Irene + Stephen",
    4: "Packet 4 - Chauncey + Eve",
    5: "Packet 5 - Tianyi",
    6: "Packet 6 - Medley",
}

SYSTEM_PROMPT = """You are answering a progressively revealed quizbowl question.
Use only clues currently visible. Give the shortest conventional answer that uniquely identifies
the entity. Return one JSON object and no other text:
{"answer": "short answer", "confidence": 0.0}
Confidence must be a number from 0 to 1."""


def run_indices_from_tokens(tokens: list[dict[str, Any]], run_length: int = 7) -> list[int]:
    """Match the progressive-reveal schedule used by the QANTA submission runner."""
    ordered = sorted(tokens, key=lambda token: token["position"])
    if not ordered:
        return []
    indices: list[int] = []
    previous = -1
    for index, token in enumerate(ordered):
        content = str(token.get("content") or "").strip()
        end_of_clue = token.get("type") == "text" and content.endswith((".", "?", "!"))
        if end_of_clue or token.get("type") == "delay" or index == previous + run_length:
            indices.append(index)
            previous = index
    if len(ordered) - 1 not in indices:
        indices.append(len(ordered) - 1)
    return sorted(set(indices))


def image_before_text_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert legacy text/image/delay slides to the competition reveal order."""
    ordered = sorted(tokens, key=lambda token: token["position"])
    slides: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for token in ordered:
        if token.get("type") == "delay":
            if current:
                slides.append(current)
                current = []
        else:
            current.append(token)
    if current:
        slides.append(current)

    result: list[dict[str, Any]] = []
    for slide in slides:
        images = [token for token in slide if token.get("type") == "image"]
        texts = [token for token in slide if token.get("type") == "text"]
        if images:
            result.extend(images)
            if texts:
                result.append({"type": "delay", "content": None})
        result.extend(texts)
    for position, token in enumerate(result):
        token = dict(token)
        token["position"] = position
        result[position] = token
    return result


def text_fragment(tokens: list[dict[str, Any]], stop_index: int) -> str:
    """Render a reveal for a text-only model while explicitly marking missing images."""
    pieces: list[str] = []
    for token in tokens[: stop_index + 1]:
        if token.get("type") == "text":
            pieces.append(str(token.get("content") or ""))
        elif token.get("type") == "image":
            pieces.append("[IMAGE OMITTED FOR TEXT-ONLY MODEL]")
    return " ".join(piece for piece in pieces if piece).strip()


def inject_packet_qid(qid: str, packet: int, task: str) -> str:
    marker = "t" if task == "tossup" else "b"
    qid = re.sub(r"^advvqa-packet\d+-[tb]-", f"advvqa-{marker}-", qid)
    prefix = f"advvqa-{marker}-"
    if not qid.startswith(prefix):
        raise ValueError(f"Unexpected {task} qid: {qid}")
    return f"advvqa-packet{packet}-{marker}-{qid.removeprefix(prefix)}"


def parse_answer_json(text: str) -> tuple[str, float]:
    """Parse strict JSON with a safe fallback for servers that add prose."""
    candidates = [text.strip()]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    candidates.append(fenced)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        answer = str(payload.get("answer") or payload.get("guess") or "").strip()
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return answer, min(1.0, max(0.0, confidence))
    return text.strip()[:200], 0.0


def _normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    normalized = normalized.lower().replace("’", "'").replace("`", "'")
    normalized = "".join(character for character in normalized if character not in string.punctuation)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def strict_correct(prediction: str, answers: list[str] | str) -> int:
    """Lightweight equivalent of the repository's strict QA matcher.

    PEDANT scoring remains a separate, non-destructive post-processing step.
    """
    references = [answers] if isinstance(answers, str) else answers
    pred = _normalize_answer(prediction)
    if not pred:
        return 0
    pred_forms = {pred, pred.removesuffix("s"), pred.removesuffix("es")}
    if not pred.endswith("s"):
        pred_forms.add(pred + "s")
        pred_forms.add(pred + "es")
    for answer in references:
        reference = _normalize_answer(answer)
        if pred == reference:
            return 1
        if any(re.search(r"\b" + re.escape(form) + r"\b", reference) for form in pred_forms if form):
            return 1
    return 0
