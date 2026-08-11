"""Assign competition packet ids and inject them into AdvVQA qids."""

from __future__ import annotations

import json
import re
from pathlib import Path

PACKET1_AUTHOR = "Jordan Lee Boyd-Gräber Ying"
PACKET5_EMAIL = "tianyizheng02@gmail.com"
PACKET6_USERNAMES = frozenset(
    {"allison.andreyev", "goyochin", "nathannguyen", "ntruong"}
)
PACKET6_BONUS_SIZE = 20

_TOSSUP_PACKET_RE = re.compile(r"^advvqa-packet(\d+)-t-(.+)$")
_BONUS_PACKET_RE = re.compile(r"^advvqa-packet(\d+)-b-(.+)$")


def author_fields(record: dict) -> tuple[str | None, str | None]:
    meta = record.get("metadata") or {}
    username = meta.get("author_username") or record.get("author_username")
    email = meta.get("author_email") or record.get("author_email")
    return username, email


def normalize_tossup_qid(qid: str) -> str:
    match = _TOSSUP_PACKET_RE.match(qid)
    if match:
        return f"advvqa-t-{match.group(2)}"
    return qid


def normalize_bonus_qid(qid: str) -> str:
    match = _BONUS_PACKET_RE.match(qid)
    if match:
        return f"advvqa-b-{match.group(2)}"
    return qid


def inject_tossup_packet_qid(qid: str, packet_id: int) -> str:
    base = normalize_tossup_qid(qid)
    prefix = f"advvqa-packet{packet_id}-t-"
    if base.startswith(prefix):
        return base
    if base.startswith("advvqa-t-"):
        return prefix + base.removeprefix("advvqa-t-")
    raise ValueError(f"Unexpected tossup qid format: {qid}")


def inject_bonus_packet_qid(qid: str, packet_id: int) -> str:
    base = normalize_bonus_qid(qid)
    prefix = f"advvqa-packet{packet_id}-b-"
    if base.startswith(prefix):
        return base
    if base.startswith("advvqa-b-"):
        return prefix + base.removeprefix("advvqa-b-")
    raise ValueError(f"Unexpected bonus qid format: {qid}")


def first_yasmine_taj_qid(records: list[dict]) -> str | None:
    for record in records:
        username, _ = author_fields(record)
        if username == "YasmineTaj":
            return normalize_tossup_qid(record["qid"])
    return None


def assign_tossup_packet(record: dict, *, first_yasmine_qid: str | None) -> int | None:
    username, email = author_fields(record)
    qid = normalize_tossup_qid(record["qid"])

    if username == PACKET1_AUTHOR:
        return 1
    if username == "kdroge":
        return 2
    if username in {"ireneying", "spachucki23"}:
        return 3
    if username in {"chauncey", "efleisig"}:
        return 4
    if email == PACKET5_EMAIL:
        return 5
    if username in PACKET6_USERNAMES:
        return 6
    if username == "YasmineTaj" and qid == first_yasmine_qid:
        return 6
    return None


def assign_bonus_packet_1_to_5(record: dict) -> int | None:
    username, email = author_fields(record)

    if username == PACKET1_AUTHOR:
        return 1
    if username == "kdroge":
        return 2
    if username in {"ireneying", "spachucki23"}:
        return 3
    if username in {"chauncey", "efleisig"}:
        return 4
    if email == PACKET5_EMAIL:
        return 5
    return None


def build_tossup_qid_mapping(
    records: list[dict],
) -> tuple[dict[str, str], dict[int, int], str | None]:
    first_yasmine = first_yasmine_taj_qid(records)
    mapping: dict[str, str] = {}
    counts = {i: 0 for i in range(1, 7)}

    for record in records:
        old_qid = record["qid"]
        packet_id = assign_tossup_packet(record, first_yasmine_qid=first_yasmine)
        if packet_id is None:
            continue
        mapping[old_qid] = inject_tossup_packet_qid(old_qid, packet_id)
        counts[packet_id] += 1

    return mapping, counts, first_yasmine


def build_bonus_qid_mapping(
    records: list[dict],
) -> tuple[dict[str, str], dict[int, int], list[str]]:
    mapping: dict[str, str] = {}
    counts = {i: 0 for i in range(1, 7)}
    packet6_qids: list[str] = []
    packet6_slots = 0

    for record in records:
        old_qid = record["qid"]
        packet_id = assign_bonus_packet_1_to_5(record)
        if packet_id is not None:
            mapping[old_qid] = inject_bonus_packet_qid(old_qid, packet_id)
            counts[packet_id] += 1
            continue

        if packet6_slots < PACKET6_BONUS_SIZE:
            mapping[old_qid] = inject_bonus_packet_qid(old_qid, 6)
            counts[6] += 1
            packet6_slots += 1
            packet6_qids.append(normalize_bonus_qid(old_qid))

    return mapping, counts, packet6_qids


def apply_qid_mapping(records: list[dict], mapping: dict[str, str]) -> int:
    updated = 0
    for record in records:
        old_qid = record["qid"]
        if old_qid in mapping:
            record["qid"] = mapping[old_qid]
            updated += 1
    return updated


def apply_tossup_packet_qids(
    records: list[dict],
    *,
    verbose: bool = True,
) -> dict[int, int]:
    mapping, counts, first_yasmine = build_tossup_qid_mapping(records)
    updated = apply_qid_mapping(records, mapping)
    if verbose:
        print(f"Injected packet ids into {updated} tossup qids")
        print(f"First YasmineTaj qid for packet 6: {first_yasmine}")
        for packet_id in range(1, 7):
            print(f"  packet{packet_id}: {counts[packet_id]}")
        unassigned = len(records) - len(mapping)
        if unassigned:
            print(f"  unassigned (qid unchanged): {unassigned}")
    return counts


def apply_bonus_packet_qids(
    records: list[dict],
    *,
    verbose: bool = True,
) -> dict[int, int]:
    mapping, counts, packet6_qids = build_bonus_qid_mapping(records)
    updated = apply_qid_mapping(records, mapping)
    if verbose:
        print(f"Injected packet ids into {updated} bonus qids")
        for packet_id in range(1, 7):
            print(f"  packet{packet_id}: {counts[packet_id]}")
        unassigned = len(records) - len(mapping)
        if unassigned:
            print(f"  unassigned (qid unchanged): {unassigned}")
        if packet6_qids:
            print(f"  packet6 range: {packet6_qids[0]} .. {packet6_qids[-1]}")
    return counts


def rewrite_jsonl(path: Path, mapping: dict[str, str]) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    updated = 0
    unchanged = 0

    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        old_qid = row["qid"]
        if old_qid in mapping:
            row["qid"] = mapping[old_qid]
            updated += 1
        else:
            unchanged += 1
        out_lines.append(json.dumps(row, ensure_ascii=False))

    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return updated, unchanged
