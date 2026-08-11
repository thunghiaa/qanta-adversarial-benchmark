from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from loguru import logger

from backend.eval_suite import _drop_multimodal_rows_requiring_audio_decode, _strip_multimodal_audio_columns
from bonus_dataset_prep import prepare_advvqa_bonus
from packet_qid_injection import inject_bonus_packet_qid, inject_tossup_packet_qid
from tossup_dataset_prep import prepare_advvqa_tossup

_PACKET_DIR_RE = re.compile(r"^packet(\d+)(?:[._-].*)?$", re.IGNORECASE)


def parse_packet_selector_string(packets: str) -> set[int]:
    packet_ids: set[int] = set()
    for part in packets.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            p_start, p_end = part.split("-", 1)
            packet_ids.update(range(int(p_start), int(p_end) + 1))
        else:
            packet_ids.add(int(part))
    return packet_ids


def resolve_packet_filter(*, packet: int | None = None, packets: str | None = None) -> str | None:
    """Combine ``--packet N`` and ``--packets`` into the selector string for ``load_packet_dataset``."""
    if packet is not None and packets:
        raise SystemExit("Use only one of --packet and --packets/-p")
    if packet is not None:
        return str(packet)
    return packets


def _qid_packet_id(qid: str) -> int:
    for part in qid.split("-"):
        if part.startswith("packet") and part[6:].isdigit():
            return int(part[6:])
    raise ValueError(f"No packet id in qid: {qid}")


def _needs_inference_prep(dataset: Dataset) -> bool:
    if "run_indices" not in dataset.column_names:
        return True
    if "multimodal_tokens" not in dataset.column_names and "tokens" in dataset.column_names:
        return True
    return False


def _inference_base_dir(competition_type: str) -> Path:
    env_dir = os.getenv("PACKET_DATASET_LOCAL_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    subdir = "tossup_export" if competition_type == "tossup" else "bonus_export"
    return Path(__file__).resolve().parent.parent / subdir


def _packet_id_from_dir(path: Path) -> int | None:
    match = _PACKET_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def resolve_local_data_file(packet_dir: Path, competition_type: str) -> Path | None:
    """Return tossup/bonus JSON for a local ``packets/packetN`` folder, if present."""
    for candidate in (
        packet_dir / f"{competition_type}.json",
        packet_dir / competition_type / "data.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def discover_local_packet_dirs(
    packets_root: Path,
    packets: str | None = None,
) -> list[tuple[int, Path]]:
    """List ``(packet_id, packet_dir)`` under a local ``packets/`` tree."""
    packets_root = packets_root.expanduser().resolve()
    if not packets_root.is_dir():
        raise FileNotFoundError(f"Local packets directory not found: {packets_root}")

    single_pid = _packet_id_from_dir(packets_root)
    if single_pid is not None:
        dirs = [(single_pid, packets_root)]
    else:
        dirs: list[tuple[int, Path]] = []
        for child in sorted(packets_root.iterdir()):
            if not child.is_dir():
                continue
            pid = _packet_id_from_dir(child)
            if pid is not None:
                dirs.append((pid, child))

    if packets:
        allowed = parse_packet_selector_string(packets)
        dirs = [(packet_id, packet_dir) for packet_id, packet_dir in dirs if packet_id in allowed]

    return dirs


def local_competition_types_available(
    local_packets_dir: str | Path,
    packets: str | None = None,
) -> list[str]:
    """Which competition types have JSON on disk under the local packet tree."""
    found: set[str] = set()
    for _packet_id, packet_dir in discover_local_packet_dirs(Path(local_packets_dir), packets):
        if resolve_local_data_file(packet_dir, "tossup"):
            found.add("tossup")
        if resolve_local_data_file(packet_dir, "bonus"):
            found.add("bonus")
    return sorted(found)


def _load_json_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _normalize_tossup_record(record: dict) -> dict:
    out = copy.deepcopy(record)
    if not out.get("image_paths"):
        out["image_paths"] = [
            img["path"]
            for img in (out.get("images") or [])
            if isinstance(img, dict) and img.get("path")
        ]
    return out


def load_local_packet_dataset(
    local_packets_dir: str | Path,
    competition_type: str,
    *,
    packets: str | None = None,
) -> Dataset:
    """Load packet questions from ``packets/packetN/{tossup,bonus}.json`` (no Hugging Face)."""
    packet_dirs = discover_local_packet_dirs(Path(local_packets_dir), packets)
    if not packet_dirs:
        raise FileNotFoundError(
            f"No packet folders (packet1, packet2, ...) under {local_packets_dir}"
            + (f" matching filter {packets!r}" if packets else "")
        )

    all_records: list[dict] = []
    for packet_id, packet_dir in packet_dirs:
        data_file = resolve_local_data_file(packet_dir, competition_type)
        if data_file is None:
            logger.warning(f"No {competition_type} data in {packet_dir}; skipping")
            continue

        records = _load_json_records(data_file)
        logger.info(f"Loaded {len(records)} {competition_type} row(s) from {data_file}")
        for record in records:
            row = (
                _normalize_tossup_record(record)
                if competition_type == "tossup"
                else copy.deepcopy(record)
            )
            if competition_type == "tossup":
                row["qid"] = inject_tossup_packet_qid(row["qid"], packet_id)
                row = prepare_advvqa_tossup(row, packet_dir, embed_images=True)
            else:
                row["qid"] = inject_bonus_packet_qid(row["qid"], packet_id)
                row = prepare_advvqa_bonus(row, packet_dir, embed_images=True)
            row["_base_dir"] = str(packet_dir)
            all_records.append(row)

    if not all_records:
        raise FileNotFoundError(
            f"No {competition_type} records found under {local_packets_dir}"
            + (f" (packets={packets!r})" if packets else "")
        )

    logger.info(
        f"Local packet dataset: {len(all_records)} {competition_type} question(s) "
        f"from {len(packet_dirs)} folder(s)"
    )
    return Dataset.from_list(all_records)


def load_packet_dataset(
    dataset_name: str,
    competition_type: str,
    *,
    split: str,
    packets: str | None = None,
    local_packets_dir: str | Path | None = None,
) -> Dataset:
    if local_packets_dir is not None:
        return load_local_packet_dataset(
            local_packets_dir,
            competition_type,
            packets=packets,
        )

    logger.info(f"Loading {dataset_name} config={competition_type} split={split}")
    dataset = load_dataset(dataset_name, competition_type, split=split)

    if "leadin_audio" in dataset.column_names or "part_audio" in dataset.column_names:
        dataset = _drop_multimodal_rows_requiring_audio_decode(dataset)
        dataset = _strip_multimodal_audio_columns(dataset)

    if competition_type == "tossup" and _needs_inference_prep(dataset):
        base_dir = _inference_base_dir(competition_type)
        logger.info(f"Preparing tossups for inference (image base_dir={base_dir})")

        def _prep_row(row: dict) -> dict:
            return prepare_advvqa_tossup(
                row,
                base_dir,
                embed_images=not _images_are_embedded(row),
            )

        dataset = dataset.map(_prep_row, desc="Prepare tossups for inference")

    if competition_type == "bonus" and _needs_bonus_inference_prep(dataset):
        base_dir = _inference_base_dir(competition_type)
        logger.info(f"Preparing bonuses for inference (image base_dir={base_dir})")

        def _prep_bonus_row(row: dict) -> dict:
            return prepare_advvqa_bonus(
                row,
                base_dir,
                embed_images=not _bonus_images_are_embedded(row),
            )

        dataset = dataset.map(_prep_bonus_row, desc="Prepare bonuses for inference")

    if packets:
        packet_ids = parse_packet_selector_string(packets)
        dataset = dataset.filter(lambda x: _qid_packet_id(x["qid"]) in packet_ids)
        logger.info(f"Filtered to packets {sorted(packet_ids)}: {len(dataset)} questions")

    return dataset


def _images_are_embedded(row: dict) -> bool:
    images = row.get("images")
    if not images:
        return False
    first = images[0]
    if hasattr(first, "size"):  # PIL Image from Hub parquet
        return True
    if isinstance(first, dict) and first.get("bytes"):
        return True
    return False


def _needs_bonus_inference_prep(dataset: Dataset) -> bool:
    if len(dataset) == 0:
        return False
    if "part_image_splits" not in dataset.column_names:
        return True
    if "leadin_images" not in dataset.column_names and row_has_bonus_image_paths(dataset[0]):
        return True
    return False


def row_has_bonus_image_paths(row: dict) -> bool:
    if row.get("leadin_image"):
        return True
    part_images = row.get("part_images")
    if part_images and isinstance(part_images[0], dict):
        return True
    return False


def _bonus_images_are_embedded(row: dict) -> bool:
    part_images = row.get("part_images")
    if part_images:
        first = part_images[0]
        if hasattr(first, "size"):
            return True
        if isinstance(first, dict) and first.get("bytes"):
            return True
    leadin_images = row.get("leadin_images")
    if leadin_images:
        first = leadin_images[0]
        if hasattr(first, "size"):
            return True
        if isinstance(first, dict) and first.get("bytes"):
            return True
    return False
