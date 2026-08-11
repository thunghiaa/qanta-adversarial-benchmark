#!/usr/bin/env python3
"""Generate and store model outputs on competition packet questions.

Examples (Hugging Face packet-questions dataset):
    python scripts/generate_outputs.py -s username/model_name --packet 2
    python scripts/generate_outputs.py --submissions-file models.txt --packets 1-6

Examples (local packets/ tree, no Hub upload):
    python scripts/generate_outputs.py --local-packets-dir packets/packet2 \\
        --hf-models-file models.txt --competition-type tossup
    python scripts/generate_outputs.py --local-packets-dir packets/packet2 \\
        --hf-model nttruong1007/qb-yi15-9b --competition-type tossup --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKET_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PACKET_ROOT / "src"))
from bootstrap import setup_paths

setup_paths()

from backend.eval_suite import generate_bonus_outputs
from dataset import (
    load_packet_dataset,
    local_competition_types_available,
    resolve_packet_filter,
)
from packet_generation import generate_packet_tossup_outputs
from packet_envs import (
    DEV_PACKET_DATASET,
    LOCAL_PACKET_OUTPUT_PATH,
    PACKET_BONUS_SPLIT,
    PACKET_DATASET,
    PACKET_DATASET_SPLIT,
    PACKET_EVAL_SPLIT,
)
from io_utils import load_jsonl, packet_outputs_path, write_packet_outputs
from submissions import build_submission_index, resolve_submission_names, sync_submissions


def _expand_submission_tokens(raw: str) -> list[str]:
    """Split user input on commas or whitespace when multiple username/model names are given."""
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    if raw.count("/") > 1 and any(ch.isspace() for ch in raw):
        return [part.strip() for part in raw.split() if part.strip()]
    return [raw]


def _resolve_input_path(path: str) -> Path:
    """Resolve a user path; try packet-outputs root when relative path is missing from cwd."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    from_packet_root = _PACKET_ROOT / path
    if from_packet_root.is_file():
        return from_packet_root
    raise SystemExit(f"File not found: {path} (also tried {_PACKET_ROOT / path})")


def _read_name_list_file(path: str, *, label: str) -> list[str]:
    filepath = _resolve_input_path(path)
    names: list[str] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    if not names:
        raise SystemExit(f"{label} {filepath} is empty (add one username/model_name per line)")
    return names


def _parse_submission_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return []
    names: list[str] = []
    if args.submission:
        names.extend(_expand_submission_tokens(args.submission))
    if args.submissions:
        for token in args.submissions.split(","):
            names.extend(_expand_submission_tokens(token))
    if args.submissions_file:
        names.extend(_read_name_list_file(args.submissions_file, label="--submissions-file"))
    if not names:
        raise SystemExit("Provide -s/--submission, --submissions, --submissions-file, or --all")
    return names


def _default_local_output_dir() -> str:
    return str(_PACKET_ROOT / "local_outputs")


def make_local_hf_submission(model_id: str, competition_type: str) -> dict:
    model_id = model_id.strip()
    # A local pipeline-repo directory (e.g. qanta-submit/build/<short>) is
    # accepted directly: username becomes "local" and the runner loads from
    # the path (submission["local_path"]), no Hub push required.
    local_dir = Path(model_id).expanduser()
    if local_dir.is_dir() and (local_dir / "config.json").exists():
        slug = local_dir.name.replace("/", "_")
        return {
            "id": f"local__{competition_type}__{slug}",
            "username": "local",
            "model_name": slug,
            "local_path": str(local_dir.resolve()),
            "competition_type": competition_type,
            "submission_type": "hf_pipeline",
        }
    if "/" not in model_id:
        raise ValueError(f"Expected username/model_name or a local pipeline dir, got {model_id!r}")
    username, _, model_name = model_id.partition("/")
    slug = model_name.replace("/", "_")
    return {
        "id": f"local__{competition_type}__{slug}",
        "username": username,
        "model_name": model_name,
        "competition_type": competition_type,
        "submission_type": "hf_pipeline",
    }


def _parse_hf_model_names(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if args.hf_model:
        models.append(args.hf_model.strip())
    if args.hf_models:
        for token in args.hf_models.split(","):
            models.extend(_expand_submission_tokens(token))
    if args.hf_models_file:
        models.extend(_read_name_list_file(args.hf_models_file, label="--hf-models-file"))
    return models


def _resolve_local_targets(
    args: argparse.Namespace,
    index: dict,
    packets: str | None,
    manager,
) -> list[tuple[str, dict]]:
    local_dir = Path(args.local_packets_dir).expanduser().resolve()
    hf_models = _parse_hf_model_names(args)
    if hf_models:
        if args.competition_type:
            competition_types = [args.competition_type]
        else:
            competition_types = local_competition_types_available(local_dir, packets)
            if not competition_types:
                raise SystemExit(f"No tossup/bonus JSON found under {local_dir}")
        targets: list[tuple[str, dict]] = []
        for model_id in hf_models:
            for ctype in competition_types:
                targets.append((ctype, make_local_hf_submission(model_id, ctype)))
        return targets

    if args.all:
        raise SystemExit(
            "--all is not supported with --local-packets-dir "
            "(use --hf-model/--hf-models/--hf-models-file or name advcal submissions)"
        )

    names = _parse_submission_names(args)
    return resolve_submission_names(
        index, names, competition_type=args.competition_type, manager=manager
    )


def _resolve_hub_targets(
    args: argparse.Namespace,
    index: dict,
    manager,
) -> list[tuple[str, dict]]:
    if args.all:
        targets: list[tuple[str, dict]] = []
        for mode, mode_index in index.items():
            if args.competition_type and mode != args.competition_type:
                continue
            for _model_name, submission in sorted(mode_index.items()):
                targets.append((mode, submission))
        return targets

    names = _parse_submission_names(args)
    return resolve_submission_names(
        index, names, competition_type=args.competition_type, manager=manager
    )


def _resolve_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset:
        return args.dataset
    if args.dev:
        return DEV_PACKET_DATASET
    return PACKET_DATASET


def _split_for_submission(competition_type: str, override: str | None) -> str:
    if override:
        return override
    if competition_type == "tossup":
        return PACKET_DATASET_SPLIT
    if competition_type == "bonus":
        return PACKET_BONUS_SPLIT
    raise ValueError(f"Invalid competition_type: {competition_type}")


def _generate_for_submission(
    submission: dict,
    *,
    dataset_name: str,
    split: str,
    packets: str | None,
    local_packets_dir: str | None,
    local_output_dir: str | None,
    debug: bool,
    reprocess: bool,
    dry_run: bool,
    upload: bool,
) -> None:
    competition_type = submission["competition_type"]
    submission_id = submission["id"]
    model_label = f"{submission['username']}/{submission['model_name']}"

    output_root = local_output_dir or LOCAL_PACKET_OUTPUT_PATH
    output_path = packet_outputs_path(
        competition_type,
        submission_id,
        PACKET_EVAL_SPLIT,
        output_root,
    )

    dataset = load_packet_dataset(
        dataset_name,
        competition_type,
        split=split,
        packets=packets,
        local_packets_dir=local_packets_dir,
    )
    if debug and len(dataset) > 2:
        logger.info("Debug mode: limiting to 2 questions")
        dataset = dataset.select(range(2))

    processed_outputs: list[dict] = []
    if not reprocess and Path(output_path).exists():
        processed_outputs = load_jsonl(output_path)
        processed_qids = {row["qid"] for row in processed_outputs}
        dataset = dataset.filter(lambda x: x["qid"] not in processed_qids)
        if not len(dataset):
            logger.info(f"{model_label}: all questions already processed at {output_path}")
            return

    if dry_run:
        logger.info(
            f"[dry-run] {model_label} ({competition_type}): "
            f"{len(dataset)} new questions -> {output_path}"
        )
        return

    logger.info(f"Generating outputs for {model_label} on {len(dataset)} questions")
    if competition_type == "tossup":
        new_outputs = generate_packet_tossup_outputs(submission, dataset)
    elif competition_type == "bonus":
        new_outputs = generate_bonus_outputs(submission, dataset)
    else:
        raise ValueError(f"Invalid competition_type: {competition_type}")

    all_outputs = processed_outputs + new_outputs
    written = write_packet_outputs(
        all_outputs,
        competition_type=competition_type,
        submission_id=submission_id,
        local_outdir=output_root,
        upload=upload,
    )
    logger.info(f"Wrote {len(all_outputs)} outputs to {written}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate packet outputs for submitted models.")
    parser.add_argument("-s", "--submission", type=str, help="Single submission as username/model_name")
    parser.add_argument("--submissions", type=str, help="Comma-separated username/model_name list")
    parser.add_argument("--submissions-file", type=str, help="File with one username/model_name per line")
    parser.add_argument("--all", action="store_true", help="Run all eligible submissions")
    parser.add_argument(
        "--competition-type",
        choices=["tossup", "bonus"],
        help="Only run submissions for this competition type (e.g. bonus only)",
    )
    parser.add_argument(
        "--packet",
        type=int,
        metavar="N",
        help="Run only packet N (matches qids like advvqa-packetN-t-... or advvqa-packetN-b-...)",
    )
    parser.add_argument("--packets", "-p", type=str, help='Packet filter, e.g. "1,3-5"')
    parser.add_argument(
        "--local-packets-dir",
        type=str,
        help=(
            "Load questions from local packets/ tree instead of Hugging Face "
            "(e.g. packets or packets/packet2)"
        ),
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        help="Run one HF pipeline model directly as username/model_name (no advcal submission required)",
    )
    parser.add_argument(
        "--hf-models",
        type=str,
        help="Comma-separated HF models for local packet runs (alternative to --hf-model)",
    )
    parser.add_argument(
        "--hf-models-file",
        type=str,
        help="File with one username/model_name per line for local packet runs",
    )
    parser.add_argument(
        "--local-output-dir",
        type=str,
        help=f"Write JSONL outputs here when using --local-packets-dir (default: {_default_local_output_dir()})",
    )
    parser.add_argument("--dataset", type=str, help="Override packet HF dataset repo")
    parser.add_argument(
        "--split",
        type=str,
        help="Override dataset split for all submissions (default: tossup->PACKET_DATASET_SPLIT, bonus->PACKET_BONUS_SPLIT)",
    )
    parser.add_argument("--dev", action="store_true", help=f"Use dev stand-in dataset ({DEV_PACKET_DATASET})")
    parser.add_argument("--debug", action="store_true", help="Limit to 2 questions per submission")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without inference or uploads")
    parser.add_argument("--no-upload", action="store_true", help="Write local JSONL only")
    parser.add_argument("--reprocess", action="store_true", help="Ignore existing outputs and rerun all")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing submissions from Hugging Face")
    args = parser.parse_args()

    if args.local_packets_dir and _parse_hf_model_names(args) and (
        args.submission or args.submissions or args.submissions_file
    ):
        raise SystemExit(
            "Use either --hf-model/--hf-models/--hf-models-file or -s/--submissions with --local-packets-dir"
        )
    if _parse_hf_model_names(args) and not args.local_packets_dir:
        raise SystemExit("--hf-model/--hf-models/--hf-models-file requires --local-packets-dir")
    if args.local_packets_dir and not (
        _parse_hf_model_names(args) or args.submission or args.submissions or args.submissions_file
    ):
        raise SystemExit(
            "With --local-packets-dir, provide --hf-model/--hf-models/--hf-models-file "
            "or -s/--submissions/--submissions-file"
        )

    manager = sync_submissions(sync_from_hub=not args.no_sync)
    index = build_submission_index(manager)
    dataset_name = _resolve_dataset_name(args)
    upload = not args.no_upload and not args.dry_run and not args.local_packets_dir
    packets = resolve_packet_filter(packet=args.packet, packets=args.packets)
    local_packets_dir = str(Path(args.local_packets_dir).expanduser()) if args.local_packets_dir else None
    local_output_dir = (
        str(Path(args.local_output_dir).expanduser())
        if args.local_output_dir
        else (_default_local_output_dir() if local_packets_dir else None)
    )

    if local_packets_dir:
        targets = _resolve_local_targets(args, index, packets, manager)
    else:
        targets = _resolve_hub_targets(args, index, manager)

    if not targets:
        logger.warning("No eligible submissions to run")
        return

    source = f"local={local_packets_dir}" if local_packets_dir else f"dataset={dataset_name}"
    logger.info(
        f"Source={source} tossup_split={PACKET_DATASET_SPLIT} "
        f"bonus_split={PACKET_BONUS_SPLIT} eval_split={PACKET_EVAL_SPLIT} "
        f"packets={packets!r} targets={len(targets)} upload={upload} "
        f"output_dir={local_output_dir or LOCAL_PACKET_OUTPUT_PATH}"
    )

    failures = 0
    failed_labels: list[str] = []
    for _mode, submission in targets:
        model_label = f"{submission['username']}/{submission['model_name']}"
        split = _split_for_submission(submission["competition_type"], args.split)
        logger.info(f"{model_label}: type={submission['competition_type']} split={split}")
        try:
            _generate_for_submission(
                submission,
                dataset_name=dataset_name,
                split=split,
                packets=packets,
                local_packets_dir=local_packets_dir,
                local_output_dir=local_output_dir,
                debug=args.debug,
                reprocess=args.reprocess,
                dry_run=args.dry_run,
                upload=upload,
            )
        except Exception as exc:
            failures += 1
            failed_labels.append(model_label)
            logger.exception(f"Failed for {model_label}: {exc}")

    if failures:
        raise SystemExit(
            f"{failures} submission(s) failed: {', '.join(failed_labels)}"
        )


if __name__ == "__main__":
    main()
