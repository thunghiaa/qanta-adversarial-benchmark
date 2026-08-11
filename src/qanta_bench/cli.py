from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from qanta_bench.registry import load_registry
from qanta_bench.validation import validate_repository


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _list_models(args: argparse.Namespace) -> None:
    models = load_registry(args.registry)
    for cohort in ("qanta_submitted", "additional_benchmark"):
        selected = [model for model in models if model.cohort == cohort]
        print(f"{cohort} ({len(selected)})")
        for model in selected:
            print(f"  {model.id:50} {','.join(model.tasks):13} {model.kind}")


def _plot(args: argparse.Namespace) -> None:
    from qanta_bench.plots import generate_adversarialness_figures

    root = _repo_root()
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = root / "analysis_inputs"
        if args.scoring != "strict":
            input_dir = input_dir / args.scoring
    output_dir = args.output_dir or root / "figures"
    outputs = generate_adversarialness_figures(input_dir, output_dir, args.scoring)
    for output in outputs:
        print(output)


def _build_analysis(args: argparse.Namespace) -> None:
    from qanta_bench.adversarialness import build_adversarialness_tables

    root = args.root
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = root / "analysis_inputs"
        if args.scoring != "strict":
            output_dir = output_dir / args.scoring
    summary = build_adversarialness_tables(root, args.scoring, output_dir)
    for key, value in asdict(summary).items():
        print(f"{key}={value}")


def _validate(args: argparse.Namespace) -> None:
    report = validate_repository(args.root)
    print(f"jsonl_files={report.jsonl_files}")
    print(f"jsonl_rows={report.jsonl_rows}")
    if report.unknown_output_models:
        names = ", ".join(report.unknown_output_models)
        raise SystemExit(f"Outputs missing from configs/models.json: {names}")


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(prog="qanta-bench")
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_parser = subcommands.add_parser("list-models", help="show explicit model cohorts")
    list_parser.add_argument("--registry", type=Path, default=root / "configs" / "models.json")
    list_parser.set_defaults(handler=_list_models)

    plot_parser = subcommands.add_parser("plot-adversarialness", help="rebuild paper figures")
    plot_parser.add_argument("--scoring", choices=("strict", "pedant"), default="strict")
    plot_parser.add_argument("--input-dir", type=Path)
    plot_parser.add_argument("--output-dir", type=Path)
    plot_parser.set_defaults(handler=_plot)

    build_analysis_parser = subcommands.add_parser(
        "build-analysis", help="fit submitted-model adversarialness tables"
    )
    build_analysis_parser.add_argument("--scoring", choices=("strict", "pedant"), required=True)
    build_analysis_parser.add_argument("--root", type=Path, default=root)
    build_analysis_parser.add_argument("--output-dir", type=Path)
    build_analysis_parser.set_defaults(handler=_build_analysis)

    validate_parser = subcommands.add_parser("validate", help="validate registry and JSONL outputs")
    validate_parser.add_argument("--root", type=Path, default=root)
    validate_parser.set_defaults(handler=_validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
