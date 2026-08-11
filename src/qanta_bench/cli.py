from __future__ import annotations

import argparse
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

    outputs = generate_adversarialness_figures(args.input_dir, args.output_dir)
    for output in outputs:
        print(output)


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
    plot_parser.add_argument("--input-dir", type=Path, default=root / "analysis_inputs")
    plot_parser.add_argument("--output-dir", type=Path, default=root / "figures")
    plot_parser.set_defaults(handler=_plot)

    validate_parser = subcommands.add_parser("validate", help="validate registry and JSONL outputs")
    validate_parser.add_argument("--root", type=Path, default=root)
    validate_parser.set_defaults(handler=_validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
