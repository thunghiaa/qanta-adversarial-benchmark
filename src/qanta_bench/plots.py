"""Paper-ready plots from versioned adversarialness tables.

This module intentionally has no dependency on compIRT or the inference stack.
The scientific tables are immutable inputs; plotting is deterministic and can
be rerun in a lightweight environment.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AI = "#0072B2"
HUMAN = "#009E73"
GAP = "#E69F00"
INK = "#222222"
MUTED = "#8A8A8A"
GRID = "#EEEEEE"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.dpi": 130,
        }
    )


def plot_difficulty_curves(curves_csv: Path, output: Path) -> None:
    rows = _read_csv(curves_csv)
    if not rows:
        raise ValueError(f"No curve rows in {curves_csv}")
    reveal = np.array([float(row["reveal_fraction"]) for row in rows])
    difficulty_ai = np.array([float(row["mean_d_ai"]) for row in rows])
    difficulty_human = np.array([float(row["mean_d_human"]) for row in rows])
    solve_ai = np.array([float(row["mean_solve_ai"]) for row in rows])
    solve_human = np.array([float(row["mean_solve_human"]) for row in rows])

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    left.fill_between(reveal, difficulty_human, difficulty_ai, color=GAP, alpha=0.25)
    left.plot(reveal, difficulty_ai, color=AI, linewidth=2.8, label=r"AI $d^{AI}(r)$")
    left.plot(
        reveal,
        difficulty_human,
        color=HUMAN,
        linewidth=2.8,
        label=r"Human $d^{H}(r)$",
    )
    left.set(
        xlabel="Reveal fraction r",
        ylabel="Item difficulty d(r) (logits)",
        title=r"Ability-normalized difficulty gap $\delta(r)$",
        xlim=(0, 1),
    )
    left.legend(frameon=False, fontsize=10)

    right.fill_between(reveal, solve_ai, solve_human, color=GAP, alpha=0.25)
    right.plot(reveal, solve_human, color=HUMAN, linewidth=2.8, label="Human solve")
    right.plot(reveal, solve_ai, color=AI, linewidth=2.8, label="AI solve")
    right.set(
        xlabel="Reveal fraction r",
        ylabel="Solve probability",
        title=r"Realized solve gap $G(r)=S^{H}(r)-S^{AI}(r)$",
        xlim=(0, 1),
        ylim=(0, 0.8),
    )
    right.legend(frameon=False, fontsize=10)

    for axis in (left, right):
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _zero_aligned_bins(values: np.ndarray, width: float = 0.15) -> np.ndarray:
    low_steps = int(np.ceil(max(0.0, -float(values.min())) / width))
    high_steps = int(np.ceil(max(0.0, float(values.max())) / width))
    if low_steps + high_steps == 0:
        high_steps = 1
    return np.arange(-low_steps, high_steps + 1, dtype=float) * width


def plot_adversarialness_histogram(items_csv: Path, output: Path) -> None:
    rows = _read_csv(items_csv)
    if not rows:
        raise ValueError(f"No item rows in {items_csv}")
    values = np.array([float(row["delta_area"]) for row in rows])
    positive = int((values > 0).sum())
    bins = _zero_aligned_bins(values)
    counts, edges = np.histogram(values, bins=bins)

    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    for count, lower in zip(counts, edges[:-1]):
        axis.bar(
            lower,
            count,
            width=edges[1] - edges[0],
            align="edge",
            color=AI if lower >= -1e-12 else MUTED,
            alpha=0.85,
        )
    axis.bar(np.nan, 0, color=AI, alpha=0.85, label=f"AI-harder: {positive}")
    axis.bar(
        np.nan,
        0,
        color=MUTED,
        alpha=0.85,
        label=f"human-harder: {len(values) - positive}",
    )
    axis.axvline(0, color=INK, linewidth=1.2, linestyle="--")
    axis.axvline(values.mean(), color=GAP, linewidth=2.5, label=f"mean = {values.mean():+.2f}")
    axis.set(
        xlabel=r"Per-item adversarialness $\delta_{area}=\int_0^1\delta_j(r)\,dr$",
        ylabel="Number of tossups",
    )
    axis.legend(frameon=False, fontsize=11)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def generate_adversarialness_figures(input_dir: Path, output_dir: Path) -> list[Path]:
    configure_style()
    outputs = [
        output_dir / "qanta_fig2_difficulty_curves_paper.png",
        output_dir / "qanta_fig3_delta_hist_paper.png",
    ]
    plot_difficulty_curves(input_dir / "qanta_fig_mean_curves.csv", outputs[0])
    plot_adversarialness_histogram(input_dir / "qanta_fig_delta_area.csv", outputs[1])
    return outputs
