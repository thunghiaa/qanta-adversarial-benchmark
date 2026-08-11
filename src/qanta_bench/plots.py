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

PRETTY_MODEL_LABELS = {
    "thunghiatruong/claude-heavyweight": "claude-heavyweight",
    "ileygreg/qanta_41mini_v2": "qb-v2 (gpt-4o)",
    "168mxie/heavy1": "heavy1 (gpt-4o)",
    "ileygreg/qanta_41miniv1": "qb-v1 (gpt-4o)",
    "168mxie/heavy2": "heavy2 (gpt-4o)",
    "ileygreg/qanta_41mini_v3": "qb-v3 (gpt-4o)",
    "168mxie/heavy3": "heavy3 (gpt-4o)",
    "eshanli/eshan_v3_calibrated_nano": "eshan-v3-nano",
    "Mokshj1/moksh_tossup_multimodal_qa": "moksh-mm",
    "sidS216/sid_tossup_agent": "sid-tossup",
    "nirjharami108/qanta41mini_v1": "nirjhar-v1 (gpt-4o)",
    "divyagoyal6224/divya6224": "divya6224",
    "divyagoyal6224/divyagoyal": "divyagoyal",
    "168mxie/anthropic": "anthropic (deployed)",
    "eshanli/calibrated_3step_tossup_v1": "eshan-3step",
    "168mxie/tw-step-simple": "tw-step-simple",
}


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


def plot_theta_ladder(theta_csv: Path, output: Path) -> None:
    rows = _read_csv(theta_csv)
    if not rows:
        raise ValueError(f"No ability rows in {theta_csv}")

    ai = [
        (row["model"], float(row["theta_qanta_scale"]))
        for row in rows
        if row["model"] not in {"HUMAN", "HUMAN_buzz_only"}
    ]
    human_rows = [row for row in rows if row["model"] == "HUMAN"]
    if not human_rows:
        raise ValueError(f"No HUMAN ability row in {theta_csv}")
    theta_human = float(human_rows[0]["theta_qanta_scale"])
    ai.sort(key=lambda item: item[1])

    labels = [PRETTY_MODEL_LABELS.get(model, model) for model, _ in ai]
    values = np.array([theta for _, theta in ai])
    best_ai = float(values.max())

    figure, axis = plt.subplots(figsize=(9.5, 6.6))
    y = np.arange(len(values))
    axis.hlines(y, 0, values, color="#D9D9D9", linewidth=2, zorder=1)
    axis.scatter(values, y, s=90, color=AI, zorder=3, label="QANTA AI system")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=11)
    for y_value, theta in zip(y, values):
        axis.annotate(
            f"{theta:+.2f}",
            (theta, y_value),
            xytext=(6 if theta >= 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if theta >= 0 else "right",
            fontsize=9,
            color=MUTED,
        )

    axis.axvline(theta_human, color=HUMAN, linewidth=2.5, zorder=2)
    axis.annotate(
        f"Humans  θ = {theta_human:+.2f}",
        (theta_human, len(values) - 0.4),
        xytext=(8, 0),
        textcoords="offset points",
        color=HUMAN,
        fontweight="bold",
        fontsize=13,
        va="center",
    )
    axis.axvline(0, color="#D9D9D9", linewidth=1, linestyle="--", zorder=0)
    axis.annotate("AI field mean = 0", (0, -0.9), ha="center", color=MUTED, fontsize=9)

    bracket_y = len(values) - 7.5
    axis.annotate(
        "",
        xy=(theta_human, bracket_y),
        xytext=(best_ai, bracket_y),
        arrowprops={"arrowstyle": "<->", "color": GAP, "lw": 2},
    )
    axis.annotate(
        f"+{theta_human - best_ai:.2f} logits\nabove best AI",
        (theta_human + 0.07, bracket_y),
        ha="left",
        va="center",
        color=GAP,
        fontsize=10,
        fontweight="bold",
    )

    axis.set_xlabel("Knowledge ability  θ  (shared IRT scale, AI field mean-centered)")
    axis.set_xlim(-1.9, 2.3)
    axis.set_ylim(-1.5, len(values) + 0.2)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


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


def generate_adversarialness_figures(
    input_dir: Path, output_dir: Path, scoring: str = "strict"
) -> list[Path]:
    if scoring not in {"strict", "pedant"}:
        raise ValueError(f"Unknown scoring regime: {scoring}")
    configure_style()
    marker = "" if scoring == "strict" else f"_{scoring}"
    outputs = [
        output_dir / f"qanta_fig1_theta_ladder{marker}_paper.png",
        output_dir / f"qanta_fig2_difficulty_curves{marker}_paper.png",
        output_dir / f"qanta_fig3_delta_hist{marker}_paper.png",
    ]
    plot_theta_ladder(input_dir / "qanta_ai_theta.csv", outputs[0])
    plot_difficulty_curves(input_dir / "qanta_fig_mean_curves.csv", outputs[1])
    plot_adversarialness_histogram(input_dir / "qanta_fig_delta_area.csv", outputs[2])
    return outputs
