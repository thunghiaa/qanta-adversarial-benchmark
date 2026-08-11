"""Build submitted-field human-vs-AI adversarialness tables from model outputs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qanta_bench.registry import models_for

PACKETS = (1, 2, 3, 4, 5)
GRID = np.linspace(0.0, 1.0, 21)
VALID_SCORING = frozenset({"strict", "pedant"})

Trace = tuple[float, int, bool]
ItemKey = tuple[int, str]


@dataclass(frozen=True)
class AnalysisSummary:
    scoring: str
    models: int
    ai_items: int
    human_items: int
    human_buzzes: int
    human_censored: int
    adversarial_items: int
    human_outsolve_items: int
    organizer_correlation: float
    human_theta: float
    best_ai_theta: float


def _sig(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value)
    return np.where(
        array >= 0,
        1 / (1 + np.exp(-np.clip(array, None, 700))),
        np.exp(np.clip(array, -700, 0)) / (1 + np.exp(np.clip(array, -700, 0))),
    )


def _model_from_filename(path: Path) -> str:
    fields = path.stem.split("__")
    if len(fields) < 4:
        raise ValueError(f"Cannot identify model from {path.name}")
    return f"{fields[-2]}/{fields[-1]}"


def _gold_token_lengths(root: Path, packet: int) -> dict[str, int]:
    path = root / "data" / "packets" / f"packet{packet}_final" / "tossup" / "data.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["qid"].split("-")[-1]: len(row.get("tokens", [])) for row in rows}


def load_ai_traces(
    root: Path, scoring: str
) -> tuple[dict[str, dict[ItemKey, list[Trace]]], tuple[str, ...]]:
    """Load packets 1-5 for exactly the registered submitted tossup systems."""
    if scoring not in VALID_SCORING:
        raise ValueError(f"Unknown scoring regime: {scoring}")

    registry_path = root / "configs" / "models.json"
    expected = tuple(
        sorted(
            model.id
            for model in models_for("qanta_submitted", "tossup", registry_path)
        )
    )
    expected_set = set(expected)
    traces: dict[str, dict[ItemKey, list[Trace]]] = defaultdict(dict)
    result_root = root / "results" / "qanta_submitted"
    output_subdir = "Tossup" if scoring == "strict" else "Tossup_pedant"

    for packet in PACKETS:
        packet_dirs = sorted(result_root.glob(f"Packet {packet} -*"))
        if len(packet_dirs) != 1:
            raise ValueError(f"Expected one submitted result directory for packet {packet}")
        token_lengths = _gold_token_lengths(root, packet)
        for path in sorted((packet_dirs[0] / output_subdir).glob("*.jsonl")):
            model = _model_from_filename(path)
            if model not in expected_set:
                raise ValueError(f"Non-submitted model in {output_subdir}: {model}")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    nano = row["qid"].split("-")[-1]
                    token_count = token_lengths.get(nano, 0)
                    if not token_count:
                        continue
                    sequence: list[Trace] = []
                    runs = sorted(
                        row.get("run_outputs", []),
                        key=lambda run: float(run.get("run_idx", 1e9)),
                    )
                    for run in runs:
                        correct = int(str(run.get("correct")) in {"1", "True", "true"})
                        if scoring == "pedant" and str(run.get("guess", "")).strip() == "ERROR":
                            correct = 0
                        reveal = min(1.0, float(run["token_position"]) / token_count)
                        buzz = str(run.get("buzz")) in {"1", "True", "true"}
                        sequence.append((reveal, correct, buzz))
                    if sequence:
                        traces[model][(packet, nano)] = sequence

    missing = sorted(expected_set - set(traces))
    if missing:
        raise ValueError(f"Missing submitted outputs: {', '.join(missing)}")
    return dict(traces), expected


def fit_ai_correctness(
    models: tuple[str, ...],
    traces: dict[str, dict[ItemKey, list[Trace]]],
    items: list[ItemKey],
) -> tuple[np.ndarray, dict[ItemKey, tuple[float, float, float]]]:
    """Fit P(correct)=sigmoid(theta_i-beta_j*sigmoid(gamma_j*(mu_j-r)))."""
    model_index = {model: index for index, model in enumerate(models)}
    item_index = {item: index for index, item in enumerate(items)}
    rows = [
        (model_index[model], item_index[item], reveal, correct)
        for model in models
        for item in items
        if item in traces[model]
        for reveal, correct, _ in traces[model][item]
    ]
    model_ids = np.array([row[0] for row in rows])
    item_ids = np.array([row[1] for row in rows])
    reveal = np.array([row[2] for row in rows])
    correct = np.array([row[3] for row in rows], dtype=float)
    model_count, item_count = len(models), len(items)

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = values[:model_count]
        beta = values[model_count : model_count + item_count]
        gamma_raw = values[model_count + item_count : model_count + 2 * item_count]
        mu_raw = values[model_count + 2 * item_count :]
        return theta, beta, gamma_raw, mu_raw

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        theta, beta, gamma_raw, mu_raw = unpack(values)
        gamma = np.logaddexp(0, gamma_raw)
        mu = _sig(mu_raw)
        shape = _sig(gamma[item_ids] * (mu[item_ids] - reveal))
        logits = theta[model_ids] - beta[item_ids] * shape
        probability = _sig(logits)
        loss = (
            np.sum(np.logaddexp(0, logits) - correct * logits)
            + np.sum(theta**2) / 2
            + np.sum(beta**2) / 2
            + np.sum(gamma_raw**2) / 18
            + np.sum(mu_raw**2) / 18
        )
        residual = probability - correct
        shape_gradient = shape * (1 - shape)

        theta_gradient = np.zeros(model_count)
        np.add.at(theta_gradient, model_ids, residual)
        theta_gradient += theta

        beta_gradient = np.zeros(item_count)
        np.add.at(beta_gradient, item_ids, residual * (-shape))
        beta_gradient += beta

        gamma_gradient = np.zeros(item_count)
        np.add.at(
            gamma_gradient,
            item_ids,
            residual * (-beta[item_ids] * shape_gradient * (mu[item_ids] - reveal)),
        )
        gamma_gradient = gamma_gradient * _sig(gamma_raw) + gamma_raw / 9

        mu_gradient = np.zeros(item_count)
        np.add.at(
            mu_gradient,
            item_ids,
            residual * (-beta[item_ids] * shape_gradient * gamma[item_ids]),
        )
        mu_gradient = mu_gradient * (mu * (1 - mu)) + mu_raw / 9
        gradient = np.concatenate(
            (theta_gradient, beta_gradient, gamma_gradient, mu_gradient)
        )
        return float(loss), gradient

    initial = np.concatenate(
        (
            np.zeros(model_count),
            np.ones(item_count),
            np.full(item_count, math.log(math.expm1(4.0))),
            np.zeros(item_count),
        )
    )
    fitted = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 800, "ftol": 1e-10},
    ).x
    theta, beta, gamma_raw, mu_raw = unpack(fitted)
    theta = theta - theta.mean()
    parameters = {
        item: (
            float(beta[index]),
            float(np.logaddexp(0, gamma_raw[index])),
            float(_sig(mu_raw[index])),
        )
        for item, index in item_index.items()
    }
    return theta, parameters


def load_human_observations(root: Path) -> dict[str, dict[str, object]]:
    path = root / "data" / "human" / "human_buzz_observations.csv"
    human: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            nano = row["nano"]
            item = human.setdefault(
                nano,
                {
                    "packet": int(row["packet"]),
                    "observations": [],
                    "n_play": 0,
                    "n_buzz": 0,
                    "n_censor": 0,
                },
            )
            censored = row["censored"] == "1"
            item["observations"].append(
                (float(row["reveal_fraction"]), int(row["correct"]), censored)
            )
            item["n_play"] += 1
            item["n_censor" if censored else "n_buzz"] += 1
    return human


def fit_human_levels(
    human: dict[str, dict[str, object]],
    ai_items: dict[ItemKey, tuple[float, float, float]],
    *,
    use_censored: bool,
) -> tuple[float, dict[str, float], dict[str, tuple[float, float]]]:
    keymap = {item[1]: parameters for item, parameters in ai_items.items()}
    items = [nano for nano in human if nano in keymap]
    item_index = {nano: index for index, nano in enumerate(items)}
    beta_ai = np.array([keymap[nano][0] for nano in items])
    gamma = np.array([keymap[nano][1] for nano in items])
    mu = np.array([keymap[nano][2] for nano in items])

    rows: list[tuple[int, float, int]] = []
    for nano in items:
        index = item_index[nano]
        for reveal, correct, censored in human[nano]["observations"]:
            if censored and not use_censored:
                continue
            shape = float(_sig(gamma[index] * (mu[index] - reveal)))
            rows.append((index, shape, correct))
    item_ids = np.array([row[0] for row in rows])
    shape = np.array([row[1] for row in rows])
    correct = np.array([row[2] for row in rows], dtype=float)

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        theta = values[0]
        beta_human = values[1:]
        logits = theta - beta_human[item_ids] * shape
        probability = _sig(logits)
        loss = float(np.sum(np.logaddexp(0, logits) - correct * logits))
        loss += 0.5 * theta**2 + float(np.sum((beta_human - beta_ai) ** 2) / 2)
        residual = probability - correct
        theta_gradient = float(np.sum(residual)) + theta
        beta_gradient = np.zeros(len(items))
        np.add.at(beta_gradient, item_ids, residual * (-shape))
        beta_gradient += beta_human - beta_ai
        return loss, np.concatenate(([theta_gradient], beta_gradient))

    initial = np.concatenate(([0.0], beta_ai.copy()))
    fitted = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 800, "ftol": 1e-10},
    ).x
    beta_human = {nano: float(fitted[1 + item_index[nano]]) for nano in items}
    shapes = {nano: (keymap[nano][1], keymap[nano][2]) for nano in items}
    return float(fitted[0]), beta_human, shapes


def _organizer_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "data" / "human" / "question_tossup_adversarialness.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["qid"].split("-")[-1]: row
            for row in csv.DictReader(handle)
            if row["packet"] != "6"
        }


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_adversarialness_tables(root: Path, scoring: str, output_dir: Path) -> AnalysisSummary:
    """Fit the submitted field and write the four scientific plotting tables."""
    traces, models = load_ai_traces(root, scoring)
    items = sorted({item for model in models for item in traces[model]})
    ai_theta, ai_items = fit_ai_correctness(models, traces, items)
    human = load_human_observations(root)
    theta_human, beta_human, shapes = fit_human_levels(
        human, ai_items, use_censored=True
    )
    theta_human_buzz_only, _, _ = fit_human_levels(
        human, ai_items, use_censored=False
    )
    organizer = _organizer_rows(root)

    keymap = {item[1]: (item[0], *parameters) for item, parameters in ai_items.items()}
    rows: list[dict[str, object]] = []
    accumulated = {name: np.zeros_like(GRID) for name in ("d_ai", "d_human", "s_ai", "s_human")}
    for nano, beta_h in beta_human.items():
        packet, beta_ai, gamma, mu = keymap[nano]
        shape = _sig(gamma * (mu - GRID))
        difficulty_ai = beta_ai * shape
        difficulty_human = beta_h * shape
        delta = difficulty_ai - difficulty_human
        solve_ai = _sig(-difficulty_ai)
        solve_human = _sig(theta_human - difficulty_human)
        gap = solve_human - solve_ai
        accumulated["d_ai"] += difficulty_ai
        accumulated["d_human"] += difficulty_human
        accumulated["s_ai"] += solve_ai
        accumulated["s_human"] += solve_human
        observation = human[nano]
        organizer_row = organizer.get(nano, {})
        rows.append(
            {
                "packet": packet,
                "nano": nano,
                "n_play": observation["n_play"],
                "n_buzz": observation["n_buzz"],
                "n_censor": observation["n_censor"],
                "beta_ai": round(beta_ai, 3),
                "beta_human": round(beta_h, 3),
                "dbeta": round(beta_ai - beta_h, 3),
                "gap_area": round(float(np.trapz(gap, GRID)), 4),
                "gap_early": round(float(gap[GRID <= 0.5].mean()), 4),
                "gap_end": round(float(gap[-1]), 4),
                "delta_area": round(float(np.trapz(delta, GRID)), 4),
                "delta_early": round(float(delta[GRID <= 0.5].mean()), 4),
                "delta_end": round(float(delta[-1]), 4),
                "ai_solve_end": round(float(solve_ai[-1]), 3),
                "human_solve_end": round(float(solve_human[-1]), 3),
                "org_human_minus_ai": organizer_row.get("raw_human_minus_ai_gap", ""),
                "org_adv_rank": organizer_row.get("adversarial_rank", ""),
            }
        )
    rows.sort(key=lambda row: row["delta_area"], reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dict_rows(output_dir / "human_ai_adversarial_qanta.csv", rows)

    with (output_dir / "qanta_ai_theta.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "theta_qanta_scale"])
        for index in np.argsort(-ai_theta):
            writer.writerow([models[index], round(float(ai_theta[index]), 4)])
        writer.writerow(["HUMAN", round(theta_human, 4)])
        writer.writerow(["HUMAN_buzz_only", round(theta_human_buzz_only, 4)])

    count = len(beta_human)
    with (output_dir / "qanta_fig_mean_curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "reveal_fraction",
                "mean_d_ai",
                "mean_d_human",
                "mean_delta",
                "mean_solve_ai",
                "mean_solve_human",
                "mean_solve_gap",
            ]
        )
        for index, reveal in enumerate(GRID):
            d_ai = accumulated["d_ai"][index] / count
            d_human = accumulated["d_human"][index] / count
            s_ai = accumulated["s_ai"][index] / count
            s_human = accumulated["s_human"][index] / count
            writer.writerow(
                [
                    round(float(reveal), 3),
                    round(float(d_ai), 4),
                    round(float(d_human), 4),
                    round(float(d_ai - d_human), 4),
                    round(float(s_ai), 4),
                    round(float(s_human), 4),
                    round(float(s_human - s_ai), 4),
                ]
            )

    with (output_dir / "qanta_fig_delta_area.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["packet", "nano", "delta_area", "gap_area", "human_buzz_frac", "org_adv_rank"]
        )
        for row in rows:
            plays = int(row["n_play"])
            buzz_fraction = round(int(row["n_buzz"]) / plays, 3) if plays else ""
            writer.writerow(
                [
                    row["packet"],
                    row["nano"],
                    row["delta_area"],
                    row["gap_area"],
                    buzz_fraction,
                    row["org_adv_rank"],
                ]
            )

    paired = [row for row in rows if row["org_human_minus_ai"] not in {"", None}]
    organizer_values = [float(row["org_human_minus_ai"]) for row in paired]
    delta_values = [float(row["delta_area"]) for row in paired]
    correlation = float(np.corrcoef(organizer_values, delta_values)[0, 1])
    return AnalysisSummary(
        scoring=scoring,
        models=len(models),
        ai_items=len(items),
        human_items=len(rows),
        human_buzzes=sum(int(item["n_buzz"]) for item in human.values()),
        human_censored=sum(int(item["n_censor"]) for item in human.values()),
        adversarial_items=sum(float(row["delta_area"]) > 0 for row in rows),
        human_outsolve_items=sum(float(row["gap_area"]) > 0 for row in rows),
        organizer_correlation=correlation,
        human_theta=theta_human,
        best_ai_theta=float(ai_theta.max()),
    )
