from collections import defaultdict
from datetime import datetime, timezone

from datasets import Dataset
from tqdm import tqdm

from shared.workflows.metrics import evaluate_prediction
from shared.workflows.qb_agents import QuizBowlBonusAgent, QuizBowlTossupAgent
from shared.workflows.structs import TossupWorkflow, Workflow
from src.backend.calculate_metric import (
    calculate_buzz_accuracy,
    calculate_win_rate_human,
    calculate_win_rate_model,
)

from .io_utils import load_model_outputs, write_model_outputs, write_model_result


def get_run_contexts(dataset: Dataset) -> dict[str, dict[int, list[str]]]:
    qid2runs = defaultdict(dict)
    for item in dataset:
        qid = item["qid"]
        question_tokens = item["question"].split()
        for run_id in range(len(item["run_indices"])):
            run_index_end = item["run_indices"][run_id] + 1
            qid2runs[qid][run_id] = " ".join(question_tokens[:run_index_end])
    return qid2runs


def generate_bonus_outputs(eval_request: dict, dataset: Dataset) -> list[dict]:
    bonus_agent = QuizBowlBonusAgent(workflow=Workflow(**eval_request["workflow"]))
    model_outputs = []
    for example in tqdm(dataset, "Running Bonus outputs"):
        qid = example["qid"]
        for i, part in enumerate(tqdm(example["parts"], desc="Running Bonus outputs", leave=False)):
            part_output = bonus_agent.run(example["leadin"], part["part"])
            model_outputs.append(
                {
                    "qid": qid,
                    "part_id": f"{qid}#{i}",
                    "answer_primary": part["answer_primary"],
                    "clean_answers": part["clean_answers"],
                    "guess": part_output["answer"],
                    "confidence": part_output["confidence"],
                    "explanation": part_output["explanation"],
                }
            )
    return model_outputs


def generate_tossup_outputs(eval_request: dict, dataset: Dataset) -> list[dict]:
    tossup_agent = QuizBowlTossupAgent(workflow=TossupWorkflow(**eval_request["workflow"]))
    model_outputs = []
    for example in tqdm(dataset, "Running Tossup outputs"):
        question_runs = []
        tokens = example["question"].split()
        for run_idx in example["run_indices"]:
            question_runs.append(" ".join(tokens[: run_idx + 1]))
        results = list(tossup_agent.run(question_runs, early_stop=True))
        run_outputs = [
            {
                "qid": example["qid"],
                "run_id": f"{example['qid']}#{rid}",
                "answer_primary": example["answer_primary"],
                "clean_answers": example["clean_answers"],
                "guess": result["answer"],
                "confidence": result["confidence"],
                "buzz": result["buzz"],
            }
            for rid, result in enumerate(results)
        ]
        model_outputs.extend(run_outputs)

    return model_outputs


def generate_model_outputs(eval_request: dict, dataset: Dataset, local_outdir: str, eval_split: str):
    config_name = eval_request["competition_type"]
    if config_name == "bonus":
        func = generate_bonus_outputs
    elif config_name == "tossup":
        func = generate_tossup_outputs
    else:
        raise ValueError(f"Invalid eval type: {config_name}")

    model_outputs = func(eval_request, dataset)
    write_model_outputs(model_outputs, eval_request, eval_split, local_outdir)


def run_tossup_eval(
    request: dict,
    other_requests: list[dict],
    dataset: Dataset,
    eval_split: str,
    local_output_dir: str,
    local_result_dir: str,
):
    current_model_outputs = load_model_outputs(request, local_output_dir, eval_split)
    other_model_outputs = {r["id"]: load_model_outputs(r, local_output_dir, eval_split) for r in other_requests}
    run_contexts = get_run_contexts(dataset)

    # Compute Metrics
    for model_output in current_model_outputs:
        model_output["run_context"] = run_contexts[model_output["qid"]][model_output["run_id"]]
    buzz_accuracy = calculate_buzz_accuracy(current_model_outputs)
    win_rate_human = calculate_win_rate_human(current_model_outputs, dataset)
    win_rate_model = calculate_win_rate_model(current_model_outputs, other_model_outputs)

    model_result = {
        "model_id": request["id"],
        "competition_type": "tossup",
        "model_name": request["model_name"],
        "eval_set": eval_split,
        "buzz_accuracy": round(buzz_accuracy, 4),
        "win_rate_human": round(win_rate_human, 4),
        "win_rate_model": round(win_rate_model, 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_model_result(model_result, request["id"], eval_split, local_result_dir)


def run_bonus_eval(request: dict, eval_split: str, local_output_dir: str, local_result_dir: str):
    model_outputs = load_model_outputs(request, local_output_dir, eval_split)
    total_correct = 0
    total_parts = 0
    for item in model_outputs:
        score = evaluate_prediction(item["guess"], item["clean_answers"])
        total_correct += score
        total_parts += 1
    print(f"Total correct: {total_correct}, Total parts: {total_parts}")
    print(f"Accuracy: {total_correct / total_parts}")
    model_result = {
        "model_id": request["id"],
        "competition_type": "bonus",
        "model_name": request["model_name"],
        "eval_set": eval_split,
        "accuracy": round(total_correct / total_parts, 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_model_result(model_result, request["id"], eval_split, local_result_dir)
