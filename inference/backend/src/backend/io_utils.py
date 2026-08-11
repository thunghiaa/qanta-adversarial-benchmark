import json
import os

from src.envs import API, LOCAL_OUTPUT_PATH, LOCAL_RESULTS_PATH, OUTPUTS_REPO, RESULTS_REPO

from .jobs import Job


def write_jsonl(objects: list[dict], filepath: str):
    with open(filepath, "w") as f:
        for obj in objects:
            f.write(json.dumps(obj) + "\n")


def write_json(obj: dict, filepath: str, indent: int = 2):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def load_jsonl(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


# def model_outputs_path(competition_type: str, eval_split: str, submission_id: str) -> str:
#     return os.path.join(competition_type, eval_split, submission_id, f"{submission_id}.jsonl")


# def model_results_path(submission_id: str, eval_split: str) -> str:
#     return os.path.join(submission_id, eval_split, f"{submission_id}.json")


def _hub_rel_path(*parts: str) -> str:
    """Path inside a Hugging Face dataset repo (always ``/``, never ``\\``)."""
    return "/".join(parts)


def model_outputs_path(job: Job, local_outdir: str | None = None) -> str:
    """Hub-relative path (no base), or absolute local path when ``local_outdir`` is set."""
    parts = (job.competition_type, job.eval_split, f"{job.submission_id}.jsonl")
    if local_outdir is None:
        return _hub_rel_path(*parts)
    return os.path.join(local_outdir, *parts)


def model_results_path(job: Job, local_resdir: str | None = None) -> str:
    """Hub-relative path (no base), or absolute local path when ``local_resdir`` is set."""
    parts = (job.competition_type, job.eval_split, f"{job.submission_id}.json")
    if local_resdir is None:
        return _hub_rel_path(*parts)
    return os.path.join(local_resdir, *parts)


def model_elo_results_path(eval_split: str, local_resdir: str | None = None) -> str:
    parts = ("tossup", eval_split, "elo_results.json")
    if local_resdir is None:
        return _hub_rel_path(*parts)
    return os.path.join(local_resdir, *parts)


def load_model_outputs(job: Job, local_outdir: str = LOCAL_OUTPUT_PATH) -> list[dict]:
    filepath = model_outputs_path(job, local_outdir)
    return load_jsonl(filepath)


def load_model_results(job: Job, local_resdir: str = LOCAL_RESULTS_PATH) -> dict:
    filepath = model_results_path(job, local_resdir)
    return load_json(filepath)


def write_model_outputs(model_outputs: list[dict], job: Job, local_outdir: str):
    config_name = job.competition_type
    rel_filepath = model_outputs_path(job)
    filepath = model_outputs_path(job, local_outdir)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    write_jsonl(model_outputs, filepath)
    API.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=rel_filepath,
        repo_id=OUTPUTS_REPO,
        repo_type="dataset",
        commit_message=f"Add {config_name} outputs for {job.submission_id} / {job.eval_split}",
    )


def write_model_result(model_result: dict, job: Job, local_result_dir: str):
    config_name = job.competition_type
    rel_filepath = model_results_path(job)
    filepath = model_results_path(job, local_result_dir)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    write_json(model_result, filepath)
    API.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=rel_filepath,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add {config_name} results for {job.submission_id} / {job.eval_split}",
    )


# Stable envelope so HF Hub dataset previews don't infer one row per submission-id column (flat maps break Arrow casting).
TOSSUP_ELO_SCHEMA_V1 = "tossup_elo_points_v1"


def write_model_tossup_elo_results(elo_results: dict, job: Job, local_resdir: str = LOCAL_RESULTS_PATH):
    config_name = job.competition_type
    if config_name != "tossup":
        raise ValueError(f"Model tossup elo results can only be written for tossup jobs, got {config_name}")
    rel_filepath = model_elo_results_path(job.eval_split)
    filepath = model_elo_results_path(job.eval_split, local_resdir)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "schema": TOSSUP_ELO_SCHEMA_V1,
        "eval_split": job.eval_split,
        "entries": [{"submission_id": k, "points": float(v)} for k, v in elo_results.items()],
    }
    write_json(payload, filepath)
    API.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=rel_filepath,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add {config_name} elo results for {job.eval_split}",
    )
