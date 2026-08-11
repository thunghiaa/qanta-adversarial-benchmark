import json
import os

from packet_envs import API, LOCAL_PACKET_OUTPUT_PATH, PACKET_EVAL_SPLIT, PACKET_OUTPUTS_REPO


def write_jsonl(objects: list[dict], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj) + "\n")


def load_jsonl(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _hub_rel_path(*parts: str) -> str:
    return "/".join(parts)


def packet_outputs_path(
    competition_type: str,
    submission_id: str,
    eval_split: str = PACKET_EVAL_SPLIT,
    local_outdir: str | None = None,
) -> str:
    parts = (competition_type, eval_split, f"{submission_id}.jsonl")
    if local_outdir is None:
        return _hub_rel_path(*parts)
    return os.path.join(local_outdir, *parts)


def check_and_create_outputs_repo() -> None:
    try:
        API.repo_info(repo_id=PACKET_OUTPUTS_REPO, repo_type="dataset")
    except Exception:
        API.create_repo(
            repo_id=PACKET_OUTPUTS_REPO,
            repo_type="dataset",
            exist_ok=True,
            private=True,
        )


def write_packet_outputs(
    model_outputs: list[dict],
    *,
    competition_type: str,
    submission_id: str,
    local_outdir: str = LOCAL_PACKET_OUTPUT_PATH,
    eval_split: str = PACKET_EVAL_SPLIT,
    upload: bool = True,
) -> str:
    rel_filepath = packet_outputs_path(competition_type, submission_id, eval_split)
    filepath = packet_outputs_path(competition_type, submission_id, eval_split, local_outdir)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    write_jsonl(model_outputs, filepath)
    if upload:
        check_and_create_outputs_repo()
        API.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=rel_filepath,
            repo_id=PACKET_OUTPUTS_REPO,
            repo_type="dataset",
            commit_message=f"Add {competition_type} packet outputs for {submission_id} / {eval_split}",
        )
    return filepath
