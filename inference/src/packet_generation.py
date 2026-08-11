"""Packet-output tossup generation (all run indices, not early-stop eval)."""

from __future__ import annotations

from datasets import Dataset

from backend.docker_runner import run_docker_tossup
from backend.hf_pipeline_runner import run_hf_pipeline_tossup
from shared.workflows.qb_agents import QuizBowlTossupAgent
from shared.workflows.runners import run_and_eval_tossup_dataset
from shared.workflows.structs import TossupWorkflow


def generate_packet_tossup_outputs(submission: dict, dataset: Dataset) -> list[dict]:
    """Generate tossup outputs with buzz evaluated at every run index."""
    submission_type = submission.get("submission_type")
    if submission_type == "docker_image":
        return run_docker_tossup(submission, dataset)
    if submission_type == "hf_pipeline":
        return run_hf_pipeline_tossup(submission, dataset, early_stop=False)
    tossup_agent = QuizBowlTossupAgent(workflow=TossupWorkflow(**submission["workflow"]))
    return run_and_eval_tossup_dataset(tossup_agent, dataset, early_stop=False)
