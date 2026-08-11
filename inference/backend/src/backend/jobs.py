# %%
import glob
import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum

import pandas as pd
from huggingface_hub import CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from src.backend.submissions import SubmissionManager

_job_counter_lock = threading.Lock()


class JobTask(str, Enum):
    """Enumeration of possible tasks for a job."""

    TOSSUP = "tossup"
    BONUS = "bonus"


class JobStatus(str, Enum):
    """Enumeration of possible states for a job."""

    # Waiting to be processed, which yields model outputs per question
    SUBMITTED = "submitted"

    # This indicates that there is a worker creating the model outputs in background
    IN_PROGRESS = "in_progress"

    # This indicates that the job has been completed
    COMPLETED = "completed"

    # This indicates that there was an error in the model output generation
    FAILED = "failed"


class JobStage(str, Enum):
    """Enumeration of possible stages for a job."""

    OUTPUTS = "outputs"
    METRICS = "metrics"
    EVALUATION = "evaluation"


class Job(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    submission_id: str  # this is same as request id
    username: str
    competition_type: str
    model_name: str
    eval_split: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: JobStatus
    stage: JobStage
    error: str = ""

    @classmethod
    def get_repo_relative_path(cls, job_id: str) -> str:
        return f"jobs/{job_id}.json"

    @property
    def repo_relative_path(self) -> str:
        return f"jobs/{self.id}.json"

    @property
    def secondary_repo_relative_path(self) -> str:
        return f"{self.competition_type}/{self.eval_split}/{self.submission_id}.json"


class JobManager:
    """Manages jobs, including status tracking and updating."""

    def __init__(self, jobs_repo_dir: str, jobs_repo: str, submission_manager: SubmissionManager, api):
        self.jobs_repo_dir = jobs_repo_dir
        self.jobs_repo = jobs_repo
        self.submission_manager = submission_manager
        self.api = api

    def _check_request_file_valid(self, data: dict, json_filepath: str) -> bool:
        try:
            if len(json_filepath.split("/")) != 4:
                return False
            if data["username"] == "":
                return False
            if data["competition_type"] not in ["tossup", "bonus"]:
                return False
            return True
        except KeyError:
            return False

    def get_local_path(self, job: str | Job) -> str:
        if isinstance(job, Job):
            job = job.id
        return os.path.join(self.jobs_repo_dir, f"jobs/{job}.json")

    def get_local_path2(self, job: Job) -> str:
        return os.path.join(self.jobs_repo_dir, job.secondary_repo_relative_path)

    def get_job(self, job_id: str) -> Job | None:
        """Get a request by its ID."""
        json_filepath = self.get_local_path(job_id)
        if not os.path.exists(json_filepath):
            return None
        with open(json_filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return Job(**data)

    def _upload_job_to_hf(self, job_id: str):
        job_path = self.get_local_path(job_id)
        job = self.get_job(job_id)
        self.api.upload_file(
            path_or_fileobj=job_path,
            path_in_repo=job.repo_relative_path,
            repo_id=self.jobs_repo,
            repo_type="dataset",
        )
        self.api.upload_file(
            path_or_fileobj=self.get_local_path2(job),
            path_in_repo=job.secondary_repo_relative_path,
            repo_id=self.jobs_repo,
            repo_type="dataset",
        )

    def add_job_to_dataset(self, job: Job):
        """Push the job object to the local directory and the Hugging Face repo."""
        job_dict = job.model_dump(mode="json")
        for job_path in [self.get_local_path(job.id), self.get_local_path2(job)]:
            os.makedirs(os.path.dirname(job_path), exist_ok=True)
            with open(job_path, "w", encoding="utf-8") as fp:
                json.dump(job_dict, fp, indent=2, ensure_ascii=False)
        self._upload_job_to_hf(job.id)

    def _delete_paths_on_hub(self, paths: list[str], commit_message: str) -> None:
        """Delete multiple paths from the jobs dataset in a single Hub commit."""
        paths = [p for p in dict.fromkeys(paths) if p]
        if not paths:
            return
        operations = [CommitOperationDelete(path_in_repo=p) for p in paths]
        try:
            self.api.create_commit(
                repo_id=self.jobs_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
        except HfHubHTTPError as e:
            # Common when cleaning many orphaned jobs: HF caps dataset commits/hour.
            logger.error(f"Hub commit failed while deleting {len(paths)} path(s): {e}")

    def remove_job(self, job_id: str):
        """Remove a job from the dataset."""
        job = self.get_job(job_id)
        if job is None:
            logger.warning(f"remove_job({job_id}): no local job JSON found; skipping Hub delete")
            return

        primary = self.get_local_path(job_id)
        secondary = self.get_local_path2(job)
        for job_path in (primary, secondary):
            if os.path.exists(job_path):
                os.remove(job_path)

        self._delete_paths_on_hub(
            [job.repo_relative_path, job.secondary_repo_relative_path],
            commit_message=f"Remove job {job_id}",
        )

    def remove_jobs(self, job_ids: list[str], *, commit_message: str | None = None) -> None:
        """Remove many jobs locally and on Hub using batched commits (avoids HF commit rate limits)."""
        hub_paths: list[str] = []
        for job_id in job_ids:
            job = self.get_job(job_id)
            if job is None:
                logger.warning(f"remove_jobs: skipping {job_id} (missing local job JSON)")
                continue

            primary = self.get_local_path(job_id)
            secondary = self.get_local_path2(job)
            for job_path in (primary, secondary):
                if os.path.exists(job_path):
                    os.remove(job_path)

            hub_paths.extend([job.repo_relative_path, job.secondary_repo_relative_path])

        if hub_paths:
            # Chunk deletes: each job maps to ~2 paths; very large single commits can fail.
            chunk_size = 80
            base_msg = commit_message or f"Remove {len(job_ids)} jobs"
            for start in range(0, len(hub_paths), chunk_size):
                chunk = hub_paths[start : start + chunk_size]
                self._delete_paths_on_hub(chunk, commit_message=f"{base_msg} ({start // chunk_size + 1})")

    def update_job(self, job: str | Job, **kwargs) -> Job:
        """Update a job."""
        if isinstance(job, str):
            job = self.get_job(job)
        job = job.model_copy(update=kwargs)
        self.add_job_to_dataset(job)
        return job

    def update_job_stage(
        self, job: str | Job, stage: JobStage, status: JobStatus = JobStatus.SUBMITTED, message: str = ""
    ):
        """Set the stage of a job."""
        if isinstance(job, str):
            job = self.get_job(job)
        job = job.model_copy(update={"stage": stage, "status": status, "error": message})
        return self.update_job(job)

    def update_job_status(self, job: str | Job, status: str, message: str = "") -> Job:
        """Set the status of a job."""
        if isinstance(job, str):
            job = self.get_job(job)
        job = job.model_copy(update={"status": status, "error": message})
        updated_at = datetime.now(timezone.utc).isoformat()
        return self.update_job(job, updated_at=updated_at)

    def get_jobs(
        self,
        status: str = None,
        stage: str = None,
        competition_type: str = None,
        eval_split: str = None,
        *,
        sort_by: str = "created_at",
        reverse: bool = False,
    ) -> list[Job]:
        """Get all requests."""
        json_files = glob.glob(f"{self.jobs_repo_dir}/jobs/*.json", recursive=True)
        jobs = []
        for json_filepath in json_files:
            with open(json_filepath, "r", encoding="utf-8") as fp:
                job_dict = json.load(fp)
            if status is not None and job_dict["status"] != status.lower():
                continue
            if stage is not None and job_dict["stage"] != stage.lower():
                continue
            if competition_type is not None and job_dict["competition_type"] != competition_type.lower():
                continue
            if eval_split is not None and job_dict["eval_split"] != eval_split:
                continue
            jobs.append(Job(**job_dict))
        jobs.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
        return jobs

    def get_jobs_by_status(self, status: str, stage: JobStage = None) -> list[Job]:
        """Get all job ids with a given status."""
        return self.get_jobs(status=status, stage=stage)

    def get_job_for_submission(self, submission_id: str, eval_split: str) -> Job | None:
        """Get a job for a submission request."""
        for config_name in os.listdir(f"{self.jobs_repo_dir}"):
            filepath = f"{self.jobs_repo_dir}/{config_name}/{eval_split}/{submission_id}.json"
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as fp:
                    job_dict = json.load(fp)
                return Job(**job_dict)
        return None

    def count_created_jobs(self) -> int:
        """Count the number of jobs created."""
        try:
            return len(os.listdir(f"{self.jobs_repo_dir}/jobs"))
        except FileNotFoundError:
            logger.warning(f"Jobs repo directory {self.jobs_repo_dir}/jobs does not exist. Count: 0")
            return 0

    def get_current_job_counter(self) -> int:
        with _job_counter_lock:
            filenames = os.listdir(f"{self.jobs_repo_dir}/jobs")
            counters = [int(suffix) for f in filenames if (suffix := f.removesuffix(".json").split("_")[-1]).isdigit()]
            return max(counters) + 1 if counters else 1

    def get_submission(self, job: str | Job) -> dict:
        """Get the submission for a job."""
        if isinstance(job, str):
            job_id = job
            job = self.get_job(job)
        if not job:
            logger.error(f"Job {job_id} not found. Returning None.")
            return None
        return self.submission_manager.get_submission(job.submission_id)

    def create_job_for_submission(self, submission: dict, eval_split: str) -> Job | None:
        """Create a job for a submission request and uploads it to the Hugging Face repo"""
        ctype = submission["competition_type"].lower()[0]
        job_id = f"{ctype}_{self.get_current_job_counter()}"
        if self.get_job(job_id):
            logger.error(f"Job {job_id} already exists. Returning None.")
            raise RuntimeError(f"Job {job_id} already exists. Returning None.")
        job = Job(
            id=job_id,
            submission_id=submission["id"],
            username=submission["username"],
            competition_type=submission["competition_type"],
            model_name=submission["model_name"],
            eval_split=eval_split,
            status=JobStatus.SUBMITTED,
            stage=JobStage.OUTPUTS,
        )
        self.add_job_to_dataset(job)
        return job

    def update_in_progress_jobs(self, timeout_hours: int = 24):
        """Update the status of all in progress jobs to failed if they have been in progress for more than 24 hours"""
        jobs = self.get_jobs(status=JobStatus.IN_PROGRESS)
        count_jobs = 0
        for job in jobs:
            time_last_updated = pd.to_datetime(job.updated_at, utc=True)
            time_now = datetime.now(timezone.utc)
            if (time_now - time_last_updated).total_seconds() >= timeout_hours * 3600:
                count_jobs += 1
                self.update_job_status(job, JobStatus.FAILED, message="Job timed out. Marked as failed.")

        logger.info(f"Found and marked {count_jobs} in-progress jobs that timed out")

    def get_remote_url(self, job_id: str) -> str:
        """Get the remote URL of a job."""
        return f"https://huggingface.co/datasets/{self.jobs_repo}/blob/main/{Job.get_repo_relative_path(job_id)}"

    def get_remote_user_url(self, username: str) -> str:
        """Get the remote URL of a user."""
        return f"https://huggingface.co/{username}"

    def get_submission_remote_url(self, job_id: str) -> str:
        """Get the remote URL of a submission."""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found. Returning empty string.")
            return ""
        return self.submission_manager.get_remote_url(job.submission_id)

    # Misc functions
    def get_orphaned_jobs(self):
        "Return the jobs that are not associated with any submission. e.g., the submissions were problematic, and were later removed."
        return [j for j in self.get_jobs() if not self.submission_manager.get_submission(j.submission_id)]
