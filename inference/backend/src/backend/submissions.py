import glob
import json
import os

from loguru import logger


class SubmissionManager:
    """Manages submissions, including status tracking and updating."""

    def __init__(self, local_outdir: str, repo: str, api):
        self.local_outdir = local_outdir
        self.repo = repo
        self.api = api

    def _check_submission_file_valid(self, data: dict, json_filepath: str) -> bool:
        try:
            # Expect layout: <local_outdir>/<username>/<submission_id>.json (2 segments under outdir).
            # Do not use split("/") only — Windows paths use "\" and would fail validation silently.
            abs_out = os.path.abspath(self.local_outdir)
            abs_fp = os.path.abspath(json_filepath)
            rel = os.path.relpath(abs_fp, abs_out)
            if rel.startswith("..") or os.path.isabs(rel):
                return False
            parts = rel.replace("\\", "/").split("/")
            if len(parts) != 2 or not parts[1].endswith(".json"):
                return False
            if data["username"] == "":
                return False
            if data["competition_type"] not in ["tossup", "bonus"]:
                return False
            return True
        except KeyError:
            return False

    def _upload_to_hf(self, submission_id: str, json_filepath: str):
        relative_path = self.get_repo_relative_submission_path(submission_id)
        self.api.upload_file(
            path_or_fileobj=json_filepath,
            path_in_repo=relative_path,
            repo_id=self.repo,
            repo_type="dataset",
        )

    @classmethod
    def get_username_from_submission_id(cls, submission_id: str) -> str:
        """Get the username from a submission ID.

        Supported shapes (see quizbowl-submission ``submit.py``)::

            {ctype}__{YYYYMMDD_HHMMSS}__{username}__{model}                     # workflow
            {ctype}__hf__{YYYYMMDD_HHMMSS}__{username}__{model}                 # Hugging Face pipeline
            {ctype}__docker__{YYYYMMDD_HHMMSS}__{username}__{model}             # Docker
        """
        parts = submission_id.split("__")
        try:
            if len(parts) >= 5 and parts[1] in ("hf", "docker"):
                return parts[3]
            if len(parts) >= 4:
                return parts[2]
        except IndexError:
            pass
        logger.error(f"Couldn't get username from submission ID: {submission_id}. Stale submission ID?")
        raise ValueError(f"Couldn't get username from submission ID: {submission_id}. Stale submission ID?")

    @classmethod
    def get_repo_relative_submission_path(cls, submission_id: str) -> str:
        """Get the repo relative path to a submission by its ID."""
        username = cls.get_username_from_submission_id(submission_id)
        return f"{username}/{submission_id}.json"

    def get_submission_path_by_id(self, submission_id: str) -> str:
        """Get the path to a submission by its ID."""
        relative_path = self.get_repo_relative_submission_path(submission_id)
        return f"{self.local_outdir}/{relative_path}"

    def get_submission(self, submission_id: str) -> dict | None:
        """Get a submission by its ID."""
        json_filepath = self.get_submission_path_by_id(submission_id)
        if not os.path.exists(json_filepath):
            return None
        with open(json_filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data

    def add_submission(self, submission: dict):
        """Add a submission to the queue."""
        submission_id = submission["id"]
        json_filepath = self.get_submission_path_by_id(submission_id)
        with open(json_filepath, "w", encoding="utf-8") as fp:
            json.dump(submission, fp, indent=2, ensure_ascii=False)
        self._upload_to_hf(submission_id, json_filepath)

    def remove_submission(self, submission_id: str):
        """Remove a submission from the queue."""
        relative_path = self.get_repo_relative_submission_path(submission_id)
        os.remove(os.path.join(self.local_outdir, relative_path))
        self.api.delete_file(relative_path, repo_id=self.repo, repo_type="dataset")

    def get_submissions(
        self, competition_type: str = None, sort_by: str = "created_at", reverse: bool = False
    ) -> list[dict]:
        """Get all submissions."""
        json_files = glob.glob(f"{self.local_outdir}/**/*.json", recursive=True)
        submissions = []
        for json_filepath in json_files:
            with open(json_filepath, "r", encoding="utf-8") as fp:
                submission_dict = json.load(fp)
            if not self._check_submission_file_valid(submission_dict, json_filepath):
                continue
            if competition_type is not None and submission_dict["competition_type"] != competition_type:
                continue
            submissions.append(submission_dict)
        submissions.sort(key=lambda x: x[sort_by], reverse=reverse)
        return submissions

    def get_submissions_by_status(self, status: str) -> list[str]:
        """Get all submission ids with a given status."""
        submissions = self.get_submissions(status=status)
        return [submission["id"] for submission in submissions]

    def get_submission_competition_type(self, submission_id: str) -> str:
        """Get the competition type of a submission."""
        submission = self.get_submission(submission_id)
        return submission["competition_type"]

    def get_remote_url(self, submission_id: str) -> str:
        """Get the remote URL of a submission."""
        return f"https://huggingface.co/datasets/{self.repo}/blob/main/{self.get_repo_relative_submission_path(submission_id)}"
