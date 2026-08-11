import os

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from src.envs import API, JOBS_REPO, LLM_CACHE_REPO, OUTPUTS_REPO, Q25_CACHE_REPO, REPO_ID, RESULTS_REPO


def check_and_create_dataset_repo(repo_id: str):
    try:
        API.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"{repo_id} exists")
    except Exception:
        print(f"Creating {repo_id}")
        API.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)


def check_and_create_repos():
    print("1. JOBS Repository")
    check_and_create_dataset_repo(JOBS_REPO)
    print("2. OUTPUTS Repository")
    check_and_create_dataset_repo(OUTPUTS_REPO)
    print("3. RESULTS Repository")
    check_and_create_dataset_repo(RESULTS_REPO)
    print("4. Q25 CACHE Repository")
    check_and_create_dataset_repo(Q25_CACHE_REPO)
    print("5. LLM CACHE Repository")
    check_and_create_dataset_repo(LLM_CACHE_REPO)


def download_dataset_snapshot(repo_id, local_dir) -> bool:
    """
    Download the latest snapshot of the dataset from HuggingFace.
    Returns True if the snapshot was downloaded, False if it was already up to date.
    """
    api = HfApi()
    try:
        logger.bind(every_n=10).info(f"Checking for updates in dataset repo {repo_id} for local dir {local_dir}")

        # Get the latest commit hash from the remote repo
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        remote_commit = repo_info.sha

        # Try to read the last commit hash from the local snapshot
        commit_file = os.path.join(local_dir, "snapshots", "latest_commit")
        local_commit = None
        if os.path.exists(commit_file):
            with open(commit_file, "r") as f:
                local_commit = f.read().strip()

        if local_commit == remote_commit:
            logger.bind(every_n=10).info(
                f"Local dataset snapshot for {repo_id} is up to date (commit: {local_commit})"
            )
            return False

        logger.info(
            f"Downloading dataset snapshot from {repo_id} to {local_dir} (remote commit: {remote_commit}, local commit: {local_commit})"
        )
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            tqdm_class=None,
            max_workers=10,
        )

        # Save the latest commit hash locally for future checks
        os.makedirs(os.path.dirname(commit_file), exist_ok=True)
        with open(commit_file, "w") as f:
            f.write(remote_commit)
        return True

    except Exception as e:
        logger.error(f"Error downloading dataset snapshot from {repo_id} to {local_dir}: {e}")
        # `repo_id` is a dataset, not a Space — restarting it 404s. Only the backend Space may be restarted.
        try:
            API.restart_space(repo_id=REPO_ID)
        except Exception as restart_err:
            logger.warning(f"No Space restart after snapshot failure (expected when running locally): {restart_err}")
        return False
