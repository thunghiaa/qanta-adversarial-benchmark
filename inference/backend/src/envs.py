import os

from huggingface_hub import HfApi

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Optional dependency for local development.
    pass

# --------------* ENV VARS *---------------------------------------
TOKEN = os.environ.get("HF_TOKEN")  # A read/write token
ENV_NAME = os.getenv("ENV_NAME", "advcal")  # Use advcal for production, test for testing
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

OWNER = "qanta-challenge"
# Same queue dataset quizbowl-submission uploads to (`{ENV_NAME}-requests`).
REQUESTS_REPO = f"{OWNER}/{ENV_NAME}-requests"
# --------------* READ-ONLY REPOS *--------------------------------
REPO_ID = f"{OWNER}/quizbowl-submission"
LEADERBOARD_REPO = f"{OWNER}/leaderboard"
USERS_REPO = f"{OWNER}/registered-users"
EVAL_DATASET = f"{OWNER}/qanta25-eval"
# One job per submission: all component splits concatenated (see EVAL_COMPONENT_SPLITS).
COMBINED_EVAL_SPLIT_NAME = os.getenv("COMBINED_EVAL_SPLIT_NAME", "combined_eval").strip()
# Source splits stacked into COMBINED_EVAL_SPLIT_NAME (order preserved; must match MULTIMODAL_EVAL_SPLIT_NAME for Hub rows).
EVAL_COMPONENT_SPLITS = [
    s.strip()
    for s in os.getenv(
        "EVAL_COMPONENT_SPLITS",
        "tiny_eval,w1_eval,w2_eval,multimodal",
    ).split(",")
    if s.strip()
]
# Default: single combined suite only. Set EVAL_SPLITS to e.g. tiny_eval,w1_eval,... for legacy per-split jobs.
EVAL_SPLITS = [
    s.strip()
    for s in os.getenv("EVAL_SPLITS", COMBINED_EVAL_SPLIT_NAME).split(",")
    if s.strip()
]
# Only enqueue/run evaluations for submissions whose timestamp/id resolves to this calendar year (leaderboard uses the same).
COMPETITION_YEAR = int(os.getenv("COMPETITION_YEAR", "2026"))

# Split evaluator: hf_space (HF Space) vs docker_worker (Linux VM with Docker Engine).
EVALUATOR_ROLE = os.getenv("EVALUATOR_ROLE", "hf_space").strip()
# Submission types this process will execute (comma-separated). HF Space: exclude docker_image; docker worker: docker_image only.
ENABLED_SUBMISSION_TYPES = frozenset(
    s.strip()
    for s in os.getenv(
        "ENABLED_SUBMISSION_TYPES",
        "simple_workflow,complex_workflow,hf_pipeline,docker_image",
    ).split(",")
    if s.strip()
)

# Align with quizbowl-submission playground: first N rows are playground-only; rows [N:pool) are the multimodal eval window when using tail_window selection.
PLAYGROUND_MAX_QUESTIONS = int(os.getenv("PLAYGROUND_MAX_QUESTIONS", "5"))
MULTIMODAL_EVAL_POOL_SIZE = int(os.getenv("MULTIMODAL_EVAL_POOL_SIZE", "100"))
MULTIMODAL_EVAL_HUB = os.getenv("MULTIMODAL_EVAL_HUB", "168mxie/qanta-multimodal-tossups")
# If true, append multimodal tail rows to qanta25-eval splits (tiny/w1/w2). Default false: multimodal items are only in MULTIMODAL_EVAL_SPLIT_NAME.
APPEND_MULTIMODAL_EVAL_TAIL = os.getenv("APPEND_MULTIMODAL_EVAL_TAIL", "false").lower() in ("1", "true", "yes")
# When job.eval_split matches this name, load only from MULTIMODAL_EVAL_HUB (subset tossups or bonus by competition_type).
MULTIMODAL_EVAL_SPLIT_NAME = os.getenv("MULTIMODAL_EVAL_SPLIT_NAME", "multimodal").strip()
# tail_window: rows [PLAYGROUND_MAX_QUESTIONS : min(len, MULTIMODAL_EVAL_POOL_SIZE)) from train (same rows as legacy tail append). full_train: entire train split.
MULTIMODAL_EVAL_ROW_SELECTION = os.getenv("MULTIMODAL_EVAL_ROW_SELECTION", "tail_window").strip().lower()


# --------------* READ-WRITE REPOS *--------------------------------
LLM_CACHE_REPO = f"{OWNER}/advcal-llm-cache"
JOBS_REPO = f"{OWNER}/{ENV_NAME}-jobs"
OUTPUTS_REPO = f"{OWNER}/{ENV_NAME}-outputs"
RESULTS_REPO = f"{OWNER}/{ENV_NAME}-results"
Q25_CACHE_REPO = f"{OWNER}/q25-backend-cache"


# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.environ.get("LLM_CACHE_PATH", ".cache")  # NAS sqlite is flaky; set LLM_CACHE_PATH to a local-disk dir for long runs
HF_HOME = os.getenv("HF_HOME", ".hf")

# Local caches
ELO_COMPUTE_REQUEST_DIR = ".elo-compute-requests"
LOCAL_REQUESTS_PATH = os.path.join(HF_HOME, "eval-queue")
LOCAL_JOBS_PATH = os.path.join(HF_HOME, "eval-jobs")
LOCAL_RESULTS_PATH = os.path.join(HF_HOME, "eval-results")
LOCAL_OUTPUT_PATH = os.path.join(HF_HOME, "eval-outputs")
LOCAL_USERS_PATH = os.path.join(HF_HOME, "eval-registered-users")

LOCAL_LOGS_DIR = f"{ENV_NAME}-logs"
LOCAL_Q25_CACHE_REPO_PATH = os.path.join(CACHE_PATH, "q25-cache")
LOCAL_WORKFLOW_CACHE_DIR = os.path.join(CACHE_PATH, "workflow-cache")
# Keep LiteLLM disk cache outside q25-cache so HF snapshot_download can replace that tree on Windows
# without hitting WinError 32 on a locked cache.db (see bonus_metrics litellm.enable_cache).
LITELLM_CACHE_DIR = os.getenv("LITELLM_CACHE_DIR", os.path.join(CACHE_PATH, "litellm-cache"))

# Gradio `launch(server_port=...)`. Precedence (see app.py): OS env GRADIO_SERVER_PORT, then this value, then free-port scan.
# This does NOT set os.environ; assigning GRADIO_SERVER_PORT here alone used to do nothing.
# Use None to auto-pick a port. On Hugging Face Spaces, set secret GRADIO_SERVER_PORT=7860 or keep None.
GRADIO_SERVER_PORT_OVERRIDE: int | None = 7890


"""
Dataset structure:

eval-queue:
    user_id/
        {request_id}.json

eval-jobs:
    competition_type/
        eval-set/
            {request_id}.json
    jobs/
        {job_id}.json

eval-outputs:
    tossup/ (config)
        eval-set/ (split)
            {request_id}.jsonl (each line is a model output)
    bonus/
        eval-set/
            {request_id}.jsonl (each line is a model output)

eval-results:
    tossup/ (config)
        eval-set/ (split; default name COMBINED_EVAL_SPLIT_NAME, e.g. combined_eval)
            {request_id}.json (metrics over all EVAL_COMPONENT_SPLITS rows concatenated)
    bonus/
        eval-set/
            {request_id}.json

"""

OUTPUT_GEN_RATE = 60 * 60  # 1 hour
METRICS_GEN_RATE = 60 * 60  # 1 hour
EVAL_GEN_RATE = 60 * 10  # 10 minutes

QUEUE_SYNC_RATE = 30 * 5  # 5 minutes
CACHE_SYNC_RATE = 60 * 60  # 1 hour
EVAL_JOB_INTERVAL = 60 * 2  # 2 minutes
NUM_LINES_VISUALIZE = 300

API = HfApi(token=TOKEN)
