import os
from pathlib import Path

from huggingface_hub import HfApi

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass

TOKEN = os.environ.get("HF_TOKEN")
OWNER = "qanta-challenge"

PACKET_DATASET = os.getenv("PACKET_DATASET", f"{OWNER}/packet-questions")
PACKET_DATASET_SPLIT = os.getenv("PACKET_DATASET_SPLIT", "tossup")
PACKET_BONUS_SPLIT = os.getenv("PACKET_BONUS_SPLIT", "bonus")
PACKET_EVAL_SPLIT = os.getenv("PACKET_EVAL_SPLIT", "competition")
PACKET_OUTPUTS_REPO = os.getenv("PACKET_OUTPUTS_REPO", f"{OWNER}/packet-outputs")
REQUESTS_REPO = os.getenv("REQUESTS_REPO", f"{OWNER}/advcal-requests")
DEV_PACKET_DATASET = os.getenv("DEV_PACKET_DATASET", f"{OWNER}/qanta25-eval")

COMPETITION_YEAR = int(os.getenv("COMPETITION_YEAR", "2026"))
ENABLED_SUBMISSION_TYPES = frozenset(
    s.strip()
    for s in os.getenv(
        "ENABLED_SUBMISSION_TYPES",
        "simple_workflow,complex_workflow,hf_pipeline,docker_image",
    ).split(",")
    if s.strip()
)

HF_HOME = os.getenv("HF_HOME", ".hf")
LOCAL_REQUESTS_PATH = os.path.join(HF_HOME, "eval-queue")
LOCAL_PACKET_OUTPUT_PATH = os.path.join(HF_HOME, "packet-outputs")

SUPPORTED_SUBMISSION_TYPES = frozenset(
    {"simple_workflow", "complex_workflow", "hf_pipeline", "docker_image"}
)

API = HfApi(token=TOKEN)
