from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from loguru import logger

from backend.submissions import SubmissionManager
from packet_envs import (
    API,
    COMPETITION_YEAR,
    ENABLED_SUBMISSION_TYPES,
    LOCAL_REQUESTS_PATH,
    REQUESTS_REPO,
    SUPPORTED_SUBMISSION_TYPES,
)
from hf_dataset_utils import download_dataset_snapshot


def submission_year(submission: dict) -> int | None:
    created_at = submission.get("created_at")
    if isinstance(created_at, str) and created_at.strip():
        try:
            normalized = created_at.strip().replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).year
        except ValueError:
            pass
    submission_id = submission.get("id", "") or ""
    for seg in submission_id.split("__"):
        if len(seg) == 15 and seg[8] == "_" and seg[:8].isdigit() and seg[9:].isdigit():
            try:
                datetime.strptime(seg, "%Y%m%d_%H%M%S")
                return int(seg[:4])
            except ValueError:
                pass
    return None


def submission_qualifies(submission: dict) -> tuple[bool, str]:
    year = submission_year(submission)
    if year is None or year != COMPETITION_YEAR:
        return False, f"not a {COMPETITION_YEAR} submission"
    submission_type = submission.get("submission_type")
    if submission_type not in SUPPORTED_SUBMISSION_TYPES:
        return False, f"unsupported submission_type={submission_type!r}"
    if submission_type not in ENABLED_SUBMISSION_TYPES:
        return False, f"submission_type={submission_type!r} not in ENABLED_SUBMISSION_TYPES"
    return True, ""


def sync_submissions(*, sync_from_hub: bool = True) -> SubmissionManager:
    if sync_from_hub:
        download_dataset_snapshot(REQUESTS_REPO, LOCAL_REQUESTS_PATH)
    return SubmissionManager(LOCAL_REQUESTS_PATH, REQUESTS_REPO, API)


def build_submission_index(
    manager: SubmissionManager,
    *,
    competition_type: str | None = None,
) -> dict[str, dict[str, dict]]:
    """Map competition_type -> {username/model_name -> submission dict}."""
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for submission in manager.get_submissions(competition_type=competition_type):
        qualified, reason = submission_qualifies(submission)
        if not qualified:
            continue
        mode = submission["competition_type"]
        model_name = f"{submission['username']}/{submission['model_name']}"
        if model_name in index[mode]:
            logger.warning(f"Duplicate submission for {model_name}; keeping first")
            continue
        index[mode][model_name] = submission
    return index


def list_eligible_submissions(
    manager: SubmissionManager,
    *,
    competition_type: str | None = None,
) -> list[dict]:
    rows = []
    for submission in manager.get_submissions(competition_type=competition_type):
        qualified, reason = submission_qualifies(submission)
        rows.append(
            {
                "model": f"{submission['username']}/{submission['model_name']}",
                "competition_type": submission["competition_type"],
                "submission_type": submission.get("submission_type"),
                "submission_id": submission["id"],
                "eligible": qualified,
                "reason": reason if not qualified else "",
            }
        )
    return rows


def explain_submission_lookup_failure(
    manager: SubmissionManager,
    name: str,
    *,
    competition_type: str | None,
) -> str:
    if "/" not in name:
        return "expected username/model_name"
    username, _, model_name = name.partition("/")
    matches = [
        sub
        for sub in manager.get_submissions()
        if sub.get("username") == username and sub.get("model_name") == model_name
    ]
    if not matches:
        return "not in local advcal-requests cache (sync from Hub or check spelling)"
    for sub in matches:
        if competition_type and sub.get("competition_type") != competition_type:
            continue
        qualified, reason = submission_qualifies(sub)
        if qualified:
            return "eligible in cache but missing from index (please report)"
        return f"ineligible: {reason}"
    types = ", ".join(sorted({sub.get('competition_type', '?') for sub in matches}))
    return f"found only as {types}; requested {competition_type}"


def resolve_submission_names(
    index: dict[str, dict[str, dict]],
    names: list[str],
    *,
    competition_type: str | None = None,
    manager: SubmissionManager | None = None,
) -> list[tuple[str, dict]]:
    resolved: list[tuple[str, dict]] = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        if "/" not in name:
            raise ValueError(f"Invalid submission name {name!r}; expected username/model_name")
        mode = None
        submission = None
        search = index.items()
        if competition_type:
            search = [(competition_type, index[competition_type])]
        for ctype, mode_index in search:
            if name in mode_index:
                mode = ctype
                submission = mode_index[name]
                break
        if submission is None:
            hint = f" ({competition_type} only)" if competition_type else ""
            detail = ""
            if manager is not None:
                detail = f"; {explain_submission_lookup_failure(manager, name, competition_type=competition_type)}"
            raise KeyError(f"Submission {name!r} not found among eligible submissions{hint}{detail}")
        resolved.append((mode, submission))
    return resolved
