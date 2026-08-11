"""Shared helpers for AdvVQA tossup/bonus export upload scripts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE_HTTP_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

# Firestore user docs that lack email (admin login email differs from stored profile).
KNOWN_EMAIL_BY_USER_ID: dict[str, str] = {
    "22": "ying@umd.edu",
}


@dataclass(frozen=True)
class AuthorFilter:
    emails: frozenset[str]
    user_ids: frozenset[str]


def parse_user_id_aliases() -> dict[str, set[str]]:
    """Parse ADVVQA_AUTHOR_USER_IDS=ying@umd.edu:22,foo@x.com:9 into email -> {user_ids}."""
    aliases: dict[str, set[str]] = {}
    for uid, email in KNOWN_EMAIL_BY_USER_ID.items():
        aliases.setdefault(email.lower(), set()).add(uid)

    env = os.getenv("ADVVQA_AUTHOR_USER_IDS", "").strip()
    for part in env.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        email, uid = part.split(":", 1)
        aliases.setdefault(email.strip().lower(), set()).add(uid.strip())
    return aliases


def parse_author_filter(values: list[str] | None) -> AuthorFilter | None:
    if not values:
        env = os.getenv("ADVVQA_AUTHOR_EMAILS", "").strip()
        if not env:
            return None
        values = [env]
    emails: set[str] = set()
    for value in values:
        for part in value.split(","):
            email = part.strip().lower()
            if email:
                emails.add(email)
    if not emails:
        return None

    uid_aliases = parse_user_id_aliases()
    user_ids: set[str] = set()
    for email in emails:
        user_ids.update(uid_aliases.get(email, set()))
    return AuthorFilter(emails=frozenset(emails), user_ids=frozenset(user_ids))


def session_author_user_id(session_data: dict) -> str | None:
    submitter = session_data.get("submitter") or {}
    uid = submitter.get("user_id") or session_data.get("user_id")
    return str(uid) if uid is not None and str(uid) != "" else None


def session_author_username(session_data: dict) -> str | None:
    submitter = session_data.get("submitter") or {}
    username = submitter.get("username") or session_data.get("username")
    if not username:
        return None
    return str(username)


def session_author_email(session_data: dict) -> str | None:
    submitter = session_data.get("submitter") or {}
    email = submitter.get("email")
    if email:
        return str(email)
    uid = session_author_user_id(session_data)
    if uid and uid in KNOWN_EMAIL_BY_USER_ID:
        return KNOWN_EMAIL_BY_USER_ID[uid]
    return None


def record_author_fields(record: dict) -> tuple[str | None, str | None]:
    meta = record.get("metadata") or {}
    email = (meta.get("author_email") or "").lower() or None
    uid = meta.get("author_user_id")
    uid = str(uid) if uid is not None and str(uid) != "" else None
    if not email and uid and uid in KNOWN_EMAIL_BY_USER_ID:
        email = KNOWN_EMAIL_BY_USER_ID[uid].lower()
    return email, uid


def matches_author_filter(
    *,
    author_filter: AuthorFilter,
    session_data: dict | None = None,
    record: dict | None = None,
) -> bool:
    if session_data is not None:
        email = (session_author_email(session_data) or "").lower()
        uid = session_author_user_id(session_data)
    elif record is not None:
        email, uid = record_author_fields(record)
    else:
        return False

    if email and email in author_filter.emails:
        return True
    if uid and uid in author_filter.user_ids:
        return True
    return False


def api_token(client: httpx.Client, api_url: str) -> str:
    token = os.getenv("ADVVQA_TOKEN", "").strip()
    if token:
        return token

    username = os.getenv("ADVVQA_USERNAME", "").strip()
    password = os.getenv("ADVVQA_PASSWORD", "")
    if not username or not password:
        raise SystemExit(
            "Set ADVVQA_TOKEN (from browser localStorage on the admin site) or "
            "ADVVQA_USERNAME + ADVVQA_PASSWORD in packet-outputs/.env"
        )

    resp = client.post(
        f"{api_url}/api/auth/login",
        data={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def auth_error_help(status_code: int, *, local_flag: str) -> str:
    if status_code == 401:
        return (
            "AdvVQA API returned 401 Unauthorized — your ADVVQA_TOKEN has likely expired. "
            "Log in at https://advvqa-author-firebase.web.app/admin, copy a fresh `token` "
            f"from browser localStorage into packet-outputs/.env, or re-run with "
            f"{local_flag} to upload an existing export without the API."
        )
    return f"AdvVQA API request failed with HTTP {status_code}."


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    attempts: int = 5,
    base_delay: float = 1.0,
    label: str = "",
    follow_redirects: bool = False,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.request(
                method,
                url,
                headers=headers,
                params=params,
                follow_redirects=follow_redirects,
            )
        except RETRYABLE_HTTP_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            who = label or url
            print(
                f"  WARN: {who} attempt {attempt}/{attempts} failed ({type(exc).__name__}); "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def get_json_with_retry(
    client: httpx.Client,
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    label: str = "",
) -> Any:
    resp = _request_with_retry(
        client, "GET", url, headers=headers, params=params, label=label
    )
    return resp


def fetch_completed_sessions(
    client: httpx.Client,
    api_url: str,
    token: str,
    *,
    question_type: str,
    local_flag: str,
) -> list[tuple[str, dict]]:
    headers = {"Authorization": f"Bearer {token}"}
    summaries: list[dict] = []
    page = 1
    page_size = 100

    while True:
        resp = get_json_with_retry(
            client,
            f"{api_url}/api/admin/submissions",
            headers=headers,
            params={
                "question_type": question_type,
                "review_status": "all",
                "page": page,
                "page_size": page_size,
            },
            label=f"submissions list page {page}",
        )
        if resp.status_code >= 400:
            raise SystemExit(auth_error_help(resp.status_code, local_flag=local_flag))
        payload = resp.json()
        summaries.extend(payload["submissions"])
        if page * payload["page_size"] >= payload["total"]:
            break
        page += 1

    total = len(summaries)
    print(f"Fetching details for {total} {question_type} sessions...")
    sessions: list[tuple[str, dict]] = []
    for i, summary in enumerate(summaries, start=1):
        session_id = summary["session_id"]
        if i == 1 or i % 10 == 0 or i == total:
            print(f"  session {i}/{total}: {session_id}")
        detail = get_json_with_retry(
            client,
            f"{api_url}/api/admin/submissions/{session_id}",
            headers=headers,
            label=f"session {session_id}",
        )
        if detail.status_code >= 400:
            raise SystemExit(auth_error_help(detail.status_code, local_flag=local_flag))
        sessions.append((session_id, detail.json()))
    return sessions


def download_image(client: httpx.Client, url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS image URL: {url!r}")
    resp = _request_with_retry(
        client, "GET", url, label=url[:80], follow_redirects=True
    )
    resp.raise_for_status()
    return resp.content


def list_authors(api_url: str, *, question_type: str, label: str) -> None:
    """Print completed session counts grouped by author email / user_id."""
    from collections import Counter

    with httpx.Client(timeout=180.0) as client:
        token = api_token(client, api_url)
        sessions = fetch_completed_sessions(
            client,
            api_url,
            token,
            question_type=question_type,
            local_flag="--from-local",
        )

    groups: Counter[tuple[str, str, str]] = Counter()
    for _sid, session_data in sessions:
        email = session_author_email(session_data) or "<no email>"
        uid = session_author_user_id(session_data) or "?"
        username = session_author_username(session_data) or "?"
        groups[(email, uid, username)] += 1

    print(f"Completed {label}: {len(sessions)}")
    for (email, uid, username), count in groups.most_common():
        print(f"  {count:3d}  email={email!r}  user_id={uid}  username={username!r}")
