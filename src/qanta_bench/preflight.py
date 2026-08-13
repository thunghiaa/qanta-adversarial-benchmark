"""Storage and remote-revision checks for frontier benchmark setup."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


GIB = 1024**3


@dataclass(frozen=True)
class ScratchReport:
    path: Path
    free_bytes: int
    required_bytes: int


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def forbidden_scratch_roots(repo_root: Path, home: Path | None = None) -> tuple[Path, ...]:
    """Return locations that are durable homes/project storage, not scratch."""
    resolved_repo = repo_root.resolve()
    resolved_home = (home or Path.home()).resolve()
    roots = {
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/nfshomes"),
        resolved_home,
        resolved_repo,
    }
    parts = resolved_repo.parts
    if len(parts) >= 5 and parts[1:4] == ("mnt", "main", "users"):
        roots.add(Path(*parts[:5]))
    extra = os.environ.get("QANTA_FORBIDDEN_SCRATCH_ROOTS", "")
    roots.update(Path(item).expanduser().resolve() for item in extra.split(":") if item)
    return tuple(sorted(roots, key=str))


def validate_scratch(
    path: Path,
    *,
    repo_root: Path,
    required_bytes: int = 5 * GIB,
    write_probe: bool = True,
) -> ScratchReport:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"QANTA_SCRATCH must be an existing directory: {resolved}")
    for forbidden in forbidden_scratch_roots(repo_root):
        if _is_within(resolved, forbidden):
            raise ValueError(f"QANTA_SCRATCH cannot be inside {forbidden}: {resolved}")
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise ValueError(f"QANTA_SCRATCH is not writable/searchable: {resolved}")
    usage = shutil.disk_usage(resolved)
    if usage.free < required_bytes:
        required_gib = required_bytes / GIB
        free_gib = usage.free / GIB
        raise ValueError(
            f"QANTA_SCRATCH has {free_gib:.1f} GiB free; this operation requires "
            f"at least {required_gib:.1f} GiB"
        )
    if write_probe:
        probe = Path(tempfile.mkdtemp(prefix=".qanta-write-test-", dir=resolved))
        probe.rmdir()
    return ScratchReport(resolved, usage.free, required_bytes)


def scratch_from_env(repo_root: Path, required_bytes: int = 5 * GIB) -> ScratchReport:
    value = os.environ.get("QANTA_SCRATCH")
    if not value:
        raise ValueError(
            "QANTA_SCRATCH is not set. Export it to an existing large scratch filesystem."
        )
    return validate_scratch(Path(value), repo_root=repo_root, required_bytes=required_bytes)


def checkpoint_scratch_bytes(model: dict[str, Any]) -> int:
    """Allow checkpoint plus ten percent staging space and 20 GiB runtime headroom."""
    return model["checkpoint_weight_bytes"] * 110 // 100 + 20 * GIB


def verify_huggingface_revision(model_id: str, revision: str, timeout: float = 30) -> str:
    """Resolve an exact Hub revision anonymously and return its canonical SHA."""
    quoted_id = "/".join(urllib.parse.quote(part, safe="") for part in model_id.split("/"))
    url = f"https://huggingface.co/api/models/{quoted_id}/revision/{revision}"
    request = urllib.request.Request(url, headers={"User-Agent": "qanta-frontier-preflight/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Hugging Face rejected {model_id}@{revision} anonymously: HTTP {error.code}"
        ) from error
    actual = payload.get("sha")
    if actual != revision:
        raise RuntimeError(f"Revision mismatch for {model_id}: expected {revision}, got {actual}")
    return actual
