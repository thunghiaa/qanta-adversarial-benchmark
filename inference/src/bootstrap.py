"""Add the vendored QANTA backend and inference helpers to ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

_PACKET_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_BACKEND = _PACKET_ROOT / "backend"
_LEGACY_BACKEND = _PACKET_ROOT.parent / "grounded_qa_backend"
_BACKEND_ROOT = _BUNDLED_BACKEND if _BUNDLED_BACKEND.is_dir() else _LEGACY_BACKEND


def setup_paths() -> tuple[Path, Path]:
    """Prepend backend and packet src roots; packet src must win import resolution."""
    env_file = _PACKET_ROOT.parent / ".env"
    try:
        from dotenv import load_dotenv

        if env_file.is_file():
            # Must run before ``backend.*`` imports: grounded_qa_backend/.env can
            # set ENABLED_SUBMISSION_TYPES=docker_image and hide workflow models.
            load_dotenv(env_file, override=True)
    except ImportError:
        pass

    packet_src = str(_PACKET_ROOT / "src")
    backend_src = str(_BACKEND_ROOT / "src")
    backend_root = str(_BACKEND_ROOT)
    # Insert backend paths first, then packet src last so `import envs` resolves to packet-outputs.
    for entry in (backend_root, backend_src, packet_src):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return _PACKET_ROOT, _BACKEND_ROOT
