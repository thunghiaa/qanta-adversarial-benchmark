import io
from pathlib import Path
import unittest
from unittest.mock import patch

from qanta_bench.preflight import (
    GIB,
    checkpoint_scratch_bytes,
    forbidden_scratch_roots,
    validate_scratch,
    verify_huggingface_revision,
)


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = io.BytesIO(body)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


class FrontierPreflightTest(unittest.TestCase):
    def test_forbidden_roots_include_tmp_home_and_project_space(self) -> None:
        roots = forbidden_scratch_roots(ROOT, Path("/home/example"))
        self.assertIn(Path("/tmp"), roots)
        self.assertIn(Path("/home/example"), roots)
        self.assertIn(Path("/mnt/main/users/ntruong8"), roots)

    def test_project_directory_is_rejected_as_scratch(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be inside"):
            validate_scratch(ROOT, repo_root=ROOT, write_probe=False)

    def test_checkpoint_space_includes_staging_and_runtime_headroom(self) -> None:
        required = checkpoint_scratch_bytes({"checkpoint_weight_bytes": 100 * GIB})
        self.assertEqual(required, 130 * GIB)

    @patch("qanta_bench.preflight.urllib.request.urlopen")
    def test_exact_anonymous_revision_must_match(self, urlopen: object) -> None:
        revision = "a" * 40
        urlopen.return_value = _Response(f'{{"sha":"{revision}"}}'.encode())
        self.assertEqual(verify_huggingface_revision("owner/model", revision), revision)
        request = urlopen.call_args.args[0]
        self.assertNotIn("Authorization", request.headers)


if __name__ == "__main__":
    unittest.main()
