import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

from datasets import Dataset
from loguru import logger

from shared.workflows.runners import get_question_runs


def _docker_pull_policy() -> str:
    return (os.environ.get("DOCKER_PULL_POLICY") or "always").strip().lower()


def _docker_cpus() -> str:
    return (os.environ.get("DOCKER_CPUS") or "2.0").strip()


def _docker_memory() -> str:
    return (os.environ.get("DOCKER_MEMORY") or "4g").strip()


def _docker_host_port() -> int:
    return int((os.environ.get("DOCKER_HOST_PORT") or "8080").strip())


def _local_endpoint_base(host_port: int | None = None) -> str:
    port = host_port if host_port is not None else _docker_host_port()
    return f"http://127.0.0.1:{port}"


def _endpoint_uses_local_docker(endpoint: str) -> bool:
    """True → spawn `docker run` and talk to localhost. False → hosted API at `endpoint` (no Docker)."""
    host = (urlparse((endpoint or "").strip()).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _docker_executable() -> str:
    """
    Resolve the Docker CLI used for `docker run` / `docker stop`.
    Set DOCKER_BIN to an absolute path, or a name on PATH (default: docker).
    """
    exe = (os.environ.get("DOCKER_BIN") or "docker").strip() or "docker"
    if os.path.sep in exe and os.path.isfile(exe):
        return exe
    resolved = shutil.which(exe)
    if resolved:
        return resolved
    space_hint = ""
    if os.environ.get("SPACE_ID"):
        space_hint = (
            " Hugging Face Spaces cannot run a Docker daemon or `docker run` inside the Space container "
            "(nested containers are not supported). Run `grounded_qa_backend` on a Linux VM, dedicated server, "
            "or CI runner with Docker Engine, or restrict Docker submissions to that environment."
        )
    raise RuntimeError(
        "Docker CLI not found (needed for docker_image submissions). "
        "Install Docker Engine and ensure `docker` is on PATH, or set DOCKER_BIN to the full path. "
        "If the evaluator runs inside a container, install the docker CLI and mount the host socket, "
        "e.g. -v /var/run/docker.sock:/var/run/docker.sock."
        f"{space_hint}"
    )


def _image_exists_locally(docker_bin: str, image: str) -> bool:
    proc = subprocess.run(
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _pull_image_if_needed(docker_bin: str, image: str) -> None:
    policy = _docker_pull_policy()
    if policy == "never":
        return
    if policy == "if-not-present" and _image_exists_locally(docker_bin, image):
        logger.info(f"Using local image (DOCKER_PULL_POLICY=if-not-present): {image}")
        return
    logger.info(f"Pulling Docker image: {image}")
    proc = subprocess.run([docker_bin, "pull", image], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Failed to pull Docker image {image!r}. "
            "Check the image reference, registry auth (docker login), and network. "
            f"Docker output: {err}"
        )


@dataclass
class DockerConfig:
    image: str = ""
    endpoint: str = "http://127.0.0.1:8080"
    timeout_seconds: int = 30
    env: dict[str, str] | None = None
    command: list[str] | None = None
    healthcheck_path: str = "/health"
    task_type: str = "tossup"


def _docker_config_from_submission_dict(d: dict[str, Any]) -> DockerConfig:
    return DockerConfig(
        image=str(d.get("image") or "").strip(),
        endpoint=str(d.get("endpoint") or "http://127.0.0.1:8080").strip(),
        timeout_seconds=int(d.get("timeout_seconds", 30)),
        env=d.get("env"),
        command=d.get("command"),
        healthcheck_path=str(d.get("healthcheck_path") or "/health"),
        task_type=str(d.get("task_type") or "tossup"),
    )


def _docker_embed_images() -> bool:
    return os.environ.get("DOCKER_EMBED_IMAGES", "true").strip().lower() in ("1", "true", "yes", "on")


def _docker_image_max_edge() -> int | None:
    raw = (os.environ.get("DOCKER_IMAGE_MAX_EDGE") or "").strip()
    if not raw:
        return 768
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return 768


def _pil_to_data_uri(img: Any, max_edge: int | None) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; cannot embed PIL images in Docker HTTP payload.")
        return None
    if not isinstance(img, Image.Image):
        return None
    im = img
    if max_edge and max(im.size) > max_edge:
        im = im.copy()
        im.thumbnail((max_edge, max_edge))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _json_safe_image_entry(img: Any) -> str | None:
    if isinstance(img, str):
        return img
    if _docker_embed_images():
        return _pil_to_data_uri(img, _docker_image_max_edge())
    return None


def _json_safe_question_runs(runs: list) -> list[dict[str, Any]]:
    """JSON-serialize question runs; embed PIL images as data:image/...;base64,... when enabled."""
    safe: list[dict[str, Any]] = []
    encoded_pil = 0
    omitted_pil = 0
    for run in runs:
        if isinstance(run, str):
            safe.append({"text": run, "images": [], "multimodal_tokens": None, "is_multimodal": False})
            continue
        if not isinstance(run, dict):
            safe.append({"text": str(run), "images": [], "multimodal_tokens": None, "is_multimodal": False})
            continue
        safe_images: list[str] = []
        for img in run.get("images") or []:
            entry = _json_safe_image_entry(img)
            if entry:
                safe_images.append(entry)
                if not isinstance(img, str):
                    encoded_pil += 1
            elif not isinstance(img, str):
                omitted_pil += 1
        safe.append(
            {
                "text": run.get("text", ""),
                "images": safe_images,
                "multimodal_tokens": run.get("multimodal_tokens"),
                "is_multimodal": bool(run.get("is_multimodal", False)),
            }
        )
    if encoded_pil:
        logger.bind(every_n=5).info(
            f"Docker HTTP payload: embedded {encoded_pil} PIL image(s) as base64 data URIs "
            f"(max edge { _docker_image_max_edge() or 'full' })."
        )
    if omitted_pil:
        logger.bind(every_n=5).warning(
            f"Docker HTTP payload: omitted {omitted_pil} non-string image(s) "
            "(set DOCKER_EMBED_IMAGES=true and install Pillow on the worker)."
        )
    return safe


class DockerRunner:
    def __init__(self, config: DockerConfig):
        self.config = config
        self.container_id: str | None = None

    def _http_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        base = self.config.endpoint.rstrip("/")
        url = f"{base}{path}"
        payload = None
        headers = {}
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url=url, method=method, data=payload, headers=headers)
        timeout = max(1, int(self.config.timeout_seconds))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read().decode("utf-8")
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code} for {url}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Docker model endpoint {url}: {e}") from e

    def start(self):
        if self.container_id:
            return
        if not _endpoint_uses_local_docker(self.config.endpoint):
            logger.info(f"Hosted API mode (no docker run): GET {self.config.endpoint.rstrip('/')}/health")
            self.wait_for_health()
            return
        if not self.config.image.strip():
            raise RuntimeError(
                "Docker image is required for platform-run evaluation. "
                "Submit a container image reference (e.g. ghcr.io/org/agent@sha256:...)."
            )
        docker_bin = _docker_executable()
        host_port = _docker_host_port()
        if _endpoint_uses_local_docker(self.config.endpoint):
            self.config.endpoint = _local_endpoint_base(host_port)
        _pull_image_if_needed(docker_bin, self.config.image)
        port_map = f"{host_port}:8080"
        cmd = [
            docker_bin,
            "run",
            "-d",
            "--rm",
            "--cpus",
            _docker_cpus(),
            "--memory",
            _docker_memory(),
            "-p",
            port_map,
        ]
        for key, value in (self.config.env or {}).items():
            cmd.extend(["-e", f"{key}={str(value)}"])
        cmd.append(self.config.image)
        if self.config.command:
            cmd.extend(self.config.command)
        logger.info(f"Starting docker container for submission image {self.config.image} (port {port_map})")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Failed to start docker container for {self.config.image!r}: {err}")
        self.container_id = proc.stdout.strip()
        self.wait_for_health()

    def wait_for_health(self, retries: int = 20, delay_seconds: float = 1.5):
        path = self.config.healthcheck_path if self.config.healthcheck_path.startswith("/") else f"/{self.config.healthcheck_path}"
        last_error = None
        for _ in range(retries):
            try:
                self._http_json("GET", path)
                return
            except Exception as e:
                last_error = e
                time.sleep(delay_seconds)
        raise RuntimeError(f"Docker model failed healthcheck at {path}: {last_error}")

    def stop(self):
        if not self.container_id:
            return
        docker_bin = _docker_executable()
        subprocess.run([docker_bin, "stop", self.container_id], capture_output=True, text=True, check=False)
        self.container_id = None

    def run_tossup_dataset(self, dataset: Dataset) -> list[dict]:
        outputs = []
        for example in dataset:
            payload = {
                "qid": example.get("qid"),
                "question_runs": _json_safe_question_runs(get_question_runs(example)),
            }
            response = self._http_json("POST", "/predict/tossup", payload)
            run_outputs = response.get("run_outputs", [])
            if not isinstance(run_outputs, list):
                raise ValueError("Docker tossup response must include list field `run_outputs`.")
            normalized = []
            for item in run_outputs:
                normalized.append(
                    {
                        "guess": item.get("guess", ""),
                        "confidence": float(item.get("confidence", 0.0)),
                        "buzz": bool(item.get("buzz", False)),
                        "run_idx": int(item.get("run_idx", 1)),
                        "correct": int(item.get("correct", 0)),
                        "token_position": int(item.get("token_position", 1)),
                    }
                )
            outputs.append({"qid": example.get("qid"), "run_outputs": normalized})
        return outputs

    def run_bonus_dataset(self, dataset: Dataset) -> list[dict]:
        outputs = []
        for example in dataset:
            payload = {
                "qid": example.get("qid"),
                "leadin": example.get("leadin"),
                "parts": example.get("parts", []),
            }
            response = self._http_json("POST", "/predict/bonus", payload)
            part_outputs = response.get("part_outputs", [])
            if not isinstance(part_outputs, list):
                raise ValueError("Docker bonus response must include list field `part_outputs`.")
            normalized = []
            for idx, item in enumerate(part_outputs, start=1):
                normalized.append(
                    {
                        "number": int(item.get("number", idx)),
                        "guess": item.get("guess", ""),
                        "confidence": float(item.get("confidence", 0.0)),
                        "explanation": item.get("explanation", ""),
                        "correct": int(item.get("correct", 0)),
                    }
                )
            outputs.append({"qid": example.get("qid"), "part_outputs": normalized})
        return outputs


def _log_container_logs(runner: DockerRunner, *, tail: int = 80) -> None:
    if not runner.container_id:
        return
    docker_bin = _docker_executable()
    proc = subprocess.run(
        [docker_bin, "logs", "--tail", str(tail), runner.container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    logs = (proc.stdout or proc.stderr or "").strip()
    if logs:
        logger.error(f"Docker container {runner.container_id} logs (last {tail} lines):\n{logs}")


def run_docker_tossup(submission: dict, dataset: Dataset) -> list[dict]:
    docker_cfg = _docker_config_from_submission_dict(submission.get("docker") or {})
    runner = DockerRunner(docker_cfg)
    try:
        runner.start()
        return runner.run_tossup_dataset(dataset)
    except Exception:
        _log_container_logs(runner)
        raise
    finally:
        runner.stop()


def run_docker_bonus(submission: dict, dataset: Dataset) -> list[dict]:
    docker_cfg = _docker_config_from_submission_dict(submission.get("docker") or {})
    runner = DockerRunner(docker_cfg)
    try:
        runner.start()
        return runner.run_bonus_dataset(dataset)
    except Exception:
        _log_container_logs(runner)
        raise
    finally:
        runner.stop()
