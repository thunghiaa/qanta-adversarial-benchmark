#!/usr/bin/env bash
set -euo pipefail

# Reproducible CPU entrypoint. Code is always executed from a fresh GitHub clone;
# model weights and build products live only in disposable scratch space.

mode="${1:-smoke}"
export_dir="${2:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "usage: $0 [smoke|full] EXPORT_DIR" >&2
  exit 2
fi
if [[ -z "$export_dir" ]]; then
  echo "EXPORT_DIR is required; only benchmark outputs are copied out of scratch" >&2
  exit 2
fi

git_url="${BENCHMARK_GIT_URL:-https://github.com/thunghiaa/qanta-adversarial-benchmark.git}"
git_ref="${BENCHMARK_GIT_REF:-main}"
threads="${LLAMA_THREADS:-150}"
run_id="${BENCHMARK_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
scratch_root="$(mktemp -d /tmp/qanta-frontier.XXXXXX)"
repo_dir="$scratch_root/repo"
runtime_dir="$scratch_root/llama.cpp"
weights_dir="$scratch_root/weights"
run_output="$scratch_root/output"
server_log="$scratch_root/llama-server.log"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$scratch_root"
}
trap cleanup EXIT

git clone --depth 1 --branch "$git_ref" "$git_url" "$repo_dir"
git clone https://github.com/poolsideai/llama.cpp.git "$runtime_dir"
git -C "$runtime_dir" checkout 06f8cebd7fe728687be3d19f8bdedb70d75883af

python -m venv --system-site-packages "$scratch_root/venv"
"$scratch_root/venv/bin/python" -m pip install --quiet "cmake>=3.28" "huggingface-hub>=0.24"

"$scratch_root/venv/bin/cmake" -S "$runtime_dir" -B "$runtime_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=ON
"$scratch_root/venv/bin/cmake" --build "$runtime_dir/build" --config Release -j "$threads" --target llama-server

mkdir -p "$weights_dir"
WEIGHTS_DIR="$weights_dir" "$scratch_root/venv/bin/python" - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="poolside/Laguna-S-2.1-GGUF",
    revision="19036076775e9ba6758595f78af99cb976ef8ff0",
    filename="laguna-s-2.1-Q4_K_M.gguf",
    local_dir=os.environ["WEIGHTS_DIR"],
)
PY

model_file="$weights_dir/laguna-s-2.1-Q4_K_M.gguf"
numactl --interleave=all "$runtime_dir/build/bin/llama-server" \
  --model "$model_file" \
  --alias poolside/Laguna-S-2.1 \
  --host 127.0.0.1 --port 8000 \
  --threads "$threads" --threads-batch "$threads" \
  --ctx-size 8192 --parallel 1 --jinja >"$server_log" 2>&1 &
server_pid="$!"

for _attempt in $(seq 1 180); do
  if curl --silent --fail http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -200 "$server_log" >&2
    exit 1
  fi
  sleep 2
done
curl --silent --fail http://127.0.0.1:8000/health >/dev/null

common_args=(
  --model poolside/Laguna-S-2.1
  --served-model poolside/Laguna-S-2.1
  --base-url http://127.0.0.1:8000/v1
  --output-root "$run_output"
  --run-id "$run_id"
)
if [[ "$mode" == "smoke" ]]; then
  "$scratch_root/venv/bin/python" "$repo_dir/scripts/benchmark_frontier.py" \
    "${common_args[@]}" --task tossup --packets 1 --limit-questions 1
else
  "$scratch_root/venv/bin/python" "$repo_dir/scripts/benchmark_frontier.py" \
    "${common_args[@]}" --task tossup --packets 1 2 3 4 5
  "$scratch_root/venv/bin/python" "$repo_dir/scripts/benchmark_frontier.py" \
    "${common_args[@]}" --task bonus --packets 1 2 3 4 5
fi

mkdir -p "$export_dir"
cp -a "$run_output"/. "$export_dir"/
cp "$server_log" "$export_dir/laguna__${run_id}__server.log"
REPO_DIR="$repo_dir" EXPORT_DIR="$export_dir" RUN_ID="$run_id" MODE="$mode" THREADS="$threads" \
  MODEL_FILE="$model_file" "$scratch_root/venv/bin/python" - <<'PY'
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

model = Path(os.environ["MODEL_FILE"])
digest = hashlib.sha256()
with model.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ["RUN_ID"],
    "mode": os.environ["MODE"],
    "benchmark_git_commit": subprocess.check_output(
        ["git", "-C", os.environ["REPO_DIR"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "model_id": "poolside/Laguna-S-2.1",
    "model_artifact": "poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-Q4_K_M.gguf",
    "model_revision": "19036076775e9ba6758595f78af99cb976ef8ff0",
    "model_sha256": digest.hexdigest(),
    "runtime": "poolsideai/llama.cpp",
    "runtime_revision": "06f8cebd7fe728687be3d19f8bdedb70d75883af",
    "track": "text_only",
    "threads": int(os.environ["THREADS"]),
    "host": platform.platform(),
}
path = Path(os.environ["EXPORT_DIR"]) / f"laguna__{os.environ['RUN_ID']}__manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "Exported benchmark artifacts to $export_dir"
