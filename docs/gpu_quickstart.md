# GPU quick start for the ten frontier models

This is the canonical GPU runbook for the post-submission frontier cohort. It
keeps model definitions, commands, benchmark outputs, and provenance in GitHub.
It deliberately keeps model weights in node-local scratch or a Hugging Face
cache: the checkpoints range from roughly 235 GB to 1.56 TB and do not belong
in Git.

All ten models are `additional_benchmark`. They are never QANTA submitted
models. The current common track is **text-only**: image tokens are represented
by an omission marker so every model receives the same information. A model's
image, video, or audio capability in the registry is not a claim that those
modalities have already been benchmarked.

## Hardware and runtime matrix

The checkpoint column is the checkpoint actually served. The canonical model
and every pinned revision live in `configs/frontier_gpu.json`. NVIDIA or other
vendor quantizations are explicitly labeled in `checkpoint_provenance`; they
are not silently treated as the original BF16 checkpoint.

| Slug | Served checkpoint | Minimum allocation for this recipe | Runtime |
|---|---|---:|---|
| `qwen3.5-397b` | `Qwen/Qwen3.5-397B-A17B-FP8` | 8 GPUs, 488 GB total | current vLLM with Qwen3.5 |
| `hy3` | `tencent/Hy3-FP8` | 8 GPUs, 354 GB total | Hy3 vLLM image/nightly |
| `deepseek-v4-pro` | `deepseek-ai/DeepSeek-V4-Pro` | 8 GPUs, 960 GB total | vLLM >=0.20 + DeepGEMM |
| `minimax-m3` | `MiniMaxAI/MiniMax-M3-MXFP8` | 8 GPUs, 513 GB total | dedicated MiniMax-M3 image |
| `glm-5.2` | `zai-org/GLM-5.2-FP8` | 8 GPUs, 893 GB total | vLLM >=0.23 |
| `inkling` | `thinkingmachines/Inkling-NVFP4` | 8 GPUs, 651 GB total | vLLM 0.26 nightly |
| `motif-3-beta` | `Motif-Technologies/Motif-3-Beta` | 8 H200/B200, 756 GB total | Motif vLLM image |
| `laguna-s-2.1` | `poolside/Laguna-S-2.1` | 4 GPUs, 282 GB total | current vLLM with Poolside parsers |
| `solar-open2-250b` | `upstage/Solar-Open2-250B` | 8 GPUs, 601 GB total | Upstage vLLM image/fork |
| `kimi-k3` | `moonshotai/Kimi-K3` | 1,680 GB total | Kimi-K3 vLLM image/nightly |

These are conservative startup floors for this launcher's default profiles at
an 8K context window, not performance guarantees. Kimi-K3 normally needs
multiple nodes; an 8xB300 node is the notable single-node option. The official
Inkling recipe also has a four-GPU GB200 profile, but this launcher's portable
default is TP8; use the linked official recipe for that special case. Always run
the preflight on the actual allocation.

## 1. Clone GitHub on the GPU machine

```bash
git clone https://github.com/thunghiaa/qanta-adversarial-benchmark.git
cd qanta-adversarial-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference]'
```

If the repository is already cloned:

```bash
git pull --ff-only
source .venv/bin/activate
```

## 2. Inspect the allocation before downloading weights

List every target, then plan one model:

```bash
python scripts/frontier_gpu.py list
python scripts/frontier_gpu.py plan hy3
```

`plan` reads visible NVIDIA GPUs using `nvidia-smi` and checks GPU count, total
VRAM, and supported GPU family. It also prints the pinned checkpoint, revision,
runtime image, and the official model/runtime recipe.

Use the scheduler to request enough GPUs first. Extra CPUs do not solve a VRAM
shortage; 100-150 CPUs may help data preparation or CPU inference, but these ten
GPU checkpoints are limited primarily by HBM capacity and interconnect.

## 3. Install the model-specific runtime

Do not assume one vLLM wheel supports all ten day-zero architectures. The
`plan` output shows the required runtime and container. Prefer that exact
dedicated container for Hy3, MiniMax-M3, Motif-3-Beta, Solar-Open2, and Kimi-K3.
Use the linked official recipe when the cluster requires Docker, Apptainer, or a
multi-node Ray deployment.

For a model supported by the active vLLM environment, print the exact command
without starting or downloading anything:

```bash
python scripts/frontier_gpu.py serve hy3 --dry-run
```

The printed command includes the pinned Hugging Face revision. If using a
container, execute the same arguments inside the model-specific image printed
by `plan`.

## 4. Start the server in terminal A

Use node-local scratch for weights. `frontier_gpu.py` refuses a cache path
inside the Git checkout.

```bash
export MODEL_CACHE="${SLURM_TMPDIR:-/tmp}/qanta-frontier-hf-cache"
python scripts/frontier_gpu.py serve hy3 --cache-dir "$MODEL_CACHE"
```

The server listens on `http://127.0.0.1:8000/v1`. Keep terminal A open. The
default context is intentionally 8K, which is sufficient for QANTA and leaves
more HBM for weights and KV cache. Do not increase it to 256K or 1M unless the
allocation has been sized for that context.

For a multi-node model such as Kimi-K3, first create the multi-node vLLM/Ray
cluster according to the official recipe linked by `plan`. Then use the same
pinned checkpoint, revision, served-model name, and 8K benchmark context from
the dry-run command. The local launcher is not a substitute for configuring the
cluster fabric.

## 5. Smoke-test in terminal B

Start with two questions and one worker:

```bash
cd qanta-adversarial-benchmark
source .venv/bin/activate
python scripts/frontier_gpu.py benchmark hy3 \
  --task tossup \
  --packets 1 \
  --limit-questions 2 \
  --workers 1
```

The client checks `/v1/models` before inference. Results go under
`results/additional_benchmarks/`, and a JSON provenance manifest is written to
`results/additional_benchmarks/_manifests/` after a successful run.

## 6. Run packets 1-5

Run tossups and bonuses separately. Use the same run ID so the pair is easy to
audit:

```bash
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_hy3_text"

python scripts/frontier_gpu.py benchmark hy3 \
  --task tossup --packets 1 2 3 4 5 --workers 4 --run-id "$RUN_ID"

python scripts/frontier_gpu.py benchmark hy3 \
  --task bonus --packets 1 2 3 4 5 --workers 4 --run-id "$RUN_ID"
```

If the server is unstable, lower `--workers` before changing inference or
scoring settings. The JSONL writer resumes by QID, so rerunning the same command
and run ID skips completed questions.

## 7. Validate and push only reproducible artifacts

```bash
qanta-bench validate
python -m unittest discover -s tests
git status --short
git add results/additional_benchmarks
git commit -m "Benchmark Hy3 on QANTA packets 1-5"
git push origin main
```

Before pushing, confirm that `git status` contains JSONL results and manifests,
not `.safetensors`, `.bin`, `.gguf`, Hugging Face cache directories, environment
files, or tokens. Those weight formats and caches are blocked by `.gitignore`.

## Canonical files

- `configs/frontier_models.json`: official model inventory and original weights.
- `configs/frontier_gpu.json`: executable GPU checkpoint, revision, runtime,
  hardware floor, command, and source recipe for all ten models.
- `configs/models.json`: cohort registration (`additional_benchmark`).
- `scripts/frontier_gpu.py`: allocation preflight, serve command, benchmark
  client, and run manifest.
- `scripts/benchmark_frontier.py`: common QANTA OpenAI-compatible inference
  protocol and strict scoring.

Raw model weights are available from the official Hugging Face repositories
linked in the registries. GitHub stores how to retrieve and run them, not a
second unlicensed copy of the checkpoints.
