# Frontier open-weight benchmark track

These models are post-competition benchmarks. They are never counted as QANTA
submitted systems. The machine-readable source of truth is
`configs/frontier_models.json`; every checkpoint and runtime is pinned to an
immutable revision.

## Verified model inventory

| Model | Official weights | Input track | CPU status on the current 1.5 TiB host |
|---|---:|---|---|
| Qwen3.5-397B-A17B FP8 | 406 GB | multimodal | No official CPU runtime/artifact |
| Hy3 FP8 | 300 GB | text-only | No official CPU runtime/artifact |
| DeepSeek-V4-Pro BF16 | 865 GB | text-only | No official CPU runtime/artifact |
| MiniMax-M3 BF16 | 854 GB | multimodal | Official 250 GB NVFP4 is GPU-oriented |
| GLM-5.2 BF16 | 1,507 GB | text-only | Full weights exceed usable host memory |
| Inkling BF16 | 1,905 GB | multimodal | Full weights exceed host memory; official NVFP4 is GPU-oriented |
| Motif-3-Beta BF16 | 630 GB | text-only | No official CPU runtime/artifact |
| Laguna-S-2.1 BF16 | 235 GB | text-only | **Ready:** official 96 GB Q4 GGUF and Poolside llama.cpp fork |
| Solar-Open2-250B BF16 | 501 GB | text-only | No official CPU runtime/artifact |
| Kimi-K3 QAT | 1,561 GB | multimodal | Artifact exceeds usable host memory |

Sizes above are sums of the model weight blobs reported by the Hugging Face API
on 2026-08-11. They are storage sizes, not complete runtime-memory estimates.
Third-party GGUF conversions exist for several models, but running them would
create a separate quantized-replication track and must not be presented as an
official-checkpoint result.

## Reproducible CPU run

The current executable target is Laguna-S-2.1. The script clones this GitHub
repository into a fresh scratch directory, builds Poolside's pinned llama.cpp
fork, downloads the pinned official GGUF to scratch, runs the benchmark, and
exports only results, logs, and provenance metadata:

```bash
scripts/run_frontier_cpu_from_github.sh smoke /absolute/output/directory
scripts/run_frontier_cpu_from_github.sh full /absolute/output/directory
```

`smoke` evaluates one progressive tossup. `full` evaluates tossup and bonus on
packets 1-5. Set `LLAMA_THREADS` to the number of CPU threads reserved for the
job; the default is 150. Model responses use a shared direct-QA prompt,
temperature zero, a 0.75 buzz threshold, and the same reveal schedule as the
submission runner. Images are explicitly omitted in the text-only track.
Laguna's official `enable_thinking` request switch is set to false so hidden
reasoning does not consume the constrained short-answer budget or inflate the
latency measurement.

## Storage boundary

Weights, builds, and packet-image downloads are disposable and never committed.
Packet JSON is already versioned in Git, so the Laguna text-only run needs no
external packet data. Result JSONL and its run manifest are the durable artifacts
that belong under `results/additional_benchmarks/`.
