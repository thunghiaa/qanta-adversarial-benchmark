# CPU smoke runs

## Laguna-S-2.1 Q4

The `20260811_cpu_smoke_v2` run evaluated all 19 progressive reveals of the
first packet-1 tossup using Poolside's official Q4_K_M GGUF on 150 CPU threads.
It was executed from GitHub commit `10a347a06a94c3478de3ae4170ba94179658b38e`.

- Total model-call latency: 238.24 seconds
- Median latency per reveal: 12.28 seconds
- Decode throughput: approximately 2.4 tokens/second
- First correct answer: token position 74 (`Apollo missions`, confidence 0.85)
- Final answer: `Apollo missions`, confidence 0.95
- Images: explicitly omitted (text-only track)
- Native reasoning: disabled with Poolside's official `enable_thinking=false`

The JSONL contains the standard QANTA `run_outputs` plus request latency and
token usage. The manifest pins the Git commit, checkpoint revision, runtime
revision, weight SHA-256, track, and CPU thread count. The server log provides
the independent runtime timing trace.
