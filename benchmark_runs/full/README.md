# Full frontier runs

## Laguna-S-2.1 Q4 — packets 1–5

Run ID: `20260811_laguna_q4_cpu`

- 100 tossups, 2,001 progressive reveal calls
- 100 bonus sets, 300 bonus-part calls
- No failed calls
- Final-clue tossup accuracy: 67/100
- Correct at the model's first confidence ≥ 0.75 point: 17/100
- Bonus-part strict accuracy: 61/300
- Total summed request latency: 17.82 hours
- Median request latency: 28.82 seconds with four concurrent server slots
- Checkpoint: official Poolside Q4_K_M GGUF, text-only track
- Execution: 150 CPU threads, four workers, code cloned from GitHub commit `ac5506f`

The ten QANTA-compatible JSONL files, server log, and immutable provenance
manifest live under `results/additional_benchmarks/`. This run remains separate
from all QANTA submitted systems.
