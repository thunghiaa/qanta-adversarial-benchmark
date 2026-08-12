# QANTA Adversarial Benchmark

A clean, reproducible home for running QANTA systems, comparing submitted
models with post-competition benchmarks, and measuring human-vs-AI
adversarialness.

The repository makes one distinction non-negotiable:

- **`qanta_submitted`**: the 16 tossup and 5 bonus systems evaluated as QANTA
  2026 submissions on packets 1-5.
- **`additional_benchmark`**: controlled backbone reruns and local models added
  after the submitted field.

The canonical assignment lives in [`configs/models.json`](configs/models.json).
Code must read that registry; filenames and usernames are not provenance.

## What is versioned

- Submission/workflow specifications, separated by cohort.
- Raw and PEDANTS-rescored model outputs, separated by cohort.
- Packet JSON metadata, human statistics, checksums, and dataset provenance.
- Lightweight analysis tables and deterministic paper-figure code.
- The inference runner and its vendored QANTA backend.

Model weights, virtual environments, API caches, and the 69 GB LangChain cache
are not source data. Full packet images are a reproducible dataset snapshot and
are downloaded on demand instead of bloating every Git clone.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

qanta-bench list-models
qanta-bench validate
qanta-bench plot-adversarialness
python -m unittest discover -s tests
```

The last command regenerates:

- `figures/qanta_fig1_theta_ladder_paper.png`
- `figures/qanta_fig2_difficulty_curves_paper.png`
- `figures/qanta_fig3_delta_hist_paper.png`

The plotting path does not import or run compIRT. The versioned CSVs under
`analysis_inputs/` are the scientific inputs to the plotting layer.

### PEDANT-scored analysis

The PEDANT variant is fitted independently from the submitted systems under
`results/qanta_submitted/**/Tossup_pedant/`; it never mixes in later benchmark
models. Rebuild its scientific tables and figures with:

```bash
qanta-bench build-analysis --scoring pedant
qanta-bench plot-adversarialness --scoring pedant
```

This writes `analysis_inputs/pedant/` and three non-destructive figure variants:

- `figures/qanta_fig1_theta_ladder_pedant_paper.png`
- `figures/qanta_fig2_difficulty_curves_pedant_paper.png`
- `figures/qanta_fig3_delta_hist_pedant_paper.png`

The de-identified `data/human/human_buzz_observations.csv` contains the human
bridge used by both scoring regimes. `scripts/build_human_observations.py` can
recreate it from the public QANTA game logs without publishing player names.
See [`docs/adversarialness.md`](docs/adversarialness.md) for scoring semantics,
the PEDANT backfill command, and the strict-vs-PEDANT comparison.

## Run models

Install inference dependencies and configure keys:

```bash
python -m pip install -e '.[inference]'
cp .env.example .env
# edit .env; never commit it
```

Fetch the packet image snapshot, then run a registered workflow:

```bash
python scripts/fetch_data.py
python scripts/fetch_model_artifacts.py local/pixtral-12b
python scripts/run_models.py \
  --cohort qanta_submitted \
  --task tossup \
  --packets 1 \
  --models ileygreg/qanta_41miniv1 \
  --debug
```

Use `--cohort additional_benchmark` for models we benchmark ourselves. The
runner rejects a model if its cohort or task does not match the registry.

### Frontier open-weight models

The ten requested 2026 frontier models are registered separately inside
`additional_benchmark`; none is presented as a submitted QANTA system. Every
served checkpoint, pinned revision, runtime, hardware floor, official recipe,
and launch command is versioned in
[`configs/frontier_gpu.json`](configs/frontier_gpu.json).

On a GPU terminal, start here:

```bash
git pull --ff-only
python scripts/frontier_gpu.py list
python scripts/frontier_gpu.py plan hy3
python scripts/frontier_gpu.py serve hy3 --dry-run
```

Then follow the two-terminal smoke/full workflow in the
**[GPU quick start](docs/gpu_quickstart.md)**. The launcher checks visible GPU
count and VRAM before it downloads weights, pins the model revision, keeps the
Hugging Face cache outside Git, and writes a provenance manifest for successful
benchmark runs.

The configured targets are Qwen3.5-397B-A17B, Hy3, DeepSeek-V4-Pro,
MiniMax-M3, GLM-5.2, Inkling, Motif-3-Beta, Laguna-S-2.1,
Solar-Open2-250B, and Kimi-K3. The common comparison track is currently
text-only even for multimodal-capable models; no image/video/audio benchmark is
claimed yet.

Raw weights are intentionally not mirrored to GitHub. They range from hundreds
of gigabytes to more than a terabyte, remain governed by their upstream
licenses, and are reproducibly downloaded from the pinned official checkpoint
into GPU scratch storage.

The broader model inventory and CPU feasibility notes are in
[`docs/frontier_models.md`](docs/frontier_models.md). The completed CPU baseline
uses Poolside's official Laguna-S-2.1 Q4 GGUF:

```bash
scripts/run_frontier_cpu_from_github.sh smoke /absolute/output/directory
```

This command clones the public repository and runs from that clone. Model
weights and builds remain in disposable scratch space; only reproducible output
artifacts are exported.

## Layout

```text
configs/models.json                 canonical model registry
submission_specs/qanta_submitted/  original competition specs
submission_specs/additional_benchmarks/
results/qanta_submitted/            competition outputs
results/additional_benchmarks/      later benchmark outputs
data/packets/                       packet JSON + downloaded images
analysis_inputs/                    immutable tables used by plots
src/qanta_bench/                    analysis, registry, validation CLI
inference/                          QANTA execution backend
scripts/                            data and model entrypoints
```

See [`docs/data.md`](docs/data.md) for storage policy and
[`docs/model_cohorts.md`](docs/model_cohorts.md) for cohort semantics.

## Current scope

The active workstream is model execution, model comparison, and
adversarialness evaluation. compIRT experiments are intentionally outside this
repository's active pipeline for now.
