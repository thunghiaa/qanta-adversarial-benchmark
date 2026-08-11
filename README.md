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

- `figures/qanta_fig2_difficulty_curves_paper.png`
- `figures/qanta_fig3_delta_hist_paper.png`

Neither figure path imports or runs compIRT. The versioned CSVs under
`analysis_inputs/` are the scientific inputs to the plotting layer.

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
