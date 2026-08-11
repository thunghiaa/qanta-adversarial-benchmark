# Adversarialness scoring regimes

The strict and PEDANT analyses use the same 16 registered QANTA submitted
tossup systems, packets 1–5, 100 AI-evaluated items, and 90 items with human
buzz observations. The only changed input is answer-equivalence scoring:

- `strict` reads `Tossup/` correctness from the original evaluation harness.
- `pedant` reads question-aware correctness from `Tossup_pedant/` at the
  standard `PEDANT > 0.3` threshold. Literal `ERROR` sentinels are always wrong.

Later controlled reruns and local models under `additional_benchmarks` are
rejected by the submitted-field loader.

## Reproduce PEDANT outputs and figures

Existing PEDANT-scored JSONL files are versioned. To score a newly added raw
submission first, install the optional scorer and run the non-destructive
backfill command:

```bash
python -m pip install -e '.[pedant]'
python scripts/rescore_pedant.py \
  --cohort qanta_submitted \
  --task tossup \
  --packets 1 2 3 4 5
```

Existing destination files are skipped unless `--overwrite` is supplied. Then
fit and plot the PEDANT variant:

```bash
qanta-bench build-analysis --scoring pedant
qanta-bench plot-adversarialness --scoring pedant
```

The build command consumes raw registered outputs and the de-identified human
buzz table; the plot command consumes only the resulting versioned CSVs.

## Current submitted-field comparison

| Metric | Strict | PEDANT |
|---|---:|---:|
| Human θ | +1.487 | +0.922 |
| Best submitted AI θ | +0.438 | +0.538 |
| Human margin over best AI | +1.049 | +0.383 |
| Items with δ_area > 0 | 69/90 | 66/90 |
| Human out-solves mean AI | 90/90 | 90/90 |
| Correlation with organizer raw gap | 0.632 | 0.274 |

These are different fitted regimes, not cosmetic redraws. Results and claims
must therefore identify which correctness scorer was used.
