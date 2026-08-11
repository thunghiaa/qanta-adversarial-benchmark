# Model cohorts

## QANTA submitted systems

`qanta_submitted` is a historical cohort. It contains only systems treated as
the QANTA 2026 field for the competition-era analysis. Adding a new model must
never change this cohort.

The current registry has 16 tossup systems and 5 bonus systems. Some system
names refer to workflows built around the same resolved backbone; a workflow
submission is still a distinct evaluated system because its prompts, steps,
calibration, and buzz policy can differ.

## Additional benchmarks

`additional_benchmark` contains models run after the competition submission
field was fixed. It currently includes:

- controlled backbone swaps (`gpt41mini-true`, Haiku 4.5, Sonnet 5, GPT-5.6);
- additional Hugging Face pipelines;
- locally executed open VLMs.

These models may be compared with submitted systems, but aggregate claims must
say whether they refer to the submitted field or the expanded benchmark field.

## Rules

1. Every runnable or analyzed model must appear in `configs/models.json`.
2. A model id belongs to exactly one cohort.
3. Task coverage is explicit (`tossup`, `bonus`, or both).
4. Partial/in-progress runs use `status: "partial"`.
5. Analysis filters by registry fields, never by timestamp or username.
