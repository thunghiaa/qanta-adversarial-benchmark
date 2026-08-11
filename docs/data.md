# Data and artifact policy

## Stored in Git

- Submission JSON specifications.
- Packet question JSON and human/adversarialness statistics.
- De-identified human buzz observations used by the strict and PEDANT fits.
- Model response JSONL, including both raw and PEDANTS-rescored variants.
- Derived CSVs required to reproduce published figures.
- Checksums for packet files and all migrated results.

These artifacts are small enough to review, diff, and preserve with the code.

## Downloaded on demand

Packet images come from the canonical `qanta-challenge/packet-questions`
dataset snapshot. They occupy roughly 1.2 GB locally. `scripts/fetch_data.py`
downloads them into `data/packets/`; the image paths are ignored by Git while
the packet JSON and `data/MANIFEST.sha256` remain versioned.

## Never committed

- Hugging Face model weights (`*.safetensors`, `*.bin`).
- SQLite/LangChain/LiteLLM caches.
- API request caches and logs that can contain sensitive payloads.
- `.env`, API tokens, virtual environments, Python bytecode, or OS metadata.

This is a storage boundary, not data loss: weights and packet images have
canonical upstream identifiers, while every irreplaceable model output is kept.
