# Active Context

## Current Focus

- Establish Memory Bank documentation for continuity across sessions.
- Land ingestion pipeline for Summa → SQLite with OpenAI rephrasing.
- Stabilize dev environment (Nix; pinned Python; tooling).

## Recent Changes

- Added `scripts/ingest_summa.py` with streaming parser, OpenAI JSON generation, and SQLite upsert logic.
- Updated `shell.nix` to pin Python 3.12 and include `openai`, `black`, and `ruff` plus SQLite tools.
- Expanded `.gitignore` to exclude local DBs and caches.

## Next Steps

- Implement minimal backend (FastAPI or Flask) exposing HTMX views for:
  - Fetch/display a question by ID or random.
  - Submit answer; return correctness + refutations partial.
- Add basic session progress (client‑side for prototype; no auth).
- Seed a small, reliable subset (e.g., Q1–Q5) and test UI.

## Decisions & Preferences

- Model: default to `gpt-4o-mini` for low cost; allow override via env.
- Content: no “authority phrasing”; questions and answers are self‑contained.
- DB: commit schema via code, not DB files; keep `*.db` ignored.

