# Tech Context

## Languages & Frameworks

- Python 3.12 (pinned in `shell.nix`).
- HTMX (planned for frontend partials over a minimal Python web app).
- SQLite for persistence.

## Key Dependencies

- `openai` — Chat Completions API for low‑cost JSON question generation.
- `black`, `ruff` — code formatting and linting.
- `sqlite`, `sqlitebrowser` — local DB tooling in dev shell.

## Development Environment

- Nix: see `shell.nix`. Launch a shell with project deps, then:
  - Run ingestion: `scripts/ingest_summa.py --source summa_theologica.txt --db lucelingo.db --max-articles 5`
  - Use `--skip-openai` for quick local smoke tests without API calls.
  - Configure env via `.env` (ignored):
    - `OPENAI_API_KEY=...`
    - `OPENAI_MODEL=gpt-4o-mini` (optional)
    - `RATE_LIMIT_CALLS=15` (optional)

## Code Quality

- Format: `black scripts/`.
- Lint: `ruff check scripts/`.
- Keep functions small and explicit; prefer pure helpers for parsing/normalization.

## Runbook Snippets

- Explore DB: `sqlite3 lucelingo.db .tables` and standard SQL.
- Reset a single article: delete from response tables for its `question_id` and re‑ingest.

