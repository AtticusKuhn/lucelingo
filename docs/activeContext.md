# Active Context

## Current Focus

- Web UI: Django + HTMX quiz flow reading from SQLite.
- Keep ingestion stable; iterate on parsing/rewrites as needed.

## Recent Changes

- Added Django project with app `quiz`; models map to existing tables (`managed = False`).
- Implemented views and HTMX endpoints for random question, answer feedback, and next question.
- Minimal templates with clean styling; wired URLs and settings to use `lucelingo.db`.

## Next Steps

- Optional: add filtering by Part/Question and a simple review mode.
- Optional: add session progress tracking (client‑side only for MVP).
- Polish prompts/ingestion for more consistent options and refutations.

## Decisions & Preferences

- Model: default to `gpt-4o-mini` for low cost; allow override via env.
- Content: no “authority phrasing”; questions and answers are self‑contained.
- DB: commit schema via code, not DB files; keep `*.db` ignored.
