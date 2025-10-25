# System Patterns

## Architecture Overview

- Source: `summa_theologica.txt` (large, regular structure).
- Ingestion: `scripts/ingest_summa.py` stream‑parses one article at a time.
- Generation: Low‑cost OpenAI model rephrases each article into a question + answers.
- Storage: SQLite with 3 tables: `Question`, `CorrectResponse`, `IncorrectResponse`.
- App (planned): Python backend with HTMX views to render questions and record answers.

## Ingestion Pattern

- Streaming parser detects article headers and section markers (Objections, On the contrary, I answer that, Replies) to build an `Article` record.
- Prompt builder crafts a compact system/user prompt carrying title, objections, main answer, and replies.
- OpenAI call returns JSON: `{question_text, correct_answer, incorrect_options[]}` with optional refutations.
- Scrubber removes authority references; whitespace normalized.
- Upsert guarantees a single `Question` row per (part, question, article) key; responses replaced atomically.
- Resume behavior: skip articles already complete in DB.

## Data Model

- `Question(id, question_text, part_number, question_number, article_number, created_at, updated_at)`
- `CorrectResponse(id, question_id, response_text)`
- `IncorrectResponse(id, question_id, response_text, refutation_text)`

## Operational Concerns

- Rate limiting: configurable `--calls-per-minute`; simple delay plus retries with backoff.
- Determinism: low temperature to keep outputs stable; heuristic fallback when `--skip-openai`.
- Safety: avoids attributions like “Aquinas says…”; outputs framed as standalone catechetical content.

## Backend

- Django app serving HTMX partials:
  - `GET /` → renders a random question and options
  - `GET /answer` → evaluates selection and returns feedback partial
  - `GET /question` → returns another random question partial
- Selection: random question for MVP; filters by Part/Question can be added.
