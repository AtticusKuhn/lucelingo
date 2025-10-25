# LuceLingo — Project Brief

LuceLingo is a prototype quiz/learning platform to teach Catholic doctrine, theology, and social teaching. It transforms articles from St. Thomas Aquinas’s Summa Theologica into concise multiple‑choice questions, storing them in SQLite and later presenting them via a simple Python backend with an HTMX front end.

## Goals

- Deliver a minimal, clean system that:
  - Parses Summa Theologica text into structured articles.
  - Rephrases each article into a question using a low‑cost OpenAI model.
  - Persists questions and answers in SQLite with a simple schema.
  - Exposes a lightweight web UI (HTMX) to practice and review.

## Non‑Goals (for the prototype)

- User accounts or authentication.
- Complex spaced repetition algorithms.
- Advanced content moderation or editorial workflow.
- Rich styling or design systems beyond basic usability.

## Acceptance Criteria (MVP)

- A repeatable ingestion script converts a subset of Summa into:
  - 1 row in `Question` with part/question/article identifiers.
  - 1 row in `CorrectResponse` and ≥1 rows in `IncorrectResponse` with optional refutations.
- Questions avoid “authority” phrasings (e.g., “According to Aquinas…”) and stand alone.
- A minimal web page can fetch and display a question with answers.
- Code quality: black + ruff clean on changed files.

## Constraints and Preferences

- Python backend, HTMX frontend, SQLite DB.
- Use inexpensive, reliable OpenAI models (e.g., `gpt-4o-mini`).
- Keep the code simple and explicit; avoid hacks and ad‑hoc glue.
- Nix‑based dev environment; all deps declared in `shell.nix`.

