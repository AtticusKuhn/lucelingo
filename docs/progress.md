# Progress

## What works

- Summa ingestion script parses articles and persists questions, correct and incorrect responses with refutations.
- OpenAI JSON generation with retries, rate limiting, and heuristic fallback when disabled.
- Dev environment updated in `shell.nix`; local DB tooling available.
- Django + HTMX web server shows random question, evaluates answers, and loads another via partials.

## What’s left to build

- Optional: filters by Part/Question, and a review page.
- Optional: basic session progress (client‑side; no auth).
- Prompt polish for even tighter, lay‑friendly phrasing.

## Current Status

- Data model in place; ingestion validated locally.
- Memory Bank added to `docs/` to ensure continuity.

## Known Issues / Notes

- Outputs vary slightly with model; low temperature mitigates but doesn’t eliminate drift.
- Source text has structural quirks; parser is conservative and may skip edge cases; extend as needed.
- Avoid checking in local DB files; they’re ignored via `.gitignore`.
