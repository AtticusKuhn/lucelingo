# Progress

## What works

- Summa ingestion script parses articles and persists questions, correct and incorrect responses with refutations.
- OpenAI JSON generation with retries, rate limiting, and heuristic fallback when disabled.
- Dev environment updated in `shell.nix`; local DB tooling available.

## What’s left to build

- Minimal backend + HTMX views for quiz flow.
- Question selection (random or by range) and answer handling.
- Basic styling and review view.

## Current Status

- Data model in place; ingestion validated locally.
- Memory Bank added to `docs/` to ensure continuity.

## Known Issues / Notes

- Outputs vary slightly with model; low temperature mitigates but doesn’t eliminate drift.
- Source text has structural quirks; parser is conservative and may skip edge cases; extend as needed.
- Avoid checking in local DB files; they’re ignored via `.gitignore`.

