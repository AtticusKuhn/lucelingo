LuceLingo — Django + HTMX web app

This project serves multiple-choice questions derived from the Summa Theologica using a lightweight Django backend and HTMX partials.

Dev setup
- Enter the Nix shell:
  - `nix-shell` (uses `shell.nix` with Python 3.12, Django, OpenAI, black, ruff)
- Optional: Populate the database (SQLite at `lucelingo.db`) using the ingestion script:
  - `scripts/ingest_summa.py --source summa_theologica.txt --db lucelingo.db --max-articles 10`
  - Set `OPENAI_API_KEY` in `.env` if not using `--skip-openai`.

Run the app
- `python manage.py runserver`
- Visit `http://127.0.0.1:8000/` to see a random question.
- Click an option to get immediate feedback; click “Another random question” to load a new one.

Code quality
- Format: `black manage.py config quiz scripts`
- Lint: `ruff check manage.py config quiz scripts`

Notes
- Django models map to the existing SQLite tables created by `scripts/ingest_summa.py`. They are marked `managed = False` to avoid Django attempting schema migrations.
- Database path can be overridden with `SQLITE_PATH` env var; defaults to `./lucelingo.db`.

