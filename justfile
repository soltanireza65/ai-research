run:
    uv run python -m app.main

lab:
    uv run jupyter lab app/

notebooks:
    uv run python scripts/generate_notebooks.py

validate-book *ARGS:
    uv run python scripts/validate_chapter.py {{ARGS}}

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format .

typecheck:
    uv run pyright

check: lint typecheck test
