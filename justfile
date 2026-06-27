run:
    uv run python -m app.main

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format .

typecheck:
    uv run pyright

check: lint typecheck test
