setup:
    uv sync --all-extras
    uv run --all-extras prek install --prepare-hooks

lint fix="":
    uv run --all-extras just --fmt {{ if fix != "--fix" { "--check" } else { "" } }} .
    uv run --all-extras ruff check {{ fix }} .
    uv run --all-extras ruff format {{ if fix == "--fix" { "" } else { "--check" } }} .
    uv run --all-extras tombi {{ if fix == "--fix" { "fmt" } else { "check" } }} .
    uv run --all-extras prek validate-config prek.toml

format:
    uv run --all-extras just --fmt .
    uv run --all-extras ruff format .
    uv run --all-extras tombi fmt .

typecheck:
    uv run --all-extras ty check
    uv run --all-extras mypy
    uv run --all-extras pyright
    uv run --all-extras pyrefly check
    uv run scripts/check_typing.py

generate:
    uv run scripts/generate.py

check-generated:
    uv run scripts/generate.py --check

test:
    uv run --all-extras pytest tests/runtime
