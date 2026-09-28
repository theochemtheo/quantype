setup:
    uv sync --all-extras
    uv run --all-extras prek install --prepare-hooks

check fix="":
    uv run --all-extras just --fmt {{ if fix != "--fix" { "--check" } else { "" } }} .
    uv run --all-extras ruff check {{ fix }} .
    uv run --all-extras ruff format {{ if fix == "--fix" { "" } else { "--check" } }} .
    uv run --all-extras tombi {{ if fix == "--fix" { "fmt" } else { "check" } }} .
    uv run --all-extras prek validate-config prek.toml

format:
    uv run --all-extras just --fmt .
    uv run --all-extras ruff format .
    uv run --all-extras tombi fmt .
