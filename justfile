setup:
    uv sync --all-extras
    uv run --all-extras prek install --prepare-hooks

lint fix="":
    uv run --all-extras just --fmt {{ if fix != "--fix" { "--check" } else { "" } }} .
    uv run --all-extras ruff check {{ fix }} .
    uv run --all-extras ruff format {{ if fix == "--fix" { "" } else { "--check" } }} .
    uv run --all-extras tombi {{ if fix == "--fix" { "fmt" } else { "check" } }} .
    uv run --all-extras prek validate-config prek.toml
    uv run --all-extras zizmor --offline .

format:
    uv run --all-extras just --fmt .
    uv run --all-extras ruff format .
    uv run --all-extras tombi fmt .

typecheck:
    uv run --all-extras ty check
    uv run --all-extras scripts/check_mypy.py
    uv run --all-extras pyright
    uv run --all-extras pyrefly check
    uv run scripts/check_typing.py

# Runtime/stub agreement via stubtest; slower, so separate from typecheck
stubcheck:
    uv run --all-extras python -m mypy.stubtest quantype --allowlist tests/typing/stubtest_allowlist.txt

generate:
    uv run scripts/generate_codata.py
    uv run scripts/generate.py
    uv run scripts/generate_docs.py

check-generated:
    uv run scripts/generate_codata.py --check
    uv run scripts/generate.py --check
    uv run scripts/generate_docs.py --check

test:
    uv run --all-extras pytest tests/runtime -n auto

coverage:
    uv sync --all-extras
    bash scripts/coverage.sh
