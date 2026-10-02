#!/usr/bin/env bash
# Measure test coverage and assemble a Markdown summary.
#
# One pytest run with every optional backend importable: a partial environment
# is not a coverage figure, so the tier jobs never pass --cov. Subprocesses the
# tests start are measured too (`patch = ["subprocess"]` in pyproject.toml).
# The summary is written before the floor is enforced, so a failing gate still
# leaves it behind.
#
# Assumes `uv sync --all-extras` has been run. Outputs: htmlcov/ (browse),
# coverage/quantype.lcov, coverage/summary.md, and coverage/badge.json (a
# shields.io endpoint).
# Run from the repository root:  bash scripts/coverage.sh
set -euo pipefail

# Minimum line-and-branch coverage, percent. Override per run via the environment.
MIN="${COVERAGE_MIN:-98}"

mkdir -p coverage
uv run --no-sync pytest tests/runtime -n auto -p no:sugar \
    --cov --cov-report=html --cov-report=lcov --cov-report=term
{
    echo "## Coverage"
    echo
    uv run --no-sync coverage report --format=markdown
} > coverage/summary.md

TOTAL="$(uv run --no-sync coverage report --format=total)"
if ((TOTAL >= MIN)); then COLOR=brightgreen; elif ((TOTAL >= 90)); then COLOR=yellow; else COLOR=red; fi
printf '{"schemaVersion": 1, "label": "coverage", "message": "%s%%", "color": "%s"}\n' \
    "$TOTAL" "$COLOR" > coverage/badge.json

uv run --no-sync coverage report --fail-under="$MIN" > /dev/null
