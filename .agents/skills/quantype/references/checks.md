# Development and conformance checks

Run commands from the repository root. Consult `justfile`, `pyproject.toml`, `docs/development.md`, and `.github/workflows/test.yml` for current definitions.

## Environment

The project requires Python 3.12 or later and uses `uv`. `uv sync --all-extras` installs development tools plus JAX and Torch. Core-only work can use `uv sync`; unavailable optional backends skip their tests. The `just test`, `just typecheck`, and `just lint` recipes request all extras, so they can change the environment or require downloads. Select commands appropriate to the available environment and report skipped coverage.

## Choose checks by the affected contract

| Change | Useful validation |
| --- | --- |
| Runtime arithmetic or units | `uv run pytest tests/runtime/test_core.py tests/runtime/test_refactor.py` |
| Registry or generator | `uv run scripts/generate.py`, then `uv run scripts/generate.py --check`; `uv run pytest tests/runtime/test_codegen.py` |
| Public signatures or inferred algebra | `uv run just typecheck` and `uv run just stubcheck`; generated API changes also need generator freshness and codegen tests |
| Constructors or storage dispatch | `uv run pytest tests/runtime/test_refactor.py tests/runtime/test_binary_failures.py` |
| JAX, Torch or import behavior | Relevant `test_autodiff.py` or `test_torch.py`, plus `test_optional_backends.py` and `test_refactor.py` |
| JSON/Pydantic/archive behavior | `uv run pytest tests/runtime/test_validation.py tests/runtime/test_binary_failures.py tests/runtime/test_refactor.py` |
| Cross-cutting package change | `uv run just check-generated`, `uv run just typecheck`, `uv run just stubcheck`, `uv run just test`, `uv run just lint`; `uv build` for packaging changes |

## Static conformance

`uv run just typecheck` combines ty, both mypy parsers via `scripts/check_mypy.py`, Pyright, Pyrefly, and `scripts/check_typing.py`. `uv run just stubcheck` runs mypy stubtest with `tests/typing/stubtest_allowlist.txt`.

Positive contracts use `assert_type` in `tests/typing/positive/`. Each invalid expression in `tests/typing/negative/invalid.py` ends with `# error`; the negative runner requires every checker to report each marked line, not merely exit unsuccessfully. `tests/typing/probes/extension_operators.py` is an investigation fixture with a known operator-fallback limitation, outside the passing contract.

Use `scripts/check_mypy.py` for the configured dual-parser run: it separates parser caches and places parallel workers' IPC files under `.mypy_cache`. `test_codegen.py` runs four-checker consumer conformance in an isolated generated project and skips unavailable tools individually.

## Wider checks

`uv run just lint` checks Just formatting, Ruff lint/format, TOML formatting/validation, hook configuration and offline workflow lint. `uv run just setup` also installs Git hooks; use it when hook installation is requested, not merely to validate a change.

CI additionally tests core-only, JAX-only, Torch-only and minimum core dependency environments. A successful all-extras run alone does not establish optional dependency isolation or compatibility with minimum versions. Use the relevant existing isolation tests and describe coverage limits instead of claiming those environments were tested locally.
