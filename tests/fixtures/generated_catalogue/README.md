# Generated-catalogue example project

`tests/runtime/test_codegen.py` copies this directory into a temporary project and
generates `labquantities` beside these files. The generated package is not checked
in or imported into the parent pytest process.

- `consumer.py` contains the `assert_type` contract checked independently by mypy,
  Pyright, Pyrefly, and ty. `pyproject.toml` supplies strict checker settings;
  mypy receives `--strict` on the command line.
- `runtime_cases.py` contains ordinary pytest assertions for the generated API.
- `jax_example.py` and `torch_example.py` exercise each optional adapter separately.

The runtime files deliberately do not use `test_*.py` names: they need the generated
package and must not be collected directly from this directory. The parent tests
invoke pytest on explicit files/node IDs in subprocesses. This keeps normal
assertion diagnostics while isolating catalogue identities and JAX registrations.

These source files are formatted and linted normally. Static conformance runs on
`consumer.py` in the generated project, not against an absent package in this
source directory. Ruff is required only for tests that generate files; a missing
checker or optional backend skips only its own test.
