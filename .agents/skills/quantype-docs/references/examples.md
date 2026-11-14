# Executable examples

Sybil checks README and published Markdown independently of rendering. Do not execute installation shell fences, proposals, decision history, skills, or source trees as documentation examples. See [Sybil Markdown syntax](https://sybil.readthedocs.io/en/latest/markdown.html) for the upstream parser contract.

## Fences and state

Use plain, top-level fenced blocks with exactly `python` as the language, including for `>>>` doctest transcripts. Do not use `pycon`, `python notest`, annotated fence attributes, nested fences in lists/admonitions, or invisible code blocks. One `PythonCodeBlockParser` handles both Python and doctest; do not add overlapping parsers that double-collect examples.

- Tutorial pages share a namespace within the page and execute in documented order. Show all imports and prerequisite steps before their use. Do not depend on another page or hidden fixture imports.
- How-to, reference, explanation, troubleshooting, and README fences are independently executable. Repeat visible imports and setup in each fence. Reset the namespace between fences, not between prompts within a single doctest transcript.
- Filesystem examples execute in a temporary working directory with pytest cleanup. Use relative paths there; do not rely on the repository cwd, write into the checkout, or require persistent files from another fence.
- Keep examples deterministic and independent of network, credentials, and GPU availability. Examples must teach the setup readers actually need, not harness-only bindings.

Prefer assertions and numerical testing helpers for arrays and approximate numbers. Use doctest expected output where representation teaches something. Do not mask meaningful differences with blanket ellipses or loose matching. Verify expected exceptions explicitly, with `pytest.raises` and a meaningful match or a doctest traceback; an intentionally invalid call is not automatically a reason to skip.

## Optional backends

Declare a requirement immediately before each fence that needs an optional backend:

````markdown
<!-- requires: jax -->
```python
import jax.numpy as jnp
assert jnp.asarray([1, 2]).shape == (2,)
```

<!-- requires: torch -->
```python
import torch
assert torch.tensor([1, 2]).shape == (2,)
```
````

A declaration applies to one fence, not the rest of the page. Declare requirements even when a tutorial fence uses backend state established in an earlier fence. Do not infer backend requirements from source text or hide conditional imports in examples. In incomplete environments, only examples requiring an unavailable backend skip; retain visible skip counts/reasons. The full-backend run must execute these examples. Avoid a tutorial dependency chain that needs an undeclared backend.

## Illustrative skips

Use Sybil's reason-bearing syntax for genuinely illustrative, non-executable patterns:

````markdown
<!-- skip: next "Illustrative pseudocode; application supplies the callback." -->
```python
result = application_callback(...)
```
````

For a genuinely inseparable collection, use `<!-- skip: start "Specific reason." -->` and close it with `<!-- skip: end -->`; prefer a single-fence skip. Reasons must explain why execution is inappropriate, not merely say “skip”. Reason-bearing skips remain visible in pytest reporting. Do not retain the legacy `python notest` convention, skip ordinary failures, or use generic skips in place of explicit backend requirements. A skip is not verification of the illustrated behavior.

## Collection and claims

Keep tutorial documents on one worker when using xdist (file-based scheduling); never split sequential examples between workers. Maintain regression coverage for tutorial sequencing, independent fence namespaces, incorrect expected output, and explicit unavailable-backend skips. Unsupported syntax must not silently disappear from collection.

Sybil establishes runtime behavior, not inferred types. Important advertised inferred types and rejected expressions need positive/negative consumer fixtures in `tests/typing` and the existing four-checker workflow; consult [typing guidance](../../quantype-typing/SKILL.md). For commands and full versus partial proof, see [publishing and review](publishing.md).
