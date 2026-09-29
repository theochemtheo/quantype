# Refactor requirements and findings

## Agreed scope

- Breaking changes are allowed; retain the useful external quantity algebra.
- Prefer `Length[np.float64](2, u.length.nanometer)`. The generic argument drives
  conversion, not merely validation. Separate scalar storage from array storage.
- Keep unit choice out of the resulting quantity type. Canonical data is distinct
  from presentation, and `from_canonical` remains a trusted, nonconverting boundary.
- Replace runtime string-expression parsing and generated-module class discovery
  with nominal identities and structured expressions. Keep numerical, conversion,
  serialization, and Pydantic responsibilities separate.
- NumPy 2.x and SciPy are core. Retain Pydantic v2 in core for this refactor.
  Torch and JAX are independent optional extras. Custom storage adapters are deferred.
- Use SciPy constants as the conversion authority and measure startup as well as
  throughput before deciding where to resolve them.
- Allow explicit user-defined units without global registration. A bleb is
  `kelvin = magnitude * sqrt(2) + 273.15`, not a scaling of Kelvin's origin.
- Investigate user-defined kinds and ordinary operators in both operand orders.
  Reuse the built-in generation machinery rather than create a second generator.
- Remove `Vector3` and `SymmetricTensor3`; shape/symmetry are not quantity kinds.
- Discrete serialization is `{kind, magnitude, unit}`. Binary arrays use NPY/NPZ;
  restoration should be typed. Show optional recorded backend metadata without
  assuming every dtype/backend can be serialized faithfully.

## Constructor feasibility

A `types.GenericAlias` subclass can intercept construction while retaining
`typing.get_origin` and `typing.get_args`. Generated quantities remain ordinary
one-parameter generic classes in stubs, so all four checkers see
`Length[np.float64]`, not a storage-specific runtime subclass. Pydantic receives
normal generic metadata. JAX registers the underlying nominal class only once.

Storage conversion is explicit at construction and typed restoration. NumPy dtype
parameters are resolved for both NumPy 2.0's aliases and current PEP 695 aliases.
Torch/JAX require a separate `dtype=` because their array classes do not encode
it. Typed construction intentionally costs more than trusted wrapping.

Pydantic 2.0 used `general_wrap_validator_function`, later renamed to
`with_info_wrap_validator_function`. The adapter selects the available name and
uses a shared wire parser. A wrapper schema describes JSON input without first
coercing booleans to numbers or losing custom-unit validation context.

## Constants measurements

Local measurements: macOS arm64, CPython 3.12.11, NumPy 2.5.3, SciPy 1.18.1.
These are observations, not performance guarantees or regression thresholds.
Cold timings are medians of seven fresh processes; steady-state timings are the
best of five 10,000-call repeats. Process launch itself is excluded.

| Measurement | Original | Refactored sample |
|---|---:|---:|
| `import quantype` | 8.3 ms | 9.1 ms |
| `import numpy` | 24.4 ms | 24.2 ms |
| `from scipy import constants` | 71.5 ms | 71.3 ms |
| First unit use after importing quantype | not recorded | 65.5 ms |
| First typed construction after importing NumPy and quantype | unavailable | 46.8 ms |
| Cached numerical factor multiplication | 6.9 ns | 6.9 ns |
| SciPy dictionary lookup + multiplication | 16.1 ns | 15.6 ns |
| `u.hartree(2.0)` | 396 ns | 383 ns |
| `u.hartree(array_of_1000)` | 924 ns | 884 ns |
| `Length[np.float64](2, u.length.nanometer)` | unavailable | 2.22 μs |
| `Length.from_canonical(array)` | not recorded | 208 ns |

Small throughput differences should be treated as measurement noise. The large
cost is importing SciPy, not reading its constants dictionary.

### Decision

Resolve SciPy constants lazily on the first unit request, then retain ordinary
numerical scale factors in immutable units. Importing quantype does not load any
numerical backend. Arithmetic does no SciPy lookups. The first unit request pays
for the whole small unit table; there is no speculative per-constant lazy framework.

Generated frozen values would avoid the first-use cost but would follow the
build-time SciPy version rather than the installed version. Eager imports would
charge that cost to every import, even when only trusted canonical boundaries are
used. Neither is the default. External generated catalogues also refer back to
SciPy-derived built-in definitions rather than copying their numerical constants.

Reproduce with:

```bash
uv run --all-extras python scripts/benchmark_constants.py
```

## Extensible operators: evidence and decision

`tests/typing/probes/extension_operators.py` is an intentionally failing research
fixture, excluded from the passing conformance suite. It defines a custom
quantity with both a left and reflected multiplication method returning a named
built-in result.

All four checkers infer the existing structural fallback for `builtin * custom`,
not the reflected method's result. Mypy additionally diagnoses an unsafe overlap
between the reflected method and the built-in overload. The reverse order can be
inferred correctly. Runtime registration cannot change this static overload set.

A combined generated catalogue solves both orders without plugins or casts in
consumer code. The runtime test generates a `SurfaceTension` extension, checks
both operand orders with all four checkers, and exercises construction, parsing,
math, temperatures, and optional autodiff adapters.

### Implemented extension workflow

`Catalogue` is immutable, validated input. `builtin_catalogue().extend(...)`
rejects replacements of existing declarations. The same source renderers produce
both the built-in and application APIs. `generate` separately owns Ruff invocation,
file writes, and stale-output checks; `render` does not invoke tools or write files.

Generated packages are **separate nominal catalogues**. Their quantity classes,
unit identities, and marker classes are distinct even when names match built-ins.
Applications must consistently import quantities from their generated package.
Importing an extension never changes the static meaning of `quantype.Length`.
Only listed physical relationships receive named results; generation is not
symbolic inference of arbitrary relationships.

This workflow is useful but deliberately limited. It does not transparently add
new overloads to installed built-ins, and it does not offer custom storage backend
registration. If compatibility with an existing library's `quantype.Length`
annotations is essential, prefer new units on the built-in kinds or explicit
application operations rather than a second catalogue.

## Serialization decisions

JSON includes the physical kind so dimensionally equal semantic kinds cannot be
silently substituted. Unit identifiers remain strings at this external boundary.
Custom definitions must be supplied for decoding; serialized data does not install
new unit definitions or execute code.

NPZ contains one numerical array per quantity and a JSON string array called
`metadata`. Version 1 metadata looks like:

```json
{
  "version": 1,
  "quantities": {
    "positions": {
      "kind": "Length",
      "unit": "angstrom",
      "array": "array_0",
      "source_backend": "numpy"
    }
  }
}
```

`source_backend` is optional provenance. Arrays themselves carry dtype and shape.
`load_npz(path, name, Length[NDArray[np.float64]])` chooses the destination storage
explicitly. There is no automatic backend import driven by metadata, and no device
or autograd-graph restoration. Torch bfloat16 and other unsupported NumPy host
representations fail rather than silently narrowing. Noncanonical conversions
can round; this is not a bit-exact model-checkpoint format.

NPY remains numerical data only. Its kind and unit must come from the caller's
schema or another external record. NumPy's normal `save`/`load` plus an explicit
typed quantity constructor is sufficient; there is no redundant NPY wrapper API.

## Compatibility and verification

Breaking changes include removal of shapes, replacement of the JSON object
format, explicit units in quantity constructors, and namespace precedence over
the three conflicting old flat names (`dimensionless`, `energy_density`,
`energy_per_volume`). Common flat unit names such as `u.nm` remain available.
Pydantic now converts storage/dtype rather than merely checking the container.

Core dependency lower bounds are NumPy 2.0, SciPy 1.13, and Pydantic 2.0.3. The
latter is a tested early-v2 version with Python 3.12 wheels; Pydantic 2.0.0's pinned
core had no usable wheel in the local binary-only test. Existing JAX/Torch lower
bounds were retained rather than promising compatibility with untested older
backend releases.

Checks include runtime contracts, strict positive/negative typing, stubtest,
generated-file freshness, lint/format checks, a core-only lower-bound environment,
and generated application-package conformance. CI is configured for separate core,
JAX, and Torch installations; hosted CI was not run from this session. The local
lower-bound test used NumPy 2.0.0, SciPy 1.13.0, and Pydantic 2.0.3 with neither
accelerator backend installed.

Completed local checks:

- 66 runtime tests passed with both extras.
- 52 tests passed, 6 skipped using the built wheel in the minimum-core environment.
- All four strict checkers passed; both mypy parsers and stubtest passed. Every
  checker rejected all 12 marked invalid expressions.
- Generated-file checks, lint/format checks, and wheel/sdist builds passed.
- A real Torch-produced NPZ archive was restored into NumPy in the environment
  without Torch or JAX. Independent-backend tests also block imports of the other
  backend while exercising each autodiff adapter.
