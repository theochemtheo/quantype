# Unit storage and display: prototype comparison

Evidence for [ADR-003](../decisions/ADR-003-unit-system-is-a-type-parameter.md)
and [ADR-004](../decisions/ADR-004-display-unit-is-presentation.md). Eleven
standalone prototypes, built outside the repository in September 2026, reused
quantype's kinds, relations, temperature rules, and unit scales, and differed
only in where magnitudes were stored and which unit was shown. One scenario
suite covered presentation, serialization, the raw-value boundary, JAX (`jit`,
`scan`, `cond`, `grad`, with x64 enabled), and cost.

## Round 1: storage and display

At the time, every value was stored in Å, eV, and fs: `Length[float](2, u.nm)`
printed `20.0 Å`, and a configuration's `"0.5 nm"` was written back as
`5.0 angstrom`.

| Variant | Storage | Shown unit | Unit in JAX pytree metadata |
| --- | --- | --- | --- |
| V1 canonical | Å, eV, fs | canonical | yes |
| V2 sticky display | Å, eV, fs | input unit; same-kind operations keep the left operand's | yes |
| V3 preferences | Å, eV, fs | per-kind application preferences | no |
| V4 native (as pint) | input unit | storage unit; products compose units | yes |
| V5 native, normalising | input unit while operands agree | storage unit | yes |
| V6 sticky, erased | Å, eV, fs | as V2 | no |

| Scenario | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|
| `Length(2, nm)` repr | 20.0 Å | 2.0 nm | 2.0 nm | 2.0 nm | 2.0 nm | 2.0 nm |
| `.value` of 2 nm vs 20 Å | same | same | same | differs | differs | same |
| config `"0.5 nm"` written back | 5.0 angstrom | 0.5 nanometer | 0.5 nanometer | 0.5 nanometer | 0.5 nanometer | 0.5 nanometer |
| `jit` traces for nm, Å and bohr inputs | 1 | 3 | 1 | 3 | 3 | 1 |
| `lax.cond` with nm and Å branches | ✓ | fails | ✓ | fails | fails | ✓ |
| force from `grad`, shown as | eV/Å | eV/Å | eV/Å | eV/Å^2·nm^2/nm | eV/nm | eV/Å |
| 10⁶-element add, mixed units | 0.49 ms | 0.50 ms | 0.50 ms | 1.04 ms | 1.03 ms | 0.52 ms |

Findings:

- Remembering the input unit (V2–V6) fixed the REPL, `.magnitude()`,
  reductions, temperatures, and configuration round trips.
- Storing the input unit (V4, V5) gave the same length a `.value` of 2.0 or
  20.0 depending on how it was built, needed unit algebra, and doubled the cost
  of mixed-unit arithmetic.
- Any unit in pytree metadata broke `lax.cond` and recompiled per input unit.
  V6, which keeps the display unit out of the metadata, traced once.
- V6's cost: a value coming out of a JAX transformation shows the storage unit.
- Global preferences (V3) made serialization depend on ambient state.
- An offset unit round-tripped with roundoff (`20.1 °C` came back as
  `20.100000000000023`). Keeping the Python scalar a quantity was built from
  (exact echo) fixed it in about 45 lines.

## Round 2: choosable storage systems

Coherent systems over the base axes: Atomistic (Å, eV, fs), LAMMPS `metal`
and `real`, SI, CGS, and Hartree atomic units.

| Variant | Storage | How the system is chosen |
| --- | --- | --- |
| V7 global | the active system | once per process, locked by the first construction |
| V8 per-quantity | the quantity's own system | an ambient default or `system=`; mixed operations convert |
| V9 boundary | Å, eV, fs | conversion helpers at kernel boundaries, and a display fallback |

- V7 kept `.value` consistent only by locking a global, so tests, notebooks, and
  libraries assuming different systems couldn't share a process.
- V8 let the same 2 nm be `2e-09` or `20.0` depending on an ambient default, and
  `lax.cond` failed across systems.
- Storing in SI underflows float32 for ordinary atomistic values: (1 meV)² is
  2.5e-44 J², below float32's smallest normal number, and a float32 variance of
  [0, 2] meV came out 1.7% low, with no error or warning. Atomistic, metal,
  real, and atomic units stayed in range.
- Install extras (`quantype[si]`) can't change a package's code, and pip merges
  conflicting extras without error.

## Round 3: the system in the static type

V11 stored each quantity in its own system, as V8, and made the system the last
type parameter, with a PEP 696 default so `Length[float]` kept its meaning.
Combining systems raised `TypeError` at runtime.

A probe with 15 positive checks and 7 lines that must be rejected ran through
Pyright 1.1.414, mypy 2.3.1, Pyrefly 1.3.2, and ty 0.0.84. All four accepted
every positive check, including the defaulted parameter and functions generic
over the system, and rejected all seven misuses: adding SI and metal lengths,
dividing across systems, a generic function given two systems, SI data passed
to a kernel typed for `real`, and the existing kind errors.

What V11 left open: SI storage still underflows float32; untyped code gets only
the runtime check; a library typed `Length[float]` rejects SI quantities unless
written generically; and autodiff, Pydantic, NPZ, and every stub must carry the
system.
