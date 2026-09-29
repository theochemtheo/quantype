# `quantype`

A Python 3.12+ prototype of **physical meaning in static types**, with canonical
numerical storage for NumPy, JAX, and PyTorch. No type-checker plugins or tensor
subclasses are required.

```python
from typing import assert_type
from quantype import Energy, Force, Length, u

distance = 2.0 * u.angstrom
energy = 1.0 * u.hartree
force = energy / distance

assert_type(distance, Length[float])
assert_type(force, Force[float])
assert_type(energy.to(u.eV), Energy[float])
print(energy.magnitude(u.eV))  # approximately 27.2114
```

Compatible units produce the same nominal quantity type. `Energy + Length`,
converting length to seconds, and passing `EnergyDensity` to a function expecting
`Pressure` are rejected statically and at runtime.

## Construction, conversion, and numerical boundaries

```python
import numpy as np
from quantype import u

positions = u.angstrom(np.array([[1.0, 2.0, 3.0]]))
also_positions = u.angstrom * np.zeros((100, 3))
areas = positions**2
lengths = u.sqrt(areas)
total = positions.sum()
```

- Use `unit(array)` as the reliable cross-backend constructor. `unit * array`
  also works; NumPy additionally supports `array * unit`. Backend-left dispatch
  is not promised for every backend.
- Unit construction normalizes Python integers to `float`. Numerical arrays
  remain owned by their backend; identity conversions do not allocate or detach.
- `.value` explicitly exposes **canonical** numerical data.
- `.magnitude(unit)` converts numerical data to a chosen compatible unit.
- `.to(unit)` chooses a presentation/serialization unit without changing the
  canonical data. `.magnitude()` uses that presentation unit, or canonical units
  when none has been chosen. Construction defaults to canonical presentation.
- `Energy.from_canonical(raw)` is an explicit, cheap, trusted model boundary.
  Hidden activations need not carry physical units.
- `sum(axis=None, keepdims=False)` and `mean(...)` preserve physical kind.
  NumPy scalar reductions are wrapped as zero-dimensional ndarrays.
- `u.sqrt(Area)` returns `Length`; `u.sin(Angle)` and `u.exp(Dimensionless)`
  return `Dimensionless`. These helpers delegate to the stored backend.
- Implicit NumPy coercion is rejected rather than silently discarding units.

Canonical units are angstrom, eV, fs, kelvin, Bohr magneton, atom, and electron.
The last two are genuine independent dimensional axes. Pressure uses eV/Å³,
force eV/Å, and force constants eV/Å².

## Meaning, dimensions, and temperatures

The catalogue includes the recommended simulation quantities plus
`EnergyPerVolume`, `Frequency`, `InverseTime`, `AtomCount`, and `ElectronCount`.
Dimensional equality never chooses a semantic result: `Force / Area` is
`Pressure`, while `Energy / Volume` is `EnergyDensity`. Their unit names are
deliberately distinct (`eV_per_angstrom_cubed` versus
`energy_density`).

```python
from quantype import TemperatureDifference, u
from typing import assert_type

cold = 0 * u.celsius
warm = 300 * u.K
delta = warm - cold
assert_type(delta, TemperatureDifference[float])
restored = cold + delta
rate = delta / (2 * u.s)
```

Absolute temperatures are affine points: two cannot be added, multiplied,
scaled, or exponentiated. Differences can be scaled and divided by time.
Absolute-temperature means are supported, but sums are rejected.

Uncatalogued products and ratios retain semantic expression markers such as
`Quantity[Mul[LengthKind, TimeKind], float]`, never `Any`. Runtime dimensional
metadata remains available through `.dimensions` (basis order given above).
This is not an arbitrary symbolic simplifier; see the prototype limits below.

## Pydantic and serialization

```python
from pydantic import BaseModel
from quantype import Force, Length, Temperature, u


class RelaxationConfig(BaseModel):
    force_tolerance: Force[float]
    max_displacement: Length[float]
    temperature: Temperature[float]


config = RelaxationConfig.model_validate(
    {
        "force_tolerance": {"value": 0.01, "units": "eV / angstrom"},
        "max_displacement": {"value": 0.1, "units": "nm"},
        "temperature": "20 degC",
    }
)
wire_json = config.model_dump_json()
restored = RelaxationConfig.model_validate_json(wire_json)

wire = config.max_displacement.to(u.nm).to_dict()
# {"value": 0.1, "units": "nanometer"}
length = Length.parse(wire)
```

Bare annotations such as `cutoff: Length` work at runtime; strict static code
should specify storage. NumPy fields use `Length[numpy.typing.NDArray[np.float64]]`.
Structured lists parse into float64 arrays. Scalar and array storage annotations
are validated, and JSON schemas describe accepted unit names and wire values.
Unit aliases normalize to registry names on serialization; no Python type
machinery appears in the payload.

`Length.parse` accepts wire data and scalar/NumPy quantities and honestly returns
scalar-or-array storage. For a specific storage type, use a typed Pydantic field
or `TypeAdapter(Length[float])`. Validation errors name the expected and received
quantity. Serialization is an explicit host boundary: Torch values are detached
and copied to CPU there, never inside arithmetic or differentiation.

## Physically typed automatic differentiation

```python
import jax
import jax.numpy as jnp
from quantype import Energy, Length, u, ujax


def harmonic(x: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


x = u.angstrom(jnp.array([1.0, 2.0, 3.0]))
energy = jax.jit(harmonic)(x)
dE_dx = ujax.grad(harmonic)(x)  # Force[jax.Array], [2, 4, 6]
hessian = ujax.hessian(harmonic)(x)  # ForceConstant[jax.Array], 2 * identity
forces = -dE_dx
```

Importing `ujax` registers quantities as single-leaf pytrees. JAX `jit` and `vmap`
then transform quantity-aware functions; only kind/display metadata is static.
Derivative adapters unwrap canonical data and use the same division registry as
ordinary arithmetic. Changing input units does not rescale the physical gradient.

```python
import torch
from quantype import Energy, Length, u, utorch


def harmonic_torch(x: Length[torch.Tensor]) -> Energy[torch.Tensor]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


x = u.angstrom(torch.tensor([1.0, 2.0, 3.0], requires_grad=True))
dE_dx = utorch.grad(harmonic_torch(x), x, create_graph=True)
```

Gradients are **positive derivatives**. Physical force is `-grad(energy)`.
Torch graph creation and retention options are forwarded to `torch.autograd.grad`.
Autodiff wrappers currently support a single quantity input and scalar output.

## Shapes

`quantype.shapes.Vector3` and `SymmetricTensor3` are reusable, runtime-validated
NumPy constructors, not physical kinds or ndarray subclasses:

```python
import numpy as np
from quantype import u
from quantype.shapes import Vector3, SymmetricTensor3

position = u.angstrom(Vector3([1, 2, 3]))
force = u.eV_per_angstrom(Vector3([0, 0, -1]))
stress = u.eV_per_angstrom_cubed(SymmetricTensor3(np.eye(3)))
```

This small demonstrator checks shape/symmetry at construction; it does not
promise compile-time tensor shapes or preserve shape constraints after arithmetic.

## Design and prototype limits

`src/quantype/_registry.py` is the single source for quantity dimensions,
canonical units, aliases, and named algebra. `scripts/generate.py` validates it
and generates nominal classes, unit definitions, arithmetic stubs, and both
autodiff stub sets. Runtime work lives in `core.py`; Pydantic, numerical helpers,
and backend transformations stay in narrow modules.

The generated stubs use ordinary generics and overloads. Python typing cannot
express “any quantity except the named cases above”, so generated files narrowly
suppress **overlapping overload** diagnostics: named cases intentionally precede
structural fallbacks. Public results do not use `Any`. Dynamic casts are confined
to runtime dispatch and third-party backend interfaces.

Deliberate limits:

- A finite physical catalogue, not a complete SI/metrology system.
- No arbitrary unit-string evaluation, symbolic cancellation, or automatic
  reinterpretation of dimensionally equal semantic kinds.
- Structural results retain typing/metadata, but the initial static API does not
  support chaining arbitrary structural expressions or serializing unnamed kinds.
  Add a named registry relationship when an operation belongs in a scientific API.
- No complete NumPy interception, implicit array conversion, arbitrary mixed
  backends, tensor subclasses, general JVP/VJP, or multi-input autodiff.
- Storage typing identifies the backend/container; arbitrary NumPy dtype
  promotion and shape algebra are outside this prototype.
- Array wire payloads preserve nested numerical values, not device, dtype,
  gradients, or zero-dimensional-array versus scalar identity.
- Pydantic validates ndarray storage, not all parametrized dtype/shape metadata.

## Development

Use **uv and just** for dependency management and project workflows:

```bash
uv sync
uv run just generate         # regenerate after registry edits
uv run just check-generated  # fail on stale generated files
uv run just test             # runtime/conversion/schema/JAX/Torch workflows
uv run just typecheck        # strict mypy, Pyright, Pyrefly, ty + negative tests
uv run just lint
uv run just format
uv build
```

`tests/typing/positive` uses `assert_type`, not merely absence of errors.
`scripts/check_typing.py` requires every checker to diagnose every marked invalid
expression in `tests/typing/negative`; a checker failing for unrelated reasons is
not sufficient. Runtime tests also validate every named relation's dimensions.
The scaffold's CPU Torch index and locked dependencies are retained.
