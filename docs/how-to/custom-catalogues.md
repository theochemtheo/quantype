# Generate an application catalogue

An application catalogue adds your own nominal
[kinds](../explanation/glossary.md#kind "Named physical meanings that remain distinct even when their dimensions match.")
and relations. Quantype generates a separate package with runtime classes,
type stubs, units, a `numpy` module, and optional `ujax` / `utorch` adapters.
No type-checker plugin is needed.
For only a new unit of an existing kind, use [define units](define-units.md)
instead.

## Install the generator prerequisite

!!! info "Generation requires Ruff"
    `generate` formats output with Ruff and requires it on `PATH`, including in
    `check=True` mode. Install it in your application environment:

```bash
uv add --dev ruff
# Or: python -m pip install ruff
```

Run generation in your application project, not Quantype's source tree.
The generation example writes to its current working directory.

## Extend the built-in definitions

Declare `SurfaceTension`, and a relation `Pressure * Length -> SurfaceTension`:

```python
from pathlib import Path

from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate

catalogue = builtin_catalogue().extend(
    quantities={
        "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0, 0), "surface_tension"),
    },
    units={"surface_tension": UnitSpec("SurfaceTension")},
    relations={("mul", "Pressure", "Length"): "SurfaceTension"},
)
changed = generate(catalogue, "src/labquantities", package="labquantities")
assert Path("src/labquantities/__init__.py").is_file()
```

`changed` lists files written by this call. An empty list means the generated
output is already current; rerunning the example does not need to rewrite it.

[Dimensions](../explanation/glossary.md#dimensions "The exponents over Quantype's eight physical base axes.")
are exponents over eight axes, in order: length, energy, time, temperature,
magnetic moment, atom, electron, charge. This definition is energy / length²;
consult [the catalogue](../reference/catalogue.md) for the built-in dimensions.

Multiplication relations apply in either order and define divisions undoing
the product, here `SurfaceTension / Length -> Pressure`. An explicit division
wins. If two products imply conflicting inverse divisions, declare that
division yourself. Definitions must have consistent dimensions and unique,
valid names; `extend` does not replace existing definitions.

## Import only from the generated API

For a `src/` project, install your application in editable mode
(`uv pip install -e .`) so Python can find `labquantities`. For a standalone
experiment, generate into `"labquantities"` beside your script instead.
This consumer is illustrative until that external application package is
installed:

<!-- skip: next "Requires the generated labquantities application package installed on the import path" -->
```python
from typing import assert_type

import labquantities.numpy as qnp
from labquantities import Length, Pressure, SurfaceTension, u

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)
assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])
assert_type(qnp.sqrt(length * length), Length[float])
```

A generated `labquantities.Length` is nominally different from `quantype.Length`;
do not mix imports. The built-in operators are unchanged. Structural results
use the generated package's own `Quantity` subclass, so use it for structural
annotations. It is still a subclass of `quantype.core.Quantity`; reciprocals
retain the generated catalogue's `DimensionlessKind`.

The package's `products` module includes an unnamed product class for every
pair of its kinds, such as `LengthTime` or `SurfaceTensionPerTime`. Generation
fails if a generated product name clashes with a named kind. Other reserved
bindings, such as `Unit`, `np`, typing imports, `u`, kind markers such as
`LengthKind`, unit aliases, and colliding quantity namespaces are rejected.

## Maintain generated files

Public generation entry points are:

- `quantype.codegen.generate(catalogue, output, *, package="quantype", check=False)
  -> list[str]`: returns changed filenames; with `check=True`, returns stale
  filenames without writing. Both modes require Ruff on `PATH`.
- `quantype.codegen.render(catalogue, *, package="quantype") -> dict[str, str]`:
  returns source strings without running tools or writing files.

The generator owns `_products/`, `_constants/`, `_kinds/`, and
`_generated.pyi`, including removal of obsolete modules in its private folders.
Exclude these minified/generated files from your own formatter and linter.

Built-in units follow the process's CODATA edition. Custom factors are written
as Python float literals fixed at generation time. NumPy-derived factors work,
rounded to binary64; factors outside binary64's range are rejected. Consult
[units and constants](../reference/units.md) and
[choose constants](choose-constants.md) when reproducibility matters.

## Use systems and backend adapters

Generated quantities accept the same storage and system parameters:
`Length[float, SI]` works without extra catalogue definitions. Systems are
specified by base axes, so each new kind has a unit in every system. Where no
named catalogue unit exists, the unit has a system-qualified identifier such as
`si:SurfaceTension` and symbol `J/m^2`. See
[systems reference](../reference/systems.md) and
[define systems](define-systems.md).

Import `numpy`, `ujax`, and `utorch` from your generated package too. Install
the corresponding backend extra to use an adapter. The tasks in
[JAX differentiation](autodiff-jax.md) and [Torch differentiation](autodiff-torch.md)
apply with those generated imports; generated JAX adapters register their own
nominal classes as pytrees. Refer to [type annotations](../reference/type-annotations.md)
for the boundary between runtime discovery and static contracts.
