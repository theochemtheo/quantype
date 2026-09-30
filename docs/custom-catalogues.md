# Custom catalogues

Application catalogues use the same definition model, validation, and renderers
as the built-in API. Generation requires Ruff for formatting, not a type-checker
plugin. Install it in the environment running generation (`uv add --dev ruff`
in an application project, or `python -m pip install ruff`).

## Generate an application catalogue

```python
from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate

catalogue = builtin_catalogue().extend(
    quantities={
        "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0), "surface_tension"),
    },
    units={"surface_tension": UnitSpec("SurfaceTension")},
    relations={("mul", "Pressure", "Length"): "SurfaceTension"},
)
generate(catalogue, "src/labquantities", package="labquantities")
```

Run generation from your application project, not from quantype's source tree.
For the `src/` layout above, install your application in editable mode (for example,
`uv pip install -e .`) so Python can import `labquantities`. Alternatively, generate
into `"labquantities"` beside a script for a standalone experiment.

Then import consistently from that generated package:

```python
from typing import assert_type
from labquantities import Length, Pressure, SurfaceTension, u

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)
assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])
```

Multiplication relations are symmetric; division relations are explicit.
Generation includes runtime classes, stubs, units, numerical helpers, and
optional autodiff adapters. `generate(..., check=True)` returns stale file names
without writing; `render(...)` returns source strings without invoking tools.
Names and aliases that collide with generated API bindings (such as `sqrt` or
`get_unit`) are rejected, as are conflicting quantity namespaces. Quantity
names also reserve imported typing bindings, package aliases such as `u`, and
all generated kind-marker names (for example, `LengthKind`). Unit scales and
offsets are emitted as Python float literals; NumPy-derived factors are supported,
with binary64 rounding. Factors outside binary64's range are rejected.

A generated catalogue is a separate, combined API, not an extension of the
installed classes. Its `Length` is a distinct nominal type from
`quantype.Length`; import quantities consistently from the generated package.
Generating a package does not change `quantype`'s operator overloads. Generated
structural results use the package's own `Quantity` subclass; import that class
for structural annotations so reciprocal types retain the catalogue's
`DimensionlessKind`. It remains a subclass of `quantype.core.Quantity`.

To define a new unit for an existing physical kind instead, see
[custom units](units.md#define-a-unit-without-global-registration).
