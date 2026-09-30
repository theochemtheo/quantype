# Custom catalogues

Application catalogues use the same definition model, validation, and renderers
as the built-in API. Generation requires Ruff for formatting, not a type-checker
plugin. Install it in the environment running generation (`uv add --dev ruff`
in an application project, or `python -m pip install ruff`).

## Generate an application catalogue

```python notest
from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate

catalogue = builtin_catalogue().extend(
    quantities={
        "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0, 0), "surface_tension"),
    },
    units={"surface_tension": UnitSpec("SurfaceTension")},
    relations={("mul", "Pressure", "Length"): "SurfaceTension"},
)
generate(catalogue, "src/labquantities", package="labquantities")
```

A `QuantitySpec`'s dimensions are exponents over quantype's eight base axes, in
order: length, energy, time, temperature, magnetic moment, atom, electron, and
charge. `SurfaceTension` above is energy / length². The
[catalogue reference](catalogue.md) shows every built-in kind's dimensions.

Run generation from your application project, not from quantype's source tree.
For the `src/` layout above, install your application in editable mode (for example,
`uv pip install -e .`) so Python can import `labquantities`. Alternatively, generate
into `"labquantities"` beside a script for a standalone experiment.

Then import consistently from that generated package:

```python notest
from typing import assert_type
import labquantities.numpy as qnp
from labquantities import Length, Pressure, SurfaceTension, u

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)
assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])
assert_type(qnp.sqrt(length * length), Length[float])
```

Multiplication relations are symmetric, and each also names the divisions that
undo it, so the catalogue above gives `SurfaceTension / Length -> Pressure`.
Declared divisions take precedence; a division two multiplications would undo
differently must be declared.

Generation includes runtime classes, stubs, units, per-kind constant classes, a
`numpy` module typed with the catalogue's classes, and optional autodiff
adapters (`ujax`, `utorch`). `generate(..., check=True)` returns stale file names
without writing; `render(...)` returns source strings without invoking tools.

Names and aliases that collide with generated API bindings (such as `Unit` or
`np`) are rejected, as are conflicting quantity namespaces. Quantity names also
reserve imported typing bindings, package aliases such as `u`, and all
generated kind-marker names (for example, `LengthKind`).

Built-in units in the package follow the process's CODATA edition, as
quantype's do. Units you define are emitted with their scales and offsets as
Python float literals, fixed at generation; NumPy-derived factors are
supported, with binary64 rounding, and factors outside binary64's range are
rejected.

A generated catalogue is a separate, combined API, not an extension of the
installed classes. Its `Length` is a distinct nominal type from
`quantype.Length`; import quantities consistently from the generated package.
Generating a package does not change `quantype`'s operator overloads. Generated
structural results use the package's own `Quantity` subclass; import that class
for structural annotations so reciprocal types retain the catalogue's
`DimensionlessKind`. It remains a subclass of `quantype.core.Quantity`.

Generated quantities take a unit system parameter like the built-in ones:
`Length[float, SI]` works with no extra definitions. Unit systems are defined
over base axes, not kinds, so every kind in the catalogue has a unit in every
system. A kind without a named catalogue unit in a system gets a
system-qualified identifier, such as `si:SurfaceTension` (symbol `J/m^2`); see
[unit systems](unit-systems.md#derived-units).

To define a new unit for an existing physical kind instead, see
[custom units](units.md#define-a-unit-without-global-registration).
