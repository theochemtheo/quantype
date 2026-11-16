# Quantity type annotations

Public imports: `from quantype.typing import QuantityType, quantity_type`.
These handwritten helpers let annotation-driven readers discover a named
quantity's [kind](../explanation/glossary.md#kind "A nominal physical meaning, not just a dimension vector.")
class, storage target, and unit system at runtime. They do not replace generated
static overloads or infer physical meaning for a raw model.

## quantity_type

`quantity_type(annotation: object) -> QuantityType | None`

- A named `Force[V, S]` returns its kind class, storage `V`, and system `S`.
  Omitted system means `NoSystem`; a bare `Force` has storage `None`.
- `Annotated` wrappers are unwrapped; their metadata is not part of the result.
- Nonquantity annotations return `None`, including unions such as
  `Force[float] | None`. Inspect union members yourself.
- Malformed quantity parameters, such as `Force[SI]` or `Force[float, int]`,
  raise `TypeError`. Structural `Quantity[...]` is not a named target and raises
  `TypeError` rather than inventing a wire kind.

`QuantityType` is a frozen dataclass with fields `kind` (the quantity class),
`storage` (the annotation object, or `None`), and `system` (the system class).
It describes a target, not a numerical instance; discovery does not convert or
allocate quantity storage.

```python
from typing import Annotated, get_type_hints

import numpy as np
import numpy.typing as npt

from quantype import Energy, Force
from quantype.systems import NoSystem, SI
from quantype.typing import QuantityType, quantity_type


class Sample:
    forces: Annotated[Force[npt.NDArray[np.float64]], "REF_forces"]
    energy: Energy[float, SI]
    label: str


hints = get_type_hints(Sample, include_extras=True)
assert quantity_type(hints["forces"]) == QuantityType(
    Force, npt.NDArray[np.float64], NoSystem
)
assert quantity_type(hints["energy"]) == QuantityType(Energy, float, SI)
assert quantity_type(hints["label"]) is None
assert quantity_type(Force) == QuantityType(Force, None, NoSystem)
assert quantity_type(Force[float] | None) is None
```

To restore into another storage, parameterize the discovered class again with
the selected storage and its system, for example
`Energy[npt.NDArray[np.float64], SI]`. See
[validation](../how-to/validate-data.md) and
[NPZ target types](serialization.md#npz-api-and-schema).

## Runtime discovery versus static checking

Named quantities use `Kind[V, S]`;
[structural products](../explanation/glossary.md#structural-product "A quantity whose physical meaning retains a product expression rather than a named kind.")
use `Quantity[K, V, S]` with `Mul`, `Div`, or `Pow` kind markers.
`S` defaults to `NoSystem` and is invariant.

!!! info "Runtime discovery does not describe static overloads"

    Runtime quantity classes are dynamically supplied; generated `.pyi` files refine
    named algebra, array operations, and derivative kinds for mypy, Pyright,
    Pyrefly, and ty without a plugin. Runtime introspection alone is not the static
    API reference.

A storage parameter is a conversion target, not merely a label. Integer/list
bare constructors leave static storage open; use a typed constructor for precise
storage. Shape inference and arbitrary dtype promotion are outside the static
contract. [Quantity construction](quantities.md#construction-and-storage) defines
constructor and conversion contracts;
[system adoption](systems.md#system-selection-and-adoption) and
[system-generic code](systems.md#write-system-generic-code) define system
boundaries; [numerical dispatch](numerical-operations.md#runtime-dispatch-versus-static-support)
distinguishes portable typed calls from native runtime dispatch.

A generated application catalogue has its own nominal classes and overloads;
subclassing or mixing its classes with built-in ones does not extend the
installed Quantype algebra. See [custom catalogues](../how-to/custom-catalogues.md).
