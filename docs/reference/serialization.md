# Serialization and validation contracts

Public APIs: quantity `.to_dict` / `.parse`, Pydantic quantity schemas, and
`from quantype.serialization import save_npz, load_npz`.
Optional numerical storage requires its own backend extra. See
[persistence tasks](../how-to/serialize.md) and
[configuration validation](../how-to/validate-data.md) for complete workflows.

## Value boundary and unit selection

!!! warning "Value interchange is not a model checkpoint"

    Only values are serialized: graphs, devices, and model state are not preserved.
    Torch tensors are detached and transferred to CPU at this host boundary.
    Unsupported host dtypes, such as Torch bfloat16, raise `TypeError` rather than
    silently approximating.

Serialization requires a **named** semantic kind;
[structural products](../explanation/glossary.md#structural-product "A quantity whose physical meaning retains a product expression rather than a named kind.")
have no named wire representation. Use an intentional `Kind.reinterpret` with
equal dimensions or an [application catalogue](../how-to/custom-catalogues.md)
to give them meaning.

Formats record units, not unit systems. A decoding target chooses storage and
system, so wire data from one system can decode into another. The wire unit is
remembered for display. Unit conversion can introduce roundoff; these formats
are not bit-exact snapshots.

`q.to_dict(unit=None) -> dict[str, object]` selects an explicit compatible unit,
otherwise the quantity's presentation unit (which defaults to its system's unit
when bound). Wrong-kind units raise `ValueError`; a non-unit argument raises
`TypeError`. Unnamed kinds or absence of a writable unit raise `ValueError`.
Unit identifiers use names, not ambiguous
symbols: a composed `eV/nm` writes `electron_volt/nanometer`. Custom-system units
can use identifiers such as `gromacs:Force`.

## JSON wire objects and strings

A written quantity has exactly these fields:

```json
{"kind": "Length", "magnitude": 0.5, "unit": "nanometer"}
```

- `kind` is the nominal class name (`Length.kind == "Length"`), not dimensions.
  It must match the target kind. Readers may omit it when the expected kind is
  already known, but writers always include it.
- `magnitude` is a real number or nested numerical arrays. Booleans, complex
  values, strings in numerical arrays, and nonnumerical data are rejected.
- `unit` is a string identifier compatible with that nominal kind. Unknown
  identifiers or wrong-kind units fail. Extra object fields are rejected.

`Length.parse(data, *, units=())` accepts a scalar string such as `"5 angstrom"`,
a wire object, or an existing quantity of the same kind and **exact system**.
Names, aliases, and printed symbols parse, including `"25 °C"` and
`"1 eV/Å^2"`. A number without a unit fails except for `Dimensionless`, whose
bare scalar strings are accepted. Decoded wire data becomes Python `float` or
a float64 NumPy array; a scalar or zero-dimensional wire magnitude parses as
`float`. Existing quantities must have Python `float` or NumPy array storage;
NumPy arrays of other dtypes, including float32, are normalized to float64.
Other storage, such as JAX/Torch tensors or NumPy float32 scalars, raises
`TypeError`; use a typed Pydantic field or `TypeAdapter` for those quantities.
Do not rely on parsing preserving an existing object's identity or dtype.

`Length[float, SI].parse(...)` chooses SI but does **not** select parse storage.
For exact storage, use Pydantic `TypeAdapter(Length[V, S])` instead. Wire decode
errors are `ValueError`; Pydantic reports them as `ValidationError`.

JSON stores no backend, dtype, device, or system metadata. Extended-precision
NumPy values are rounded to Python binary64. A finite value that would become
infinity or zero raises `ValueError`; NPZ is the extended-precision alternative.
Original scalar display values are retained where possible: configuration
`"0.5 nm"` writes back as `0.5 nanometer` in any system, and `"20.1 degC"` as
exactly `20.1 celsius`.

## Pydantic schemas and target storage

`TypeAdapter(Length[V, S]).validate_python(data)` and `.validate_json(data)`
accept the inputs above and convert to annotated storage, dtype, and system.
`BaseModel` fields use the same schema. Python/NumPy scalar targets reject
nonscalar arrays; an array annotation preserves zero-dimensional array storage.
A bare `Length` annotation has a runtime schema, but is not a precise static
storage annotation.

!!! warning "Fields do not adopt a quantity's system"

    At this [validation boundary](../explanation/glossary.md#validation-boundary "An entry point that checks external data and converts it into a declared quantity target."),
    existing quantity instances must already have the field's system; validators
    do not adopt one or implicitly bridge systems. Convert with
    [`.to_system(S)`](quantities.md#conversion-and-numerical-access) first, or provide
    wire data that can decode into the field's system.

`model_dump()` keeps quantity objects; `model_dump(mode="json")`,
`model_dump_json()`, and a type adapter's JSON writer emit wire objects.

## Custom-unit decoding

Custom units are explicit inputs, not global registration:

| Reader | Supply definitions |
| --- | --- |
| `Kind.parse(data, *, units=())` | `units=(custom_unit, ...)` |
| `load_npz(..., units=())` | `units=(custom_unit, ...)` |
| Pydantic validation | `context={"units": (custom_unit, ...)}` |

The target custom system's own units are found automatically, including
system-qualified derived identifiers. Duplicate definitions and shadowing of
built-in identifiers are rejected. When an identifier occurs across kinds,
lookup prefers definitions of the requested kind. See
[unit definitions](units.md#define-a-unit) and
[system-unit decoding](systems.md#derived-units-and-decoding); the
[define-units task](../how-to/define-units.md) provides a complete example.

## NPZ API and schema

- `save_npz(path: str | Path, *, record_backend=False, **quantities) -> None`
  writes numerical arrays in each quantity's display unit and metadata.
- `load_npz(path: str | Path, name: str, target: type[Q], *, units=(), dtype=None)
  -> Q` reads one entry, converting to an explicit parameterized target such as
  `Length[npt.NDArray[np.float64], SI]`. A bare kind target raises `TypeError`.
  NumPy dtype annotations govern dtype; conflicting `dtype=` fails. JAX/Torch
  targets use `dtype=` when an explicit dtype is required.

NPZ uses `allow_pickle=False`. The `metadata` array holds a JSON string:

```json
{
  "version": 1,
  "codata": "2022",
  "quantities": {
    "positions": {
      "kind": "Length",
      "unit": "nanometer",
      "array": "array_0",
      "source_backend": "numpy"
    }
  }
}
```

`version` must be integer `1`. `quantities` maps names to objects whose required
fields `kind`, `unit`, `array` are strings; optional `source_backend` is a string.
`codata` is optional string provenance for the unit definitions' CODATA edition.
Other top-level or entry fields, invalid metadata, missing entries/arrays,
wrong kinds/units, and object, boolean, complex, or nonnumerical arrays fail.
`source_backend` is provenance, never an instruction to import or restore that
backend. A Torch archive can load into NumPy with no Torch installed.

Dtype and shape, including zero-dimensional and empty arrays, are retained
subject to the requested target storage and unit conversion. No dtype is
silently substituted when the requested backend cannot provide it.

## NPY boundary

NPY contains numbers only, with no kind, unit, system, or CODATA metadata.
Save `q.magnitude(unit)` using `np.save(..., allow_pickle=False)` and reconstruct
from `np.load(..., allow_pickle=False)` plus an external unit agreement. See
[the NPY workflow](../how-to/serialize.md#use-npy-with-an-explicit-unit-agreement).
