# Autodiff adapters

Public imports: `from quantype import ujax` (requires `quantype[jax]`) and
`from quantype import utorch` (requires `quantype[torch]`). Each adapter works
without the other backend installed. Complete tasks:
[JAX](../how-to/autodiff-jax.md), [Torch](../how-to/autodiff-torch.md).

## Physical derivatives and systems

Derivatives divide the output kind by the input kind using declared physical
relations. Hessians divide twice. `Energy / Length -> Force` and
`Force / Length -> ForceConstant`.

!!! warning "A gradient is not physical force"

    The returned gradient is **positive**: physical force is `-gradient`.
    The `Force` [kind](../explanation/glossary.md#kind "A nominal physical meaning, not just a dimension vector.")
    specifies the derivative's physical units, not its sign convention.

The output must be a scalar quantity; differentiated arguments must be
quantities with backend-compatible numerical storage. Input and output systems
must match exactly, including `NoSystem`; mismatches raise `TypeError`.

!!! info "Derivative units follow storage, not presentation"

    An explicit system produces derivatives in that system's units, rescaled for
    any overridden kinds (for example LAMMPS pressure in bar). With `NoSystem`,
    derivatives use [reference units](units.md#reference-units-and-conversion-precision)
    (Å, eV, fs, etc.), independently of displayed input units: energy in kcal/mol
    versus length in nm still gives eV/Å.

Named derivative results and the system parameter are refined in generated
stubs, including system-generic functions. Unlisted relations retain structural
physical types. Precision is advertised for the default first-argument JAX
path and a single Torch input; other argument selections / heterogeneous
sequences do not have equally precise static result kinds. See
[type annotations](type-annotations.md) and [systems](systems.md).

## JAX API

| Signature | Result and scope |
| --- | --- |
| `ujax.grad(function, argnums=0)` | Callable returning the derivative of a scalar quantity output |
| `ujax.value_and_grad(function, argnums=0)` | Callable returning `(output, derivatives)` from one model evaluation |
| `ujax.hessian(function, argnums=0)` | Callable returning a full Hessian for one positional quantity argument |
| `ujax.jit(fun)` | Compiled callable with the original callable's static signature |
| `ujax.vmap(fun, in_axes=0, out_axes=0)` | Vectorized callable; static signature preserved, shapes not tracked |

`grad` / `value_and_grad` accept integer or tuple-of-integer `argnums`; an integer
returns one derivative and a tuple returns a tuple, even for a one-item tuple.
`hessian` accepts one integer, not a tuple. Other arguments and keyword arguments
pass through to the model. Default first-argument stubs use a ParamSpec to
preserve further parameter types; explicit nondefault `argnums` uses an untyped
result signature. These are not wrappers for arbitrary `jax.grad` options;
JVP/VJP are not part of this public adapter surface.

Nonquantity differentiated arguments or nonquantity outputs raise `TypeError`.
Nonscalar quantity outputs fail JAX's scalar-output differentiation requirement.
Use `ujax.grad`, not native `jax.grad`, for quantity outputs: before registration
JAX rejects quantities as invalid JAX types; after registration it treats the
output as a container rather than the scalar numerical output it requires.

## JAX pytrees and transformations

Importing `ujax` registers quantities as single-leaf pytrees. The leaf contains
raw system values, or reference-unit values in `NoSystem`. Static metadata is
the semantic kind and system, **not the display unit**. `jit` and `vmap` accept
these quantities; `lax.cond` and `lax.scan` can handle compatible branches/states
with differing display units. Different systems have different tree structures.

One trace can serve inputs given in nm, Å, or bohr. Flattening a `NoSystem`
quantity stored in another unit rescales outside the trace; this can round the
numbers, and buffer donation acts on the rescaled copy. Reconstructed results
are displayed in reference units without a system, or system units when bound.
`jax.device_put` and `jax.tree_util.tree_map` also flatten and reconstruct.

!!! warning "Name magnitude units inside JAX transformations"

    Inside a transformation, `.magnitude()` on a `NoSystem` quantity sees reference
    units, unlike the original outside object. Name the unit explicitly
    (`x.magnitude(u.nm)`) in traced code. `jax.numpy` cannot dispatch quantities;
    use [quantype.numpy](numerical-operations.md#runtime-dispatch-versus-static-support).
    Quantities are unhashable and cannot be `jax.jit` static arguments; pass an
    appropriate hashable numerical representation and reconstruct if a static
    argument is truly needed.

JAX float64 requires x64 support enabled. Explicit unavailable dtype requests
raise, rather than silently fall back. Constructors avoid host transfer of
existing backend arrays; pytree reconstruction does not coerce tracer leaves.

## Torch API and graph lifetime

`utorch.grad(output, inputs, *, create_graph=False, retain_graph=None)`
differentiates an already evaluated zero-dimensional quantity output.
One quantity input returns one derivative; a sequence returns a tuple.
Nonscalar output raises `ValueError`; incompatible systems raise `TypeError`.
Source tensors need `requires_grad=True`; ordinary list construction does not
turn it on. Torch's ordinary failures for missing/unused graphs still apply.

`create_graph=True` makes the derivative differentiable; `retain_graph` follows
Torch's graph-retention semantics, including its default. Arithmetic, storage
conversion, and unit conversion preserve graph connectivity and device placement
where supported. A tensor annotation does not encode dtype: use explicit
`dtype=` when necessary.

`.to(unit)` changes presentation, not device. For device moves or intentional
detaching use `q.apply(lambda raw: raw.to(device))` or
`q.apply(lambda raw: raw.detach())`. A raw-model output may be wrapped using
`Energy[torch.Tensor](raw, u.eV)`, preserving its graph while asserting physical
meaning. `Energy.from_value(raw, Atomistic)` instead asserts raw values in the
system's units. [Serialization](serialization.md#value-boundary-and-unit-selection)
is an explicit host boundary that loses graph/device state; it is not a model
checkpoint API.
