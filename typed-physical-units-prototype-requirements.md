# Typed Physical Units Prototype — Requirements

## Purpose

Prototype a Python physical-units library for scientific and machine-learning code, with a particular focus on atomistic materials simulation.

The prototype should explore whether physical quantities and units can be represented naturally in Python's static type system while retaining excellent runtime ergonomics, Pydantic-style validation, NumPy compatibility, and typed automatic differentiation with JAX and PyTorch.

The prototype is not intended to provide a complete SI or metrology system. Its purpose is to validate the core type model, developer experience, interoperability, and physical correctness of the design.

## Design principles

- Physical quantity semantics should be represented in the static type system.
- Units should be used primarily for construction, conversion, serialization, and boundary validation.
- Quantity types should describe physical meaning rather than merely dimensionality.
- Invalid operations should be rejected as early as possible, preferably by the type checker.
- The public API should not require a type-checker plugin.
- The same source should work under strict configurations of mypy, Pyright, Pyrefly, and ty.
- Scalar, NumPy, JAX, and PyTorch values should share one quantity abstraction rather than parallel class hierarchies.
- Backend numerical libraries should continue to perform the underlying numerical work.
- Automatic differentiation should preserve physically correct derivative units.
- The prototype should optimize for scientific Python ergonomics rather than imitate another language literally.

---

## User stories

### 1. Construct quantities naturally

**As a scientific Python user, I want to construct physical quantities using readable unit expressions, so that code resembles the scientific notation I already use.**

The prototype should support APIs such as:

```python
length = 5.0 * u.angstrom
energy = 2.0 * u.eV
force = 0.1 * u.eV_per_angstrom
```

For array backends, the prototype should also support an unambiguous constructor form:

```python
positions = u.angstrom(array)
```

and preferably:

```python
positions = u.angstrom * array
```

Support for `array * u.angstrom` is desirable where backend dispatch permits it, but should not be required if it compromises reliability.

#### Acceptance criteria

```python
assert_type(length, Length[float])
assert_type(energy, Energy[float])
assert_type(force, Force[float])
```

Construction should not require spelling out both the quantity type and the unit.

---

### 2. Treat quantities as the primary static types

**As a library author, I want function signatures to describe physical quantities rather than concrete units, so that callers can use any compatible unit.**

For example:

```python
def potential_energy(
    distance: Length[float],
) -> Energy[float]:
    ...
```

The following should be accepted:

```python
potential_energy(2.0 * u.angstrom)
potential_energy(0.2 * u.nm)
potential_energy(2e-10 * u.m)
```

The following should be rejected statically:

```python
potential_energy(3.0 * u.eV)
```

#### Acceptance criteria

- Compatible units produce the same quantity type.
- Unit choice does not appear in ordinary function annotations.
- Incompatible physical quantities are rejected by every supported type checker.

---

### 3. Distinguish physical meaning from dimensionality

**As a domain modeller, I want quantities with identical dimensionality to remain distinct when their physical meaning differs, so that dimensionally valid but semantically incorrect substitutions can be rejected.**

Examples include:

```text
Pressure
EnergyDensity
EnergyPerVolume
```

which may share dimensionality while representing different concepts.

Similarly:

```text
Frequency
InverseTime
```

may be dimensionally equivalent while remaining distinct nominal quantity kinds.

#### Acceptance criteria

```python
def consume_pressure(x: Pressure[float]) -> None:
    ...

pressure: Pressure[float]
energy_density: EnergyDensity[float]

consume_pressure(pressure)         # accepted
consume_pressure(energy_density)   # static error
```

The library should retain dimensional metadata internally, but dimensional equality alone must not imply type equality.

---

### 4. Support domain-specific base dimensions

**As an atomistic-simulation developer, I want concepts such as `atom` and `electron` to participate in dimensional analysis, so that per-atom and per-electron quantities are represented correctly.**

The prototype should be able to distinguish:

```text
Energy
EnergyPerAtom
ParticleDensity
ElectronDensity
```

Examples:

```python
cohesive_energy = -4.2 * u.eV_per_atom
number_density = 0.06 * u.atom_per_angstrom_cubed
```

#### Acceptance criteria

```python
assert_type(cohesive_energy, EnergyPerAtom[float])
```

A function expecting `EnergyPerAtom` should reject `Energy`, even if the numerical values happen to be comparable.

---

### 5. Infer named quantity types through arithmetic

**As a scientist, I want arithmetic between quantities to produce physically meaningful static result types, so that derived quantities remain type-safe.**

Representative relationships should include:

```text
Length × Length -> Area
Area × Length -> Volume
Length / Time -> Velocity
Energy / Length -> Force
Force × Length -> Energy
Force / Area -> Pressure
Energy / Volume -> EnergyDensity
MagneticMoment / Volume -> Magnetization
```

#### Acceptance criteria

```python
length = 2.0 * u.angstrom
time = 4.0 * u.fs
energy = 3.0 * u.eV

assert_type(length**2, Area[float])
assert_type(length**3, Volume[float])
assert_type(length / time, Velocity[float])
assert_type(energy / length, Force[float])
```

Known relationships should resolve to named quantity types rather than opaque symbolic expressions.

---

### 6. Preserve type information for uncommon derived quantities

**As an advanced user, I want physically valid operations outside the predefined catalogue to retain dimensional type information, so that the library does not fall back to `Any`.**

Where no named quantity exists, the prototype may use a structural dimension-expression type such as:

```python
Quantity[Div[FooKind, BarKind], V]
```

or an equivalent representation.

#### Acceptance criteria

- Valid but unnamed quantity arithmetic must not degrade to `Any`.
- Named relationships may canonicalize structural results into named quantities.
- The structural representation should not dominate the normal user-facing API.

---

### 7. Use one generic quantity abstraction for scalars and arrays

**As a numerical Python user, I want the same quantity type to work with scalars and array-like values, so that I do not need separate scalar and array class hierarchies.**

The desired model is:

```python
Energy[float]
Energy[np.ndarray]
Energy[jax.Array]
Energy[torch.Tensor]
```

rather than separate concepts such as scalar-energy and energy-array classes.

#### Acceptance criteria

```python
scalar_energy = 1.0 * u.eV
array_energy = u.eV(np.array([1.0, 2.0]))

assert_type(scalar_energy, Energy[float])
assert_type(array_energy, Energy[np.ndarray])
```

The quantity kind and numerical storage type should be orthogonal.

---

### 8. Keep tensor shape semantics separate from quantity semantics

**As an atomistic-simulation developer, I want vector and tensor shape semantics to be independent of units, so that positions, forces, and stresses compose cleanly.**

The prototype should investigate a representation such as:

```python
Length[Vector3]
Force[Vector3]
Pressure[SymmetricTensor3]
```

or an equivalent typed representation.

#### Acceptance criteria

- A vector should not require a distinct physical-quantity class solely because it has three components.
- Shape-aware wrappers should be reusable across different quantity kinds.
- The prototype should demonstrate at least positions, force vectors, and a 3x3 stress-like tensor.

---

### 9. Support physically correct temperature semantics

**As a scientific user, I want affine units and temperature differences to be modelled correctly, so that subtraction of absolute temperatures does not silently produce another absolute temperature.**

Example:

```python
t1 = 300 * u.K
t2 = 280 * u.K

delta = t1 - t2
```

The preferred result is:

```python
TemperatureDifference[float]
```

rather than `Temperature[float]`.

#### Acceptance criteria

```python
assert_type(delta, TemperatureDifference[float])
assert_type(t1 + delta, Temperature[float])
assert_type(delta / (2 * u.s), TemperatureRate[float])
```

At minimum, Kelvin and Celsius should be included in the prototype.

---

### 10. Convert units explicitly without changing physical type

**As a user, I want to convert the presentation unit of a quantity without changing its physical quantity type.**

For example:

```python
energy = 1.0 * u.hartree
converted = energy.to(u.eV)
```

#### Acceptance criteria

```python
assert_type(converted, Energy[float])
```

The library should provide a straightforward way to obtain the numerical magnitude in a chosen unit:

```python
value = energy.magnitude(u.eV)
```

Unit conversion should be rejected for incompatible quantity families.

---

### 11. Use canonical internal units

**As a numerical-computing user, I want quantities to have a predictable canonical representation internally, so that compiled and differentiable code does not carry dynamic unit-conversion machinery.**

The prototype should support an atomistic canonical system such as:

```text
length            angstrom
energy            eV
time              fs
force             eV / angstrom
pressure          eV / angstrom^3
magnetic moment   Bohr magneton
```

#### Acceptance criteria

- Conversion to canonical units occurs at quantity construction or another well-defined boundary.
- Numerical kernels should operate primarily on canonical magnitudes.
- Changing an input display unit should not alter the static quantity type.

---

### 12. Integrate naturally with Pydantic

**As a schema author, I want quantity types to work directly as Pydantic fields, so that physical validation is expressed in annotations rather than custom validators.**

Example:

```python
class MDConfig(BaseModel):
    timestep: Time
    cutoff: Length
    temperature: Temperature
```

The model should be able to accept appropriate serialized representations and produce strongly typed quantity objects.

#### Acceptance criteria

- Quantity classes work as normal Pydantic field annotations.
- Validation errors identify both the expected and received physical quantity where possible.
- The prototype should support at least one structured representation such as:

```json
{
  "value": 5.0,
  "units": "angstrom"
}
```

- Support for parsing strings such as `"5 angstrom"` is desirable.

---

### 13. Preserve a simple serialization boundary

**As an API or storage-system developer, I want quantities to serialize to a simple value-plus-unit representation, so that wire formats remain independent of Python implementation details.**

The prototype should support output resembling:

```json
{
  "value": 500.0,
  "units": "electron_volt"
}
```

#### Acceptance criteria

- Scalar quantities round-trip through serialization.
- Array quantities round-trip through serialization.
- Serialization does not expose internal phantom types or dimension-expression classes.
- The serialized unit may be configurable independently of the internal canonical unit.

---

### 14. Support NumPy without erasing physical types

**As a NumPy user, I want common operations to preserve or transform physical quantity types correctly, so that I can use the library in ordinary scientific code.**

The prototype should support a deliberately limited, well-typed set of operations rather than attempting to intercept the entire NumPy API.

Examples may include:

```python
quantity.sum()
quantity.mean()
u.sqrt(area)
u.sin(angle)
u.exp(dimensionless)
```

#### Acceptance criteria

```python
assert_type(u.sqrt(area), Length[np.ndarray])
assert_type(u.sin(angle), Dimensionless[np.ndarray])
```

The library should avoid implicit conversion mechanisms that silently discard unit information.

---

### 15. Work with JAX arrays

**As a JAX user, I want quantities to contain `jax.Array` values and participate in JAX transformations, so that physical typing can be used in differentiable simulation and ML code.**

Example:

```python
positions = u.angstrom(jax_array)

assert_type(
    positions,
    Length[jax.Array],
)
```

#### Acceptance criteria

- Quantities can contain `jax.Array`.
- Quantity objects can be registered as JAX pytrees where useful.
- `jit` works for representative quantity-aware functions.
- `vmap` works for representative quantity-aware functions.
- Traced numerical computation should operate on raw arrays and simple numerical scale factors rather than runtime unit registries.

---

### 16. Provide physically typed JAX gradients

**As an MLIP developer, I want differentiating an energy with respect to positions to produce a statically typed force quantity, so that automatic differentiation respects physical dimensions.**

Example:

```python
def energy(
    positions: Length[jax.Array],
) -> Energy[jax.Array]:
    ...

gradient = ujax.grad(energy)
```

The type should be:

```python
Callable[[Length[jax.Array]], Force[jax.Array]]
```

rather than a quantity with the same kind as the input.

#### Acceptance criteria

```python
assert_type(
    ujax.grad(energy),
    Callable[[Length[jax.Array]], Force[jax.Array]],
)
```

At runtime:

```python
dE_dx = ujax.grad(energy)(positions)
assert_type(dE_dx, Force[jax.Array])
```

The wrapper may internally unwrap canonical magnitudes, call JAX autodiff, and re-wrap the result.

---

### 17. Support higher-order JAX differentiation

**As an atomistic-simulation researcher, I want second derivatives to acquire the correct physical quantity type, so that Hessians and force constants can be represented safely.**

For:

```python
Energy / Length -> Force
```

a second derivative with respect to length has units:

```text
Energy / Length^2
```

which may be represented by a named quantity such as `ForceConstant`.

#### Acceptance criteria

```python
hessian = ujax.hessian(energy)(positions)

assert_type(
    hessian,
    ForceConstant[jax.Array],
)
```

The prototype should demonstrate at least one higher-order derivative.

---

### 18. Work with PyTorch tensors

**As a PyTorch user, I want quantities to contain `torch.Tensor` values without breaking autograd, so that physical typing can be used in neural-network models and atomistic potentials.**

Example:

```python
positions = u.angstrom(
    torch.tensor(
        [[0.0, 0.0, 0.0]],
        requires_grad=True,
    )
)
```

#### Acceptance criteria

```python
assert_type(
    positions,
    Length[torch.Tensor],
)
```

Operations on quantities must not detach the underlying tensor or otherwise break its autograd graph.

---

### 19. Provide physically typed PyTorch gradients

**As an MLIP developer, I want differentiating an energy with respect to positions in PyTorch to produce a typed force quantity.**

Example:

```python
energy = model(positions)
gradient = utorch.grad(energy, positions)
```

#### Acceptance criteria

```python
assert_type(energy, Energy[torch.Tensor])
assert_type(gradient, Force[torch.Tensor])
```

The implementation may delegate to `torch.autograd.grad` on underlying tensors and wrap the result according to the physical derivative relationship.

---

### 20. Avoid requiring tensor subclasses

**As a library maintainer, I want the initial design to avoid deep backend-specific subclassing mechanisms, so that compatibility with JAX, PyTorch, compilation, and future backend changes remains manageable.**

For the prototype:

```python
Quantity[EnergyKind, torch.Tensor]
```

should be an ordinary wrapper around a tensor rather than a `torch.Tensor` subclass.

Likewise, JAX integration should not require pretending that a quantity object is itself a JAX array.

#### Acceptance criteria

- PyTorch support does not depend on `torch.Tensor` subclassing.
- JAX support does not depend on implicit array coercion that drops physical metadata.
- Backend-specific integrations should live in narrow adapter modules.

---

### 21. Allow raw tensors inside ML models

**As an ML model author, I want physical quantities at meaningful model boundaries without forcing every hidden activation to carry a physical unit.**

For example:

```python
def model(
    positions: Length[jax.Array],
) -> Energy[jax.Array]:

    x = positions.magnitude

    features = encoder(x)
    raw_energy = decoder(features)

    return Energy.from_canonical(raw_energy)
```

#### Acceptance criteria

- The library makes it easy to unwrap canonical magnitudes intentionally.
- Re-wrapping a result in a known quantity is explicit and cheap.
- Latent neural-network features do not need units unless the model author wants them.

---

### 22. Share one physical algebra across normal arithmetic and autodiff

**As a library maintainer, I want arithmetic result types and derivative result types to derive from one physical relationship registry, so that the system has a single source of truth.**

For example, the relationship:

```text
Energy / Length -> Force
```

should drive both:

```python
energy / length
```

and:

```python
grad(
    Callable[[Length], Energy]
) -> Callable[[Length], Force]
```

#### Acceptance criteria

- Result-type relations are defined declaratively.
- The same relation definitions can generate ordinary arithmetic overloads and autodiff overloads.
- Adding a named quantity relationship should not require hand-editing every backend integration.

---

### 23. Generate repetitive static typing code

**As a maintainer, I want repetitive overloads to be generated from a declarative quantity registry, so that the public typing surface remains comprehensive without becoming manually unmaintainable.**

The registry should be capable of representing:

- quantity name;
- physical dimension;
- semantic quantity kind;
- canonical unit;
- accepted units and aliases;
- known arithmetic relations.

#### Acceptance criteria

The prototype should demonstrate generation of at least:

```text
quantity definitions
unit definitions
ordinary arithmetic overloads
JAX autodiff overloads
PyTorch autodiff overloads
```

Generated code should remain ordinary standards-compliant Python typing code.

---

### 24. Require no type-checker plugins

**As a library consumer, I want the library to work with my chosen modern Python type checker without installing a checker-specific plugin.**

The implementation should use standard typing constructs wherever possible, such as:

```text
Generic
TypeVar / PEP 695 type parameters
overload
Protocol
Self
Final
Literal
Annotated
type aliases
```

#### Acceptance criteria

No supported workflow may require:

- a mypy plugin;
- a Pyright extension;
- a Pyrefly-specific transform;
- a ty-specific extension.

Checker-specific optimizations may be explored only if the portable behaviour remains correct without them.

---

### 25. Pass all supported type checkers in their strict configurations

**As a library author, I want the same public API to type-check consistently under mypy, Pyright, Pyrefly, and ty, so that the library is not coupled to one interpretation of Python typing.**

The prototype should maintain a checker conformance suite.

#### Acceptance criteria

Representative examples must be checked with:

```text
mypy --strict
Pyright strict configuration
Pyrefly strict configuration
ty with its strictest relevant checks enabled
```

The suite should include both positive and negative examples.

A successful prototype should not merely produce no checker errors; all checkers should infer comparably useful public types for representative expressions.

---

### 26. Make negative typing tests part of the specification

**As a maintainer, I want examples of invalid physical programs to be checked continuously, so that regressions in type safety are caught.**

Examples should include:

```python
def needs_energy(x: Energy[float]) -> None:
    ...

needs_energy(1.0 * u.angstrom)
```

and invalid additions such as:

```python
energy + length
```

#### Acceptance criteria

The test suite should verify that every supported checker rejects representative invalid programs.

---

### 27. Preserve runtime physical validation

**As a user receiving untyped external data, I want runtime validation to reject incompatible units even when static typing cannot help, so that serialization and API boundaries remain safe.**

Examples:

```python
Length.parse({"value": 2.0, "units": "second"})
```

should fail with a useful error.

#### Acceptance criteria

Errors should communicate:

- expected physical quantity;
- supplied unit;
- supplied dimensionality or inferred quantity where useful.

Runtime validation should complement static typing rather than substitute for it.

---

### 28. Provide good representations and diagnostics

**As an interactive Python user, I want quantities to have concise, readable representations, so that debugging and notebook use do not require understanding implementation internals.**

Examples should be close to:

```text
5.0 Å
2.3 eV
[1.0, 2.0, 3.0] eV/Å
```

or another similarly compact format.

#### Acceptance criteria

- `repr()` should expose the physical value and unit clearly.
- It should not expose phantom marker classes or generated type machinery by default.
- Error messages should use scientific quantity names rather than internal generic type parameters wherever possible.

---

## Prototype quantity catalogue

The first prototype should include enough quantities to exercise the design without attempting comprehensive unit coverage.

Recommended initial quantities:

```text
Dimensionless
Length
Area
Volume
Time
Velocity
Energy
EnergyPerAtom
Force
ForceConstant
Pressure
EnergyDensity
Temperature
TemperatureDifference
TemperatureRate
MagneticMoment
Magnetization
ParticleDensity
ElectronDensity
Angle
```

Recommended initial units include:

```text
Length:
    meter
    centimeter
    nanometer
    angstrom
    bohr

Time:
    second
    picosecond
    femtosecond

Energy:
    joule
    electron_volt
    millielectron_volt
    hartree
    rydberg

EnergyPerAtom:
    eV / atom
    hartree / atom
    J / atom

Force:
    newton
    eV / angstrom
    hartree / bohr

Pressure:
    pascal
    gigapascal
    eV / angstrom^3
    hartree / bohr^3

Temperature:
    kelvin
    celsius

MagneticMoment:
    ampere meter^2
    Bohr magneton

Angle:
    radian
    degree
```

---

## Representative end-to-end prototype scenarios

### Typed scalar arithmetic

```python
r = 2.0 * u.angstrom
t = 4.0 * u.fs
e = -3.2 * u.eV

v = r / t
f = e / r
a = r**2
volume = r**3
p = f / a

assert_type(r, Length[float])
assert_type(v, Velocity[float])
assert_type(f, Force[float])
assert_type(a, Area[float])
assert_type(volume, Volume[float])
assert_type(p, Pressure[float])
```

### Typed NumPy values

```python
positions = u.angstrom(np.zeros((100, 3)))
forces = u.eV_per_angstrom(np.zeros((100, 3)))

assert_type(positions, Length[np.ndarray])
assert_type(forces, Force[np.ndarray])
```

### Pydantic integration

```python
class RelaxationConfig(BaseModel):
    force_tolerance: Force
    max_displacement: Length


config = RelaxationConfig.model_validate(
    {
        "force_tolerance": {
            "value": 0.01,
            "units": "eV / angstrom",
        },
        "max_displacement": {
            "value": 0.1,
            "units": "angstrom",
        },
    }
)
```

### JAX energy and forces

```python
def harmonic(
    x: Length[jax.Array],
) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


x = u.angstrom(jnp.array([1.0, 2.0, 3.0]))

energy = harmonic(x)
gradient = ujax.grad(harmonic)(x)
hessian = ujax.hessian(harmonic)(x)

assert_type(energy, Energy[jax.Array])
assert_type(gradient, Force[jax.Array])
assert_type(hessian, ForceConstant[jax.Array])
```

### PyTorch energy and forces

```python
x = u.angstrom(
    torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )
)

energy = harmonic(x)
gradient = utorch.grad(energy, x)

assert_type(energy, Energy[torch.Tensor])
assert_type(gradient, Force[torch.Tensor])
```

---

## Explicit non-goals for the first prototype

The first prototype does not need to:

- provide an exhaustive SI unit catalogue;
- replace a mature runtime unit-conversion library in every respect;
- support arbitrary symbolic simplification of all possible unit expressions;
- intercept every NumPy function;
- subclass `torch.Tensor`;
- make every JAX or PyTorch hidden activation unit-aware;
- fully type JVP and VJP tangent/cotangent semantics;
- solve compile-time tensor shape typing beyond a small demonstrator;
- guarantee compatibility with every third-party array library;
- provide type-checker-specific plugins.

---

## Prototype success criteria

The prototype should be considered successful if it demonstrates all of the following:

1. Natural quantity construction with good scientific UX.
2. Named physical quantities visible in normal type annotations.
3. Static rejection of representative physically invalid programs.
4. Correct type propagation through representative arithmetic.
5. A clean distinction between semantic quantity kind and dimensionality.
6. Generic scalar and array storage without parallel quantity hierarchies.
7. Pydantic parsing and serialization.
8. NumPy support for representative operations.
9. JAX `jit`, `vmap`, gradient, and Hessian examples.
10. PyTorch autograd examples.
11. Physically correct static derivative types.
12. Equivalent behaviour under mypy, Pyright, Pyrefly, and ty without plugins.
13. A declarative registry capable of generating repetitive typing code.
14. A public API that remains readable without exposing the internal type machinery.
