# Testing quantities

Public import: `from quantype.testing import assert_allclose`.
This NumPy-based assertion compares physical quantities, not plain numerical
inputs. For task-oriented examples, see [test quantities](../how-to/test-quantities.md).

## assert_allclose

```text
assert_allclose[Q: Quantity[Any, Any, Any]](
    actual: Q,
    desired: Q,
    *,
    rtol: float = 1e-7,
    atol: Quantity[Any, Any, Any] | Literal[0] = 0,
    equal_nan: bool = True,
    err_msg: str = "",
) -> None
```

`actual` and `desired` must be quantities of the same nominal kind (or the same
structural expression). Display units may differ. A `NoSystem` operand adopts
the other's bound system; two different bound systems do not mix. Convert
explicitly with [`.to_system`](quantities.md#to_system) when necessary.

After adoption and unit alignment, the assertion delegates to
`numpy.testing.assert_allclose`. For finite values, its tolerance is
`abs(actual - desired) <= atol + rtol * abs(desired)`. NumPy's shape/broadcasting
rules apply, including scalar comparisons against arrays. `equal_nan=True`
allows NaNs at matching positions. The function returns `None` on success.

### Tolerances and compared units

- `rtol` is dimensionless and relative to the **desired** numbers.
- `atol` is integer zero or a scalar quantity of the compared kind. A bare
  nonzero number (or `0.0`) is not accepted. A tolerance must have the adopted
  comparison system or `NoSystem`; an unbound tolerance adopts that system.
- For absolute temperatures, use a `TemperatureDifference` tolerance, such as
  `0.1 * u.delta_K`. The implementation also accepts a same-kind absolute
  temperature tolerance; differences express the intended physical meaning.
- Bound quantities are compared in the adopted system's storage units.
  Without a system, they are compared in the **desired** quantity's storage
  unit, not necessarily its display unit. For absolute temperatures supplied
  in °C, storage is kelvin-scaled: `rtol` is relative to absolute zero, not to
  the Celsius display value. Difference tolerances are converted to the same
  scale without a temperature-point offset.

```python
from quantype import Length, u
from quantype.systems import SI
from quantype.testing import assert_allclose

# The unbound actual quantity adopts SI; different display units are fine.
assert_allclose(2 * u.nm, Length[float, SI](20, u.angstrom))
# Absolute tolerance is a physical difference; disable relative tolerance here.
assert_allclose(20 * u.celsius, 20.05 * u.celsius,
                rtol=0, atol=0.1 * u.delta_K)
```

### Failures and diagnostics

A kind mismatch, two different bound systems, or a numerical mismatch raises
`AssertionError`. Numerical failure messages include `err_msg`, the compared
kind/system, and the compared unit when a named or storage unit is available.
An unnamed structural result in a bound system has no named comparison unit.
Non-quantity operands and invalid tolerance kinds/systems raise `TypeError`;
backend conversion and NumPy numerical errors are not suppressed. Use
`numpy.testing` directly for plain numbers.

### Backend and ownership boundary

Comparison materializes host NumPy values with `numpy.asarray`. Torch storage
is first detached and moved to CPU (`detach().cpu()`); JAX arrays cross the
host-conversion boundary. This helper is for tests outside differentiation,
`jit`, and other compiled/traced computation, not a differentiable backend
operation. Host conversion can transfer/copy data; it does not promise an
independent copy of existing NumPy storage and does not modify inputs.

For backend boolean results instead of an assertion, see
[`quantype.numpy.isclose` and `allclose`](numerical-operations.md#plain-numerical-and-boolean-results),
whose defaults and tolerance contracts differ.
