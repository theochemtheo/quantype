"""JAX transformations on canonical magnitudes with physical derivative kinds.

Importing this module registers quantity classes as single-leaf pytrees.
Only unary, scalar-output functions are supported by the autodiff adapters.
Gradients are positive derivatives; physical forces are their explicit negation.
"""

# Internal pytree registration deliberately shares core's wrapping/metadata.
# pyright: reportPrivateUsage=false

from collections.abc import Callable
from typing import Any, Protocol, TypeGuard, cast

import jax

from quantype import _generated
from quantype.core import Quantity, _wrap, result_kind

# These aliases describe the intentionally dynamic backend integration boundary.
type _Quantity = Quantity[Any, Any]
type _Metadata = tuple[str, str | None]
type _RawFunction = Callable[[Any], tuple[Any, _Quantity]]


class _TreeUtil(Protocol):
    def register_pytree_node(
        self,
        nodetype: type[_Quantity],
        flatten_func: Callable[[_Quantity], tuple[tuple[Any], _Metadata]],
        unflatten_func: Callable[[_Metadata, tuple[Any]], _Quantity],
    ) -> None: ...


class _Jax(Protocol):
    tree_util: _TreeUtil

    def grad(self, fun: _RawFunction, *, has_aux: bool) -> _RawFunction: ...

    def jacfwd(self, fun: _RawFunction, *, has_aux: bool) -> _RawFunction: ...


# JAX's upstream annotations leave transformation results and tree definitions
# unknown. Describe only the has_aux transformations used at this boundary.
_jax = cast("_Jax", jax)


def _is_quantity_class(candidate: object) -> TypeGuard[type[_Quantity]]:
    return isinstance(candidate, type) and issubclass(candidate, Quantity)


def _flatten(quantity: _Quantity) -> tuple[tuple[Any], _Metadata]:
    return (quantity.value,), (quantity.kind, quantity._display)  # noqa: SLF001


def _unflatten(metadata: _Metadata, leaves: tuple[Any]) -> _Quantity:
    kind, display = metadata
    quantity = _wrap(kind, leaves[0])
    # JAX may supply sentinel objects during vmap tree manipulation. Neither
    # reconstruction nor restoring display metadata may coerce the leaf.
    object.__setattr__(quantity, "_display", display)
    return quantity


def _register_pytrees() -> None:
    classes: set[type[_Quantity]] = {Quantity}
    for candidate in vars(_generated).values():
        if _is_quantity_class(candidate):
            classes.add(candidate)
    for cls in classes:
        _jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)


_register_pytrees()


def _raw_function(
    function: Callable[[_Quantity], _Quantity], argument: _Quantity
) -> Callable[[Any], tuple[Any, _Quantity]]:
    metadata = (argument.kind, argument._display)  # noqa: SLF001

    def evaluate(value: Any) -> tuple[Any, _Quantity]:  # noqa: ANN401
        output = function(_unflatten(metadata, (value,)))
        # Validate untyped callers as well as the statically checked contract.
        if not isinstance(output, Quantity):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Differentiated functions must return a Quantity.")
        # Carry output metadata through has_aux, rather than evaluating the
        # user's function a second time just to discover its physical kind.
        return output.value, output

    return evaluate


def grad(
    function: Callable[[_Quantity], _Quantity],
) -> Callable[[_Quantity], _Quantity]:
    """Differentiate a unary scalar quantity function in canonical units."""

    def derivative(argument: _Quantity) -> _Quantity:
        value, output = _jax.grad(_raw_function(function, argument), has_aux=True)(
            argument.value
        )
        return _wrap(result_kind("div", output.kind, argument.kind), value)

    return derivative


def hessian(
    function: Callable[[_Quantity], _Quantity],
) -> Callable[[_Quantity], _Quantity]:
    """Return the full Hessian of a unary scalar quantity function."""

    def derivative(argument: _Quantity) -> _Quantity:
        first = _jax.grad(_raw_function(function, argument), has_aux=True)
        value, output = _jax.jacfwd(first, has_aux=True)(argument.value)
        first_kind = result_kind("div", output.kind, argument.kind)
        return _wrap(result_kind("div", first_kind, argument.kind), value)

    return derivative
