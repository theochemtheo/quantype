"""JAX transformations on raw magnitudes with physical derivative kinds.

Importing this module registers quantity classes as single-leaf pytrees whose
static metadata is the kind and the unit system. Display units are not part of
the tree structure, so values presented differently still share one trace.
Only unary, scalar-output functions are supported by the autodiff adapters.
Gradients are positive derivatives; physical forces are their explicit negation.
Inside one unit system the raw derivative is already in that system's units.
"""

# Internal pytree registration deliberately shares core's wrapping/metadata.
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

import math
from collections.abc import Callable
from typing import Any, Protocol, TypeGuard, cast

try:
    import jax
except ModuleNotFoundError as exc:
    if exc.name != "jax":
        raise
    raise ModuleNotFoundError("Install quantype[jax] to use quantype.ujax") from exc

from quantype import _generated
from quantype._internal._semantics import Semantic
from quantype._internal._systems import UnitSystem, coherence
from quantype.core import Quantity, _rescaled, _wrap, result_kind

# These aliases describe the intentionally dynamic backend integration boundary.
type _Quantity = Quantity[Any, Any, Any]
type _Metadata = tuple[Semantic, type[UnitSystem]]
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

    def jit[**P, T](self, fun: Callable[P, T]) -> Callable[P, T]: ...

    def vmap[**P, T](
        self, fun: Callable[P, T], in_axes: int | None = 0, out_axes: int = 0
    ) -> Callable[P, T]: ...

    def grad(self, fun: _RawFunction, *, has_aux: bool) -> _RawFunction: ...

    def jacfwd(self, fun: _RawFunction, *, has_aux: bool) -> _RawFunction: ...


# JAX's upstream annotations leave transformation results and tree definitions
# unknown. Describe only the has_aux transformations used at this boundary.
_jax = cast("_Jax", jax)


def _is_quantity_class(candidate: object) -> TypeGuard[type[_Quantity]]:
    return isinstance(candidate, type) and issubclass(candidate, Quantity)


def _flatten(quantity: _Quantity) -> tuple[tuple[Any], _Metadata]:
    return (quantity.value,), (quantity._semantic, quantity._system)


def _unflatten(metadata: _Metadata, leaves: tuple[Any]) -> _Quantity:
    kind, system = metadata
    # JAX may supply sentinel objects during vmap tree manipulation, so
    # reconstruction must not coerce the leaf.
    return _wrap(kind, leaves[0], system)


def _register_quantity(cls: type[_Quantity]) -> None:
    _jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)


def _register_pytrees() -> None:
    classes: set[type[_Quantity]] = {Quantity}
    for candidate in vars(_generated).values():
        if _is_quantity_class(candidate):
            classes.add(candidate)
    for cls in classes:
        _register_quantity(cls)


_register_pytrees()


def jit[**P, T](fun: Callable[P, T]) -> Callable[P, T]:
    """JIT-compile a function operating on registered quantities."""
    return _jax.jit(fun)


def vmap[**P, T](
    fun: Callable[P, T], in_axes: int | None = 0, out_axes: int = 0
) -> Callable[P, T]:
    """Vectorize a function operating on registered quantities."""
    return _jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)


def _raw_function(
    function: Callable[[_Quantity], _Quantity], argument: _Quantity
) -> Callable[[Any], tuple[Any, _Quantity]]:
    metadata = (argument._semantic, argument._system)

    def evaluate(value: Any) -> tuple[Any, _Quantity]:  # noqa: ANN401
        output = function(_unflatten(metadata, (value,)))
        # Validate untyped callers as well as the statically checked contract.
        if not isinstance(output, Quantity):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Differentiated functions must return a Quantity.")
        if output._system is not argument._system:
            raise TypeError(
                "Differentiated functions must return a quantity in their "
                f"argument's unit system ({argument._system.__name__}); "
                f"received {output._system.__name__}"
            )
        # Carry output metadata through has_aux, rather than evaluating the
        # user's function a second time just to discover its physical kind.
        return output.value, output

    return evaluate


def _derivative_scale(
    argument: _Quantity, output: _Quantity, kind: Semantic, *, order: int
) -> float:
    """Raw derivatives are coherent, except where the system overrides a kind."""
    system = argument._system
    inputs = math.pow(coherence(system, argument._semantic), order)
    return coherence(system, output._semantic) / inputs / coherence(system, kind)


def grad(
    function: Callable[[_Quantity], _Quantity],
) -> Callable[[_Quantity], _Quantity]:
    """Differentiate a unary scalar quantity function in its unit system."""

    def derivative(argument: _Quantity) -> _Quantity:
        value, output = _jax.grad(_raw_function(function, argument), has_aux=True)(
            argument.value
        )
        kind = result_kind("div", output._semantic, argument._semantic)
        scale = _derivative_scale(argument, output, kind, order=1)
        return _wrap(kind, _rescaled(value, scale), argument._system)

    return derivative


def hessian(
    function: Callable[[_Quantity], _Quantity],
) -> Callable[[_Quantity], _Quantity]:
    """Return the full Hessian of a unary scalar quantity function."""

    def derivative(argument: _Quantity) -> _Quantity:
        first = _jax.grad(_raw_function(function, argument), has_aux=True)
        value, output = _jax.jacfwd(first, has_aux=True)(argument.value)
        first_kind = result_kind("div", output._semantic, argument._semantic)
        kind = result_kind("div", first_kind, argument._semantic)
        scale = _derivative_scale(argument, output, kind, order=2)
        return _wrap(kind, _rescaled(value, scale), argument._system)

    return derivative
