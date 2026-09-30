"""JAX transformations on raw magnitudes with physical derivative kinds.

Importing this module registers quantity classes as single-leaf pytrees whose
static metadata is the kind and the unit system. Display units are not part of
the tree structure, so values presented differently still share one trace.

The autodiff adapters differentiate scalar-output functions with respect to the
quantities at ``argnums``; other arguments, such as model parameters or neighbour
lists, pass through untouched. Gradients are positive derivatives; physical
forces are their explicit negation. Inside one unit system the raw derivative
is already in that system's coherent units.
"""

# Internal pytree registration deliberately shares core's wrapping/metadata.
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeGuard, cast

import jax

from quantype import _generated
from quantype._internal._semantics import Semantic
from quantype._internal._systems import UnitSystem, coherence
from quantype.core import Quantity, _rescaled, _wrap, result_kind

# These aliases describe the intentionally dynamic backend integration boundary.
type _Quantity = Quantity[Any, Any, Any]
type _Metadata = tuple[Semantic, type[UnitSystem]]
type _RawFunction = Callable[..., tuple[Any, _Quantity]]
type _Argnums = int | tuple[int, ...]


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

    def grad(
        self, fun: _RawFunction, argnums: _Argnums = 0, *, has_aux: bool
    ) -> Callable[..., tuple[Any, _Quantity]]: ...

    def value_and_grad(
        self, fun: _RawFunction, argnums: _Argnums = 0, *, has_aux: bool
    ) -> Callable[..., tuple[tuple[Any, _Quantity], Any]]: ...

    def jacfwd(
        self, fun: Callable[..., tuple[Any, _Quantity]], *, has_aux: bool
    ) -> Callable[..., tuple[Any, _Quantity]]: ...


# JAX's upstream annotations leave transformation results and tree definitions
# unknown. Describe only the has_aux transformations used at this boundary.
_jax = cast("_Jax", jax)


def _is_quantity(value: object) -> TypeGuard[_Quantity]:
    return isinstance(value, Quantity)


def _is_quantity_class(candidate: object) -> TypeGuard[type[_Quantity]]:
    return isinstance(candidate, type) and issubclass(candidate, Quantity)


def _flatten(quantity: _Quantity) -> tuple[tuple[Any], _Metadata]:
    return (quantity.value,), (quantity._semantic, quantity._system)


def _unflatten(metadata: _Metadata, leaves: tuple[Any]) -> _Quantity:
    kind, system = metadata
    # JAX may supply sentinel objects during vmap tree manipulation, so
    # reconstruction must not coerce the leaf.
    return _wrap(kind, leaves[0], system)


def register_quantity(cls: type[_Quantity]) -> None:
    """Register a quantity class as a single-leaf pytree."""
    _jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)


def _register_pytrees() -> None:
    classes: set[type[_Quantity]] = {Quantity}
    for candidate in vars(_generated).values():
        if _is_quantity_class(candidate):
            classes.add(candidate)
    for cls in classes:
        register_quantity(cls)


_register_pytrees()


def jit[**P, T](fun: Callable[P, T]) -> Callable[P, T]:
    """JIT-compile a function operating on registered quantities."""
    return _jax.jit(fun)


def vmap[**P, T](
    fun: Callable[P, T], in_axes: int | None = 0, out_axes: int = 0
) -> Callable[P, T]:
    """Vectorize a function operating on registered quantities."""
    return _jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)


def _positions(argnums: _Argnums) -> tuple[int, ...]:
    return (argnums,) if isinstance(argnums, int) else tuple(argnums)


def _inputs(args: Sequence[object], positions: tuple[int, ...]) -> list[_Quantity]:
    inputs: list[_Quantity] = []
    for position in positions:
        argument = args[position]
        if not _is_quantity(argument):
            raise TypeError(
                f"Differentiated argument {position} must be a quantity; "
                f"received {type(argument).__name__}"
            )
        inputs.append(argument)
    return inputs


def _raw_function(
    function: Callable[..., _Quantity],
    args: Sequence[object],
    kwargs: dict[str, object],
    positions: tuple[int, ...],
    inputs: Sequence[_Quantity],
) -> _RawFunction:
    """The function on raw values at ``positions``, returning the output as aux."""
    metadata = [(argument._semantic, argument._system) for argument in inputs]

    def evaluate(*values: Any) -> tuple[Any, _Quantity]:
        call = list(args)
        for position, meta, value in zip(positions, metadata, values, strict=True):
            call[position] = _unflatten(meta, (value,))
        output = function(*call, **kwargs)
        # Validate untyped callers as well as the statically checked contract.
        if not isinstance(output, Quantity):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Differentiated functions must return a Quantity.")
        for argument in inputs:
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


def _derivative(
    argument: _Quantity, output: _Quantity, value: Any, *, order: int = 1
) -> _Quantity:
    """A raw derivative as a quantity; coherent except where a kind is overridden."""
    kind: Semantic = output._semantic
    for _ in range(order):
        kind = result_kind("div", kind, argument._semantic)
    system = argument._system
    inputs = math.pow(coherence(system, argument._semantic), order)
    scale = coherence(system, output._semantic) / inputs / coherence(system, kind)
    return _wrap(kind, _rescaled(value, scale), system)


def _results(
    argnums: _Argnums, inputs: Sequence[_Quantity], output: _Quantity, values: Any
) -> Any:
    derivatives = tuple(
        _derivative(argument, output, value)
        for argument, value in zip(inputs, values, strict=True)
    )
    return derivatives[0] if isinstance(argnums, int) else derivatives


def grad(
    function: Callable[..., _Quantity], argnums: _Argnums = 0
) -> Callable[..., Any]:
    """Differentiate a scalar quantity function at the quantities ``argnums`` names.

    An integer gives one derivative; a tuple gives a tuple of them.
    """

    def derivative(*args: Any, **kwargs: Any) -> Any:
        positions = _positions(argnums)
        inputs = _inputs(args, positions)
        raw = _raw_function(function, args, kwargs, positions, inputs)
        values, output = _jax.grad(raw, tuple(range(len(positions))), has_aux=True)(
            *(argument.value for argument in inputs)
        )
        return _results(argnums, inputs, output, values)

    return derivative


def value_and_grad(
    function: Callable[..., _Quantity], argnums: _Argnums = 0
) -> Callable[..., tuple[_Quantity, Any]]:
    """The function's value and its derivatives, from one evaluation.

    This is the usual energy-and-forces call: the energy and its gradient with
    respect to positions, without evaluating the model twice.
    """

    def evaluate(*args: Any, **kwargs: Any) -> tuple[_Quantity, Any]:
        positions = _positions(argnums)
        inputs = _inputs(args, positions)
        raw = _raw_function(function, args, kwargs, positions, inputs)
        transformed = _jax.value_and_grad(
            raw, tuple(range(len(positions))), has_aux=True
        )
        (_, output), values = transformed(*(argument.value for argument in inputs))
        return output, _results(argnums, inputs, output, values)

    return evaluate


def hessian(
    function: Callable[..., _Quantity], argnums: int = 0
) -> Callable[..., _Quantity]:
    """The full Hessian of a scalar quantity function at one quantity argument."""

    def derivative(*args: Any, **kwargs: Any) -> _Quantity:
        inputs = _inputs(args, (argnums,))
        raw = _raw_function(function, args, kwargs, (argnums,), inputs)
        first = _jax.grad(raw, 0, has_aux=True)
        value, output = _jax.jacfwd(first, has_aux=True)(inputs[0].value)
        return _derivative(inputs[0], output, value, order=2)

    return derivative
