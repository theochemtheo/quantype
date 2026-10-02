"""PyTorch autograd on raw tensors without graph-detaching conversions."""

# Internal differentiation deliberately shares core's wrapping and metadata.
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from collections.abc import Sequence
from typing import Any

import torch

from quantype._internal._systems import coherence
from quantype.core import Quantity, _rescaled, _wrap, result_kind

type _Quantity = Quantity[Any, Any, Any]


def grad(
    output: _Quantity,
    inputs: _Quantity | Sequence[_Quantity],
    *,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> Any:
    """Differentiate one scalar output with respect to one or more quantities.

    One quantity gives one derivative; a sequence gives a tuple of them. Each is
    the positive derivative, not the negative physical force.
    ``create_graph=True`` keeps derivatives differentiable for higher orders;
    ``retain_graph`` follows PyTorch's normal semantics (including its default).
    Every quantity must share the output's unit system.
    """
    single = isinstance(inputs, Quantity)
    arguments: list[_Quantity] = []
    if isinstance(inputs, Quantity):
        arguments.append(inputs)
    else:
        arguments.extend(inputs)
    for argument in arguments:
        if output.system is not argument.system:
            raise TypeError(
                f"Cannot differentiate an output in {output.system.__name__} with "
                f"respect to an input in {argument.system.__name__}; convert one "
                "with .to_system(...)"
            )
    if output.value.ndim != 0:
        raise ValueError("utorch.grad requires a scalar (zero-dimensional) output.")
    values = torch.autograd.grad(
        output.value,
        [argument.value for argument in arguments],
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    derivatives = tuple(
        _derivative(output, argument, value)
        for argument, value in zip(arguments, values, strict=True)
    )
    return derivatives[0] if single else derivatives


def _derivative(output: _Quantity, argument: _Quantity, value: Any) -> _Quantity:
    kind = result_kind("div", output._semantic, argument._semantic)
    # Raw derivatives are coherent, except where the system overrides a kind.
    system = argument.system
    scale = (
        coherence(system, output._semantic)
        / coherence(system, argument._semantic)
        / coherence(system, kind)
    )
    return _wrap(kind, _rescaled(value, scale), system)
