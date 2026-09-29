"""PyTorch autograd on canonical tensors without graph-detaching conversions."""

from typing import Any

import torch

from quantype.core import (
    Quantity,
    _wrap,  # pyright: ignore[reportPrivateUsage]
    result_kind,
)


def grad(
    output: Quantity[Any, torch.Tensor],
    inputs: Quantity[Any, torch.Tensor],
    *,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> Quantity[Any, torch.Tensor]:
    """Differentiate one scalar output with respect to one quantity.

    The result is the positive derivative, not the negative physical force.
    ``create_graph=True`` keeps the derivative differentiable for higher orders;
    ``retain_graph`` follows PyTorch's normal semantics (including its default).
    """
    if output.value.ndim != 0:
        raise ValueError("utorch.grad requires a scalar (zero-dimensional) output.")
    (value,) = torch.autograd.grad(
        output.value,
        inputs.value,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return _wrap(result_kind("div", output.kind, inputs.kind), value)
