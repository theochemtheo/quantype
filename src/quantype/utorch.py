"""PyTorch autograd on raw tensors without graph-detaching conversions."""

from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise ModuleNotFoundError("Install quantype[torch] to use quantype.utorch") from exc

from quantype._internal._systems import coherence
from quantype.core import (
    Quantity,
    _rescaled,  # pyright: ignore[reportPrivateUsage]
    _wrap,  # pyright: ignore[reportPrivateUsage]
    result_kind,
)


def grad(
    output: Quantity[Any, torch.Tensor, Any],
    inputs: Quantity[Any, torch.Tensor, Any],
    *,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> Quantity[Any, torch.Tensor, Any]:
    """Differentiate one scalar output with respect to one quantity.

    The result is the positive derivative, not the negative physical force.
    ``create_graph=True`` keeps the derivative differentiable for higher orders;
    ``retain_graph`` follows PyTorch's normal semantics (including its default).
    Both quantities must share a unit system; the derivative is in that system.
    """
    if output.system is not inputs.system:
        raise TypeError(
            f"Cannot differentiate an output in {output.system.__name__} with "
            f"respect to an input in {inputs.system.__name__}; convert one with "
            ".to_system(...)"
        )
    if output.value.ndim != 0:
        raise ValueError("utorch.grad requires a scalar (zero-dimensional) output.")
    (value,) = torch.autograd.grad(
        output.value,
        inputs.value,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    kind = result_kind("div", output._semantic, inputs._semantic)  # noqa: SLF001
    # Raw derivatives are coherent, except where the system overrides a kind.
    system = inputs.system
    scale = (
        coherence(system, output._semantic)  # noqa: SLF001
        / coherence(system, inputs._semantic)  # noqa: SLF001
        / coherence(system, kind)
    )
    return _wrap(kind, _rescaled(value, scale), system)
