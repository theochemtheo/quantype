"""PyTorch autograd with physical derivative kinds.

``grad`` differentiates a scalar output with respect to one quantity, or to a
sequence of them, keeping the graph unless asked to detach.
"""

try:
    from quantype._internal._torch import grad
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise ModuleNotFoundError("Install quantype[torch] to use quantype.utorch") from exc

__all__ = ["grad"]
