"""JAX transformations with physical derivative kinds.

Importing this module registers quantity classes as single-leaf pytrees whose
static metadata is the kind and the unit system. ``grad``, ``value_and_grad``
and ``hessian`` differentiate scalar-output functions with respect to the
quantities ``argnums`` names; other arguments pass through untouched. Gradients
are positive derivatives; physical forces are their explicit negation.
"""

try:
    from quantype._internal._jax import (
        grad,
        hessian,
        jit,
        value_and_grad,
        vmap,
    )

    # Generated application catalogues register their classes through this.
    from quantype._internal._jax import (
        register_quantity as _register_quantity,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )
except ModuleNotFoundError as exc:
    if exc.name != "jax":
        raise
    raise ModuleNotFoundError("Install quantype[jax] to use quantype.ujax") from exc

__all__ = ["grad", "hessian", "jit", "value_and_grad", "vmap"]
