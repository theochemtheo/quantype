"""Unit-aware counterparts of NumPy functions, for any array backend.

Each function applies its physical rule to quantities, and hands plain numbers
and arrays straight to their own backend: NumPy, ``jax.numpy``, or Torch. The
same rules serve ``quantype.numpy`` and NumPy's and Torch's dispatch protocols,
so ``np.cos(angle)`` and ``torch.cos(angle)`` behave like ``qnp.cos(angle)``.

Arithmetic is on coherent raw numbers, so kinds a unit system overrides (such as
LAMMPS pressure in bar) are rescaled on the way in and out.
"""

# Package-private wrapping is shared by the numerical adapters.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from quantype._internal._semantics import (
    EXPONENTS,
    TEMPERATURE_DIFFERENCES,
    Expression,
    Kind,
    Semantic,
    dimensionless_kind,
    named_kind,
    product,
)
from quantype._internal._storage import keep_storage
from quantype._internal._systems import coherence
from quantype.core import Quantity, _rescaled, _wrap

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from quantype._internal._systems import UnitSystem

type _Quantity = Quantity[Any, Any, Any]


def _is_quantity(value: object) -> TypeGuard[_Quantity]:
    return isinstance(value, Quantity)


__all__ = [
    "abs",
    "absolute",
    "allclose",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "clip",
    "concatenate",
    "cos",
    "cosh",
    "cross",
    "cumsum",
    "diff",
    "dot",
    "exp",
    "expand_dims",
    "expm1",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "linalg",
    "log",
    "log1p",
    "log2",
    "log10",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "reshape",
    "sin",
    "sinh",
    "sqrt",
    "squeeze",
    "stack",
    "std",
    "sum",
    "tan",
    "tanh",
    "transpose",
    "where",
    "zeros_like",
]


class _Torch:
    """NumPy's names and arguments for the functions Torch spells differently."""

    @staticmethod
    def module() -> Any:
        import torch

        return torch

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module(), name)

    def transpose(self, value: Any, axes: Sequence[int] | None = None) -> Any:
        order = range(value.ndim)[::-1] if axes is None else axes
        return value.permute(*order)

    def expand_dims(self, value: Any, axis: int) -> Any:
        return self.module().unsqueeze(value, axis)

    def squeeze(self, value: Any, axis: int | None = None) -> Any:
        return value.squeeze() if axis is None else value.squeeze(axis)

    def stack(self, values: Sequence[Any], axis: int = 0) -> Any:
        return self.module().stack(list(values), dim=axis)

    def concatenate(self, values: Sequence[Any], axis: int = 0) -> Any:
        return self.module().cat(list(values), dim=axis)

    def cumsum(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self.module().cumsum(value.flatten(), dim=0)
        return self.module().cumsum(value, dim=axis)

    def diff(self, value: Any, n: int = 1, axis: int = -1) -> Any:
        return self.module().diff(value, n=n, dim=axis)

    def std(
        self,
        value: Any,
        axis: int | None = None,
        *,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> Any:
        return self.module().std(value, dim=axis, correction=ddof, keepdim=keepdims)

    def dot(self, left: Any, right: Any) -> Any:
        return self.module().matmul(left, right)

    def cross(self, left: Any, right: Any, axis: int = -1) -> Any:
        return self.module().linalg.cross(left, right, dim=axis)

    def norm(
        self, value: Any, axis: int | None = None, *, keepdims: bool = False
    ) -> Any:
        return self.module().linalg.vector_norm(value, dim=axis, keepdim=keepdims)


_TORCH = _Torch()


def _xp(*values: Any) -> Any:
    """The array namespace of the values: NumPy, jax.numpy, or Torch's adapter."""
    for value in values:
        module = type(value).__module__
        if module.startswith("torch"):
            return _TORCH
        if module.startswith(("jax", "jaxlib")):
            import jax.numpy as jnp

            return jnp
    import numpy as np

    return np


def _norm(namespace: Any) -> Callable[..., Any]:
    return namespace.norm if namespace is _TORCH else namespace.linalg.norm


def _raw(quantity: _Quantity) -> Any:
    """Raw numbers in the system's coherent units."""
    return _rescaled(quantity.value, coherence(quantity.system, quantity._semantic))


def _result(
    kind: Semantic,
    coherent: Any,
    system: type[UnitSystem],
    like: Any,
    *,
    display: Any = None,
) -> _Quantity:
    """Wrap coherent raw numbers, keeping the operand's storage type and dtype."""
    raw = _rescaled(coherent, 1 / coherence(system, kind))
    return _wrap(kind, keep_storage(like, raw), system, display=display)


def _require(quantity: _Quantity, name: str, function: str) -> None:
    if quantity._semantic is not named_kind(quantity._semantic, name):
        raise TypeError(f"{function} expects {name}, received {quantity.kind}")


def _same(function: str, first: _Quantity, *others: object) -> None:
    """Every operand is a quantity of the first's kind and system."""
    for other in others:
        if not _is_quantity(other):
            raise TypeError(
                f"{function} expects quantities of one kind; "
                f"received {first.kind} and {type(other).__name__}"
            )
        quantity = other
        first._same_system(quantity)
        if quantity._semantic != first._semantic:
            raise TypeError(
                f"{function} expects quantities of one kind; "
                f"received {first.kind} and {quantity.kind}"
            )


def _not_affine(quantity: _Quantity, function: str) -> None:
    if isinstance(quantity._semantic, Kind) and quantity._semantic.affine:
        raise TypeError(f"{function} is meaningless for absolute Temperatures")


def _elementwise(name: str, source: str, target: str) -> Callable[[Any], Any]:
    def function(x: Any, /) -> Any:
        if not _is_quantity(x):
            return getattr(_xp(x), name)(x)
        quantity = x
        _require(quantity, source, name)
        raw = getattr(_xp(quantity.value), name)(_raw(quantity))
        kind = named_kind(quantity._semantic, target)
        return _result(kind, raw, quantity.system, quantity.value)

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"``{name}`` of a {source} quantity, as {target}."
    return function


sin = _elementwise("sin", "Angle", "Dimensionless")
cos = _elementwise("cos", "Angle", "Dimensionless")
tan = _elementwise("tan", "Angle", "Dimensionless")
arcsin = _elementwise("arcsin", "Dimensionless", "Angle")
arccos = _elementwise("arccos", "Dimensionless", "Angle")
arctan = _elementwise("arctan", "Dimensionless", "Angle")
exp = _elementwise("exp", "Dimensionless", "Dimensionless")
expm1 = _elementwise("expm1", "Dimensionless", "Dimensionless")
log = _elementwise("log", "Dimensionless", "Dimensionless")
log1p = _elementwise("log1p", "Dimensionless", "Dimensionless")
log2 = _elementwise("log2", "Dimensionless", "Dimensionless")
log10 = _elementwise("log10", "Dimensionless", "Dimensionless")
sinh = _elementwise("sinh", "Dimensionless", "Dimensionless")
cosh = _elementwise("cosh", "Dimensionless", "Dimensionless")
tanh = _elementwise("tanh", "Dimensionless", "Dimensionless")


def _root(semantic: Semantic) -> Semantic:
    """The kind whose square is ``semantic``: declared powers, or a structural one."""
    if isinstance(semantic, Expression):
        if semantic.operation == "pow" and semantic.right == 2:  # noqa: PLR2004
            return semantic.left
        if semantic.operation == "mul" and semantic.left == semantic.right:
            return semantic.left
    elif semantic is dimensionless_kind(semantic):
        return semantic
    else:
        for (base, exponent), result in EXPONENTS.items():
            if exponent == 2 and result is semantic:  # noqa: PLR2004
                return base
    raise TypeError(f"sqrt needs a squared kind, such as Area; received {semantic}")


def sqrt(x: Any, /) -> Any:
    """Square root: Area to Length, and any declared or structural square."""
    if not _is_quantity(x):
        return x**0.5 if isinstance(x, (int, float)) else _xp(x).sqrt(x)
    quantity = x
    kind = _root(quantity._semantic)
    raw = _raw(quantity)
    root = math.sqrt(raw) if isinstance(raw, (int, float)) else _xp(raw).sqrt(raw)
    return _result(kind, root, quantity.system, quantity.value)


def absolute(x: Any, /) -> Any:
    """Magnitude, keeping the kind and display unit."""
    # The builtin, which this module rebinds to `abs` below.
    return operator.abs(x) if _is_quantity(x) else _xp(x).abs(x)


abs = absolute  # noqa: A001 -- NumPy's name


def arctan2(y: Any, x: Any, /) -> Any:
    """The angle of (x, y), which share a kind and system."""
    if not _is_quantity(y):
        return _xp(y, x).arctan2(y, x)
    first = y
    _same("arctan2", first, x)
    second = x
    raw = _xp(first.value, second.value).arctan2(first.value, second.value)
    return _result(named_kind(first._semantic, "Angle"), raw, first.system, first.value)


def _pairwise(name: str) -> Callable[[Any, Any], Any]:
    def function(a: Any, b: Any, /) -> Any:
        if not _is_quantity(a):
            return getattr(_xp(a, b), name)(a, b)
        first = a
        _same(name, first, b)
        second = b
        raw = getattr(_xp(first.value, second.value), name)(first.value, second.value)
        return first._with(raw)

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"Element-wise ``{name}`` of quantities of one kind."
    return function


maximum = _pairwise("maximum")
minimum = _pairwise("minimum")


def clip(x: Any, a_min: Any, a_max: Any, /) -> Any:
    """Limit values to bounds of the same kind; either bound may be None."""
    if not _is_quantity(x):
        return _xp(x).clip(x, a_min, a_max)
    quantity = x
    bounds = [bound for bound in (a_min, a_max) if bound is not None]
    _same("clip", quantity, *bounds)
    low, high = (None if b is None else b.value for b in (a_min, a_max))
    return quantity._with(_xp(quantity.value).clip(quantity.value, low, high))


def where(condition: Any, x: Any, y: Any, /) -> Any:
    """Choose from two quantities of one kind; ``condition`` is plain booleans."""
    if not _is_quantity(x):
        return _xp(condition, x, y).where(condition, x, y)
    first = x
    _same("where", first, y)
    second = y
    raw = _xp(first.value, second.value).where(condition, first.value, second.value)
    return first._with(raw)


def _predicate(name: str) -> Callable[[Any], Any]:
    def function(x: Any, /) -> Any:
        value: Any = x.value if _is_quantity(x) else x
        return getattr(_xp(value), name)(value)

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"``{name}`` of the raw numbers: plain booleans, no units."
    return function


isnan = _predicate("isnan")
isfinite = _predicate("isfinite")
isinf = _predicate("isinf")


def _tolerances(a: _Quantity, b: object, atol: object) -> tuple[Any, Any]:
    _same("isclose", a, b)
    other: Any = b  # _same checked that it is a quantity of a's kind
    if _is_quantity(atol):
        _same("isclose", a, atol)
        return other.value, atol.value
    if atol != 0:
        raise TypeError("atol must be a quantity of the compared kind, or 0")
    return other.value, 0.0


def isclose(
    a: Any, b: Any, *, rtol: float = 1e-05, atol: Any = 0, equal_nan: bool = False
) -> Any:
    """Element-wise closeness; an absolute tolerance must share the kind."""
    if not _is_quantity(a):
        return _xp(a, b).isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    first = a
    other, tolerance = _tolerances(first, b, atol)
    namespace = _xp(first.value, other)
    return namespace.isclose(
        first.value, other, rtol=rtol, atol=tolerance, equal_nan=equal_nan
    )


def allclose(
    a: Any, b: Any, *, rtol: float = 1e-05, atol: Any = 0, equal_nan: bool = False
) -> bool:
    """Whether every element is close; an absolute tolerance must share the kind."""
    if not _is_quantity(a):
        return bool(_xp(a, b).allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
    first = a
    other, tolerance = _tolerances(first, b, atol)
    namespace = _xp(first.value, other)
    close = namespace.allclose(
        first.value, other, rtol=rtol, atol=tolerance, equal_nan=equal_nan
    )
    return bool(close)


def dot(a: Any, b: Any, /) -> Any:
    """Dot product; declared products name the result, as for ``*``."""
    if not _is_quantity(a) or not _is_quantity(b):
        if _is_quantity(a) or _is_quantity(b):
            raise TypeError("dot of a quantity needs another quantity; use *")
        return _xp(a, b).dot(a, b)
    first, second = a, b
    first._same_system(second)
    kind = product("mul", first._semantic, second._semantic)
    raw = _xp(first.value, second.value).dot(_raw(first), _raw(second))
    return _result(kind, raw, first.system, first.value)


def cross(a: Any, b: Any, /, axis: int = -1) -> Any:
    """Cross product. The result stays unnamed: r cross F is a torque, not work."""
    if not _is_quantity(a) or not _is_quantity(b):
        if _is_quantity(a) or _is_quantity(b):
            raise TypeError("cross of a quantity needs another quantity")
        return _xp(a, b).cross(a, b, axis=axis)
    first, second = a, b
    first._same_system(second)
    kind = Expression("mul", first._semantic, second._semantic)
    raw = _xp(first.value, second.value).cross(_raw(first), _raw(second), axis=axis)
    return _result(kind, raw, first.system, first.value)


def _norm_of(x: Any, axis: int | None = None, *, keepdims: bool = False) -> Any:
    if not _is_quantity(x):
        return _norm(_xp(x))(x, axis=axis, keepdims=keepdims)
    quantity = x
    _not_affine(quantity, "norm")
    raw = _norm(_xp(quantity.value))(quantity.value, axis=axis, keepdims=keepdims)
    return quantity._with(keep_storage(quantity.value, raw))


class _Linalg:
    """``qnp.linalg``: the linear-algebra functions with unit rules."""

    @staticmethod
    def norm(x: Any, axis: int | None = None, *, keepdims: bool = False) -> Any:
        """Vector norm along an axis, keeping the kind."""
        return _norm_of(x, axis, keepdims=keepdims)


linalg = _Linalg()


def _joined(name: str) -> Callable[..., Any]:
    def function(arrays: Sequence[Any], axis: int = 0) -> Any:
        items = list(arrays)
        first = items[0] if items else None
        if not _is_quantity(first):
            return getattr(_xp(*items), name)(items, axis=axis)
        _same(name, first, *items[1:])
        raws = [item.value for item in items]
        return first._with(getattr(_xp(*raws), name)(raws, axis=axis))

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"``{name}`` quantities of one kind and system."
    return function


stack = _joined("stack")
concatenate = _joined("concatenate")


def _reshaping(name: str) -> Callable[..., Any]:
    def function(x: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_quantity(x):
            return getattr(_xp(x), name)(x, *args, **kwargs)
        quantity = x
        return quantity._with(
            getattr(_xp(quantity.value), name)(quantity.value, *args, **kwargs)
        )

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"``{name}``, keeping the kind and display unit."
    return function


reshape = _reshaping("reshape")
transpose = _reshaping("transpose")
squeeze = _reshaping("squeeze")
expand_dims = _reshaping("expand_dims")


def zeros_like(x: Any) -> Any:
    """Zeros shaped like ``x``, keeping its kind and display unit."""
    if not _is_quantity(x):
        return _xp(x).zeros_like(x)
    return x._with(_xp(x.value).zeros_like(x.value))


def _reduction(name: str) -> Callable[..., Any]:
    def function(x: Any, axis: int | None = None, *, keepdims: bool = False) -> Any:
        if not _is_quantity(x):
            return getattr(_xp(x), name)(x, axis=axis, keepdims=keepdims)
        return getattr(x, name)(axis, keepdims=keepdims)

    function.__name__ = function.__qualname__ = name
    function.__doc__ = f"``{name}`` over an axis, keeping the kind."
    return function


sum = _reduction("sum")  # noqa: A001 -- NumPy's name
mean = _reduction("mean")
max = _reduction("max")  # noqa: A001 -- NumPy's name
min = _reduction("min")  # noqa: A001 -- NumPy's name


def _spread(quantity: _Quantity) -> Semantic:
    """A spread of absolute temperatures is a temperature difference."""
    semantic = quantity._semantic
    if isinstance(semantic, Kind) and semantic.affine:
        return TEMPERATURE_DIFFERENCES[semantic]
    return semantic


def std(
    x: Any, axis: int | None = None, *, ddof: int = 0, keepdims: bool = False
) -> Any:
    """Standard deviation; of absolute temperatures, a temperature difference."""
    if not _is_quantity(x):
        return _xp(x).std(x, axis=axis, ddof=ddof, keepdims=keepdims)
    quantity = x
    raw = _xp(quantity.value).std(
        quantity.value, axis=axis, ddof=ddof, keepdims=keepdims
    )
    raw = keep_storage(quantity.value, _as_like(quantity.value, raw))
    return _wrap(_spread(quantity), raw, quantity.system)


def diff(x: Any, n: int = 1, axis: int = -1) -> Any:
    """Differences along an axis; of absolute temperatures, temperature differences."""
    if not _is_quantity(x):
        return _xp(x).diff(x, n=n, axis=axis)
    quantity = x
    raw = _xp(quantity.value).diff(quantity.value, n=n, axis=axis)
    return _wrap(_spread(quantity), raw, quantity.system)


def cumsum(x: Any, axis: int | None = None) -> Any:
    """Cumulative sums; absolute temperatures do not add."""
    if not _is_quantity(x):
        return _xp(x).cumsum(x, axis=axis)
    quantity = x
    _not_affine(quantity, "cumsum")
    return quantity._with(_xp(quantity.value).cumsum(quantity.value, axis=axis))


def _as_like(like: Any, result: Any) -> Any:
    """NumPy reductions of arrays give scalars; keep array storage an array."""
    if type(like).__module__.startswith("numpy"):
        import numpy as np

        if isinstance(like, np.ndarray):
            return np.asarray(result)
    return result


# NumPy's ufuncs and array functions, by name, that quantities support.
UFUNCS: dict[str, Callable[..., Any]] = {
    **{
        name: globals()[name]
        for name in (
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "arctan2",
            "exp",
            "expm1",
            "log",
            "log1p",
            "log2",
            "log10",
            "sinh",
            "cosh",
            "tanh",
            "sqrt",
            "absolute",
            "maximum",
            "minimum",
            "isnan",
            "isfinite",
            "isinf",
        )
    },
    **{
        name: cast("Callable[..., Any]", getattr(operator, method))
        for name, method in (
            ("add", "add"),
            ("subtract", "sub"),
            ("multiply", "mul"),
            ("divide", "truediv"),
            ("true_divide", "truediv"),
            ("negative", "neg"),
            ("positive", "pos"),
            ("power", "pow"),
            ("less", "lt"),
            ("less_equal", "le"),
            ("greater", "gt"),
            ("greater_equal", "ge"),
            ("equal", "eq"),
            ("not_equal", "ne"),
        )
    },
}
FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: globals()[name]
    for name in (
        "sum",
        "mean",
        "max",
        "min",
        "std",
        "cumsum",
        "diff",
        "reshape",
        "transpose",
        "squeeze",
        "expand_dims",
        "stack",
        "concatenate",
        "where",
        "clip",
        "dot",
        "cross",
        "zeros_like",
        "isclose",
        "allclose",
    )
}
FUNCTIONS["amax"], FUNCTIONS["amin"] = max, min


def _shape(x: _Quantity) -> tuple[int, ...]:
    return x.shape


def _ndim(x: _Quantity) -> int:
    return x.ndim


FUNCTIONS["shape"], FUNCTIONS["ndim"] = _shape, _ndim
LINALG: dict[str, Callable[..., Any]] = {"norm": _norm_of}

# Reflected operators for ufuncs whose left operand is a plain array.
REFLECTED = {
    "add": "__radd__",
    "subtract": "__rsub__",
    "multiply": "__rmul__",
    "divide": "__rtruediv__",
    "true_divide": "__rtruediv__",
    "less": "__gt__",
    "less_equal": "__ge__",
    "greater": "__lt__",
    "greater_equal": "__le__",
    "equal": "__eq__",
    "not_equal": "__ne__",
}

# Torch's names and keywords for the same functions.
_TORCH_NAMES = {
    "abs": "absolute",
    "acos": "arccos",
    "asin": "arcsin",
    "atan": "arctan",
    "atan2": "arctan2",
    "clamp": "clip",
    "cat": "concatenate",
    "concat": "concatenate",
    "unsqueeze": "expand_dims",
    "matmul": "dot",
    "linalg_cross": "cross",
    "linalg_vector_norm": "norm",
    "vector_norm": "norm",
}
_TORCH_KEYWORDS = {"dim": "axis", "keepdim": "keepdims", "correction": "ddof"}


def torch_function(func: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
    """Torch's dispatch protocol, onto the same unit rules as NumPy's."""
    name = str(getattr(func, "__name__", ""))
    if name.startswith("__"):
        # Tensor operators: Python then tries the quantity's reflected operator.
        return NotImplemented
    name = _TORCH_NAMES.get(name, name)
    function = UFUNCS.get(name) or FUNCTIONS.get(name) or LINALG.get(name)
    if function is None:
        raise TypeError(
            f"torch.{name} does not know the units of a quantity; use "
            "quantype.numpy, quantity methods, or .value and .magnitude(unit)"
        )
    keywords = {_TORCH_KEYWORDS.get(key, key): value for key, value in kwargs.items()}
    arguments = list(args)
    if name == "clip":
        arguments += [keywords.pop("min", None), keywords.pop("max", None)]
        arguments = arguments[:3]
    if name == "std" and "ddof" not in keywords:
        # Torch's standard deviation is unbiased unless told otherwise.
        keywords["ddof"] = 0 if keywords.pop("unbiased", True) is False else 1
    return function(*arguments, **keywords)
