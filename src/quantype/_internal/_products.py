"""Runtime side of a catalogue's naming table: pair keys, entries and classes.

A generated ``products`` module calls ``register`` with its table, and resolves
its public names through ``pair_class``, which creates each pair's class on
first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantype._internal._semantics import (
    ENTRIES,
    PAIR_CLASSES,
    PAIRS,
    Expression,
    Kind,
    Semantic,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from quantype.core import Quantity

# Generated tables: (name, operation, left, right) and (operation, left, right,
# result), where entry operands name kinds or pairs.
type PairRow = tuple[str, str, str, str | int]
type EntryRow = tuple[str, str, str, str]

_CLASSES: dict[tuple[str, str], type[Quantity[Any, Any, Any]]] = {}
_SEMANTICS: dict[tuple[str, str], Expression] = {}
# Called with each pair class when it is created, so JAX can register it.
CLASS_HOOKS: list[Callable[[type[Quantity[Any, Any, Any]]], None]] = []


def register(
    kinds: Mapping[str, Kind],
    pairs: tuple[PairRow, ...],
    entries: tuple[EntryRow, ...],
    module: str,
) -> None:
    """Register a catalogue's pairs and entries with the physical engine."""
    by_name: dict[str, Semantic] = dict(kinds)
    for name, operation, left, right in pairs:
        first = kinds[left]
        if operation == "pow":
            exponent = int(right)
            pair = Expression("pow", first, exponent)
            PAIRS["pow", first, exponent] = pair
            if exponent == 2:  # noqa: PLR2004 -- x * x is x ** 2
                PAIRS["mul", first, first] = pair
            elif exponent == -1:
                PAIRS["div", kinds["Dimensionless"], first] = pair
        else:
            second = kinds[str(right)]
            pair = Expression("mul" if operation == "mul" else "div", first, second)
            PAIRS[operation, first, second] = pair
            if operation == "mul":
                PAIRS["mul", second, first] = pair
        PAIR_CLASSES[pair] = (module, name)
        _SEMANTICS[module, name] = pair
        by_name[name] = pair
    for operation, left, right, result in entries:
        named = kinds[result]
        ENTRIES[operation, by_name[left], by_name[right]] = named


def pair_class(
    module: str, name: str, base: type[Quantity[Any, Any, Any]]
) -> type[Quantity[Any, Any, Any]]:
    """The class for a registered pair, created on first use."""
    key = (module, name)
    cls = _CLASSES.get(key)
    if cls is None:
        semantic = _SEMANTICS.get(key)
        if semantic is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        bases: tuple[Any, ...] = (base,)
        cls = type(name, bases, {"__module__": module, "_semantic": semantic})
        _CLASSES[key] = cls
        for hook in CLASS_HOOKS:
            hook(cls)
    return cls


def class_for(semantic: Expression) -> type[Quantity[Any, Any, Any]] | None:
    """The runtime class of a pair, or None for any other unnamed product."""
    owner = PAIR_CLASSES.get(semantic)
    if owner is None:
        return None
    cls = _CLASSES.get(owner)
    if cls is None:
        import importlib  # noqa: PLC0415 -- the products module defines the base

        cls = getattr(importlib.import_module(owner[0]), owner[1])
    return cls


def created() -> tuple[type[Quantity[Any, Any, Any]], ...]:
    """The pair classes created so far."""
    return tuple(_CLASSES.values())
