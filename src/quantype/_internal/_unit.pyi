from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quantype._internal._semantics import Kind

@dataclass(frozen=True, init=False, repr=False, match_args=False)
class Unit[K]:
    name: str
    kind: str
    scale: float
    offset: float
    symbol: str
    semantic: Kind
    def __init__(
        self,
        name: str,
        kind: str | Kind,
        scale: float = ...,
        offset: float = ...,
        symbol: str | None = ...,
    ) -> None: ...

def get_unit(name: str, *, kind: Kind | None = ...) -> Unit[Any]: ...
def units_of(kind: Kind) -> tuple[Unit[Any], ...]: ...
def known_kinds() -> tuple[Kind, ...]: ...

_UNIT_LOOKUPS: dict[Kind, Callable[[str], Unit[Any]]]
_CATALOGUE_UNITS: dict[Kind, tuple[Unit[Any], ...]]
