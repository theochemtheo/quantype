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
    def canonical(self, raw: Any) -> Any: ...

def get_unit(name: str, *, kind: Kind | None = ...) -> Unit[Any]: ...
def canonical_unit(kind: Kind) -> Unit[Any]: ...

_CANONICAL_UNITS: dict[Kind, Unit[Any]]
_UNIT_LOOKUPS: dict[Kind, Callable[[str], Unit[Any]]]
