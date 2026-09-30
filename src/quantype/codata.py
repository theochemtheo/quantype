"""The CODATA edition behind every unit conversion, fixed once per process.

quantype carries the CODATA 2014, 2018 and 2022 recommended values, so results
never depend on another installed package. One edition applies to the whole
process: every unit, unit system and generated catalogue reads the same values,
so a single process never mixes editions.

The edition is fixed the first time any unit is used. Choose another one before
that, either in the environment::

    QUANTYPE_CODATA=2018 python simulate.py

or in code, before the first unit is used::

    import quantype.codata

    quantype.codata.use("2018")
"""

from __future__ import annotations

import os
from typing import TypeGuard

from quantype._internal._codata import CODATA, EDITIONS, Codata, Edition

__all__ = ["DEFAULT", "EDITIONS", "ENVIRONMENT_VARIABLE", "Codata", "Edition"]
__all__ += ["edition", "is_edition", "use", "values"]

#: The edition used when none is chosen.
DEFAULT: Edition = "2022"
#: Read on first use when :func:`use` has not chosen an edition.
ENVIRONMENT_VARIABLE = "QUANTYPE_CODATA"

_fixed: Edition | None = None


def is_edition(candidate: object) -> TypeGuard[Edition]:
    """Whether ``candidate`` names an edition quantype carries."""
    return candidate in EDITIONS


def _require(candidate: object, source: str) -> Edition:
    if not is_edition(candidate):
        raise ValueError(
            f"Unknown CODATA edition {candidate!r} from {source}; "
            f"choose one of {', '.join(EDITIONS)}"
        )
    return candidate


def edition() -> Edition:
    """The process's edition. Asking fixes it, as using any unit does."""
    global _fixed  # noqa: PLW0603 -- one edition per process, by design
    if _fixed is None:
        chosen = os.environ.get(ENVIRONMENT_VARIABLE)
        _fixed = DEFAULT if chosen is None else _require(chosen, ENVIRONMENT_VARIABLE)
    return _fixed


def use(chosen: Edition) -> None:
    """Choose the process's edition; allowed only before it is fixed.

    A choice made here takes precedence over the environment variable.
    """
    global _fixed  # noqa: PLW0603 -- one edition per process, by design
    selected = _require(chosen, "quantype.codata.use()")
    if _fixed is not None and _fixed != selected:
        raise RuntimeError(
            f"The CODATA edition is already fixed at {_fixed}; call "
            "quantype.codata.use() before using any unit, or set "
            f"{ENVIRONMENT_VARIABLE}={selected}"
        )
    _fixed = selected


def values(of: Edition | None = None) -> Codata:
    """The recommended values of an edition; by default, the process's."""
    return CODATA[edition() if of is None else _require(of, "quantype.codata.values()")]
