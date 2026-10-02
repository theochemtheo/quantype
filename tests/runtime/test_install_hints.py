"""The optional adapters name the extra to install when their backend is absent.

The imports are blocked in-process, rather than in a subprocess, so coverage
attributes the guards to this run; the tests pass whether or not the backends
are installed.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from importlib.machinery import ModuleSpec
    from types import ModuleType


class _Blocker:
    """A meta-path finder that makes one top-level package unimportable."""

    def __init__(self, root: str) -> None:
        self._root = root

    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        del path, target
        if fullname.partition(".")[0] == self._root:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@contextlib.contextmanager
def _hidden(root: str, adapter: str) -> Generator[None]:
    """Hide ``root`` and force ``adapter`` to import afresh, then restore both."""
    names = [
        name
        for name in sys.modules
        if name.partition(".")[0] == root
        or name == adapter
        or name.startswith(f"quantype._internal._{root}")
    ]
    saved = {name: sys.modules.pop(name) for name in names}
    blocker = _Blocker(root)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if name in names:
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.mark.parametrize(
    ("root", "adapter"), [("jax", "quantype.ujax"), ("torch", "quantype.utorch")]
)
def test_missing_backend_names_the_extra(root: str, adapter: str) -> None:
    with (
        _hidden(root, adapter),
        pytest.raises(ModuleNotFoundError, match=rf"Install quantype\[{root}\]"),
    ):
        importlib.import_module(adapter)
    assert not any(isinstance(finder, _Blocker) for finder in sys.meta_path)


class _ModuleBlocker:
    """A meta-path finder that makes one module unimportable."""

    def __init__(self, name: str) -> None:
        self._name = name

    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        del path, target
        if fullname == self._name:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.mark.parametrize(
    ("root", "adapter"), [("jax", "quantype.ujax"), ("torch", "quantype.utorch")]
)
def test_other_missing_modules_are_not_reported_as_the_extra(
    root: str, adapter: str
) -> None:
    internal = f"quantype._internal._{root}"
    saved = {
        name: sys.modules.pop(name)
        for name in (adapter, internal)
        if name in sys.modules
    }
    blocker = _ModuleBlocker(internal)
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ModuleNotFoundError) as raised:
            importlib.import_module(adapter)
        assert raised.value.name == internal
    finally:
        sys.meta_path.remove(blocker)
        for name in (adapter, internal):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
