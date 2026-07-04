"""
Utilities for importing optional or expensive dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_UNSET = object()


class _LazyImport:
    """Callable proxy which imports an object on first use."""

    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self._target: Any = _UNSET

    def resolve(self) -> Any:
        """Return the wrapped object, importing it if needed."""
        if self._target is _UNSET:
            module = importlib.import_module(self._module_name)
            self._target = getattr(module, self._attr_name)
        return self._target

    def __call__(self, *args, **kwargs) -> Any:
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)


def lazy_import(module_name: str, attr_name: str) -> _LazyImport:
    """Return a callable proxy for an object imported on first use."""
    return _LazyImport(module_name, attr_name)
