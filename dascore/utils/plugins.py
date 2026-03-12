"""Utilities for loading plugins from Python entry points."""

from __future__ import annotations

import functools
import warnings
from importlib.metadata import entry_points
from typing import Any


@functools.cache
def get_entry_point_loaders(entry_point_group: str) -> dict[str, Any]:
    """Return cached entry-point loaders keyed by entry-point name."""
    out: dict[str, Any] = {}
    duplicate_names: set[str] = set()
    for entry_point in entry_points(group=entry_point_group):
        if entry_point.name in out:
            duplicate_names.add(entry_point.name)
        out[entry_point.name] = entry_point.load
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        msg = (
            f"Duplicate entry points found in group {entry_point_group!r}: {names}. "
            "Using the last registered entry point for each name."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    return out


@functools.cache
def maybe_load_entry_point(entry_point_group: str, name: str) -> Any:
    """
    Load and cache a single entry-point target by group and name.

    If it does not exist, simply return None.
    """
    loader = get_entry_point_loaders(entry_point_group).get(name)
    if loader is not None:
        return loader()
    return None
