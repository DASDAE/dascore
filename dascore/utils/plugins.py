"""Utilities for loading plugins from Python entry points."""

from __future__ import annotations

import functools
from importlib.metadata import entry_points
from typing import Any


@functools.cache
def get_entry_point_loaders(entry_point_group: str) -> dict[str, Any]:
    """Return cached entry-point loaders keyed by entry-point name."""
    return {x.name: x.load for x in entry_points(group=entry_point_group)}


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
