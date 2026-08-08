"""Utilities for loading plugins from Python entry points."""

from __future__ import annotations

import functools
import warnings
from importlib.metadata import entry_points
from typing import Any

from dascore.utils.mapping import FrozenDict

# These caches are deliberately unsynchronized, unlike the rest of the
# concurrency work in this subsystem. Concurrent misses on the same key can run
# the wrapped function more than once, which is harmless here. The only side
# effect either has is importing the plugin, and CPython runs a module body
# exactly once behind its per-module import lock; DASCore plugins are
# module-level classes, so once that import succeeds every racing caller
# resolves the same already-registered object. An import that raises is dropped
# from `sys.modules` and simply retried by the next caller, which is the
# behavior we want anyway. Electing a single loader would only save a duplicate
# `entry_points()` scan and, for duplicate names, a repeated warning below --
# not worth the lock choreography it takes to arrange.


@functools.cache
def get_entry_point_loaders(entry_point_group: str) -> FrozenDict[str, Any]:
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
    # Every caller shares this one mapping, so it must not be mutable.
    return FrozenDict(out)


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
