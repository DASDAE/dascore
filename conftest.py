"""
Root pytest configuration.

`dascore/xarray/index.py` subclasses xarray classes, so it cannot import
without xarray; the doctest run (`pytest dascore --doctest-modules`) must
skip collecting it where that optional dependency is absent.
"""

from __future__ import annotations

from importlib.util import find_spec

collect_ignore: list[str] = []
if find_spec("xarray") is None:
    collect_ignore.append("dascore/xarray/index.py")
