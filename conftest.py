"""
Root pytest configuration.

`dascore/xarray/index.py` subclasses xarray classes, so it cannot import
without xarray, and `dascore/xarray/patch.py` has examples which convert
to xarray; the doctest run (`pytest dascore --doctest-modules`) must skip
collecting both where that optional dependency is absent. The ARROW
format's example needs pyarrow the same way.
"""

from __future__ import annotations

from importlib.util import find_spec

collect_ignore: list[str] = []
if find_spec("xarray") is None:
    collect_ignore += ["dascore/xarray/index.py", "dascore/xarray/patch.py"]
if find_spec("pyarrow") is None:
    collect_ignore += ["dascore/io/arrow/core.py"]
