"""
Root pytest configuration.

`dascore/xarray/index.py` subclasses xarray classes, so it cannot import
without xarray, and `dascore/xarray/patch.py` has examples which convert
to xarray; the doctest run (`pytest dascore --doctest-modules`) must skip
collecting both where that optional dependency is absent. The DASARROW
format's example needs pyarrow the same way.
"""

from __future__ import annotations

import tempfile
from importlib.util import find_spec

import pytest

collect_ignore: list[str] = []
if find_spec("xarray") is None:
    collect_ignore += ["dascore/xarray/index.py", "dascore/xarray/patch.py"]
if find_spec("pyarrow") is None:
    collect_ignore += ["dascore/io/arrow/core.py"]
if find_spec("msgpack") is None:
    collect_ignore += ["dascore/io/nokia/__init__.py"]


@pytest.fixture(autouse=True)
def _example_tempdir(request):
    """
    Make a doc example's temporary folders in its tmp_path, which pytest removes.

    Docstring and qmd examples make them with `tempfile`, as a reader would,
    and leave them for the reader to remove.
    """
    node = request.node
    if isinstance(node, pytest.DoctestItem) or node.get_closest_marker("docs_examples"):
        tmp_path = request.getfixturevalue("tmp_path")
        request.getfixturevalue("monkeypatch").setattr(
            tempfile, "tempdir", str(tmp_path)
        )
