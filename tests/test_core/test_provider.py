"""Tests for building another package's objects through a provider."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import dascore as dc
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import CoordString, NumericCoord, get_coord
from dascore.core.source import ArraySource
from dascore.lazy.array import LazyArray
from dascore.lazy.recipe import Recipe


@pytest.fixture
def recipe():
    """A recipe of a small array, nothing read."""
    return Recipe.from_source(ArraySource.full((3, 4), 1.0))


@pytest.fixture
def parts():
    """The coordinates and dims of a small patch."""
    coords = {"time": np.arange(4), "distance": np.arange(3)}
    return coords, ("distance", "time")


class TestDascoreBuildsItsOwn:
    """What a provider must not change where none is stated."""

    def test_builders_keep_their_signatures(self):
        """The builders keep their names, parameters and docs."""
        assert get_coord.__name__ == "get_coord"
        assert list(inspect.signature(get_coord).parameters) == [
            "data",
            "values",
            "start",
            "min",
            "stop",
            "max",
            "step",
            "units",
            "shape",
            "dtype",
            "segments",
            "runs",
            "sources",
            "snap",
        ]
        kinds = {x.kind for x in inspect.signature(get_coord).parameters.values()}
        assert kinds == {inspect.Parameter.KEYWORD_ONLY}
        assert get_coord_manager.__name__ == "get_coord_manager"
        params = inspect.signature(get_coord_manager).parameters
        assert list(params) == ["coords", "dims", "shape"]
        assert get_coord.__doc__ and get_coord_manager.__doc__

    def test_builders_build_dascore_classes(self):
        """The builders make dascore's coordinates and managers."""
        coord = get_coord(start=0, stop=10, step=1)
        assert isinstance(coord, NumericCoord)
        assert isinstance(get_coord(data=np.array(["a", "b"])), CoordString)
        manager = get_coord_manager({"x": coord}, dims=("x",))
        assert isinstance(manager, CoordManager)

    def test_patch_builds_dascore_objects(self, recipe, parts):
        """A patch holds dascore's manager, and lazy array of a recipe."""
        coords, dims = parts
        patch = dc.Patch(data=recipe, coords=coords, dims=dims)
        assert isinstance(patch.coords, CoordManager)
        assert type(patch.data) is LazyArray
        meta = dc.PatchMeta(coords=coords, dims=dims, dtype=np.float64)
        assert isinstance(meta.coords, CoordManager)

    def test_subclass_operation_builds_dascore_objects(self):
        """An operation on a subclass stating nothing builds dascore's objects."""

        class Sub(dc.Patch):
            """A patch subclass adding nothing."""

        out = Sub(dc.get_example_patch()).dft("time")
        assert type(out) is Sub and isinstance(out.coords, CoordManager)
        assert isinstance(out.get_coord("ft_time"), NumericCoord)
