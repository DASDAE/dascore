"""Tests for a patch whose data are a lazy array."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.lazy.array import LazyArray
from dascore.lazy.recipe import Recipe
from dascore.utils.identity import operation_context

from .conftest import lazy_patch


@pytest.fixture()
def lazy(stored_float):
    """The stored float patch, its data not read yet."""
    return lazy_patch(stored_float)


class TestRecipeInAPatch:
    """A recipe handed to a patch is held as a lazy array with nothing pending."""

    def test_wrapped(self, stored_float, recipe_loads):
        """The patch holds a lazy array of the recipe, and reads nothing."""
        recipe = Recipe.from_source(stored_float._source)
        patch = dc.Patch(data=recipe, coords=stored_float.coords)
        assert isinstance(patch.data, LazyArray)
        assert patch.data.recipe is recipe
        assert len(patch.data.chain) == 0
        assert patch.backend == "lazy"
        assert patch.shape == recipe.shape
        assert patch.dtype == recipe.dtype
        assert not recipe_loads

    def test_to_patch_wraps(self, lazy, recipe_loads):
        """Giving metadata a recipe wraps it too."""
        assert isinstance(lazy.data, LazyArray)
        assert not recipe_loads

    def test_len_is_first_axis(self, lazy):
        """The data's length is the first axis, not the number of members."""
        assert len(lazy.data) == lazy.shape[0]
        assert len(lazy.data.recipe) == 1

    def test_bool_refused(self, lazy, recipe_loads):
        """As for an eager patch of more than one element, and nothing is read."""
        with pytest.raises(ValueError, match="ambiguous"):
            bool(lazy)
        assert not recipe_loads


class TestLoad:
    """`Patch.load` reads the data into memory and keeps the id."""

    def test_keeps_id_and_attrs(self, lazy, stored_float, recipe_loads):
        """The array read is the one described, so the id stands."""
        loaded = lazy.load()
        assert len(recipe_loads) == 1
        assert isinstance(loaded.data, np.ndarray)
        assert np.array_equal(loaded.data, stored_float.data)
        assert loaded.attrs == lazy.attrs
        assert loaded.attrs.data_id == lazy.attrs.data_id
        assert loaded.coords == lazy.coords

    def test_in_memory_is_itself(self, stored_float):
        """A patch already in memory is handed back as it is."""
        assert stored_float.load() is stored_float
        loaded = stored_float.load()
        assert loaded.load() is loaded

    def test_dask_is_read(self):
        """Dask data are not in memory either."""
        da = pytest.importorskip("dask.array")
        coords = {"x": np.arange(10)}
        patch = dc.Patch(data=da.arange(10, chunks=5), coords=coords, dims=("x",))
        loaded = patch.load()
        assert isinstance(loaded.data, np.ndarray)
        assert loaded.attrs.data_id == patch.attrs.data_id

    def test_subclass_kept(self):
        """A patch subclass comes back that subclass."""

        class Mine(dc.Patch):
            """A patch subclass."""

        recipe = Recipe.from_source(ArraySource.full((3,), 1.0))
        with operation_context():
            patch = Mine(data=recipe, coords={"x": range(3)}, dims=("x",))
        assert type(patch.load()) is Mine
