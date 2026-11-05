"""Tests for a patch whose data are a lazy array."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.exceptions import LazyOperationError
from dascore.lazy.array import LazyArray
from dascore.lazy.contract import Pointwise
from dascore.lazy.recipe import Recipe
from dascore.utils.identity import operation_context

from .conftest import lazy_patch


class Negate(dc.PatchProcessor):
    """Negate every sample, one sample at a time."""

    name = None

    def chunk_contract(self, dtype, plan):
        """Each sample needs only itself."""
        return Pointwise()

    def kernel(self, data):
        """Negate."""
        return -data


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


class TestNoFallback:
    """Nothing outside an operation reads lazy data unasked."""

    @pytest.mark.parametrize("show", (repr, str, lambda x: x._repr_html_()))
    def test_repr_reads_nothing(self, lazy, recipe_loads, show):
        """The data are described, not printed."""
        text = show(Negate()(lazy))
        assert "not read yet; 1 operation pending" in text
        assert "float64" in text
        assert not recipe_loads

    def test_repr_counts_steps(self, lazy):
        """The number of pending operations is stated."""
        assert "0 operations pending" in str(lazy)
        assert "2 operations pending" in str(Negate()(Negate()(lazy)))

    @pytest.mark.parametrize(
        "run",
        (
            lambda x: x + 1,
            lambda x: -x,
            lambda x: x * x,
            lambda x: np.abs(x),
            lambda x: np.mean(x),
            lambda x: np.cumsum(x, axis=0),
        ),
        ids=("add", "negative", "multiply", "ufunc", "reduce", "array_function"),
    )
    def test_numpy_fallback_refused(self, lazy, recipe_loads, run):
        """What would convert to numpy raises the lazy error instead."""
        with pytest.raises(LazyOperationError, match=r"patch\.load\(\)"):
            run(lazy)
        assert not recipe_loads

    @pytest.mark.parametrize(
        "run",
        (
            lambda x, y: y + x,
            lambda x, y: x + y,
            lambda x, y: np.add(y, x),
            lambda x, y: np.subtract(y, x),
        ),
        ids=("other_first", "lazy_first", "np_add", "np_subtract"),
    )
    def test_any_operand_lazy_refused(self, lazy, recipe_loads, run):
        """Whichever operand comes first, a lazy one is refused, not read."""
        da = pytest.importorskip("dask.array")
        eager = lazy.load()
        recipe_loads.clear()
        dask_patch = eager.new(data=da.from_array(np.asarray(eager.data)))
        with pytest.raises(LazyOperationError):
            run(lazy, dask_patch)
        assert not recipe_loads

    def test_names_the_ufunc(self, lazy):
        """The message names the operation refused."""
        with pytest.raises(LazyOperationError, match=r"^add cannot run"):
            lazy + 1

    def test_asarray_reads(self, lazy, stored_float, recipe_loads):
        """Asking numpy for the values reads them; the caller asked."""
        out = np.asarray(Negate()(lazy))
        assert len(recipe_loads) == 1
        assert np.array_equal(out, -np.asarray(stored_float.data))

    def test_write_reads(self, lazy, stored_float, tmp_path):
        """So does writing the patch to a file."""
        path = tmp_path / "written.h5"
        dc.write(Negate()(lazy), path, "dasdae")
        assert np.array_equal(dc.read(path)[0].data, -stored_float.data)
