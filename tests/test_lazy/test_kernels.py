"""
Tests for operations on lazy data: each matches the eager result.

The eager reference starts from the same patch loaded with `Patch.load`,
so both start from one id and every difference is the lazy path's.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.processor import PatchProcessor, _subclasses
from dascore.exceptions import LazyOperationError
from dascore.lazy.array import LazyArray
from dascore.lazy.contract import Pointwise, Unchanged

from .conftest import lazy_patch, stored_patch

# The dtypes whose result dtype a pointwise operation could get wrong: the
# integers and bool which angle promotes, and the complex ones whose part
# or magnitude is real.
DTYPES = (
    "int8",
    "int16",
    "int64",
    "uint8",
    "uint16",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
)

# The operations which state a chunk contract.
DECLARED = ("abs", "real", "imag", "angle", "conj")


def _values(shape, dtype):
    """Distinct values of a dtype, negative ones and both parts included."""
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    values = (values - values.mean()) / values.size * 200
    if dtype == "bool":
        return values > 0
    if np.dtype(dtype).kind == "u":
        values = np.abs(values)
    if np.dtype(dtype).kind == "c":
        values = values + 1j * values[::-1]
    return values.astype(dtype)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """A stored patch of each dtype."""
    shape = dc.get_example_patch().shape
    directory = tmp_path_factory.mktemp("dtypes")
    return {x: stored_patch(directory, _values(shape, x), x) for x in DTYPES}


def assert_same(lazy, eager):
    """The loaded lazy patch is the eager one: values, metadata and id."""
    loaded = lazy.load()
    assert isinstance(loaded.data, np.ndarray)
    assert loaded.dtype == eager.dtype
    assert loaded.coords == eager.coords
    assert loaded.attrs == eager.attrs
    assert loaded.attrs.data_id == eager.attrs.data_id
    dtype = np.dtype(eager.dtype)
    if dtype.kind in "fc":
        # Scaled to the dtype: a chunked run may round differently.
        rtol = 8 * np.finfo(dtype).eps
        np.testing.assert_allclose(loaded.data, eager.data, rtol=rtol, atol=0)
    else:
        np.testing.assert_array_equal(loaded.data, eager.data)


class TestEachOperation:
    """Every declared operation, on every dtype, matches the eager one."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("name", DECLARED)
    def test_matches_eager(self, stored, recipe_loads, name, dtype):
        """Coords, attrs, dtype and data_id are the eager ones."""
        lazy = lazy_patch(stored[dtype])
        out = getattr(lazy, name)()
        assert not recipe_loads
        assert isinstance(out.data, LazyArray)
        assert out is lazy or out.data.chain.last.contract == Pointwise()
        eager = getattr(lazy.load(), name)()
        # Known without reading: metadata is never lazy.
        assert out.dtype == eager.dtype
        assert out.shape == eager.shape
        assert out.attrs.data_id == eager.attrs.data_id
        assert_same(out, eager)

    @pytest.mark.parametrize("dtype", ("int16", "float32", "bool"))
    @pytest.mark.parametrize("name", ("real", "conj"))
    def test_no_op_on_real(self, stored, recipe_loads, name, dtype):
        """Real data come back as the patch itself, as they do eagerly."""
        lazy = lazy_patch(stored[dtype])
        assert getattr(lazy, name)() is lazy
        eager = lazy.load()
        assert getattr(eager, name)() is eager
        assert len(recipe_loads) == 1


class TestChain:
    """Several operations run as one chain, step by step as eagerly."""

    @staticmethod
    def _steps(patch):
        """Each patch along the way; a metadata-only one in the middle."""
        out = [patch]
        out.append(out[-1].conj())
        out.append(out[-1].update_attrs(station="here"))
        out.append(out[-1].abs())
        out.append(out[-1].rename_coords(time="t"))
        out.append(out[-1].angle())
        return out

    def test_each_step(self, stored, recipe_loads):
        """Every step's patch and id are the eager step's, nothing read."""
        lazy_steps = self._steps(lazy_patch(stored["complex64"]))
        assert not recipe_loads
        eager_steps = self._steps(lazy_steps[0].load())
        for lazy, eager in zip(lazy_steps, eager_steps, strict=True):
            assert lazy.attrs == eager.attrs
            assert lazy.attrs.data_id == eager.attrs.data_id
            assert lazy.dtype == eager.dtype
        last = lazy_steps[-1]
        assert len(last.data.chain) == 3
        computed = [lazy_steps[x].attrs.data_id for x in (1, 3, 5)]
        assert [x.data_id for x in last.data.chain] == computed
        assert_same(last, eager_steps[-1])
        assert len(recipe_loads) == 2

    def test_metadata_only_keeps_the_array(self, stored):
        """The metadata-only step holds the same lazy array as the step before."""
        steps = self._steps(lazy_patch(stored["complex64"]))
        assert steps[2].data is steps[1].data
        # The patch's id moved on; the array's is still the step's.
        assert steps[2].attrs.data_id != steps[2].data.data_id

    def test_refused_after_steps(self, stored, recipe_loads):
        """An operation with no contract is refused partway, reading nothing."""
        out = lazy_patch(stored["float64"]).abs()
        with pytest.raises(LazyOperationError, match="detrend"):
            out.detrend("time")
        assert not recipe_loads


# Every one of DASCore's own operations which states a chunk contract.
CONTRACTED = sorted(
    {
        x
        for x in _subclasses(PatchProcessor)
        if "chunk_contract" in x.__dict__ and x.__module__.startswith("dascore.")
    },
    key=lambda x: x.__name__,
)


class TestContractsHold:
    """
    What each declared contract promises, checked on its numpy kernel.

    Driven by the registry, so an operation declared later is checked too.
    """

    def test_the_declared_are_found(self):
        """The registry walk finds the five declared here."""
        names = {x.__name__ for x in CONTRACTED}
        assert {"Abs", "Real", "Imag", "Angle", "Conj"} <= names

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("cls", CONTRACTED, ids=lambda x: x.__name__)
    def test_contract_holds(self, cls, dtype):
        """Unchanged hands the input back; Pointwise works on any halves."""
        data = _values((6, 4), dtype)
        coords = {"a": np.arange(6), "b": np.arange(4)}
        patch = dc.Patch(data=data, coords=coords, dims=("a", "b"))
        processor = cls()
        out, plan = processor.get_metadata(patch.drop_data())
        contract = processor.chunk_contract(np.dtype(dtype), plan)
        kernel = cls.kernel_for("numpy")
        result = kernel(processor, data, **plan)
        if isinstance(contract, Unchanged):
            assert result is data
            return
        assert isinstance(contract, Pointwise)
        assert result.shape == data.shape
        assert np.dtype(result.dtype) == out.dtype
        for axis, cut in ((0, 3), (1, 1)):
            first, second = np.split(data, [cut], axis=axis)
            halves = [kernel(processor, x, **plan) for x in (first, second)]
            joined = np.concatenate(halves, axis=axis)
            np.testing.assert_array_equal(joined, result)
