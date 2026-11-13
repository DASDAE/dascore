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
from dascore.lazy.contract import Pointwise
from dascore.lazy.execute import compute

from .conftest import grid, lazy_patch, stored_patch

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
    # Not a ramp, which a smoothing filter would hand back as it was.
    values = values + 20 * np.sin(np.arange(values.size) * 0.7).reshape(shape)
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
    _assert_close(loaded.data, eager.data)


def _assert_close(found, expected):
    """The arrays agree, to the rounding of their dtype."""
    dtype = np.dtype(expected.dtype)
    if dtype.kind in "fc":
        # Scaled to the dtype: a chunked run may round differently.
        rtol = 8 * np.finfo(dtype).eps
        np.testing.assert_allclose(found, expected, rtol=rtol, atol=0)
    else:
        np.testing.assert_array_equal(found, expected)


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


# How each of them is called. An operation declared later needs an entry.
CALLS = {
    "Abs": lambda x: x.abs(),
    "Real": lambda x: x.real(),
    "Imag": lambda x: x.imag(),
    "Angle": lambda x: x.angle(),
    "Conj": lambda x: x.conj(),
    "_Query": lambda x: x.select(time=(5, 300), distance=(2, 90), samples=True),
    "PassFilter": lambda x: x.pass_filter(time=(10, 100)),
    "MedianFilter": lambda x: x.median_filter(time=5, distance=3, samples=True),
    "SavgolFilter": lambda x: x.savgol_filter(
        polyorder=2, time=7, distance=5, samples=True, mode="mirror"
    ),
    "GaussianFilter": lambda x: x.gaussian_filter(time=1.5, distance=1, samples=True),
    "Windowed": lambda x: x.window(time=64, distance=8, samples=True).detrend("time"),
}

# Cuts of the two axes: uneven ones, and pieces shorter than any halo.
GRIDS = (((7, 150), (500,)), ((1, 2, 3), (1, 2, 1000)), ((), (3, 4, 5, 1999)))


class TestContractsHold:
    """
    What each declared contract promises: any chunks give the eager result.

    Driven by the registry, so an operation declared later is checked too.
    """

    def test_the_declared_are_called(self):
        """Every operation which states a contract has a call here."""
        assert {x.__name__ for x in CONTRACTED} == set(CALLS)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("cls", CONTRACTED, ids=lambda x: x.__name__)
    def test_contract_holds(self, stored, cls, dtype):
        """Computed in boxes of any grid, the result is the eager one."""
        call = CALLS[cls.__name__]
        lazy_in = lazy_patch(stored[dtype])
        eager_in = lazy_in.load()
        try:
            lazy = call(lazy_in)
        except LazyOperationError:
            # Only the filters refuse a dtype, and never these two.
            assert cls.__name__.endswith("Filter")
            assert dtype not in ("float32", "float64")
            return
        eager = call(eager_in)
        if lazy is lazy_in:
            assert eager is eager_in  # Unchanged: a no-op eagerly too
            return
        assert lazy.dtype == eager.dtype and lazy.shape == eager.shape
        assert lazy.attrs.data_id == eager.attrs.data_id
        for edges in GRIDS:
            out = np.empty(lazy.shape, lazy.dtype)
            # A selection is shorter, so only the cuts inside it apply.
            inside = [[x for x in e if x < n] for e, n in zip(edges, lazy.shape)]
            for low, high in grid(lazy.shape, inside):
                box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
                out[box] = compute(lazy.data, box)
            _assert_close(out, eager.data)
