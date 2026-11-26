"""
Tests for another package's implementation of an interface, run on DASCore's too.

A package's suite subclasses a class here and gives the fixture named in
its docstring; see [Interfaces](`dascore/docs/notes/interfaces.qmd`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.backends import LazyBackend
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.core.interfaces import (
    BaseCoord,
    BaseCoordManager,
    BaseLazyArray,
    BaseNumericCoord,
    BaseSpool,
    BaseStringCoord,
)
from dascore.utils.identity import H


def _missing(value) -> bool:
    """Whether a value is a NaN or NaT scalar; None is not."""
    scalar = value is not None and pd.api.types.is_scalar(value)
    return scalar and bool(pd.isna(value))


def _assert_same(theirs, ours, where: str = "result") -> None:
    """Assert an implementation's result is what DASCore's gives."""
    if isinstance(ours, BaseCoord):
        numeric = isinstance(ours, BaseNumericCoord)
        kind = BaseNumericCoord if numeric else BaseStringCoord
        assert isinstance(theirs, kind), f"{where} is not a {kind.__name__}"
        names = (
            *("data_id", "units", "dtype", "shape", "step", "values"),
            *("evenly_sampled", "sorted", "reverse_sorted", "_partial", "_grid"),
        )
        for name in (*names, "runs") if numeric else names:
            _assert_same(getattr(theirs, name), getattr(ours, name), f"{where}.{name}")
    elif isinstance(ours, BaseCoordManager):
        assert isinstance(theirs, BaseCoordManager), f"{where} is not a manager"
        for name in ("dims", "shape", "dim_map", "coord_map"):
            _assert_same(getattr(theirs, name), getattr(ours, name), f"{where}.{name}")
    elif isinstance(ours, dc.Patch):
        assert isinstance(theirs, dc.Patch), f"{where} is not a Patch"
        _assert_same(theirs.backend, ours.backend, f"{where}.backend")
        _assert_same(theirs.coords, ours.coords, f"{where}.coords")
        # The history is left out: it is being retired.
        mine, expected = (
            x.attrs.model_dump(exclude={"history"}) for x in (theirs, ours)
        )
        _assert_same(mine, expected, f"{where}.attrs")
        _assert_same(np.asarray(theirs.data), np.asarray(ours.data), f"{where}.data")
    elif isinstance(ours, BaseSpool):
        assert isinstance(theirs, BaseSpool), f"{where} is not a spool"
        _assert_same(theirs.backend, ours.backend, f"{where}.backend")
        _assert_same(list(theirs), list(ours), where)
    elif isinstance(ours, pd.DataFrame):
        pd.testing.assert_frame_equal(theirs, ours, check_exact=True, obj=where)
    elif isinstance(ours, pd.Series):
        pd.testing.assert_series_equal(theirs, ours, check_exact=True, obj=where)
    elif isinstance(ours, np.ndarray):
        # Strict, so neither a scalar nor another dtype passes for an array.
        np.testing.assert_array_equal(np.asarray(theirs), ours, where, strict=True)
    elif isinstance(ours, tuple | list):
        assert isinstance(theirs, type(ours)), f"{where} is not a {type(ours)}"
        assert len(theirs) == len(ours), f"{where} has {len(theirs)} items"
        for number, (one, other) in enumerate(zip(theirs, ours, strict=True)):
            _assert_same(one, other, f"{where}[{number}]")
    elif isinstance(ours, Mapping):
        assert isinstance(theirs, Mapping), f"{where} is not a mapping"
        assert set(theirs) == set(ours), f"{where} has keys {sorted(theirs)}"
        for key, value in ours.items():
            _assert_same(theirs[key], value, f"{where}[{key!r}]")
    elif _missing(ours):
        # NaN for NaN, NaT for NaT.
        kinds = (np.asarray(x).dtype.kind for x in (theirs, ours))
        assert _missing(theirs) and len(set(kinds)) == 1, f"{where} is {theirs!r}"
    else:
        # A number of the same kind (an int for an int, not a bool or an
        # array), or else an instance of the same class.
        if isinstance(ours, np.generic | int | float):
            kinds = (np.asarray(x).dtype.kind for x in (theirs, ours))
            same = np.ndim(theirs) == 0 and len(set(kinds)) == 1
        else:
            same = isinstance(theirs, type(ours))
        assert same and theirs == ours, f"{where} is {theirs!r}, not {ours!r}"


def _check(call, theirs, ours) -> None:
    """Assert a call gives on an implementation what it gives on DASCore's."""
    try:
        expected = call(ours)
    except Exception as error:
        # Raised as DASCore raises it, so a caller catching it still does.
        with pytest.raises(type(error)):
            call(theirs)
    else:
        _assert_same(call(theirs), expected)


def _names(table: dict):
    """Return a fixture giving each name in a table in turn."""

    @pytest.fixture(params=sorted(table))
    def name(self, request) -> str:
        return request.param

    return name


def _gapped(make):
    """A coordinate of two evenly sampled runs either side of a gap."""
    one = np.int64(1)
    starts = (np.int64(0), np.int64(10))
    return make(segments=tuple(make(start=x, stop=x + 4, step=one) for x in starts))


# Each case is built by a function taking get_coord's arguments; integer
# ends are int64, so a 32-bit platform builds the same.
_NUMERIC_CASES = {
    "ints": lambda make: make(
        start=np.int64(0), stop=np.int64(10), step=np.int64(1), units="m"
    ),
    "floats": lambda make: make(start=0.0, stop=2.0, step=0.25, units="s"),
    "times": lambda make: make(
        start=np.datetime64("2020-01-01T00:00:00"),
        stop=np.datetime64("2020-01-01T00:00:10"),
        step=np.timedelta64(1, "s"),
    ),
    "descending": lambda make: make(data=np.arange(10.0)[::-1], units="m"),
    "labels": lambda make: make(data=np.array([0.0, 1.5, 4.0, 4.5, 7.0, 9.0])),
    "gapped": _gapped,
    "partial": lambda make: make(shape=(6,), step=1.0),
}
_TEXT_CASES = {
    "unsorted": lambda make: make(data=np.array(["c", "a", "b", "e", "d", "f"])),
    "sorted": lambda make: make(data=np.array(["a", "b", "c", "d", "e"])),
}
_COORD_PROPERTIES = (
    "units",
    "step",
    "shape",
    "dtype",
    "size",
    "ndim",
    "data_id",
    "evenly_sampled",
    "step_exact",
    "degenerate",
    "sorted",
    "reverse_sorted",
    "_partial",
    "_rich_style",
)
# Each call is made on the implementation's coordinate and on DASCore's.
_COORD_CALLS = {
    "properties": lambda c: {x: getattr(c, x) for x in _COORD_PROPERTIES},
    "values": lambda c: c.values,
    "len": len,
    "min_max": lambda c: (c.min(), c.max()),
    "coord_range": lambda c: c.coord_range(),
    "getitem": lambda c: c[1:4],
    "index": lambda c: c.index(np.array([0, 2, 3])),
    "select_samples": lambda c: c.select((1, 4), samples=True),
    "select_values": lambda c: c.select((c.values[1], c.values[4])),
    "select_nothing": lambda c: c.select((1, 1), samples=True),
    "order": lambda c: c.order(c.values[[3, 1]]),
    "sort": lambda c: c.sort(),
    "sort_reverse": lambda c: c.sort(reverse=True),
    "snap": lambda c: c.snap(),
    "update": lambda c: c.update(data=c.values[::-1]),
    "update_limits": lambda c: c.update_limits(min=c.values[1]),
    "new": lambda c: c.new(units="ft"),
    "new_unchanged": lambda c: c.new(),
    "set_units": lambda c: c.set_units("km"),
    "convert_units": lambda c: c.convert_units("ft"),
    "simplify_units": lambda c: c.simplify_units(),
    "change_length": lambda c: c.change_length(3),
    "reduce": lambda c: c.reduce_coord(),
    "align": lambda c: c.align_to(c[2:5]),
    "approx_equal": lambda c: (c.approx_equal(c), c.approx_equal(c[1:])),
    "approx_equal_dascore": lambda c: get_coord(data=c).approx_equal(c),
    "next_index": lambda c: c.get_next_index(c.values[2]),
    "summary": lambda c: c.to_summary(),
    "rebuilt": lambda c: get_coord(data=c),
    "identity": lambda c: c._identity(),
    "index_values": lambda c: c._get_index_values(np.array([0, 2])),
    "compatible_value": lambda c: c._get_compatible_value(c.values[1]),
    "index_of": lambda c: c._get_index(c.values[2]),
    "repr_fields": lambda c: c._repr_fields(),
}
_NUMERIC_CALLS = {
    **_COORD_CALLS,
    "sample_count": lambda c: c.get_sample_count(c.values[3] - c.values[0]),
    "runs": lambda c: (c.runs, c.runs_count),
    "segments": lambda c: c.segments,
    "run_offsets": lambda c: c._run_offsets(),
    "direction": lambda c: c._direction(),
    "grid": lambda c: c._grid,
}


class _CoordTests:
    """The tests either kind of coordinate runs, on its kind's cases."""

    kind: type[BaseCoord]
    cases: dict
    calls: dict

    @pytest.fixture
    def coords(self, make_coord, case):
        """A case built by the implementation, and by DASCore."""
        build = self.cases[case]
        return build(make_coord), build(get_coord)

    def test_kind(self, coords):
        """The coordinate is of its kind, and not of the other."""
        theirs, _ = coords
        other = BaseStringCoord if self.kind is BaseNumericCoord else BaseNumericCoord
        assert isinstance(theirs, self.kind) and not isinstance(theirs, other)

    def test_call(self, coords, call):
        """A call gives what it gives on DASCore's coordinate."""
        _check(self.calls[call], *coords)

    def test_equal(self, make_coord, case):
        """Coordinates built alike are equal, and others are not."""
        build = self.cases[case]
        first, second = build(make_coord), build(make_coord)
        assert first == second and not first == second[1:]


class BaseNumericCoordTests(_CoordTests):
    """
    The tests a numeric coordinate runs.

    A subclass gives a `make_coord` fixture: a function taking the arguments
    of [`get_coord`](`dascore.core.coords.get_coord`) and returning the
    package's numeric coordinate.
    """

    kind = BaseNumericCoord
    cases = _NUMERIC_CASES
    calls = _NUMERIC_CALLS
    case = _names(_NUMERIC_CASES)
    call = _names(_NUMERIC_CALLS)


class BaseStringCoordTests(_CoordTests):
    """
    The tests a coordinate of text runs.

    A subclass gives a `make_coord` fixture: a function taking the arguments
    of [`get_coord`](`dascore.core.coords.get_coord`) and returning the
    package's coordinate of text.
    """

    kind = BaseStringCoord
    cases = _TEXT_CASES
    calls = _COORD_CALLS
    case = _names(_TEXT_CASES)
    call = _names(_COORD_CALLS)


def _patch_coords(make):
    """The coordinates of a patch: dimensions, and others along them."""
    time = np.datetime64("2020-01-01") + np.arange(6) * np.timedelta64(500, "ms")
    coords = {
        "distance": np.arange(4) * 2.0,
        "time": time,
        "latitude": ("distance", np.array([1.0, 3.0, 2.0, 4.0])),
        "station": ("distance", np.array(["a", "b", "c", "d"])),
        "quality": (("distance", "time"), np.arange(24.0).reshape(4, 6)),
    }
    return make(coords, dims=("distance", "time"))


_MANAGER_CASES = {
    "patch": _patch_coords,
    # A dimension of length one with no coordinate, as a reduction leaves.
    "flat": lambda make: make({"x": np.arange(5)}, dims=("x", "y"), shape=(5, 1)),
}
_MANAGER_PROPERTIES = (
    "dims",
    "shape",
    "size",
    "scalar",
    "dim_map",
    "coord_shapes",
    "dim_to_coord_map",
)


def _first(manager) -> str:
    """The name of a manager's first dimension."""
    return manager.dims[0]


_MANAGER_CALLS = {
    "properties": lambda m: {x: getattr(m, x) for x in _MANAGER_PROPERTIES},
    "iter": lambda m: dict(iter(m)),
    "getitem": lambda m: m[_first(m)],
    "contains": lambda m: (_first(m) in m, "nothing" in m),
    "get_coord": lambda m: m.get_coord(m.dims[-1]),
    "get_array": lambda m: m.get_array(_first(m)),
    "get_axis": lambda m: m.get_axis(m.dims[-1]),
    "coord_range": lambda m: m.coord_range(_first(m)),
    "coord_size": lambda m: m.coord_size(_first(m)),
    "min_max_step": lambda m: [f(_first(m)) for f in (m.min, m.max, m.step)],
    "tuple_map": lambda m: m.get_coord_tuple_map(),
    "summary": lambda m: m.to_summary_dict(),
    "validate_data": lambda m: m.validate_data(np.zeros(m.shape)),
    "select": lambda m: m.select(**{_first(m): (1, 3)}, array=np.ones(m.shape)),
    "select_samples": lambda m: m.select(**{_first(m): (1, 3)}, samples=True),
    "isel": lambda m: m.isel(**{_first(m): slice(1, 3)}),
    "select_indexers": lambda m: m.select_indexers(**{_first(m): (1, 3)}),
    "update": lambda m: m.update(**{_first(m): m.get_array(_first(m)) * 2}),
    "update_new": lambda m: m.update(new=(_first(m), np.zeros(m.shape[0]))),
    "drop_coords": lambda m: m.drop_coords(m.dims[-1], array=np.ones(m.shape)),
    "drop_private_coords": lambda m: m.drop_private_coords(),
    "drop_associated": lambda m: m.drop_associated(_first(m), None),
    "disassociate_coord": lambda m: m.disassociate_coord(_first(m)),
    "rename_coord": lambda m: m.rename_coord(**{_first(m): "renamed"}),
    "set_dims": lambda m: m.set_dims(distance="latitude"),
    "transpose": lambda m: m.transpose(*m.dims[::-1]),
    "squeeze": lambda m: m.squeeze(),
    "flip": lambda m: m.flip(_first(m)),
    "sort": lambda m: m.sort(_first(m), reverse=True),
    "snap": lambda m: m.snap(_first(m)),
    "decimate": lambda m: m.decimate(**{_first(m): 2}),
    "set_units": lambda m: m.set_units(**{_first(m): "m"}),
    "convert_units": lambda m: m.set_units(**{_first(m): "m"}).convert_units(
        **{_first(m): "ft"}
    ),
    "simplify_units": lambda m: m.simplify_units(),
    "broadcast": lambda m: m.make_broadcastable_to((m.shape[0], 3)),
    "id": lambda m: H("conformance", m),
    "dim_summary": lambda m: m._get_dim_summary(),
    "repr_section": lambda m: str(m._repr_section()),
    "update_grid": lambda m: m._update_grid(_first(m)),
}


class BaseCoordManagerTests:
    """
    The tests a coordinate manager runs.

    A subclass gives a `make_manager` fixture: a function taking the
    arguments of
    [`get_coord_manager`](`dascore.core.coordmanager.get_coord_manager`) and
    returning the package's manager.
    """

    case = _names(_MANAGER_CASES)
    call = _names(_MANAGER_CALLS)

    @pytest.fixture
    def managers(self, make_manager, case):
        """A case built by the implementation, and by DASCore."""
        build = _MANAGER_CASES[case]
        return build(make_manager), build(get_coord_manager)

    def test_kind(self, managers):
        """The manager is a BaseCoordManager."""
        assert isinstance(managers[0], BaseCoordManager)

    def test_call(self, managers, call):
        """A call gives what it gives on DASCore's manager."""
        _check(_MANAGER_CALLS[call], *managers)

    def test_equal(self, make_manager, case):
        """Managers built alike are equal, and others are not."""
        build = _MANAGER_CASES[case]
        first, second = build(make_manager), build(make_manager)
        assert first == second and not first == second.rename_coord(
            **{_first(second): "other"}
        )


def _patch_of(data):
    """A patch holding data of shape (4, 6)."""
    coords = {"x": np.arange(4), "y": np.arange(6)}
    return dc.Patch(data=data, coords=coords, dims=("x", "y"))


class BaseLazyArrayTests:
    """
    The tests a lazy array runs.

    A subclass gives a `make_lazy_array` fixture: a function taking a NumPy
    array and returning the package's lazy array of its values, unread.
    """

    @pytest.fixture
    def array(self):
        """The values of the lazy array the tests make."""
        return np.arange(24.0).reshape(4, 6)

    @pytest.fixture
    def lazy(self, make_lazy_array, array):
        """A lazy array of the values."""
        return make_lazy_array(array)

    @pytest.fixture
    def patch(self, lazy):
        """A patch holding the lazy array."""
        return _patch_of(lazy)

    def test_kind(self, lazy):
        """The array is a BaseLazyArray."""
        assert isinstance(lazy, BaseLazyArray)

    def test_shape_and_dtype(self, lazy, array):
        """The array states the shape and dtype of its values."""
        assert (lazy.shape, lazy.dtype, lazy.ndim) == (array.shape, array.dtype, 2)

    def test_read(self, lazy, array):
        """Reading the array gives its values."""
        _assert_same(np.asarray(lazy), array)

    @pytest.mark.parametrize("query", [{"x": (1, 2)}, {"y": (1, 3), "samples": True}])
    def test_selection_stays_lazy(self, patch, array, query):
        """A selection from a patch holding the array is a lazy array of its values."""
        out = patch.select(**query)
        assert isinstance(out.data, BaseLazyArray)
        _assert_same(out.load().data, _patch_of(array).select(**query).data)

    def test_backends(self, patch, array):
        """The lazy backend keeps the array; one in memory reads it."""
        lazy = patch.set_backend("lazy")
        assert isinstance(lazy.data, BaseLazyArray) and lazy.backend == LazyBackend()
        numpy = patch.set_backend("numpy")
        assert isinstance(numpy.data, np.ndarray)
        _assert_same(numpy.data, array)


_START = np.datetime64("2020-01-03")
_MS = np.timedelta64(1, "ms")
_SPOOL_CALLS = {
    "len": len,
    "patches": list,
    "getitem": lambda s: s[1],
    "slice": lambda s: s[1:],
    "select": lambda s: s.select(time=(_START + 200 * _MS, _START + 1600 * _MS)),
    "select_distance": lambda s: s.select(distance=(2, 5)),
    "chunk": lambda s: s.chunk(time=0.2),
    "chunk_whole": lambda s: s.chunk(time=None),
    "concatenate": lambda s: s.concatenate(time=None),
    "update": lambda s: s.update(),
    "split": lambda s: list(s.split(size=2)),
    "map": lambda s: s.map(lambda patch: patch.attrs.data_id),
    "contents": lambda s: s.get_contents(),
    "gaps": lambda s: s.get_gaps(),
    "coverage": lambda s: s.get_coverage(),
    "patch_names": lambda s: s.get_patch_names(),
    "path_and_backend": lambda s: (s.spool_path, s.backend),
    "set_backend": lambda s: s.set_backend("numpy"),
    "add": lambda s: s + s[1:],
}


class BaseSpoolTests:
    """
    The tests a spool runs.

    A subclass gives a `make_spool` fixture: a function taking what
    [`dascore.spool`](`dascore.spool`) takes and returning the package's
    spool. The tests hand it patches, and a directory of DASDAE files.
    """

    call = _names(_SPOOL_CALLS)

    @pytest.fixture(scope="class")
    @classmethod
    def patches(cls):
        """Three small patches, a second apart."""
        kwargs = {"time_gap": 1, "shape": (10, 100), "time_min": _START}
        return list(dc.get_example_spool("random_das", **kwargs))

    @pytest.fixture(scope="class")
    @classmethod
    def directory(cls, patches, tmp_path_factory):
        """A directory of the patches, each in its own DASDAE file."""
        path = tmp_path_factory.mktemp("spool")
        for number, patch in enumerate(patches):
            dc.write(patch, path / f"patch_{number}.h5", "DASDAE")
        return path

    @pytest.fixture(params=["patches", "directory"])
    def spools(self, request, make_spool):
        """A spool built by the implementation, and by DASCore."""
        source = request.getfixturevalue(request.param)
        return make_spool(source), dc.spool(source)

    def test_kind(self, spools):
        """The spool is a BaseSpool."""
        assert isinstance(spools[0], BaseSpool)

    def test_call(self, spools, call):
        """A call gives what it gives on DASCore's spool."""
        _check(_SPOOL_CALLS[call], *spools)
