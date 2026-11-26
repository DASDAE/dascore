"""Tests for the interfaces another package's classes can stand in for."""

from __future__ import annotations

import io
from functools import singledispatch

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import dascore as dc
from dascore.core import interfaces
from dascore.core.coordmanager import (
    BaseCoordManager,
    CoordManager,
    get_coord_manager,
)
from dascore.core.coords import (
    BaseCoord,
    BaseNumericCoord,
    BaseStringCoord,
    CoordString,
    NumericCoord,
    concat_coords,
    get_coord,
)
from dascore.core.source import ArraySource
from dascore.core.spool import BaseSpool, Spool
from dascore.io.dasdae.utils import _save_coord
from dascore.lazy.array import BaseLazyArray, LazyArray
from dascore.lazy.recipe import Recipe
from dascore.utils.identity import H


class _StandIn:
    """Another package's object: plain Python, answering as one of dascore's."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def __getitem__(self, item):
        return self._inner[item]

    def __eq__(self, other):
        return self._inner == getattr(other, "_inner", other)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._inner, dtype=dtype)


@BaseNumericCoord.register
class _NumericCoord(_StandIn):
    """Another package's numeric coordinate."""


@BaseStringCoord.register
class _StringCoord(_StandIn):
    """Another package's text coordinate."""


@BaseCoordManager.register
class _Manager(_StandIn):
    """Another package's coordinate manager."""


@BaseSpool.register
class _Spool(_StandIn):
    """Another package's spool."""


@BaseLazyArray.register
class _LazyArray:
    """Another package's lazy array, counting the times it is read."""

    def __init__(self, array):
        self._array = np.asarray(array)
        self.reads = 0

    shape = property(lambda self: self._array.shape)
    dtype = property(lambda self: self._array.dtype)
    ndim = property(lambda self: self._array.ndim)

    def __getitem__(self, item):
        return _LazyArray(self._array[item])

    def __array__(self, dtype=None, copy=None):
        self.reads += 1
        return self._array


@pytest.fixture(scope="module")
def coords():
    """A range, labels and text: one of each shape a coordinate takes."""
    return {
        # int64 ends, so the pinned dtype and id hold on 32-bit platforms.
        "range": get_coord(
            start=np.int64(0), stop=np.int64(10), step=np.int64(1), units="m"
        ),
        "labels": NumericCoord.from_labels(np.array([0.0, 1.5, 4.0])),
        "text": get_coord(data=np.array(["a", "b"])),
    }


@pytest.fixture(scope="module")
def labelled_patch():
    """The example patch with a text coordinate beside its distances."""
    patch = dc.get_example_patch()
    names = np.array([f"s{x}" for x in range(len(patch.get_coord("distance")))])
    return patch.update_coords(station=("distance", names))


class TestDascoreClassesUnchanged:
    """What the interfaces must not change about dascore's own classes."""

    def test_coordinates_are_their_classes(self, coords):
        """Each coordinate is a BaseCoord of its own class, and only that."""
        assert all(isinstance(x, BaseCoord) for x in coords.values())
        assert isinstance(coords["range"], NumericCoord)
        assert isinstance(coords["labels"], NumericCoord)
        assert isinstance(coords["text"], CoordString)
        assert not isinstance(coords["text"], NumericCoord)
        assert not isinstance(coords["range"], CoordString)
        assert issubclass(NumericCoord, BaseCoord)
        assert issubclass(CoordString, BaseCoord)

    def test_managers_lazy_arrays_and_spools(self, random_patch):
        """A manager, a lazy array and a spool are each their class."""
        assert isinstance(random_patch.coords, CoordManager)
        lazy = LazyArray(Recipe.from_source(ArraySource.full((2, 3), 1.0)))
        assert isinstance(lazy, LazyArray) and not hasattr(lazy, "__dict__")
        spool = dc.spool(random_patch)
        assert isinstance(spool, Spool) and isinstance(spool, BaseSpool)
        assert isinstance(spool, dc.Spool) and isinstance(spool, dc.BaseSpool)
        assert type(spool) is Spool

    def test_names_import_from_where_they_live(self):
        """The package and module names keep naming the same classes."""
        assert dc.CoordManager is CoordManager
        assert dc.core.NumericCoord is NumericCoord
        assert dc.Spool is Spool and dc.BaseSpool is BaseSpool
        assert dc.get_coord is get_coord
        assert dc.get_coord_manager is get_coord_manager

    def test_constructors(self, coords):
        """Each class builds from the arguments it always took."""
        numeric = NumericCoord(runs=coords["range"].runs, units="m", dtype="int64")
        assert numeric == coords["range"]
        text = CoordString(values=np.array(["a", "b"]))
        assert text == coords["text"]
        manager = CoordManager(
            dims=("x",), coord_map={"x": numeric}, dim_map={"x": ("x",)}
        )
        assert manager.dims == ("x",) and manager.get_coord("x") == numeric
        patch = dc.get_example_patch()
        assert len(Spool()) == 0 and len(Spool([patch])) == 1
        assert LazyArray(Recipe.from_source(ArraySource.full((2,), 0.0))).shape == (2,)

    def test_coordinate_dumps(self, coords):
        """A coordinate dumps the fields it always dumped."""
        assert coords["range"].model_dump() == {
            "units": "m",
            "step": 1,
            "shape": (10,),
            "dtype": np.dtype("int64"),
            "runs": (
                {
                    "origin": 0,
                    "step_num": 1,
                    "step_den": 1,
                    "count": 10,
                    "k0": 0,
                    "phase": 0,
                    "parent_count": None,
                    "stride": 1,
                },
            ),
            "sources": {},
        }
        text = coords["text"].model_dump()
        assert list(text) == ["units", "step", "shape", "dtype", "values"]
        assert text["shape"] == (2,) and text["values"].tolist() == ["a", "b"]
        labels = coords["labels"].model_dump()
        assert list(labels) == list(NumericCoord.model_fields)
        assert list(NumericCoord.model_fields) == [
            "units",
            "step",
            "shape",
            "dtype",
            "runs",
            "sources",
        ]
        assert list(CoordManager.model_fields) == [
            "dims",
            "scalar",
            "coord_map",
            "dim_map",
        ]

    def test_ids(self, coords):
        """A coordinate's id is the one it always had."""
        assert coords["range"].data_id == "a35135840e219da406983d6c52b8f0c2"
        assert coords["labels"].data_id == "1847518703da22cf084bc58f9710ddce"
        assert coords["text"].data_id == "86bec1c5bf1c0ed3bbc928779dffa55f"

    def test_coordinates_and_managers_are_not_hashable(self, coords, random_patch):
        """A coordinate or manager is named by its id, never hashed."""
        with pytest.raises(TypeError, match="data_id"):
            hash(coords["range"])
        with pytest.raises(TypeError):
            hash(random_patch.coords)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_dasdae_round_trip(self, labelled_patch, tmp_path, version):
        """Numbers and text come back from DASDAE as they went in."""
        path = tmp_path / "out.h5"
        dc.write(labelled_patch, path, "DASDAE", file_version=version)
        back = dc.spool(path)[0]
        assert back.equals(labelled_patch)
        assert isinstance(back.get_coord("station"), CoordString)
        assert isinstance(back.get_coord("time"), NumericCoord)

    def test_dasdae_nodes_state_their_class(self, labelled_patch, tmp_path):
        """A version 2 node names a range, a numeric array or text."""
        patch = labelled_patch.update_coords(
            distance=np.cumsum(np.arange(len(labelled_patch.get_coord("distance"))))
        )
        path = tmp_path / "out.h5"
        dc.write(patch, path, "DASDAE", file_version="2")
        tags = {}
        with h5py.File(path) as h5:
            h5.visititems(
                lambda x, y: (
                    tags.update({x.rsplit("_", 1)[-1]: y.attrs["object_type"]})
                    if "object_type" in y.attrs
                    else None
                )
            )
        assert tags == {
            "distance": "NumericCoord",
            "station": "CoordString",
            "time": "CoordRange",
        }


class TestInterfaces:
    """dascore's classes are implementations of the interfaces."""

    def test_classes_subclass_their_interfaces(self):
        """Each class is its interface's, and each kind is a coordinate."""
        assert issubclass(NumericCoord, BaseNumericCoord)
        assert issubclass(CoordString, BaseStringCoord)
        assert issubclass(CoordManager, BaseCoordManager)
        assert issubclass(LazyArray, BaseLazyArray)
        assert issubclass(Spool, BaseSpool)
        assert issubclass(BaseNumericCoord, BaseCoord)
        assert issubclass(BaseStringCoord, BaseCoord)
        assert not issubclass(CoordString, BaseNumericCoord)

    def test_names_import_where_the_classes_live(self):
        """Each interface is the one in dascore.core.interfaces."""
        assert BaseCoord is interfaces.BaseCoord
        assert BaseNumericCoord is interfaces.BaseNumericCoord
        assert BaseStringCoord is interfaces.BaseStringCoord
        assert BaseCoordManager is interfaces.BaseCoordManager
        assert BaseLazyArray is interfaces.BaseLazyArray
        assert dc.BaseSpool is BaseSpool is interfaces.BaseSpool

    def test_a_subclass_missing_a_member_is_refused(self):
        """A class built on an interface must give its members."""

        class Partial(BaseSpool):
            """A spool which states none of its members."""

        with pytest.raises(TypeError, match="abstract"):
            Partial()


class TestRegistered:
    """Another package's classes, registered, pass dascore's checks."""

    @pytest.fixture(scope="class")
    def gappy(self):
        """A numeric coordinate of two runs around a gap."""
        first = get_coord(start=0.0, stop=10.0, step=1.0)
        return concat_coords(first, get_coord(start=15.0, stop=25.0, step=1.0))

    @pytest.fixture(scope="class")
    def gappy_patch(self, gappy):
        """A patch whose registered manager holds a registered gapped coordinate."""
        coord = _NumericCoord(gappy)
        manager = CoordManager(
            dims=("x",), coord_map={"x": coord}, dim_map={"x": ("x",)}
        )
        return dc.Patch(data=np.arange(20.0), coords=_Manager(manager))

    def test_each_interface_accepts_a_registered_class(self, coords, random_patch):
        """A registered object is an instance of its interface."""
        assert isinstance(_NumericCoord(coords["range"]), BaseNumericCoord)
        assert isinstance(_StringCoord(coords["text"]), BaseStringCoord)
        assert isinstance(_Manager(random_patch.coords), BaseCoordManager)
        assert isinstance(_LazyArray(np.zeros(2)), BaseLazyArray)
        assert isinstance(_Spool(dc.spool(random_patch)), BaseSpool)

    def test_a_kind_reaches_its_parent(self, coords):
        """A coordinate registered with a kind is a coordinate."""
        numeric, text = _NumericCoord(coords["range"]), _StringCoord(coords["text"])
        assert isinstance(numeric, BaseCoord) and isinstance(text, BaseCoord)
        assert not isinstance(numeric, BaseStringCoord)
        assert not isinstance(text, BaseNumericCoord)

    def test_concrete_classes_mean_dascore_own(self, coords, random_patch):
        """A registered object is not one of dascore's concrete classes."""
        assert not isinstance(_NumericCoord(coords["range"]), NumericCoord)
        assert not isinstance(_StringCoord(coords["text"]), CoordString)
        assert not isinstance(_Manager(random_patch.coords), CoordManager)
        assert not isinstance(_LazyArray(np.zeros(2)), LazyArray)
        assert not isinstance(_Spool(dc.spool(random_patch)), Spool)

    def test_singledispatch_honours_registration(self, coords):
        """A function dispatched on an interface takes a registered class."""

        @singledispatch
        def kind(obj):
            return "other"

        @kind.register
        def _(obj: BaseCoord):
            return "coord"

        assert kind(_NumericCoord(coords["range"])) == "coord"
        assert kind(np.zeros(2)) == "other"

    def test_spool_returns_a_registered_spool(self, random_patch):
        """dc.spool hands back another package's spool as it is."""
        spool = _Spool(dc.spool(random_patch))
        assert dc.spool(spool) is spool

    def test_a_model_field_takes_a_registered_class(self, coords):
        """A model field annotated with an interface takes a registered class."""

        class Holder(BaseModel):
            """A model holding one coordinate."""

            model_config = ConfigDict(arbitrary_types_allowed=True)
            coord: BaseCoord

        numeric = _NumericCoord(coords["range"])
        assert Holder(coord=numeric).coord is numeric
        with pytest.raises(ValidationError):
            Holder(coord=np.zeros(2))

    @pytest.mark.parametrize("name", ["range", "labels", "even", "gappy"])
    def test_coordinates_are_converted(self, coords, gappy, name):
        """get_coord and get_coord_manager rebuild a coordinate as dascore's."""
        even = NumericCoord.from_labels(np.array([0.0, 1.0, 2.0, 3.0]))
        coord = {**coords, "even": even, "gappy": gappy}[name]
        out = get_coord(data=_NumericCoord(coord))
        assert isinstance(out, NumericCoord) and out.data_id == coord.data_id
        assert out.units == coord.units
        managers = [
            get_coord_manager({"x": x}, dims=("x",))
            for x in (_NumericCoord(coord), coord)
        ]
        theirs, ours = (x.coord_map["x"] for x in managers)
        assert isinstance(theirs, NumericCoord) and theirs.data_id == ours.data_id

    def test_numeric_kind_concatenates(self, gappy):
        """Coordinates are joined by the members their kind states."""
        first, second = gappy.segments
        out = concat_coords(_NumericCoord(first), _NumericCoord(second))
        assert out == gappy

    def test_registered_coordinate_keeps_its_gaps(self, gappy_patch):
        """Operations on a patch read a registered coordinate's runs."""
        assert len(gappy_patch.split_gaps()) == 2
        assert gappy_patch.fill_gaps("x").shape == (25,)
        assert gappy_patch.isel(x=slice(2, 18)).get_coord("x").runs_count == 2

    def test_patch_keeps_a_registered_manager(self, random_patch):
        """A patch or its metadata takes its dims from another's manager."""
        manager = _Manager(random_patch.coords)
        patch = dc.Patch(
            data=random_patch.data, coords=manager, attrs=random_patch.attrs
        )
        assert patch.coords is manager and patch.dims == random_patch.dims
        meta = dc.PatchMeta(coords=manager, dtype=random_patch.dtype)
        assert meta.coords is manager
        flipped = _Manager(random_patch.coords.transpose(*reversed(patch.dims)))
        out = random_patch.replace(data=random_patch.data.T, coords=flipped)
        assert out.coords is flipped and out.dims == tuple(reversed(patch.dims))
        assert random_patch.equals(patch) and patch.equals(random_patch)
        empty = _Manager(get_coord_manager())
        assert get_coord_manager(empty, shape=()) is empty

    def test_registered_manager_has_dascore_id(self, random_patch):
        """Another package's manager has the id dascore's has."""
        coords = random_patch.coords
        assert H("test", _Manager(coords)) == H("test", coords)

    def test_lazy_array_read_when_asked(self, random_patch):
        """A selection keeps another package's lazy array unread."""
        lazy = _LazyArray(random_patch.data)
        patch = random_patch.new(data=lazy)
        selections = [
            patch.select(distance=(10, 20), samples=True, copy=True),
            patch.isel(distance=[0, 2, 3], time=[0, 2, 3]),
        ]
        for out in selections:
            assert isinstance(out.data, _LazyArray) and not out.data.reads
            assert isinstance(out.load().data, np.ndarray)
        assert not lazy.reads
        assert patch.set_backend(None).data is lazy
        assert patch.set_backend("lazy").data is lazy
        numpy = patch.set_backend("numpy")
        assert isinstance(numpy.data, np.ndarray) and numpy.equals(random_patch)

    def test_patch_function_returns_a_registered_spool(self, random_patch):
        """A patch function hands back another package's spool as it is."""
        spool = _Spool(dc.spool(random_patch))

        @dc.patch_function()
        def to_spool(patch):
            return spool

        assert to_spool(random_patch) is spool

    def test_registered_spool_scans_and_writes(self, random_patch, tmp_path):
        """A registered spool is scanned and written as any spool."""
        inner = dc.spool(random_patch)
        spool = _Spool(inner)
        assert dc.scan_to_df(spool).equals(inner.get_contents())
        path = dc.write(spool, tmp_path / "out.h5", "DASDAE")
        assert dc.spool(path)[0].equals(random_patch)

    def test_dasdae_tags_coordinates_by_kind(self, coords, gappy):
        """A node names the kind of its coordinate, whichever class holds it."""
        h5 = h5py.File(io.BytesIO(), "w")
        _save_coord(_NumericCoord(gappy), "numeric", h5, compact=True)
        _save_coord(_StringCoord(coords["text"]), "text", h5, compact=True)
        assert h5["numeric"].attrs["object_type"] == "NumericCoord"
        assert h5["text"].attrs["object_type"] == "CoordString"
