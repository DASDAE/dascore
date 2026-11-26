"""Tests for the interfaces another package's classes can stand in for."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import BaseCoord, CoordString, NumericCoord, get_coord
from dascore.core.source import ArraySource
from dascore.core.spool import BaseSpool, Spool
from dascore.lazy.array import LazyArray
from dascore.lazy.recipe import Recipe


@pytest.fixture(scope="module")
def coords():
    """A range, labels and text: one of each shape a coordinate takes."""
    return {
        "range": get_coord(start=0, stop=10, step=1, units="m"),
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
