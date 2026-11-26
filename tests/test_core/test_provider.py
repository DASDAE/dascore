"""
Tests for building another package's objects through a provider.

`_provider` is for implementers, like the interfaces' underscore members,
so these set it on subclasses as another package does.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import BaseNumericCoord, NumericCoord, get_coord
from dascore.core.provider import (
    Provider,
    Readers,
    active_readers,
    providing,
    reading_with,
)
from dascore.core.source import ArraySource
from dascore.exceptions import PatchCoordinateError
from dascore.io import FiberIO
from dascore.lazy.array import LazyArray, lazy_array
from dascore.lazy.recipe import Recipe


class _Recorder:
    """dascore's own builders, noting each call a provider hands them."""

    def __init__(self):
        self.calls = []
        self.built = []
        # What each manager was built from.
        self.given = []

    def _note(self, name, out):
        """Note a call and what it built."""
        self.calls.append(name)
        self.built.append(out)
        return out

    def get_coord(self, **kwargs):
        """Build a coordinate with dascore's builder."""
        return self._note("get_coord", get_coord(**kwargs))

    def get_coord_manager(self, coords=None, *args, **kwargs):
        """Build a manager with dascore's builder."""
        self.given.append(coords)
        out = get_coord_manager(coords, *args, **kwargs)
        return self._note("get_coord_manager", out)

    def lazy_array(self, recipe):
        """Build a lazy array with dascore's builder."""
        return self._note("lazy_array", lazy_array(recipe))

    @property
    def provider(self):
        """The provider of these builders."""
        return Provider(self.get_coord, self.get_coord_manager, self.lazy_array)


@BaseNumericCoord.register
class _Coord:
    """Another package's numeric coordinate, reading through to dascore's."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __len__(self):
        return len(self._inner)


def _their_coord(**kwargs):
    """Build another package's coordinate, of dascore's labels."""
    return _Coord(get_coord(**kwargs))


@pytest.fixture
def recorder():
    """A provider which notes what it builds."""
    return _Recorder()


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

    def test_subclass_operation_builds_dascore_objects(self):
        """An operation on a subclass stating nothing builds dascore's objects."""

        class Sub(dc.Patch):
            """A patch subclass adding nothing."""

        out = Sub(dc.get_example_patch()).dft("time")
        assert type(out) is Sub and isinstance(out.coords, CoordManager)
        assert isinstance(out.get_coord("ft_time"), NumericCoord)


class TestProviding:
    """The active provider builds what dascore's builders are asked for."""

    def test_provider_builds_inside(self, recorder, recipe):
        """Inside the block, each builder hands its arguments over once."""
        with providing(recorder.provider):
            coord = get_coord(start=0, stop=10, step=1)
            manager = get_coord_manager({"x": coord}, dims=("x",))
            array = lazy_array(recipe)
        # One call each: the provider's fallback to dascore's is not handed back.
        assert recorder.calls == ["get_coord", "get_coord_manager", "lazy_array"]
        assert all(x is y for x, y in zip(recorder.built, (coord, manager, array)))

    def test_none_restores_dascore(self, recorder):
        """A block with no provider inside one with a provider is dascore's."""
        with providing(recorder.provider), providing(None):
            get_coord(start=0, stop=10, step=1)
        assert recorder.calls == []


class TestOperation:
    """An operation builds the objects of its patch's class."""

    @pytest.fixture
    def provided_patch(self, recorder):
        """A patch of a class stating the recording provider."""

        class Provided(dc.Patch):
            _provider = recorder.provider

        return Provided(dc.get_example_patch())

    def test_operation_uses_provider(self, provided_patch, recorder):
        """A transform builds its new coordinates and manager with the provider."""
        recorder.calls.clear()
        out = provided_patch.dft("time")
        assert type(out) is type(provided_patch)
        assert {"get_coord", "get_coord_manager"} <= set(recorder.calls)
        assert out.coords is recorder.built[-1]
        assert not np.any(np.isnan(out.get_array("ft_time")))

    def test_metadata_uses_provider(self, provided_patch, recorder):
        """Operations on a provided patch's metadata build with the provider."""
        distance = provided_patch.get_array("distance")
        meta = provided_patch.drop_data().update_coords(distance=distance * 2)
        recorder.calls.clear()
        out = meta.update_coords(distance=distance * 3)
        assert "get_coord" in recorder.calls
        assert out.coords is recorder.built[-1]

    def test_plain_patch_uses_dascore(self, recorder):
        """An operation on dascore's own patch builds dascore's, a provider active."""
        patch = dc.get_example_patch()
        with providing(recorder.provider):
            out = patch.dft("time")
        assert recorder.calls == []
        assert isinstance(out.get_coord("ft_time"), NumericCoord)

    def test_provider_ends_with_operation(self, provided_patch, recorder):
        """Once the operation returns, or raises, dascore builds its own again."""
        provided_patch.dft("time")
        with pytest.raises(PatchCoordinateError):
            provided_patch.dft("not_a_dim")
        count = len(recorder.calls)
        get_coord(start=0, stop=10, step=1)
        assert len(recorder.calls) == count


class TestConstruction:
    """A patch builds its manager and lazy array with its class's provider."""

    @pytest.fixture
    def classes(self, recorder):
        """A patch class and a metadata class stating the provider."""

        class ProvidedMeta(dc.PatchMeta):
            _provider = recorder.provider

        class Provided(dc.Patch):
            _provider = recorder.provider

        return Provided, ProvidedMeta

    def test_patch(self, classes, parts, recorder):
        """The constructor and `update` build the manager from what they are given."""
        coords, dims = parts
        patch = classes[0](data=np.zeros((3, 4)), coords=coords, dims=dims)
        assert recorder.given[0] is coords and patch.coords is recorder.built[-1]
        recorder.given.clear()
        out = patch.update(coords=coords)
        assert recorder.given[0] is coords and out.coords is recorder.built[-1]

    def test_meta(self, classes, parts, recorder):
        """Metadata builds its manager with the provider too."""
        coords, dims = parts
        meta = classes[1](coords=coords, dims=dims, dtype=np.float64)
        assert recorder.given[0] is coords and meta.coords is recorder.built[-1]
        recorder.given.clear()
        out = meta.update(coords=coords)
        assert recorder.given[0] is coords and out.coords is recorder.built[-1]

    def test_recipe(self, classes, recipe, parts, recorder):
        """A recipe becomes the provider's lazy array."""
        coords, dims = parts
        patch = classes[0](data=recipe, coords=coords, dims=dims)
        assert recorder.calls == ["lazy_array", "get_coord_manager"]
        assert patch.data is recorder.built[0] and patch.shape == (3, 4)

    def test_dascore_patch_inside(self, recipe, parts, recorder):
        """Dascore's own patch and metadata build their own, a provider active."""
        coords, dims = parts
        with providing(recorder.provider):
            patch = dc.Patch(data=recipe, coords=coords, dims=dims)
            out = patch.update(attrs={"station": "x"})
            meta = dc.PatchMeta(coords=coords, dims=dims, dtype=np.float64)
        assert recorder.calls == []
        assert isinstance(patch.coords, CoordManager)
        assert isinstance(out.coords, CoordManager)
        assert isinstance(meta.coords, CoordManager)
        assert type(patch.data) is LazyArray


class TestPartialProvider:
    """A provider replacing only `get_coord` leaves dascore's arithmetic dascore's."""

    @pytest.fixture
    def provider(self):
        """A provider building another package's coordinates, and nothing else."""
        return Provider(_their_coord, get_coord_manager, lazy_array)

    def test_step_update(self, provider):
        """An operation gives a dascore coordinate of the patch a new step."""

        class Provided(dc.Patch):
            _provider = provider

        out = Provided(dc.get_example_patch()).update_coords(distance_step=2)
        assert out.get_coord("distance").step == 2

    def test_limits(self, provider):
        """A new step or new limits of dascore's coordinate are dascore's."""
        coord = get_coord(start=0.0, stop=10.0, step=1.0)
        with providing(provider):
            stepped = coord.update_limits(step=2.0)
            moved = coord.update_limits(min=0.0, max=5.0)
        assert isinstance(stepped, NumericCoord) and stepped.step == 2.0
        assert isinstance(moved, NumericCoord) and moved.max() < 5.0


def _none(*args):
    """Leave it to dascore."""


class _Readers:
    """Readers answering for a file with the metadata dascore read of another."""

    def __init__(self, path):
        name, version = dc.get_format(path)
        fiber_io = dc.io.core.FiberIO.manager.get_fiberio(format=name, version=version)
        self.metadata = fiber_io.get_metadata(path)
        self.format = (name, version)
        self.asked = []

    def get_format(self, manager, *args):
        self.asked.append(("format", manager.source))
        return self.format

    def prepare_read(self, fiber_io, manager, snap, keys):
        self.asked.append(("read", fiber_io.name))

        def load(requests):
            for windows, _ in requests:
                yield np.zeros([stop - start for start, stop in windows])

        return self.metadata, load

    def reads(self, fiber_io, manager):
        return True


class _ScansItself(FiberIO):
    """A reader whose own scan takes a path, as a plugin's may."""

    name = "_test_scans_itself"
    version = "1"

    def get_version(self, resource, **kwargs):
        """Claim no file; the readers name the format."""

    def scan(self, resource, **kwargs):
        """Summarize the example patch, if handed the path."""
        assert isinstance(resource, Path)
        return [dc.get_example_patch().drop_data()]


@pytest.fixture
def dasdae_file(tmp_path):
    """A DASDAE file of the example patch."""
    path = tmp_path / "example.h5"
    dc.write(dc.get_example_patch(), path, "DASDAE")
    return path


class TestReaders:
    """Another package's readers answer for dascore's while active."""

    def test_inactive_outside(self, dasdae_file):
        """Outside the block dascore's readers read."""
        readers = _Readers(dasdae_file)
        with reading_with(Readers(readers.get_format, _none, _none)):
            assert active_readers() is not None
        dc.read(dasdae_file)
        assert readers.asked == [] and active_readers() is None

    def test_get_format(self, dasdae_file):
        """A file's format is the active readers' answer."""
        readers = _Readers(dasdae_file)
        readers.format = ("DASDAE", "answered")
        with reading_with(Readers(readers.get_format, _none, _none)):
            assert dc.get_format(dasdae_file) == ("DASDAE", "answered")
        assert readers.asked == [("format", dasdae_file)]

    def test_read_and_scan(self, dasdae_file):
        """A read loads the readers' arrays and a scan takes their metadata."""
        readers = _Readers(dasdae_file)
        active = Readers(_none, readers.prepare_read, readers.reads)
        with reading_with(active):
            patch = dc.read(dasdae_file)[0]
            summary = dc.scan(dasdae_file)[0]
        assert np.all(patch.data == 0)
        assert readers.asked == [("read", "DASDAE")] * 2
        # What dascore makes of the metadata, the ids included, is its own.
        expected = dc.read(dasdae_file)[0]
        assert patch.coords == expected.coords
        assert patch.attrs.origin_id == expected.attrs.origin_id
        assert summary == dc.scan(dasdae_file)[0]

    def test_unopened(self, dasdae_file, tmp_path):
        """A file the readers read is handed to them unopened."""
        readers = _Readers(dasdae_file)
        # Not an HDF5 file, so it fails wherever dascore opens it.
        path = tmp_path / "unopenable.h5"
        path.write_bytes(b"not a file dascore reads")
        active = Readers(readers.get_format, readers.prepare_read, readers.reads)
        with reading_with(active):
            patch = dc.read(path)[0]
            (summary,) = dc.scan(path)
        assert patch.shape == summary.shape == readers.metadata[0].shape

    def test_own_scan_gets_its_resource(self, dasdae_file):
        """A reader with a scan of its own is handed its resource, as without."""
        answer = ("_test_scans_itself", "1")
        active = Readers(lambda *args: answer, _none, lambda *args: True)
        with reading_with(active):
            (summary,) = dc.scan(dasdae_file, on_error="raise")
        assert summary.shape == dc.get_example_patch().shape

    def test_left_to_dascore(self, dasdae_file):
        """A file the readers say they read but then leave is dascore's."""
        nothing = Readers(_none, _none, lambda *args: True)
        with reading_with(nothing):
            patch = dc.read(dasdae_file)[0]
            summary = dc.scan(dasdae_file)[0]
        assert patch == dc.read(dasdae_file)[0]
        assert summary == dc.scan(dasdae_file)[0]

    def test_directory_scan(self, tmp_path):
        """A directory's scan, readers active which read nothing, is dascore's."""
        nothing = Readers(_none, _none, _none)
        spool = dc.get_example_spool("random_das")
        path = dc.examples.spool_to_directory(spool, tmp_path / "dir")
        with reading_with(nothing):
            summaries = dc.scan(path)
        assert summaries == dc.scan(path)
